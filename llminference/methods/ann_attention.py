# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
)

from .. import utility
from ..models import llama_attention, mistral_attention
from . import sparse_attention


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    """A broadcasting version of torch.gather."""
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


class LowRank(nn.Module):
    """Use a random orthonormal projection to down-project Q & K."""

    @dataclass
    class Settings:
        rank: int
        name: str = "low_rank"

    def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.weight = nn.Parameter(torch.empty(n_kv_heads, 1, head_size, settings.rank))
        for i in range(n_kv_heads):  # can't batch this!
            nn.init.orthogonal_(self.weight[i, 0])  # type:ignore[no-untyped-call]

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_kv_heads, n_heads_per_kv, query, head_size)

        key -- (batch, n_kv_heads, 1, key, head_size)

        returns -- (batch, n_kv_heads, n_heads_per_kv, query, key)
        """
        head_size = query.shape[-1]
        query_proj = query.to(self.weight.dtype) @ self.weight
        key_proj = key.to(self.weight.dtype) @ self.weight
        return cast(Tensor, query_proj @ key_proj.transpose(-1, -2) * head_size**-0.5)


class SparseQ(nn.Module):
    """Gather the top (absolute) components of Q from Q & K."""

    @dataclass
    class Settings:
        rank: int
        name: str = "sparse_q"

    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_kv_heads, n_heads_per_kv, 1, head_size)

        key -- (batch, n_kv_heads, 1, key, head_size)

        returns -- (batch, n_kv_heads, n_heads_per_kv, 1, key)
        """
        assert query.shape[-2] == 1, "no support for multiple queries"
        head_size = query.shape[-1]

        # Sum the magnitudes within KV groups before top-k
        # shape -- (batch, n_kv_heads, 1, 1, rank)
        topk = query.abs().sum(dim=2, keepdim=True).topk(dim=-1, k=self.settings.rank)

        query_proj = gather(query, -1, topk.indices)
        key_proj = gather(key, -1, topk.indices)

        # Scale could be:
        #  - sqrt(head_size) -- if we think our approximation is exact
        #  - sqrt(rank)      -- if our approximation is no better than random
        #  - sqrt(q_coverage * head_size) -- used below
        #       q_coverage estimates the variance of Q K^T from the approximated
        #       product, and the L1 coverage of Q by the topk components
        scale = (
            query_proj.abs()
            .sum(-1)
            .div_(query.abs().sum(-1))
            .mul_(head_size)
            .pow_(0.5)
            .unsqueeze(-1)
        )
        return (query_proj @ key_proj.transpose(-1, -2)).div_(scale)


ScoreSettings = Union[LowRank.Settings, SparseQ.Settings]


@dataclass
class Settings:
    k: int
    local_k: int
    reallocate_to_mean_value: bool
    score: ScoreSettings

    def __init__(
        self,
        k: int,
        local_k: int,
        reallocate_to_mean_value: bool,
        score: Union[ScoreSettings, str],
        **args: Any,
    ):
        if isinstance(score, str):
            ctor: Any = dict(low_rank=LowRank.Settings, sparse_q=SparseQ.Settings)[
                score
            ]
            score_settings: ScoreSettings = ctor(**args)
        else:
            assert (
                not args
            ), "ann_attention.Setting only accepts **args when `score` is a string"
            score_settings = score
        self.k = k
        self.local_k = local_k
        self.reallocate_to_mean_value = reallocate_to_mean_value
        self.score = score_settings


class AnnAttention(nn.Module):
    """Generic ANN with local windowing and masking."""

    def __init__(self, settings: Settings, n_kv_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.score: nn.Module
        if isinstance(settings.score, LowRank.Settings):
            self.score = LowRank(settings.score, n_kv_heads, head_size)
        elif isinstance(settings.score, SparseQ.Settings):
            self.score = SparseQ(settings.score)
        else:
            raise ValueError(f"Unexpected settings.score = {settings.score}")
        # Set to an empty list to turn on ANN index logging
        self.debug_indices: Optional[List[Tensor]] = None

    def _attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        logmask: Tensor,
        kv_weight: Tensor,
        mean_value: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Dense attention, with left-over weight reallocation.

        query -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)

        key -- (batch, n_kv_heads, 1, n_kv, head_size)

        value -- (batch, n_kv_heads, 1, n_heads, n_kv, head_size)

        logmask -- (batch, n_kv_heads, n_heads_per_kv, n_query, n_kv)

        kv_weight -- (batch, n_kv_heads, n_heads_per_kv, n_query) | ()
                  -- 1.0 for regular attention (no reallocation)

        mean_value -- (batch, n_kv_heads, n_heads_per_kv, n_query, head_size)
        """
        scores = (
            (query @ key.transpose(-1, -2)).div_(query.shape[-1] ** 0.5).add_(logmask)
        )
        weights = torch.softmax(scores, -1, dtype=torch.float32).to(value.dtype)
        # Value-mixing with reallocation
        weights *= kv_weight[..., None]
        output = weights @ value
        output += (1 - kv_weight[..., None]) * mean_value
        return output, weights

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Preprocess (key, value, mask) for ANN attention.

        query -- (batch, n_heads, 1, head_size)

        key -- (batch, n_kv_heads, seq, head_size)

        value -- (batch, n_kv_heads, seq, head_size)

        logmask -- (batch, n_heads, 1, seq)

        returns -- (output, weights)
                   output -- (batch, n_heads, 1, head_size)
                   weights -- (batch, n_heads, 1, seq)
        """
        batch, n_kv_heads, seq, head_size = key.shape
        n_heads_per_kv = query.shape[1] // n_kv_heads

        # Group by KV head
        query, key, value, logmask = map(
            partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
            [query, key, value, logmask],
        )

        assert query.shape == (batch, n_kv_heads, n_heads_per_kv, 1, head_size)
        assert key.shape == (batch, n_kv_heads, 1, seq, head_size)
        assert value.shape == (batch, n_kv_heads, 1, seq, head_size)
        assert logmask.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq)

        # Calculate an approximate score for each (query, key) pair
        # shape -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)
        score = (self.score(query, key) + logmask).float()

        # Set the score of local keys (+1 current) to max
        causal_index = sparse_attention.causal_index(logmask)
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k + 1)
        topk_score = score.masked_fill(is_local, torch.finfo(score.dtype).max).sum(
            dim=2, keepdim=True
        )
        # Find max-score keys (note: +1 because the current token's k comes "for free")
        indices = topk_score.topk(
            min(self.settings.k + 1, score.shape[-1]), -1
        ).indices  # (batch, n_kv_heads, 1, 1, k+1)
        if self.debug_indices is not None:
            self.debug_indices.append(indices)

        # Optional "mean_value" kv
        # Note: assumes same logmask for all heads
        value_mask = (
            logmask[:, :, :1].squeeze(-2).unsqueeze(-1).exp()
        )  # (batch, n_kv_heads, 1, seq, 1)
        mean_value = ((value * value_mask).sum(-2) / value_mask.sum(-2)).unsqueeze(
            -2
        )  # (batch, n_kv_heads, 1, 1, 1)
        kv_weight = torch.tensor(1.0, device=query.device)
        if self.settings.reallocate_to_mean_value:
            kv_weight = (
                gather(torch.softmax(score, -1), -1, indices)  # no need to expand here
                .sum(-1)
                .to(value.dtype)
            )  # (batch, n_kv_heads, n_heads_per_kv, 1)

        # Slice key, value, logmask for attention
        kv_indices = indices.squeeze(-2).unsqueeze(-1)  # (batch, n_kv_heads, 1, k+1, 1)
        output, weights = self._attention(
            query,
            gather(key, -2, kv_indices),
            gather(value, -2, kv_indices),
            gather(logmask, -1, indices),
            kv_weight=kv_weight,
            mean_value=mean_value,
        )
        # Note: expand indices as scatter does not broadcast (!)
        return output.flatten(1, 2), torch.zeros_like(logmask).scatter(
            -1, indices.expand_as(weights), weights
        ).flatten(1, 2)


Model = Union[GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM]


class GPTNeoXAttentionWithANN(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.ann = AnnAttention(settings, self.num_attention_heads, self.head_size)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert attention_mask is not None
        assert head_mask is None

        # Only enable ANN during autoregressive generation
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.broadcast_to(key.unsqueeze(-3).shape[:-1]),
            )

        return super()._attn(  # type:ignore[no-any-return]
            query, key, value, attention_mask, head_mask
        )


class LlamaAttentionWithANN(llama_attention.LlamaAttention):
    def __init__(self, config: LlamaConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.settings = settings
        self.ann = AnnAttention(settings, self.num_heads, self.head_dim)

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query, key, value, logmask.broadcast_to(key.unsqueeze(-3).shape[:-1])
            )
        return super()._attn(query, key, value, logmask)


class MistralAttentionWithANN(mistral_attention.MistralAttention):
    def __init__(self, config: MistralConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.settings = settings
        self.ann = AnnAttention(
            settings,
            self.num_key_value_heads,
            self.head_dim,
        )

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query,
                key,
                value,
                # reshape to (batch, n_heads, 1, seq_len)
                logmask.broadcast_to(*query.shape[:-1], key.shape[-2]),
            )
        return super()._attn(query, key, value, logmask)


def convert(model: Model, settings: Settings) -> Model:
    """Convert a model to use KV cache compression using ANN."""

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXAttentionWithANN(model.config, settings)
        if isinstance(m, LlamaAttention):
            return LlamaAttentionWithANN(model.config, settings)
        if isinstance(m, MistralAttention):
            return MistralAttentionWithANN(model.config, settings)

    return utility.convert_module(model, _replace)
