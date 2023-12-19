# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

from dataclasses import dataclass
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

from . import llama_attention, sparse_attention, utility


class LowRank(nn.Module):
    """Use a random orthonormal projection to down-project Q & K."""

    @dataclass
    class Settings:
        rank: int
        name: str = "low_rank"

    def __init__(self, settings: Settings, n_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.weight = nn.Parameter(torch.empty(n_heads, head_size, settings.rank))
        for i in range(n_heads):  # can't batch this!
            nn.init.orthogonal_(self.weight[i])  # type:ignore[no-untyped-call]

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        """Compute approximate score for each (query, key).

        query -- (batch, n_heads, query, head_size)

        key -- (batch, n_heads, key, head_size)

        returns -- (batch, n_heads, query, key)
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

        query -- (batch, n_heads, 1, head_size)

        key -- (batch, n_heads, key, head_size)

        returns -- (batch, n_heads, 1, key)
        """
        assert query.shape[-2] == 1, "no support for multiple queries"
        head_size = query.shape[-1]
        topk = query.abs().topk(dim=-1, k=self.settings.rank)
        query_proj = query.gather(-1, topk.indices)
        key_proj = key.gather(
            -1, topk.indices.expand(key.shape[:-1] + (self.settings.rank,))
        )
        # Scale could be:
        #  - sqrt(head_size) -- if we think our approximation is exact
        #  - sqrt(rank)      -- if our approximation is no better than random
        #  - sqrt(q_coverage * head_size) -- used below
        #       q_coverage estimates the variance of Q K^T from the approximated
        #       product, and the L1 coverage of Q by the topk components
        scale = (
            topk.values.sum(-1)
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


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    """A broadcasting version of torch.gather."""
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


class AnnAttention(nn.Module):
    """Generic ANN with local windowing and masking."""

    def __init__(self, settings: Settings, n_heads: int, head_size: int):
        super().__init__()
        self.settings = settings
        self.score: nn.Module
        if isinstance(settings.score, LowRank.Settings):
            self.score = LowRank(settings.score, n_heads, head_size)
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

        query -- (batch, n_heads, n_query, head_size)

        key -- (batch, n_heads, n_kv, head_size)

        value -- (batch, n_heads, n_kv, head_size)

        logmask -- (batch, n_heads, n_query, n_kv)

        kv_weight -- (batch, n_heads, n_query) | ()
                  -- 1.0 for regular attention (no reallocation)

        mean_value -- (batch, n_heads, n_query, head_size)
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

        key -- (batch, n_heads, seq, head_size)

        value -- (batch, n_heads, seq, head_size)

        logmask -- (batch, n_heads, 1, seq)

        returns -- (output, weights)
                   output -- (batch, n_heads, 1, head_size)
                   weights -- (batch, n_heads, 1, seq)
        """
        batch, n_heads, seq, head_size = key.shape
        assert query.shape == (batch, n_heads, 1, head_size)
        assert value.shape == (batch, n_heads, seq, head_size)
        assert logmask.shape == (batch, n_heads, 1, seq)

        # Calculate an approximate score for each (query, key) pair
        score = (self.score(query, key) + logmask).float()

        # Set the score of local keys (+1 current) to max
        causal_index = sparse_attention.causal_index(logmask[:, :, -1, :])
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k + 1)
        topk_score = score.masked_fill(
            is_local[:, :, None, :], torch.finfo(score.dtype).max
        )
        # Find max-score keys (note: +1 because the current token's k comes "for free")
        indices = topk_score.topk(min(self.settings.k + 1, score.shape[-1]), -1).indices
        if self.debug_indices is not None:
            self.debug_indices.append(indices)

        # Optional "mean_value" kv
        value_mask = logmask.squeeze(-2).unsqueeze(-1).exp()
        mean_value = ((value * value_mask).sum(-2) / value_mask.sum(-2)).unsqueeze(-2)
        kv_weight = torch.tensor(1.0)
        if self.settings.reallocate_to_mean_value:
            kv_weight = (
                gather(torch.softmax(score, -1), -1, indices).sum(-1).to(value.dtype)
            )

        # Slice key, value, logmask for attention
        kv_indices = indices.squeeze(-2).unsqueeze(-1)
        output, weights = self._attention(
            query,
            gather(key, -2, kv_indices),
            gather(value, -2, kv_indices),
            gather(logmask, -1, indices),
            kv_weight=kv_weight,
            mean_value=mean_value,
        )
        return output, torch.zeros_like(logmask).scatter(-1, indices, weights)


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


def convert(
    model: Union[GPTNeoXForCausalLM, LlamaForCausalLM], settings: Settings
) -> Union[GPTNeoXForCausalLM, LlamaForCausalLM]:
    """Convert a model to use KV cache compression using ANN."""

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXAttentionWithANN(model.config, settings)
        if isinstance(m, LlamaAttention):
            return LlamaAttentionWithANN(model.config, settings)

    return utility.convert_module(model, _replace)
