"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

import copy
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

from . import llama_attention, sparse_attention


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
        components = query.abs().topk(dim=-1, k=self.settings.rank).indices
        query_proj = query.gather(-1, components)
        key_proj = key.gather(
            -1, components.expand(key.shape[:-1] + (self.settings.rank,))
        )
        return cast(Tensor, query_proj @ key_proj.transpose(-1, -2) * head_size**-0.5)


ScoreSettings = Union[LowRank.Settings, SparseQ.Settings]


@dataclass
class Settings:
    k: int
    local_k: int
    add_remainder: bool
    score: ScoreSettings

    def __init__(
        self,
        k: int,
        local_k: int,
        add_remainder: bool,
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
        self.add_remainder = add_remainder
        self.score = score_settings


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    """A broadcasting version of torch.gather."""
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


class ANN(nn.Module):
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
        self.log_indices: Optional[List[Tensor]] = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Preprocess (key, value, mask) for ANN attention.

        query -- (batch, n_heads, 1, head_size)

        key -- (batch, n_heads, seq, head_size)

        value -- (batch, n_heads, seq, head_size)

        logmask -- (batch, n_heads, 1, seq)

        returns -- (query, key, value, logmask)
        """
        batch, n_heads, seq, head_size = key.shape
        assert query.shape == (batch, n_heads, 1, head_size)
        assert value.shape == (batch, n_heads, seq, head_size)
        assert logmask.shape == (batch, n_heads, 1, seq)

        # Calculate an approximate score for each (query, key) pair
        score = (self.score(query, key) + logmask).float()

        # Set the score of local keys to max
        causal_index = sparse_attention.causal_index(logmask[:, :, -1, :])
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k)
        topk_score = score.masked_fill(
            is_local[:, :, None, :], torch.finfo(score.dtype).max
        )
        # Find max-score keys
        indices = topk_score.topk(min(self.settings.k, score.shape[-1]), -1).indices
        if self.log_indices is not None:
            self.log_indices.append(indices)

        # Slice key, value, logmask
        kv_indices = indices.squeeze(-2).unsqueeze(-1)
        key = gather(key, -2, kv_indices)
        mean_value = value.mean(-2, keepdim=True)  # before gather
        value = gather(value, -2, kv_indices)
        logmask = gather(logmask, -1, indices)

        # Optional "remainder" kv
        if self.settings.add_remainder:
            key = torch.cat([torch.zeros_like(mean_value), key], -2)
            value = torch.cat([mean_value, value], -2)
            norm = torch.logsumexp(score, -1, keepdim=True)
            remainder = (
                1 - gather((score - norm).exp(), -1, indices).sum(-1, keepdim=True)
            ).log() + norm
            logmask = torch.cat([remainder.to(logmask.dtype), logmask], -1)

        return query, key, value, logmask


class GPTNeoXAttentionWithANN(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        super().__init__(config)
        self.ann = ANN(settings, self.num_attention_heads, self.head_size)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert attention_mask is not None

        # Only enable ANN during autoregressive generation
        if query.shape[-2] == 1:
            query, key, value, attention_mask = self.ann(
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
        super().__init__(config)
        self.settings = settings
        self.ann = ANN(settings, self.num_heads, self.head_dim)

    def _process_qkv(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        if query.shape[-2] == 1:
            return self.ann(  # type:ignore[no-any-return]
                query, key, value, logmask.broadcast_to(key.unsqueeze(-3).shape[:-1])
            )
        return query, key, value, logmask


def convert_module(
    model: nn.Module, replace: Callable[[nn.Module], Optional[nn.Module]]
) -> nn.Module:
    """Generic recursive module conversion."""

    def _convert(original: nn.Module) -> nn.Module:
        replacement = replace(original)
        if replacement is not None:
            replacement.to(next(original.parameters()).dtype)
            replacement.load_state_dict(original.state_dict(), strict=False)
            return replacement

        # Recursive (lazy) copy
        result = original
        for name, child in original.named_children():
            replacement = _convert(child)
            if replacement is not child:
                if result is original:
                    result = copy.copy(original)
                setattr(result, name, replacement)
        return result

    return _convert(model)


def convert(
    model: Union[GPTNeoXForCausalLM, LlamaForCausalLM], settings: Settings
) -> Union[GPTNeoXForCausalLM, LlamaForCausalLM]:
    """Convert a model to use KV cache compression using ANN."""

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXAttentionWithANN(model.config, settings)
        if isinstance(m, LlamaAttention):
            return LlamaAttentionWithANN(model.config, settings)

    return convert_module(model, _replace)
