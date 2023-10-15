"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""

import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
)

from . import sparse_attention


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
        query_proj = query.to(self.weight.dtype) @ self.weight
        key_proj = key.to(self.weight.dtype) @ self.weight
        score: Tensor = query_proj @ key_proj.transpose(-1, -2)
        return score


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
        components = query.abs().topk(dim=-1, k=self.settings.rank).indices
        query_proj = query.gather(-1, components)
        key_proj = key.gather(
            -1, components.expand(key.shape[:-1] + (self.settings.rank,))
        )
        return query_proj @ key_proj.transpose(-1, -2)


ScoreSettings = Union[LowRank.Settings, SparseQ.Settings]


@dataclass
class Settings:
    k: int
    local_k: int
    score: ScoreSettings


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

    def forward(self, query: Tensor, key: Tensor, logmask: Tensor) -> Tensor:
        """Compute an attention mask for ANN attention.

        query -- (batch, n_heads, 1, head_size)

        key -- (batch, n_heads, key, head_size)

        logmask -- (batch, n_heads, 1, key)

        returns -- bool(batch, n_heads, 1, key) -- true for unmasked
        """
        # Calculate an approximate score for each (query, key) pair
        score = self.score(query, key) + logmask
        # Set the score of local keys to max
        causal_index = sparse_attention.causal_index(logmask[:, :, -1, :])
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k)
        score.masked_fill_(is_local[:, :, None, :], torch.finfo(score.dtype).max)
        # Mask to select max-score keys
        return sparse_attention.topk_mask(
            score, k=self.settings.k
        ) & sparse_attention.score_to_mask(logmask)


class GPTNeoXAttentionWithANN(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        super().__init__(config)
        self.ann = ANN(settings, self.num_attention_heads, self.head_size)
        # Set to an empty list to turn on ANN mask logging
        self.ann_masks: Optional[List[Tensor]] = None

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
            ann_mask = self.ann(query, key, attention_mask)
            if self.ann_masks is not None:
                self.ann_masks.append(ann_mask)
            attention_mask = torch.finfo(attention_mask.dtype).min * ~ann_mask

        return super()._attn(  # type:ignore[no-any-return]
            query, key, value, attention_mask, head_mask
        )


def convert_gptneox(
    model: GPTNeoXForCausalLM, settings: Settings
) -> GPTNeoXForCausalLM:
    """Convert a GPT-NeoX model to use (simulated) KV cache compression using ANN."""

    def _convert(m: nn.Module, **args: Any) -> None:
        for name, child in m.named_children():
            if isinstance(child, GPTNeoXAttention):
                replacement = GPTNeoXAttentionWithANN(**args)
                replacement.to(next(child.parameters()).dtype)
                replacement.load_state_dict(child.state_dict(), strict=False)
                setattr(m, name, replacement)
            _convert(child, **args)

    model = copy.deepcopy(model)
    _convert(model, config=model.config, settings=settings)
    return model
