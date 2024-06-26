# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Implements KV cache eviction schemes.

Note that these schemes only evict KV entries between model.forward() calls,
never within a call.

Rough usage:

    eviction_model = convert_gptneox(model, Settings(k=128, local_k=64))
    prefill = eviction_model(...)
    with generation_context(eviction_model):
        gen = eviction_model(..., past_key_values=prefill.past_key_values)

See: H20 (https://arxiv.org/abs/2306.14048)
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma.configuration_gemma import GemmaConfig
from transformers.models.gemma.modeling_gemma import GemmaAttention, GemmaForCausalLM
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
from ..models import gemma_attention, llama_attention, mistral_attention
from . import sparse_attention


@dataclass
class Settings:
    k: int
    local_k: int
    strategy: str  # "sum_weight|lru"


class SumWeight:
    def __init__(self, shape: Tuple[int, ...], device: torch.device):
        self.score = torch.zeros(shape, device=device)

    def update(self, weight: Tensor) -> None:
        # Update the score of each KV (summed over Q)
        self.score[..., : weight.shape[-1]] += weight.sum(-2)


class LRU:
    def __init__(self, shape: Tuple[int, ...], device: torch.device):
        # Store timestamps as float for ease-of-conversion to scores
        self.last_used = torch.zeros(shape, device=device)
        # Note: range [1, N], so that 'use' at timestep 0 is better than 'never used'
        self._t = 1 + torch.arange(shape[-1], device=device, dtype=torch.float32)

    @property
    def score(self) -> Tensor:
        return self.last_used

    def update(self, weight: Tensor) -> None:
        _, _, query_length, key_length = weight.shape

        # Compute a mask of 'use' for each key (weight >= 1/sequence_length)
        average_weight = (
            (weight > 1e-9).sum(-1, keepdim=True, dtype=weight.dtype).reciprocal_()
        )
        used = (weight >= average_weight).float()

        # Update the timestamp for the most recent 'use' of each key
        used.mul_(self._t[key_length - query_length : key_length, None])
        self.last_used[..., :key_length] = torch.maximum(
            self.last_used[..., :key_length], used.max(dim=-2).values
        )


class EvictionMask:
    """Maintain a KV cache eviction mask for one attention layer & context.

    Note: `shape` should be `(batch_size, kv_heads, max_k_length)`

    Use `eviction.mask` to get the boolean mask representing token positions
    that have been evicted ('1' for retained, '0' for evicted KVs).
    """

    def __init__(
        self,
        settings: Settings,
        shape: Tuple[int, ...],
        device: torch.device,
    ):
        self.settings = settings
        self.strategy: Union[SumWeight, LRU]
        if settings.strategy == "sum_weight":
            self.strategy = SumWeight(shape, device)
        elif settings.strategy == "lru":
            self.strategy = LRU(shape, device)
        else:
            raise ValueError(f"Unexpected eviction strategy {settings.strategy}")
        self.mask = torch.ones(shape, dtype=torch.bool, device=device)
        self._last_length = 0

    def update_mask(self, causal_index: Tensor) -> None:
        """Update the eviction mask.

        causal_index: shape (batch, kv_head, key), -1 for masked-out tokens
        """
        if self._last_length > causal_index.shape[-1]:
            raise ValueError(
                "An eviction mask is being updated with a shorter context."
                " Please use `eviction_attention.generation_context` during"
                " generation to ensure the eviction mask is reset."
            )
        finfo = torch.finfo(torch.float32)
        key_length = causal_index.shape[-1]

        # Combine locality and permadeath into score
        # (note: k+1 as the current token is "for free")
        score = self.strategy.score[:, :, :key_length].clone()
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k + 1)
        score.masked_fill_(is_local, finfo.max)  # local KVs
        score.masked_fill_(~self.mask[..., :key_length], finfo.min)  # dead KVs

        # Update the mask
        self.mask[..., :key_length] &= sparse_attention.topk_mask(
            score, min(key_length, self.settings.k + 1)
        )

    def update_scores(self, attention_weight: Tensor) -> None:
        """Update eviction scores, from a step's attention weight matrix.

        attention_weight: shape (batch, kv_head, query, key)
        """
        # Update the score of each KV (summed over Q)
        self.strategy.update(attention_weight)
        self._last_length = attention_weight.shape[-1]


class Eviction:
    """Create/update/reset eviction mask state for a single attention layer."""

    def __init__(self, settings: Settings, max_sequence_length: int):
        self.settings = settings
        self.max_sequence_length = max_sequence_length
        self.enabled = False
        self.eviction_mask: Optional[EvictionMask] = None
        # Set to an empty list to turn on eviction mask logging
        self.debug_masks: Optional[List[Tensor]] = None

    def enable(self, enabled: bool) -> None:
        self.enabled = enabled
        if not self.enabled:
            self.eviction_mask = None

    def update_mask(self, attention_mask: Tensor) -> Tensor:
        # Disabled or non-causal => no masking
        if (
            not self.enabled
            or self.eviction_mask is None
            or attention_mask.shape[-2] != 1
        ):
            return attention_mask

        # Assume attention_mask is the same across heads
        self.eviction_mask.update_mask(
            sparse_attention.causal_index(attention_mask[:, :1, 0, :]),
        )
        num_kv_heads = self.eviction_mask.mask.shape[1]
        # Copy mask for heads within the same KV group
        eviction_mask = (
            self.eviction_mask.mask[..., None, None, : attention_mask.shape[-1]]
            .expand_as(torch.unflatten(attention_mask, dim=1, sizes=(num_kv_heads, -1)))
            .flatten(1, 2)
        )
        if self.debug_masks is not None:
            self.debug_masks.append(eviction_mask.clone())
        # Apply the mask to remove previously evicted values
        return attention_mask + torch.finfo(attention_mask.dtype).min * ~eviction_mask

    def update(self, weights: Tensor, attention_mask: Tensor) -> None:
        # When disabled or missing, reset eviction statistics
        if self.eviction_mask is None or not self.enabled:
            self.eviction_mask = EvictionMask(
                self.settings,
                weights.shape[:2] + (self.max_sequence_length,),
                weights.device,
            )
        # Most of the code doesn't care about "junk" queries, but it could confuse
        # eviction models, so we mask them out here, assuming that the last N values
        # correspond to the last N queries
        query_mask = sparse_attention.score_to_mask(
            attention_mask[..., -weights.shape[2] :]
        ).swapdims(-1, -2)
        self.eviction_mask.update_scores(weights * query_mask)


Model = Union[
    GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM
]


def get_max_sequence_length(config: PretrainedConfig) -> int:
    try:
        return cast(int, config.max_sequence_length)
    except AttributeError:
        return cast(int, config.max_position_embeddings)


class GPTNeoXAttentionWithEviction(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.eviction = Eviction(settings, get_max_sequence_length(config))

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert attention_mask is not None
        output, weights = super()._attn(
            query,
            key,
            value,
            self.eviction.update_mask(
                attention_mask.expand(*query.shape[:-1], key.shape[-2])
            ),
            head_mask,
        )
        self.eviction.update(weights, attention_mask)
        return output, weights


class LlamaAttentionWithEviction(llama_attention.LlamaAttention):
    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.eviction = Eviction(settings, get_max_sequence_length(config))

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output, weights = super()._attn(
            query,
            key,
            value,
            self.eviction.update_mask(
                attention_mask.expand(*query.shape[:-1], key.shape[-2])
            ),
        )

        self.eviction.update(
            torch.unflatten(weights, dim=1, sizes=(self.num_key_value_heads, -1)).mean(
                dim=2
            ),
            attention_mask[:, :, -1:, :],
        )
        return output, weights


class GemmaAttentionWithEviction(gemma_attention.GemmaAttention):
    def __init__(
        self, config: GemmaConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.eviction = Eviction(settings, get_max_sequence_length(config))

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        output, weights = super()._attn(
            query,
            key,
            value,
            self.eviction.update_mask(
                attention_mask.expand(*query.shape[:-1], key.shape[-2])
            ),
        )

        self.eviction.update(
            torch.unflatten(weights, dim=1, sizes=(self.num_key_value_heads, -1)).mean(
                dim=2
            ),
            attention_mask[:, :, -1:, :],
        )
        return output, weights


class MistralAttentionWithEviction(mistral_attention.MistralAttention):
    def __init__(
        self, config: MistralConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.eviction = Eviction(settings, get_max_sequence_length(config))

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        output, weights = super()._attn(
            query,
            key,
            value,
            self.eviction.update_mask(
                attention_mask.expand(*query.shape[:-1], key.shape[-2])
            ),
        )
        self.eviction.update(
            torch.unflatten(weights, dim=1, sizes=(self.num_key_value_heads, -1)).mean(
                dim=2
            ),
            attention_mask[:, :, -1:, :],
        )
        return output, weights


@contextmanager
def generation_context(model: Model) -> Iterator[Model]:
    """(Context manager) enable KV eviction during this scope."""
    attns = [
        m
        for m in model.modules()
        if isinstance(
            m,
            (
                GPTNeoXAttentionWithEviction,
                LlamaAttentionWithEviction,
                MistralAttentionWithEviction,
                GemmaAttentionWithEviction,
            ),
        )
    ]
    for m in attns:
        m.eviction.enable(True)
    yield model
    for m in attns:
        m.eviction.enable(False)


def convert(model: Model, settings: Settings) -> Model:
    """Convert the model to use (simulated) KV cache eviction during generation."""

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXAttentionWithEviction(model.config, settings)
        if isinstance(m, LlamaAttention):
            return LlamaAttentionWithEviction(model.config, m.layer_idx, settings)
        if isinstance(m, MistralAttention):
            return MistralAttentionWithEviction(model.config, m.layer_idx, settings)
        if isinstance(m, GemmaAttention):
            return GemmaAttentionWithEviction(model.config, m.layer_idx, settings)

    model = utility.convert_module(model, _replace)
    model.generation_context = generation_context  # type:ignore[assignment]
    return model
