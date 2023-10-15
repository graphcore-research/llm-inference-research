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

import copy
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple, Union

import torch
import transformers
from torch import Tensor, nn
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXForCausalLM,
)

from . import sparse_attention


@dataclass
class Settings:
    k: int
    local_k: int
    strategy: str  # "sum_weight|lru"


class SumWeight:
    def __init__(self, shape: Tuple[int, ...], device: torch.device):
        self.score = torch.zeros(shape, device=device)

    def update(self, weight: Tensor) -> Tensor:
        # Update the score of each KV (summed over Q)
        key_length = weight.shape[-1]
        self.score[..., : weight.shape[-1]] += weight.sum(-2)
        return self.score[..., :key_length].clone()


class LRU:
    def __init__(self, shape: Tuple[int, ...], device: torch.device):
        self.last_used = torch.zeros(shape, device=device, dtype=torch.long)
        # Note: 1-N, so that 'use' at timestep 0 is better than 'never used'
        self._t = 1 + torch.arange(shape[-1], device=device)

    def update(self, weight: Tensor) -> Tensor:
        _, _, query_length, key_length = weight.shape

        # Compute a mask of 'use' for each key (weight >= 1/sequence_length)
        sequence_length = (weight > 1e-9).sum(-1, keepdim=True)
        used = (weight * sequence_length) >= 1

        # Update the timestamp for the most recent 'use' of each key
        last_used = (
            (used * self._t[key_length - query_length : key_length, None])
            .max(dim=-2)
            .values
        )
        self.last_used[..., :key_length] = torch.maximum(
            self.last_used[..., :key_length], last_used
        )
        return self.last_used[..., :key_length].float()


class Eviction:
    """Maintain a KV cache eviction mask for one attention layer & context.

    Note: `shape` should be `(batch_size, heads, max_k_length)`

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
        self._timestamp = 1 + torch.arange(shape[-1])

    def update(self, attention_weight: Tensor, causal_index: Tensor) -> None:
        """Update the eviction mask, from a step's attention weight matrix.

        attention_weight: shape (batch, head, query, key)

        causal_index: shape (batch, head, key), -1 for masked-out tokens
        """

        if self._last_length > attention_weight.shape[-1]:
            raise ValueError(
                "An eviction mask is being updated with a shorter context."
                " Please use `eviction_attention.generation_context` during"
                " generation to ensure the eviction mask is reset."
            )
        self._last_length = attention_weight.shape[-1]
        key_length = attention_weight.shape[-1]
        finfo = torch.finfo(torch.float32)

        # Update the score of each KV (summed over Q)
        score = self.strategy.update(attention_weight)

        # Combine locality and permadeath into score
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k)
        score.masked_fill_(is_local, finfo.max)  # local KVs
        score.masked_fill_(~self.mask[..., :key_length], finfo.min)  # dead KVs

        # Update the mask
        self.mask[..., :key_length] &= sparse_attention.topk_mask(
            score, min(key_length, self.settings.k)
        )


class GPTNeoXAttentionWithEviction(GPTNeoXAttention):  # type:ignore[misc]
    TRANSFORMERS_VERSION = "4.32.1"

    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        assert transformers.__version__ == self.TRANSFORMERS_VERSION, (
            "GPTNeoXAttentionWithEviction is version-locked to"
            f" transformers=={self.TRANSFORMERS_VERSION} for your safety"
        )
        super().__init__(config)
        self.max_sequence_length = config.max_position_embeddings
        self.eviction_settings = settings
        self.enable_eviction = False
        self.eviction: Optional[Eviction] = None
        # Set to an empty list to turn on eviction mask logging
        self.eviction_masks: Optional[List[Tensor]] = None

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        assert attention_mask is not None

        if self.eviction is None or not self.enable_eviction:
            # When disabled or missing, we should reset eviction statistics
            self.eviction = Eviction(
                self.eviction_settings,
                key.shape[:2] + (self.max_sequence_length,),
                key.device,
            )

        modified_attention_mask = attention_mask
        if self.enable_eviction:
            eviction_mask = self.eviction.mask[..., None, : attention_mask.shape[-1]]
            if self.eviction_masks is not None:
                self.eviction_masks.append(eviction_mask.clone())
            # Apply the mask to remove previously evicted values
            modified_attention_mask = (
                attention_mask + torch.finfo(attention_mask.dtype).min * ~eviction_mask
            )

        output, weights = super()._attn(
            query, key, value, modified_attention_mask, head_mask
        )
        # Most of the code doesn't care about "junk" queries, but it could confuse
        # eviction models, so we mask them out here, assuming that the last N values
        # correspond to the last N queries
        weights *= sparse_attention.score_to_mask(
            attention_mask[..., -weights.shape[2] :]
        ).swapdims(-1, -2)
        self.eviction.update(
            weights, sparse_attention.causal_index(attention_mask.squeeze(2))
        )
        return output, weights


@contextmanager
def generation_context(model: GPTNeoXForCausalLM) -> Iterator[GPTNeoXForCausalLM]:
    """(Context manager) enable KV eviction during this scope."""
    attns = [m for m in model.modules() if isinstance(m, GPTNeoXAttentionWithEviction)]
    for m in attns:
        assert not m.enable_eviction
        m.enable_eviction = True
    yield model
    for m in attns:
        m.enable_eviction = False
        m.eviction = None


def convert_gptneox(
    model: GPTNeoXForCausalLM, settings: Settings
) -> GPTNeoXForCausalLM:
    """Convert a GPT-NeoX model to use (simulated) KV cache eviction during generation.

    Note that the returned model should use `with generation_context(model)` during
    autoregressive generation to enable eviction.
    """

    def _convert(m: nn.Module, **args: Any) -> None:
        for name, child in m.named_children():
            if isinstance(child, GPTNeoXAttention):
                replacement = GPTNeoXAttentionWithEviction(**args)
                replacement.to(next(child.parameters()).dtype)
                replacement.load_state_dict(child.state_dict())
                setattr(m, name, replacement)
            _convert(child, **args)

    model = copy.deepcopy(model)
    _convert(model, config=model.config, settings=settings)
    model.generation_context = generation_context
    return model
