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
from typing import Any, Iterator, List, Optional, Tuple

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


@dataclass
class HistoryState:
    mask: Tensor
    score: Tensor


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
        self.score = torch.zeros(shape, device=device)
        self.mask = torch.ones(shape, dtype=torch.bool, device=device)
        self._last_length = 0

    @property
    def history_state(self) -> HistoryState:
        return HistoryState(
            mask=self.mask[..., : self._last_length],
            score=self.score[..., : self._last_length],
        )

    def update(self, attention_weight: Tensor, causal_index: Tensor) -> None:
        """Update the eviction mask, from a step's attention weight matrix.

        attention_weight: shape (batch, head, query, key)

        causal_index: shape (batch, query, key), -1 for masked-out tokens
        """

        if self._last_length > attention_weight.shape[-1]:
            raise ValueError(
                "An eviction mask is being updated with a shorter context."
                " Please use `eviction_attention.generation_context` during"
                " generation to ensure the eviction mask is reset."
            )
        self._last_length = attention_weight.shape[-1]
        context_len = attention_weight.shape[-1]
        finfo = torch.finfo(self.score.dtype)

        # Update the score of each KV (summed over Q)
        self.score[..., :context_len] += attention_weight.float().sum(-2)

        # Combine score, locality and permadeath into 'total_score'
        total_score = self.score[..., :context_len].clone()  # popular KVs
        is_local = (0 <= causal_index) & (causal_index < self.settings.local_k)
        total_score += finfo.max * is_local  # local KVs
        total_score += finfo.min * ~self.mask[..., :context_len]  # dead KVs

        # Update the mask
        self.mask[..., :context_len] &= sparse_attention.topk_mask(
            total_score, min(context_len, self.settings.k)
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
        self.history: List[HistoryState] = []

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

        if self.enable_eviction:
            # Apply the mask to remove previously evicted values
            fmin = torch.finfo(attention_mask.dtype).min
            attention_mask = (
                attention_mask
                + fmin * ~self.eviction.mask[..., None, : attention_mask.shape[-1]]
            )

        output, weights = super()._attn(query, key, value, attention_mask, head_mask)
        self.eviction.update(
            weights, sparse_attention.causal_index(attention_mask.squeeze(2))
        )
        return output, weights


@contextmanager
def generation_context(
    model: GPTNeoXForCausalLM, keep_history: bool = False
) -> Iterator[GPTNeoXForCausalLM]:
    """(Context manager) enable KV eviction during this scope."""
    attns = [m for m in model.modules() if isinstance(m, GPTNeoXAttentionWithEviction)]
    for m in attns:
        assert not m.enable_eviction
        m.enable_eviction = True
    yield model
    for m in attns:
        m.enable_eviction = False
        if keep_history:
            assert m.eviction
            m.history.append(m.eviction.history_state)
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
