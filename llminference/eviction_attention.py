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
import math
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
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
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
        return self.score[..., :key_length]


class LRU:
    def __init__(self, shape: Tuple[int, ...], device: torch.device):
        # Store timestamps as float for ease-of-conversion to scores
        self.last_used = torch.zeros(shape, device=device)
        # Note: range [1, N], so that 'use' at timestep 0 is better than 'never used'
        self._t = 1 + torch.arange(shape[-1], device=device, dtype=torch.float32)

    def update(self, weight: Tensor) -> Tensor:
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
        return self.last_used[..., :key_length]


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
        score = self.strategy.update(attention_weight).clone()

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


class LlamaAttentionWithEviction(LlamaAttention):  # type:ignore[misc]
    TRANSFORMERS_VERSION = "4.32.1"

    def __init__(self, config: LlamaConfig, settings: Settings):
        assert transformers.__version__ == self.TRANSFORMERS_VERSION, (
            "LlamaAttentionWithEviction is version-locked to"
            f" transformers=={self.TRANSFORMERS_VERSION} for your safety"
        )
        super().__init__(config)
        self.max_sequence_length = config.max_position_embeddings
        self.eviction_settings = settings
        self.enable_eviction = False
        self.eviction: Optional[Eviction] = None
        # Set to an empty list to turn on eviction mask logging
        self.eviction_masks: Optional[List[Tensor]] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.eviction is None or not self.enable_eviction:
            # When disabled or missing, we should reset eviction statistics
            self.eviction = Eviction(
                self.eviction_settings,
                key_states.shape[:2] + (self.max_sequence_length,),
                key_states.device,
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

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if modified_attention_mask is not None:
            attn_weights = attn_weights + modified_attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_weights *= sparse_attention.score_to_mask(
            attention_mask[..., -attn_weights.shape[2] :]
        ).swapdims(-1, -2)
        self.eviction.update(
            attn_weights, sparse_attention.causal_index(attention_mask[:,:,-1,:])
        )

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    


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


@contextmanager
def generation_context_llama(model: LlamaForCausalLM) -> Iterator[LlamaForCausalLM]:
    """(Context manager) enable KV eviction during this scope."""
    attns = [m for m in model.modules() if isinstance(m, LlamaAttentionWithEviction)]
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

def convert_llama(
    model: LlamaForCausalLM, settings: Settings
) -> LlamaForCausalLM:
    """Convert a Llama model to use (simulated) KV cache compression using ANN."""

    def _convert(m: nn.Module, **args: Any) -> None:
        for name, child in m.named_children():
            if isinstance(child, LlamaAttention):
                replacement = LlamaAttentionWithEviction(**args)
                replacement.to(next(child.parameters()).dtype)
                replacement.load_state_dict(child.state_dict(), strict=False)
                setattr(m, name, replacement)
            _convert(child, **args)

    model = copy.deepcopy(model)
    _convert(model, config=model.config, settings=settings)
    model.generation_context = generation_context_llama
    return model