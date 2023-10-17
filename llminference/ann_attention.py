"""Approximate nearest neighbour methods that approximate `Q @ K.T`."""
import math
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
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
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

    def __init__(
        self, k: int, local_k: int, score: Union[ScoreSettings, str], **args: Any
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
        self.score = score_settings


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

class LlamaAttentionWithANN(LlamaAttention):
    def __init__(self, config: LlamaConfig, settings: Settings):
        super().__init__(config)
        self.ann = ANN(settings, self.num_heads, self.head_dim)
        # Set to an empty list to turn on ANN mask logging
        self.ann_masks: Optional[List[Tensor]] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
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

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # print(query_states.shape)
        
        # Only enable ANN during autoregressive generation
        if query_states.shape[-2] == 1:
            ann_mask = self.ann(query_states, key_states, attention_mask)
            if self.ann_masks is not None:
                self.ann_masks.append(ann_mask)
            attention_mask = torch.finfo(attention_mask.dtype).min * ~ann_mask

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            # if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            #     raise ValueError(
            #         f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            #     )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

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


def convert_llama(
    model: LlamaForCausalLM, settings: Settings
) -> LlamaForCausalLM:
    """Convert a Llama model to use (simulated) KV cache compression using ANN."""

    def _convert(m: nn.Module, **args: Any) -> None:
        for name, child in m.named_children():
            if isinstance(child, LlamaAttention):
                replacement = LlamaAttentionWithANN(**args)
                replacement.to(next(child.parameters()).dtype)
                replacement.load_state_dict(child.state_dict(), strict=False)
                setattr(m, name, replacement)
            _convert(child, **args)

    model = copy.deepcopy(model)
    _convert(model, config=model.config, settings=settings)
    return model
