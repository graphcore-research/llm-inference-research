# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""Sparse attention mechanisms.

Note the following definitions:

 - 'mask' is `True` or `1` for unmasked/retained tokens, `False` or `0` for
   masked-out tokens
 - `score` (often `x`) is near `FP16_MIN` for masked-out tokens
"""
from dataclasses import asdict, dataclass
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
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


def topk_mask(x: Tensor, k: int, dim: int = -1) -> Tensor:
    """Mask selecting top-k positions along a dimension of `x` (default: last).

    Ties are broken arbitarily, enforcing the constraint
    `(out.sum(-1) <= k).all()`
    """
    fmin = torch.finfo(x.dtype).min
    topk_idxs = x.nan_to_num(fmin).topk(k, dim).indices
    return torch.zeros_like(x, dtype=torch.bool).scatter(dim, topk_idxs, 1)


def score_to_mask(score: Tensor, threshold: float = 0.5) -> Tensor:
    """Convert a score tensor to a mask.

    threshold (float): how strict to be (values below `threshold*FP16_MIN` are
    considered masked-out); between 0 and 1.

    Returns: boolean 'mask' tensor, the same shape as `score`
    """
    return threshold * torch.finfo(torch.float16).min < score


def causal_index(score: Tensor) -> Tensor:
    """Get the 'causal index' of a tokens in history, from masked attention scores.

    The causal index is the number of unmasked tokens between key token and
    the query, counting backwards

    Args:
        score (Tensor): shape (*, q_len, k_len), should be set to -finfo.min
        for masked-out values

    Returns:
        Tensor: shape (*, q_len, k_len), containing the caual index of each
        token according to the mask, or -1 if the token is masked-out
    """
    mask = score_to_mask(score)
    cumsum = mask.flip(-1).cumsum(-1, dtype=torch.int32).flip(-1)
    return cumsum.sub_(1).masked_fill_(mask.logical_not_(), -1)


@dataclass
class SparseSettings:
    k: int
    apply_after_softmax: bool
    reallocate_to_mean: bool


@dataclass
class LocalSettings:
    k: int
    initial_k: int
    apply_after_softmax: bool


Settings = Union[SparseSettings, LocalSettings]


class SparseAttention(nn.Module):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        if isinstance(settings, SparseSettings):
            self.softmax = partial(sparse_softmax, **asdict(settings))
        elif isinstance(settings, LocalSettings):
            self.softmax = partial(local_softmax, **asdict(settings))
        else:
            raise ValueError(f"Unexpected settings = {settings}")
        # Set to an empty list to turn on mask logging
        self.debug_masks: Optional[List[Tensor]] = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, logmask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Attention with sparse softmax scores.

        query -- (batch, n_heads, 1, head_size)
        key -- (batch, n_kv_heads, seq, head_size)
        value -- (batch, n_kv_heads, seq, head_size)
        logmask -- (batch, n_heads, 1, seq)

        returns -- (output, weights)
                   output -- (batch, n_heads, 1, head_size)
                   weights -- (batch, n_heads, 1, seq)
        """
        # Group by KV head
        n_kv_heads = key.shape[1]
        query, key, value, logmask = map(
            partial(torch.unflatten, dim=1, sizes=(n_kv_heads, -1)),
            [query, key, value, logmask],
        )

        scores = (query.div(query.shape[-1] ** 0.5) @ key.transpose(-1, -2)).add_(
            logmask
        )
        weights = self.softmax(scores).to(value.dtype)
        if self.debug_masks is not None:
            self.debug_masks.append(weights != 0)
        output = weights @ value
        return output.flatten(1, 2), weights.flatten(1, 2)


def sparse_softmax(
    scores: Tensor, k: int, apply_after_softmax: bool, reallocate_to_mean: bool
) -> Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output.

    scores -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)
    k -- number of attention scores to keep
    apply_after_softmax -- apply the top-k mask after softmax op
    reallocate_to_mean -- reallocate leftover weights to non-top-k

    returns -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)

    """
    assert (
        apply_after_softmax or not reallocate_to_mean
    ), "reallocate_to_mean can only be used with apply_after_softmax"

    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)

    if k >= scores.shape[-1]:
        return weights

    mask = topk_mask(weights.sum(dim=2, keepdim=True), k)

    if not apply_after_softmax:
        return torch.softmax(
            scores.masked_fill(~mask, torch.finfo(scores.dtype).min), dim=-1
        )

    weights *= mask

    if reallocate_to_mean:
        removed_by_topk = (~mask) & score_to_mask(scores)
        weights += (
            removed_by_topk
            * (1 - weights.sum(dim=-1, keepdim=True))
            / removed_by_topk.sum(dim=-1, keepdim=True)
        )

    return weights


def local_softmax(
    scores: Tensor, k: int, initial_k: int, apply_after_softmax: bool
) -> Tensor:
    """Applies softmax across last dimension, keeping k most recent and oldest values.

    Always keeps a maximum of `k` values.
        - `k - initial_k` most recent
        - `initial_k` oldest

    scores -- (batch, n_kv_heads, n_heads_per_kv, 1, seq)
    apply_after_softmax -- apply the mask after softmax op
    """
    index = causal_index(scores)
    max_index = index.max(dim=-1, keepdim=True).values
    local_mask = (index < k - initial_k).logical_or_(max_index - initial_k < index)
    if apply_after_softmax:
        return (
            torch.softmax(scores, dim=-1, dtype=torch.float32)
            .to(scores.dtype)
            .mul_(local_mask)
        )
    return torch.softmax(
        scores.masked_fill(local_mask.logical_not_(), torch.finfo(scores.dtype).min),
        dim=-1,
        dtype=torch.float32,
    ).to(scores.dtype)


Model = Union[
    GPTNeoXForCausalLM, LlamaForCausalLM, MistralForCausalLM, GemmaForCausalLM
]


class GPTNeoXSparseAttention(GPTNeoXAttention):  # type:ignore[misc]
    def __init__(self, config: GPTNeoXConfig, settings: Settings):
        utility.check_transformers_version(type(self))
        super().__init__(config)
        self.sparse_attention = SparseAttention(settings)

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
        # Only enable sparse attention during autoregressive generation
        if query.shape[-2] == 1:
            return self.sparse_attention(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.expand(*query.shape[:-1], key.shape[-2]),
            )

        return super()._attn(  # type:ignore[no-any-return]
            query, key, value, attention_mask, head_mask
        )


class LlamaSparseAttention(llama_attention.LlamaAttention):
    def __init__(
        self, config: LlamaConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.sparse_attention = SparseAttention(settings)

    def _attn(
        self, query: Tensor, key: Tensor, value: Tensor, attention_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Only enable sparse attention during autoregressive generation
        if query.shape[-2] == 1:
            return self.sparse_attention(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.expand(*query.shape[:-1], key.shape[-2]),
            )

        return super()._attn(query, key, value, attention_mask)


class MistralSparseAttention(mistral_attention.MistralAttention):
    def __init__(
        self, config: MistralConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.sparse_attention = SparseAttention(settings)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Only enable sparse attention during autoregressive generation
        if query.shape[-2] == 1:
            return self.sparse_attention(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.expand(*query.shape[:-1], key.shape[-2]),
            )

        return super()._attn(query, key, value, attention_mask)


class GemmaSparseAttention(gemma_attention.GemmaAttention):
    def __init__(
        self, config: GemmaConfig, layer_idx: Optional[int], settings: Settings
    ):
        utility.check_transformers_version(type(self))
        super().__init__(config, layer_idx)
        self.sparse_attention = SparseAttention(settings)

    def _attn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Only enable sparse attention during autoregressive generation
        if query.shape[-2] == 1:
            return self.sparse_attention(  # type:ignore[no-any-return]
                query,
                key,
                value,
                attention_mask.expand(*query.shape[:-1], key.shape[-2]),
            )
        return super()._attn(query, key, value, attention_mask)


def convert(model: Model, settings: Settings) -> Model:
    """Convert the model to use Sparse Attention during generation."""

    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, GPTNeoXAttention):
            return GPTNeoXSparseAttention(model.config, settings)
        if isinstance(m, LlamaAttention):
            return LlamaSparseAttention(model.config, m.layer_idx, settings)
        if isinstance(m, MistralAttention):
            return MistralSparseAttention(model.config, m.layer_idx, settings)
        if isinstance(m, GemmaAttention):
            return GemmaSparseAttention(model.config, m.layer_idx, settings)

    return utility.convert_module(model, _replace)
