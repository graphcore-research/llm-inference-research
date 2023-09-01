"""Sparse attention mechanisms.

Note the following definitions:

 - 'mask' is `True` or `1` for unmasked/retained tokens, `False` or `0` for
   masked-out tokens
 - `score` (often `x`) is near `finfo.min` for masked-out tokens
"""
import unittest.mock as um
from contextlib import contextmanager
from functools import partial
from typing import Any, Iterator, List, Optional, Tuple, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXAttention,
    GPTNeoXModel,
)

_softmax = F.softmax


def score_to_mask(score: Tensor, threshold: float = 0.5) -> Tensor:
    """Convert a score tensor to a mask.

    threshold (float): how strict to be (values below `threshold*fmin` are considered
    masked-out), between 0 and 1.

    Returns: boolean 'mask' tensor, the same shape as `score`
    """
    return threshold * torch.finfo(score.dtype).min < score


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
    cumsum = mask.flip(-1).cumsum(-1).flip(-1)
    return (cumsum - 1).masked_fill(~mask, -1)


def sparse_softmax_fixed_k(
    x: Tensor,
    k: int,
    dim: int = -1,
    add_avg: bool = False,
    apply_after_softmax: bool = True,
    out_weights: Optional[Tensor] = None,
) -> Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output.

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        k (int): Number of attention scores to keep
        dim (int, optional): Assumed dim = -1
        add_avg (bool, optional): If True, assign the non-top-k
        softmax weight equally to non-top-k tokens.
        apply_after_softmax (bool, optional): apply the top-k mask after softmax
        out_weights (Optional[Tensor], optional): shape (broadcastable to x)
        If passed, multiplies softmax scores with out_weights
        before choosing top-k.

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    assert not (
        add_avg and not apply_after_softmax
    ), "add_avg requires apply_after_softmax"
    if out_weights is None:
        out_weights = torch.tensor(1.0, dtype=x.dtype, device=x.device)

    if k >= x.shape[-1]:
        return _softmax(x, dim=-1)

    score = x + torch.log(out_weights)
    kth_val = -torch.kthvalue(-score, k, keepdim=True).values
    mask = kth_val <= score

    if not apply_after_softmax:
        return _softmax(x.masked_fill(~mask, torch.finfo(x.dtype).min), dim=-1)

    y = _softmax(x, dim=-1)
    y *= mask
    if add_avg:
        # '1' for tokens that were removed by the topk operation, which can
        # receive the average reallocation
        removed_by_topk = (~mask) & score_to_mask(x)
        y += (
            removed_by_topk
            * (1 - y.sum(dim=-1, keepdim=True))
            / removed_by_topk.sum(-1, keepdim=True)
        )
    return y


def sparse_softmax_fixed_p(
    x: Tensor, p: float, k_min: int = 1, dim: int = -1
) -> Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output.
    k is calculated as p * num of tokens attended to by the query

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        p (float): Proportion of attention scores to keep
        k_min (int): Minimum number of attention scores kept
        dim (int, optional): Assumed dim = -1

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    assert len(x.shape) >= 2
    q_len, k_len = x.shape[-2:]
    y = _softmax(x, dim=-1)
    if k_min >= x.shape[-1]:
        return y

    k = (torch.arange(start=k_len - q_len + 1, end=k_len + 1) * p).long()
    k[k < k_min] = k_min
    topk_vals = torch.topk(y, int(k[-1].item()), dim=-1).values
    index = k.unsqueeze(-1)
    for _ in range(len(x.shape) - 2):
        index = index.unsqueeze(0)
    index = index.expand(*x.shape[:-2], -1, -1)
    kth_vals = topk_vals.gather(dim=-1, index=index - 1)
    return cast(Tensor, (y >= kth_vals) * y)


def local_softmax(
    x: Tensor,
    context_len: int,
    apply_after_softmax: bool = True,
    dim: int = -1,
) -> Tensor:
    """Applies softmax across last dimension, keeping past context_len number
    of values in the output.

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        context_len (int): Keep context_len past output scores (t-context_len : t)
        apply_after_softmax (bool, optional):
        If True, set corresponding softmax output elements to 0.
        If False, mask corresponding inputs to softmax to `finfo.min`.
        Defaults to True.
        dim (int, optional): Assumed dim = -1.

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    assert len(x.shape) >= 2
    local_mask = causal_index(x) < context_len
    if apply_after_softmax:
        return _softmax(x, dim=-1) * local_mask
    return _softmax(x.masked_fill(~local_mask, torch.finfo(x.dtype).min), dim=-1)


@contextmanager
def number_attention_layers(model: GPTNeoXModel) -> Iterator[GPTNeoXModel]:
    try:
        for i, layer in enumerate(model.layers):
            layer.attention.layer_idx = i
        yield model
    finally:
        for layer in model.layers:
            del layer.attention.layer_idx


original_attn = GPTNeoXAttention._attn


def sparse_attn(
    self: GPTNeoXAttention,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Optional[Tensor] = None,
    head_mask: Optional[Tensor] = None,
    use_v_mag: bool = False,
    k_per_layer: Optional[List[int]] = None,
    **sa_args: Any
) -> Tuple[Tensor, Tensor]:
    if use_v_mag:
        # value.shape = (batch_size, num_heads, k_len, head_dim)
        # v_norm.shape = (batch_size, num_heads, 1, k_len)
        v_norm = value.norm(dim=-1).unsqueeze(dim=-2)
    else:
        v_norm = None
    if k_per_layer is not None:
        assert hasattr(self, "layer_idx"), (
            "Attention object does not have layer_idx attribute,"
            " wrap within number_attention_layers contextmanager beforehand."
        )
        sa_args["k"] = k_per_layer[self.layer_idx]
    with um.patch(
        "torch.nn.functional.softmax",
        partial(sparse_softmax_fixed_k, **sa_args, out_weights=v_norm),
    ):
        out: Tuple[Tensor, Tensor] = original_attn(
            self, query, key, value, attention_mask, head_mask
        )
    return out
