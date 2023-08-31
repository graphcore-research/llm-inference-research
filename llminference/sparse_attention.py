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


def sparse_softmax_fixed_k(
    x: Tensor,
    k: int,
    dim: int = -1,
    add_avg: bool = False,
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
        out_weights (Optional[Tensor], optional): shape (broadcastable to x)
        If passed, multiplies softmax scores with out_weights
        before choosing top-k.

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    if out_weights is None:
        out_weights = torch.tensor(1.0, dtype=x.dtype, device=x.device)

    y = _softmax(x, dim=-1)
    if k >= x.shape[-1]:
        return y

    y_weighted = y * out_weights
    kth_val = -torch.kthvalue(-y_weighted, k, keepdim=True).values

    mask = y_weighted >= kth_val
    out: Tensor = mask * y
    if add_avg:
        out += ~mask * (1 - out.sum(dim=-1, keepdim=True)) / (out.shape[-1] - k)
    return out


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
    x: Tensor, context_len: int, apply_after_softmax: bool = True, dim: int = -1
) -> Tensor:
    """Applies softmax across last dimension, keeping past context_len number
    of values in the output.

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        context_len (int): Keep past context_len output scores (t-context_len : t)
        apply_after_softmax (bool, optional):
        If True, set corresponding softmax output elements to 0.
        If False, mask corresponding inputs to softmax to -1e9.
        Defaults to True.
        dim (int, optional): Assumed dim = -1.

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    assert len(x.shape) >= 2
    q_len, k_len = x.shape[-2:]
    local_mask = torch.tensor(
        [
            [(j <= i - context_len) for j in range(k_len)]
            for i in range(k_len - q_len, k_len)
        ]
    )
    if apply_after_softmax:
        return _softmax(x, dim=-1) * local_mask.logical_not()
    else:
        return _softmax(x.masked_fill(local_mask, -1e9), -1)


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
