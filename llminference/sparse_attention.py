import torch
import torch.nn.functional as F
from torch import Tensor
from typing import cast

_softmax = F.softmax


def sparse_softmax_fixed_k(x: Tensor, k: int, dim: int = -1) -> Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output.

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        k (int): Number of attention scores to keep
        dim (int, optional): Assumed dim = -1

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    y = _softmax(x, dim=-1)
    if k >= x.shape[-1]:
        return y

    kth_val = -torch.kthvalue(-y, k, keepdim=True).values
    return cast(Tensor, (y >= kth_val) * y)


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
