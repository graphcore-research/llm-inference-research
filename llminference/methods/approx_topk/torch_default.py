"""Wrapper around torch.topk() to match our topk Protocol,
as well as the reference bucketed top-k implementation using torch.topk
"""

import torch
from approx_topk.autobucket import bucket
from torch import Tensor


def topk(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    return torch.topk(xs, k, dim, sorted=False)


def bucket_topk(
    xs: Tensor,
    k: int,
    dim: int,
    k_mult: int,
    k_per_bucket: int,
    interleaved: bool,
) -> tuple[Tensor, Tensor]:
    return bucket(
        topk,
        k_mult=k_mult,
        k_per_bucket=k_per_bucket,
        interleaved=interleaved,
    )(xs, k, dim)
