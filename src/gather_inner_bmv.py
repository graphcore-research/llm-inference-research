import math

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def gather_inner_bmv_kernel(
    a_ptr,
    b_ptr,
    i_ptr,
    o_ptr,
    k: tl.constexpr,
    n: int,
    n_chunk: tl.constexpr,
    a_s0: int,
    a_s2: int,
    b_s0: int,
    b_s1: int,
    b_s2: int,
    i_s0: int,
    i_s1: int,
    o_s0: int,
):
    pid = tl.program_id(axis=0)
    i = tl.load(i_ptr + pid * i_s0 + tl.arange(0, k) * i_s1)  # (k)
    a = tl.load(a_ptr + pid * a_s0 + i * a_s2)  # (k)
    for chunk in range(0, tl.cdiv(n, n_chunk)):
        chunk_idx = chunk * n_chunk + tl.arange(0, n_chunk)
        b = tl.load(
            b_ptr + pid * b_s0 + (i * b_s1)[:, None] + (chunk_idx * b_s2)[None, :]
        )  # (k x n_chunk)
        # As tl.dot() unavailable for matrix-vector
        o = tl.sum(a[:, None] * b, 0)  # (n_chunk)
        tl.store(o_ptr + pid * o_s0 + chunk_idx, o, mask=(chunk_idx < n))


def gather_inner_bmv(a: Tensor, b: Tensor, i: Tensor, chunk: int = 1024) -> Tensor:
    """Batched vector-matrix multiplication, with a gather on the inner dimension.

    Note that this recompiles the kernel when either K (`i.shape[1]`) or `chunk` change.

    a -- (B, 1, K*)
    b -- (B, K*, N)
    i -- int(B, K)         in [0, K*), where K <= K*
    chunk -- int           size of chunks of `b` to be processed at a time

    returns -- (B, 1, N)   the inner product of `a` and `b`, after gathering the inner dimension
                           according to `i`
    """
    assert a.ndim == 3 and b.ndim == 3 and a.shape[1] == 1 and a.shape[2] == b.shape[1]
    assert (
        i.ndim == 2
        and i.shape[0] == a.shape[0]
        and 2 ** int(math.log2(i.shape[1])) == i.shape[1]
    )
    assert (
        b.stride(2) == 1
    ), "`b` should be contiguous; although it works otherwise, it's very slow"

    o = torch.empty((a.shape[0], 1, b.shape[2]), dtype=a.dtype, device=a.device)
    assert o.stride(2) == 1
    gather_inner_bmv_kernel[(a.shape[0],)](
        a_ptr=a,
        b_ptr=b,
        i_ptr=i,
        o_ptr=o,
        k=i.shape[1],
        n=b.shape[2],
        n_chunk=chunk,
        a_s0=a.stride(0),
        a_s2=a.stride(2),
        b_s0=b.stride(0),
        b_s1=b.stride(1),
        b_s2=b.stride(2),
        i_s0=i.stride(0),
        i_s1=i.stride(1),
        o_s0=o.stride(0),
    )
    return o
