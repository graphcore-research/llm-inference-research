# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import math
import warnings

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _kernel_gather_inner_bmv(
    A_ptr,
    B_ptr,
    I_ptr,
    Y_ptr,
    k: tl.constexpr,  # int
    n: int,
    n_chunk: tl.constexpr,  # int
    A_s0: int,
    A_s2: int,
    B_s0: int,
    B_s1: int,
    B_s2: int,
    I_s0: int,
    I_s1: int,
    gather_A: tl.constexpr,  # bool
):
    pid = tl.program_id(axis=0).to(tl.int64)
    i = tl.load(I_ptr + pid * I_s0 + tl.arange(0, k) * I_s1)  # (k)
    a = tl.load(A_ptr + pid * A_s0 + (i if gather_A else tl.arange(0, k)) * A_s2)  # (k)
    for chunk in range(0, tl.cdiv(n, n_chunk)):
        chunk_idx = chunk * n_chunk + tl.arange(0, n_chunk)
        b = tl.load(  # (k x n_chunk)
            B_ptr + pid * B_s0 + (i * B_s1)[:, None] + (chunk_idx * B_s2)[None, :]
        )
        # As tl.dot() is unavailable for matrix-vector
        y = tl.sum(a[:, None] * b, 0)  # (n_chunk)
        tl.store(Y_ptr + pid * n + chunk_idx, y, mask=(chunk_idx < n))


def gather_inner_bmv(
    A: Tensor, B: Tensor, I: Tensor, chunk: int, _matrix_only: bool = False
) -> Tensor:
    """Batched vector-matrix multiplication, with a gather on the inner dimension.

    Dimensions:
       b -- batch
       k* -- (pre-gather) inner dimension
       k -- (post-gather) inner dimension (k <= k*), must be a power of two
       n -- outer dimension

    A -- (b, 1, k*)        batch of vectors
    B -- (b, k*, n)        batch of matrices
    I -- int(b, k)         indices, in [0, k*)
    chunk -- int           size of chunks of `B` (along dimension `n`) to be processed at a time
    _matrix_only -- bool   don't use (see `gather_inner_matrix_only_bmv`)

    returns -- (b, 1, n)   the inner product of `A` and `B`, after gathering the inner dimension
                           according to `I`
    """
    if A.ndim > 3:
        assert B.ndim == A.ndim and I.ndim == A.ndim - 1
        return gather_inner_bmv(
            A.flatten(end_dim=-3),
            B.flatten(end_dim=-3),
            I.flatten(end_dim=-2),
            chunk=chunk,
            _matrix_only=_matrix_only,
        ).unflatten(0, A.shape[:-2])
    assert A.ndim == 3 and B.ndim == 3 and A.shape[1] == 1
    assert (
        I.ndim == 2
        and I.shape[0] == A.shape[0]
        and 2 ** int(math.log2(I.shape[1])) == I.shape[1]
    )
    assert A.shape[2] == (I.shape[1] if _matrix_only else B.shape[1])
    if B.stride(2) != 1:
        warnings.warn(
            "gather_inner_bmv(A, B, ...) `B` should be contiguous in the last dimension"
            ", otherwise it is very slow"
        )

    b, k, n = A.shape[0], I.shape[1], B.shape[2]
    Y = torch.empty((b, 1, n), dtype=A.dtype, device=A.device)
    assert Y.stride(0) == n and Y.stride(2) == 1

    _kernel_gather_inner_bmv[(b,)](
        A_ptr=A,
        B_ptr=B,
        I_ptr=I,
        Y_ptr=Y,
        k=k,
        n=n,
        n_chunk=chunk,
        A_s0=A.stride(0),
        A_s2=A.stride(2),
        B_s0=B.stride(0),
        B_s1=B.stride(1),
        B_s2=B.stride(2),
        I_s0=I.stride(0),
        I_s1=I.stride(1),
        gather_A=not _matrix_only,
    )
    return Y


def gather_inner_matrix_only_bmv(A: Tensor, B: Tensor, I: Tensor, chunk: int) -> Tensor:
    """Batched vector-matrix multiplication, with a gather on the inner dimension of the matrix.

    Dimensions:
       b -- batch
       k* -- (pre-gather) inner dimension
       k -- (post-gather) inner dimension (k <= k*), must be a power of two
       n -- outer dimension

    A -- (b, 1, k)         batch of vectors
    B -- (b, k*, n)        batch of matrices
    I -- int(b, k)         indices, in [0, k*)
    chunk -- int           size of chunks of `B` (along dimension `n`) to be processed at a time

    returns -- (b, 1, n)   the inner product of `A` and `B`, after gathering the inner dimension
                           of `B` according to `I`
    """
    return gather_inner_bmv(A, B, I, chunk=chunk, _matrix_only=True)


@triton.jit
def _kernel_gather_outer_bmv(
    A_ptr,
    B_ptr,
    I_ptr,
    Y_ptr,
    k: tl.constexpr,
    n: int,
    n_chunk: tl.constexpr,
    A_s0: int,
    A_s2: int,
    B_s0: int,
    B_s1: int,
    B_s2: int,
    I_s0: int,
    I_s1: int,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    a = tl.load(A_ptr + pid * A_s0 + tl.arange(0, k) * A_s2)  # (k)
    for chunk in range(0, tl.cdiv(n, n_chunk)):
        chunk_idx = chunk * n_chunk + tl.arange(0, n_chunk)
        i = tl.load(I_ptr + pid * I_s0 + chunk_idx * I_s1)  # (n_chunk)
        b = tl.load(  # (k x n_chunk)
            B_ptr
            + pid * B_s0
            + (tl.arange(0, k) * B_s1)[:, None]
            + (i * B_s2)[None, :],
            mask=(chunk_idx < n)[None, :],
        )
        # # As tl.dot() is unavailable for matrix-vector
        y = tl.sum(a[:, None] * b, 0)  # (n_chunk)
        tl.store(Y_ptr + pid * n + chunk_idx, y, mask=(chunk_idx < n))


def gather_outer_bmv(A: Tensor, B: Tensor, I: Tensor, chunk: int) -> Tensor:
    """Batched vector-matrix multiplication, with a gather on the matrix outer dimension.

    Dimensions:
       b -- batch
       k -- inner dimension, must be a power of two
       n* -- (pre-gather) outer dimension
       n -- (post-gather) outer dimension (n <= n*)

    A -- (b, 1, k)         batch of vectors
    B -- (b, k, n*)        batch of matrices
    I -- int(b, n)         indices, in [0, n*)
    chunk -- int           size of chunks of `B` (along dimension `n`) to be processed at a time

    returns -- (b, 1, n)   the inner product of `A` and `B`, after gathering the outer dimension
                           according to `I`
    """
    if A.ndim > 3:
        assert B.ndim == A.ndim and I.ndim == A.ndim - 1
        return gather_outer_bmv(
            A.flatten(end_dim=-3),
            B.flatten(end_dim=-3),
            I.flatten(end_dim=-2),
            chunk=chunk,
        ).unflatten(0, A.shape[:-2])
    assert A.ndim == 3 and B.ndim == 3 and A.shape[1] == 1 and A.shape[2] == B.shape[1]
    assert I.ndim == 2 and I.shape[0] == A.shape[0]

    b, k, n = A.shape[0], A.shape[2], I.shape[1]
    Y = torch.empty((b, 1, n), dtype=A.dtype, device=A.device)
    assert Y.stride(0) == n and Y.stride(2) == 1

    _kernel_gather_outer_bmv[(b,)](
        A_ptr=A,
        B_ptr=B,
        I_ptr=I,
        Y_ptr=Y,
        k=k,
        n=n,
        n_chunk=chunk,
        A_s0=A.stride(0),
        A_s2=A.stride(2),
        B_s0=B.stride(0),
        B_s1=B.stride(1),
        B_s2=B.stride(2),
        I_s0=I.stride(0),
        I_s1=I.stride(1),
    )
    return Y
