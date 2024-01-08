import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import tqdm
from torch import Tensor

from gather_matmul import gather_inner_bmv, gather_outer_bmv

# Methods


def attn(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    QK = (Q @ K.transpose(2, 3)).div_(Q.shape[-1] ** 0.5)
    return torch.softmax(QK, dim=-1) @ V


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


def sparq_attn(
    Q: Tensor,
    K1: Tensor,
    K2: Tensor,
    V: Tensor,
    V_mean: Tensor,
    k1: int,
    k2: int,
    gather_matmul: str,  # "auto|torch|custom"
) -> Tensor:
    if gather_matmul == "auto":
        gather_matmul = "custom" if Q.device.type == "cuda" else "torch"

    # 1. Approximate attention scores using k1 largest components of Q
    absQ = torch.abs(Q)
    absQ_hat, i1 = torch.topk(absQ, k1, -1)
    scale = torch.sqrt(
        Q.shape[-1]
        * absQ_hat.sum(dim=-1, keepdim=True)
        / absQ.sum(dim=-1, keepdim=True)
    )
    if gather_matmul == "torch":
        QK_hat = gather(Q, -1, i1) @ gather(K1, -1, i1).transpose(-1, -2)
    elif gather_matmul in ("custom", "custom2"):
        QK_hat = gather_inner_bmv(Q, K1.transpose(-1, -2), i1.squeeze(2), chunk=512)
    s_hat = torch.softmax(QK_hat.div_(scale), dim=-1)

    # 2. Gather top k2 positions based on approximate attention scores & run attention
    s_hat_i2, i2 = torch.topk(s_hat, k2, -1)
    iKV = i2[..., 0, :, None]
    if gather_matmul in ("torch", "custom"):
        QK = Q @ gather(K2, -2, iKV).transpose(2, 3)
    elif gather_matmul == "custom2":
        QK = gather_outer_bmv(Q, K2.transpose(2, 3), iKV.squeeze(-1), chunk=128)
    s = torch.softmax(QK.div_(Q.shape[-1] ** 0.5), dim=-1)
    y_ = s @ gather(V, -2, iKV)

    # 3. Estimate the total score of the top k, and interpolate with V_mean
    alpha = s_hat_i2.sum(-1, keepdim=True)
    return alpha * y_ + (1 - alpha) * V_mean


# Execution


@dataclass
class Benchmark:
    method: str  # "empty|dense|sparq"
    kernel: str  # "empty|vanilla|compiled|auto|flash|math|mem_efficient"
    # Generic
    batch_size: int
    n_head: int
    sequence_length: int
    head_dim: int
    dtype: str  # "float16|float32"
    device: str  # "cpu|cuda"
    # Benchmark
    reps: int
    # Method-specific
    k1: Optional[int] = None
    k2: Optional[int] = None
    store_k_twice: Optional[bool] = None
    gather_matmul: Optional[str] = None  # "auto|torch|custom"

    @property
    def dense_flops(self) -> float:
        return (
            2 * 2 * self.batch_size * self.n_head * self.sequence_length * self.head_dim
        )

    @property
    def dense_transfer_bytes(self) -> float:
        scalar_size = dict(float16=2, float32=4)[self.dtype]
        return (
            scalar_size
            * 2
            * self.batch_size
            * self.n_head
            * self.sequence_length
            * self.head_dim
        )


@dataclass
class Results:
    duration: List[float]
    std: List[float]
    device_name: str
    torch_version: str
    revision: str
    error: Optional[str]


def get_runner(b: Benchmark, K: Tensor, V: Tensor) -> Callable[[Tensor], Tensor]:
    if b.method == "sparq":
        if (
            b.k1 is None
            or b.k2 is None
            or b.store_k_twice is None
            or b.gather_matmul is None
        ):
            raise ValueError(
                "Must specify {k1, k2, store_k_twice, gather_matmul} for sparq attention"
            )
        V_mean = V.mean(dim=-2, keepdim=True)
        K1 = K2 = K
        if b.store_k_twice:
            K1 = K.swapdims(-1, -2).contiguous().swapdims(-1, -2)

    attn_ = torch.compile(attn) if b.kernel == "compiled" else attn
    sparq_attn_ = torch.compile(sparq_attn) if b.kernel == "compiled" else sparq_attn

    def run(Q: Tensor) -> Tensor:
        if b.method == "empty" and b.kernel == "empty":
            return Q
        if b.method == "dense" and b.kernel in ("vanilla", "compiled"):
            return attn_(Q, K, V)
        if b.method == "dense" and b.kernel in [
            "auto",
            "flash",
            "math",
            "mem_efficient",
        ]:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=(b.kernel == "auto" or b.kernel == "flash"),
                enable_math=(b.kernel == "auto" or b.kernel == "math"),
                enable_mem_efficient=(
                    b.kernel == "auto" or b.kernel == "mem_efficient"
                ),
            ):
                return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        if b.method == "sparq" and b.kernel in ("vanilla", "compiled"):
            assert b.k1 and b.k2
            return sparq_attn_(
                Q,
                K1,
                K2,
                V,
                V_mean=V_mean,
                k1=b.k1,
                k2=b.k2,
                gather_matmul=b.gather_matmul,
            )
        raise ValueError(f"(method, kernel) = ({b.method}, {b.kernel}) was not found")

    return run


def run(b: Benchmark, progress: bool = True) -> Results:
    device = torch.device(b.device)
    dtype = getattr(torch, b.dtype)
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else f"cpu-{os.cpu_count()}"
    )
    revision = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().rstrip("\n")
    )
    results = Results(
        # Metadata
        device_name=device_name,
        torch_version=torch.__version__,
        revision=revision,
        error=None,
        # Stats
        duration=[],
        std=[],
    )
    try:
        Q = torch.empty(
            (b.batch_size, b.n_head, 1, b.head_dim), device=device, dtype=dtype
        )
        K = torch.randn(
            (b.batch_size, b.n_head, b.sequence_length, b.head_dim),
            device=device,
            dtype=dtype,
        )
        V = torch.randn_like(K)
        runner = get_runner(b, K, V)
        for _ in tqdm.tqdm(list(range(b.reps)), disable=not progress):
            torch.randn(*Q.shape, out=Q)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.time()
            y = runner(Q)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            results.duration.append(time.time() - t0)
            results.std.append(float(y.std()))
    except torch.cuda.OutOfMemoryError as error:
        results.error = repr(error)
    return results
