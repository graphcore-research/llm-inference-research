import os
import subprocess
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import tqdm
from torch import Tensor

# Methods


def attn(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    QK = (Q @ K.transpose(2, 3)).div_(Q.shape[-1] ** 0.5)
    return torch.softmax(QK, dim=-1) @ V


def gather(t: Tensor, dim: int, i: Tensor) -> Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


def sparq_attn(
    Q: Tensor, K1: Tensor, K2: Tensor, V: Tensor, V_mean: Tensor, r: int, k: int
) -> Tensor:
    # 1. Approximate attention scores using r largest components of Q
    absQ = torch.abs(Q)
    absQ_hat, i1 = torch.topk(absQ, r, -1)
    Q_hat, K_hat = gather(Q, -1, i1), gather(K1, -1, i1)
    scale = torch.sqrt(
        Q.shape[-1]
        * absQ_hat.sum(dim=-1, keepdim=True)
        / absQ.sum(dim=-1, keepdim=True)
    )
    s_hat = torch.softmax((Q_hat @ K_hat.transpose(-1, -2)).div_(scale), dim=-1)

    # 2. Gather top k positions based on approximate attention scores & run attention
    s_hat_i2, i2 = torch.topk(s_hat, k, -1)
    iKV = i2[..., 0, :, None]
    K, V = gather(K2, -2, iKV), gather(V, -2, iKV)
    y_ = attn(Q, K, V)

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


def get_runner(b: Benchmark, K: Tensor, V: Tensor) -> Callable[[Tensor], Tensor]:
    if b.method == "sparq":
        if b.k1 is None or b.k2 is None or b.store_k_twice is None:
            raise ValueError("Must specify {k1, k2, store_k_twice} for sparq attention")
        V_mean = V.mean(dim=-2, keepdim=True)
        K1 = K2 = K
        if b.store_k_twice:
            K1 = K.swapdims(-1, -2).contiguous().swapdims(-1, -2)

    attn_compiled = torch.compile(attn)
    sparq_attn_compiled = torch.compile(sparq_attn)

    def run(Q: Tensor) -> Tensor:
        if b.method == "empty" and b.kernel == "empty":
            return Q
        if b.method == "dense" and b.kernel == "vanilla":
            return attn(Q, K, V)
        if b.method == "dense" and b.kernel == "compiled":
            return attn_compiled(Q, K, V)
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
        if b.method == "sparq" and b.kernel == "vanilla":
            assert b.k1 and b.k2
            return sparq_attn(Q, K1, K2, V, V_mean=V_mean, r=b.k1, k=b.k2)
        if b.method == "sparq" and b.kernel == "compiled":
            assert b.k1 and b.k2
            return sparq_attn_compiled(Q, K1, K2, V, V_mean=V_mean, r=b.k1, k=b.k2)
        raise ValueError(f"(method, kernel) = ({b.method}, {b.kernel}) was not found")

    return run


def run(b: Benchmark, progress: bool = True) -> Results:
    device = torch.device(b.device)
    dtype = getattr(torch, b.dtype)

    Q = torch.empty((b.batch_size, b.n_head, 1, b.head_dim), device=device, dtype=dtype)
    K = torch.randn(
        (b.batch_size, b.n_head, b.sequence_length, b.head_dim),
        device=device,
        dtype=dtype,
    )
    V = torch.randn_like(K)
    runner = get_runner(b, K, V)

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
        # Stats
        duration=[],
        std=[],
    )
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
    return results
