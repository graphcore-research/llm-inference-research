import os
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


def gather(t: Tensor, dim: int, i: int) -> Tensor:
    dim += (dim < 0) * t.ndim
    return t.gather(dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1 :]))


def sparq_attn(
    Q: Tensor, K1: Tensor, K2: Tensor, V: Tensor, V_mean: Tensor, r: int, k: int
) -> Tensor:
    # 1. Approximate attention scores using r largest components of Q
    i1 = torch.topk(torch.abs(Q), r, -1).indices
    Q_hat, K_hat = gather(Q, -1, i1), gather(K1, -1, i1)
    scale = torch.sqrt(
        Q.shape[-1]
        * torch.abs(Q_hat).sum(dim=-1, keepdim=True)
        / torch.abs(Q).sum(dim=-1, keepdim=True)
    )
    s_hat = torch.softmax(Q_hat @ K_hat.transpose(-1, -2) / scale, dim=-1)

    # 2. Gather top k positions based on approximate attention scores & run attention
    i2 = torch.topk(s_hat, k, -1).indices
    iKV = i2[..., 0, :, None]
    K, V = gather(K2, -2, iKV), gather(V, -2, iKV)
    y_ = attn(Q, K, V)

    # 3. Estimate the total score of the top k, and interpolate with V_mean
    alpha = gather(s_hat, -1, i2).sum(-1, keepdim=True)
    return alpha * y_ + (1 - alpha) * V_mean


# Execution


@dataclass
class Benchmark:
    method: str  # "empty|dense|sparq"
    kernel: str  # "empty|auto|flash|math|mem_efficient|vanilla"
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
    rank: Optional[int] = None
    k_values: Optional[int] = None
    double_k: Optional[bool] = None

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


def get_runner(b: Benchmark, K: Tensor, V: Tensor) -> Callable[[Tensor], Tensor]:
    if b.method == "sparq" and b.kernel == "vanilla":
        if b.rank is None or b.k_values is None or b.double_k is None:
            raise ValueError(
                "Must specify {rank, k_values, double_k} for sparq attention"
            )
        V_mean = V.mean(dim=-2, keepdim=True)
        K1 = K2 = K
        if b.double_k:
            K1 = K.swapdims(-1, -2).contiguous().swapdims(-1, -2)

    def run(Q: Tensor) -> Tensor:
        if b.method == "empty" and b.kernel == "empty":
            return Q
        if b.method == "dense" and b.kernel == "vanilla":
            return attn(Q, K, V)
        if b.method == "dense" and b.kernel in [
            "auto",
            "flash",
            "math",
            "mem_efficient",
        ]:
            with torch.backends.cuda.sdp_kernel(
                enable_flash=(kernel == "auto" or kernel == "flash"),
                enable_math=(kernel == "auto" or kernel == "math"),
                enable_mem_efficient=(kernel == "auto" or kernel == "mem_efficient"),
            ):
                return torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        if b.method == "sparq" and b.kernel == "vanilla":
            return sparq_attn(Q, K1, K2, V, V_mean=V_mean, r=b.rank, k=b.k_values)
        raise ValueError(f"(method, kernel) = ({b.method}, {b.kernel}) was not found")

    return run


def run(b: Benchmark) -> List[float]:
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

    results = Results(
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else f"cpu-{os.cpu_count()}",
        duration=[],
        std=[],
    )
    for _ in tqdm.tqdm(list(range(b.reps))):
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
