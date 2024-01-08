import time
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor

from gather_matmul import _kernel_gather_inner_bmv, gather_inner_bmv
from sparq_benchmark import gather


def profile(fn: Callable[..., None], reps: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    duration = 0
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        duration += time.time() - t0
    return duration / reps


def reference_gather_inner_bmv(a: Tensor, b: Tensor, i: Tensor) -> Tensor:
    return gather(a, 2, i[:, None, :]) @ gather(b, 1, i[:, :, None])


def reference_slice(a: Tensor, b: Tensor, i: Tensor) -> Tensor:
    return a[:, :, : i.shape[1]] @ b[:, : i.shape[1], :]


if __name__ == "__main__":
    b, n, d, s = 32, 32, 128, 4096
    k1 = 16
    chunk = 512

    torch.manual_seed(42)
    A = torch.randn(b * n, 1, d, device="cuda", dtype=torch.float16)
    B = torch.randn(b * n, d, s, device="cuda", dtype=torch.float16)
    # B = B.transpose(-1, -2).contiguous().transpose(-1, -2)  # SLOW!
    I = torch.randint(0, d, (b * n, k1), device="cuda")

    bytes_rw = A.itemsize * b * n * (k1 * s + s)

    t = profile(lambda: reference_gather_inner_bmv(A, B, I), reps=100, warmup=10)
    print(f" reference_gather_inner_bmv: {bytes_rw / t / 1e9:.0f} GB/s")

    t = profile(lambda: reference_slice(A, B, I), reps=100, warmup=10)
    print(f"            reference_slice: {bytes_rw / t / 1e9:.0f} GB/s")

    t = profile(lambda: gather_inner_bmv(A, B, I, chunk=chunk), reps=100, warmup=10)
    print(f"gather_inner_bmv(chunk={chunk}): {bytes_rw / t / 1e9:.0f} GB/s")

    # Dump ptx, for interest
    out = Path("out/gather_matmul_benchmark")
    out.mkdir(parents=True, exist_ok=True)
    (cache,) = _kernel_gather_inner_bmv.cache.values()
    (kernel,) = cache.values()
    (out / "gather_inner_bmv.s").write_text(kernel.asm["ptx"])
