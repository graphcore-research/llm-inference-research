import json
import sys
from pathlib import Path
from typing import List

import sparq_benchmark as B
import torch
import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
methods = [dict(method="empty"), dict(method="dense")] + [
    dict(method="sparq", k1=k1, k2=k2, store_k_twice=store_k_twice)
    for k1 in [16, 32, 64]
    for k2 in [64, 128]
    for store_k_twice in [False, True]
]
dtypes = dict(cpu=["float32"], cuda=["float32", "float16"])[device]


def get_kernels(method: str) -> List[str]:
    if method == "empty":
        return ["empty"]
    if method == "dense" and device == "cpu":
        return ["vanilla", "compiled"]
    if method == "dense" and device == "cuda":
        return ["vanilla", "compiled", "flash", "math", "mem_efficient"]
    if method == "sparq":
        return ["vanilla", "compiled"]
    assert False, f"bad method {method}"


benchmarks = [
    B.Benchmark(
        **method,
        kernel=kernel,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_head=32,
        head_dim=128,
        dtype=dtype,
        device=device,
        reps=dict(cuda=1000, cpu=100)[device],
    )
    for sequence_length in [2048, 4096, 8192, 16384]
    for batch_size in [1, 4, 16, 64]
    for dtype in dtypes
    for method in methods
    for kernel in get_kernels(method["method"])
    if (kernel, dtype) not in {("flash", "float32")}
]

if __name__ == "__main__":
    print(f"Running {len(benchmarks)} benchmarks", file=sys.stderr)
    # for b in benchmarks:
    #     print(b)
    with Path("sweep.jsonl").open("w") as f:
        for benchmark in tqdm.tqdm(benchmarks):
            results = B.run(benchmark, progress=False)
            f.write(json.dumps(dict(**benchmark.__dict__, **results.__dict__)) + "\n")
            f.flush()
