import json
import sys
from typing import List
from pathlib import Path
import tqdm

import sparq_benchmark as B

device = "cpu"
methods = [dict(method="empty"), dict(method="dense")] + [
    dict(method="sparq", rank=r, k_values=k, double_k=double_k)
    for r in [16, 32, 64]
    for k in [64, 128]
    for double_k in [False, True]
]
dtypes = dict(cpu=["float32"], cuda=["float32", "float16"])[device]
def get_kernels(method: str) -> List[str]:
    if method == "empty":
        return ["empty"]
    if method == "dense" and device == "cpu":
        return ["vanilla"]
    if method == "dense" and device == "cuda":
        return ["vanilla", "auto", "flash", "math", "mem_efficient"]
    if method == "sparq":
        return ["vanilla"]

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
        reps=100,
    )
    for dtype in dtypes
    for batch_size in [1, 4, 16]
    for sequence_length in [2048, 4096]
    for method in methods
    for kernel in get_kernels(method["method"])
]
print(f"Running {len(benchmarks)} benchmarks", file=sys.stderr)
with Path("sweep.jsonl").open("w") as f:
    for benchmark in tqdm.tqdm(benchmarks):
        results = B.run(benchmark, progress=False)
        f.write(json.dumps(dict(**benchmark.__dict__, **results.__dict__)) + "\n")
        f.flush()
