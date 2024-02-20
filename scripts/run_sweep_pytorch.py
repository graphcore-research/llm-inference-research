# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import torch
import tqdm

import sparq_benchmark as B

device = "cuda" if torch.cuda.is_available() else "cpu"
dtypes = dict(cpu=["float32"], cuda=["float16"])[device]


def methods() -> Iterable[Dict[str, Any]]:
    yield dict(method="empty", kernel="empty")

    # Dense
    yield dict(method="dense", kernel="vanilla")
    if device == "cpu":
        # The compiled kernel is generally worse for CPU, so we skip it
        yield dict(method="dense", kernel="nn")
        gather_matmuls = ["torch"]
        sparq_kernels = ["vanilla"]
    if device == "cuda":
        yield dict(method="dense", kernel="compiled")
        yield dict(method="dense", kernel="flash")
        yield dict(method="dense", kernel="math")
        yield dict(method="dense", kernel="mem_efficient")
        gather_matmuls = ["torch", "custom"]
        sparq_kernels = ["vanilla", "compiled"]

    # SparQ
    for store_k_twice in [True, False]:
        for k1 in [16, 32, 64]:
            for k2 in [64, 128, 256, 512]:
                for gather_matmul in gather_matmuls:
                    for kernel in sparq_kernels:
                        if not (gather_matmul == "custom" and kernel == "compiled"):
                            yield dict(
                                method="sparq",
                                kernel=kernel,
                                k1=k1,
                                k2=k2,
                                store_k_twice=store_k_twice,
                                gather_matmul=gather_matmul,
                            )


benchmarks = [
    B.Benchmark(
        **settings,
        batch_size=batch_size,
        sequence_length=sequence_length,
        n_head=32,
        head_dim=128,
        dtype=dtype,
        device=device,
        reps=200,
        warmup=20,
    )
    for sequence_length in [1024, 2048, 4096, 8192, 16384, 65536]
    for batch_size in [1, 4, 16, 64]
    for dtype in dtypes
    for settings in methods()
    if (settings["kernel"], dtype) not in {("flash", "float32")}
]

if __name__ == "__main__":
    print(f"Running {len(benchmarks)} benchmarks", file=sys.stderr)
    for b in benchmarks:
        print(b, file=sys.stderr)
    with Path("sweep_pytorch.jsonl").open("w") as f:
        for benchmark in tqdm.tqdm(benchmarks):
            results = B.run(benchmark)
            f.write(json.dumps(dict(**benchmark.__dict__, **results.__dict__)) + "\n")
            f.flush()
