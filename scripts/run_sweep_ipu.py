# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import tqdm


def kv_size_bytes(c: Dict[str, Any]) -> int:
    element_size = dict(float16=2, float32=4)[c["dtype"]]
    return (
        2
        * c["batch_size"]
        * c["n_head"]
        * c["sequence_length"]
        * c["head_dim"]
        * element_size
    )


def run_benchmark(config: Dict[str, Any]) -> Dict[str, Any]:
    pconfig = config.copy()
    for k in ["chunk_size", "k1", "k2"]:
        pconfig.setdefault(k, 0)
    p = subprocess.Popen(
        ["./build/sparq_benchmark", json.dumps(pconfig)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    timeout = 60 * 60  # seconds
    try:
        stdout, stderr = p.communicate(timeout=timeout)
        if p.returncode:
            result = dict(error=str(stdout.decode()) + "\n" + str(stderr.decode()))
        else:
            result = json.loads(stdout.decode())
    except subprocess.TimeoutExpired:
        p.terminate()
        p.wait()
        result = dict(error=f"timeout expired ({timeout} s)")
    revision = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().rstrip("\n")
    )
    return dict(
        **config,
        device="ipu",
        **result,
        revision=revision,
    )


def kernels() -> Iterable[Dict[str, Any]]:
    yield dict(kernel="attn-local")
    # Performance is not very sensitive to chunk size
    yield dict(kernel="attn-remote", chunk_size=128)
    for k1 in [16, 32, 64]:
        for k2 in [64, 128, 256, 512]:
            yield dict(kernel="sparq-attn", k1=k1, k2=k2)


benchmarks = [
    dict(
        n_ipu=1,
        n_head=32,
        head_dim=128,
        dtype="float16",
        batch_size=batch_size,
        sequence_length=sequence_length,
        **kernel,
        inner_reps=4 if kernel["kernel"] == "attn-remote" else (1024 // batch_size),
        warmup=2,
        reps=10,
    )
    for sequence_length in (2**n for n in range(10, 17))
    for batch_size in [1, 4, 16, 64]
    for kernel in kernels()
]
# Remove benchmarks that will certainly OOM for attn-local
benchmarks = [
    b
    for b in benchmarks
    if not (b["kernel"] == "attn-local" and kv_size_bytes(b) > 900e6)
]

if __name__ == "__main__":
    print(f"Running {len(benchmarks)} benchmarks", file=sys.stderr)
    for b in benchmarks:
        print(b, file=sys.stderr)
    subprocess.check_call(["ninja", "-f", "ipu/build.ninja"])
    with Path("sweep_ipu.jsonl").open("w") as f:
        for b in tqdm.tqdm(benchmarks):
            f.write(json.dumps(run_benchmark(b)) + "\n")
            f.flush()
