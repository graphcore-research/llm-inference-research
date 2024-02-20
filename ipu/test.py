# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import json
import subprocess
from pathlib import Path


def test_ipu_benchmark() -> None:
    sparq_benchmark_exe = Path(__file__).parent.parent / "build/sparq_benchmark"
    mean_duration = {}
    for kernel in ["attn-local", "attn-remote", "sparq-attn"]:
        config = dict(
            n_ipu=1,
            batch_size=1,
            n_head=32,
            head_dim=128,
            sequence_length=256,
            dtype="float16",
            kernel=kernel,
            chunk_size=64 if kernel == "attn-remote" else 0,
            k1=8 if kernel == "sparq-attn" else 0,
            k2=16 if kernel == "sparq-attn" else 0,
            inner_reps=5,
            warmup=2,
            reps=10,
        )
        proc = subprocess.Popen(
            [str(sparq_benchmark_exe), json.dumps(config)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = proc.communicate()
        if proc.returncode:
            raise Exception(
                "sparq_benchmark failed, with:"
                f"\nSTDOUT:\n{stdout.decode()}"
                f"\nSTDERR:\n{stderr.decode()}"
            )
        result = json.loads(stdout.decode())
        assert len(result["duration"]) == 10
        mean_duration[kernel] = sum(result["duration"]) / len(result["duration"])

    # Even on this small example, the speed difference should be clear
    assert (
        mean_duration["attn-local"]
        < mean_duration["sparq-attn"]
        < mean_duration["attn-remote"]
    )
