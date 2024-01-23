import pandas as pd
import numpy as np


def gpu_kernel_name(s: pd.Series) -> str:
    if s.method == "dense":
        return "attn-" + dict(vanilla="torch", mem_efficient="mem-efficient").get(
            s.kernel, s.kernel
        )
    if s.method == "sparq":
        return (
            "sparq-attn-"
            + {
                ("vanilla", "custom"): "triton",
                ("vanilla", "torch"): "torch",
                ("compiled", "torch"): "compiled",
                ("compiled", "torch"): "compiled",
            }[(s.kernel, s.gather_matmul)]
            + ("-oneK" if not s.store_k_twice else "")
        )


ipu = (
    pd.read_json("data/20240123_sweep_m2000.jsonl", lines=True)
    .pipe(lambda d: d[~d.duration.isna()])
    .pipe(lambda d: d.assign(store_k_twice=d.kernel.apply(lambda x: "sparq" in x)))
    .drop(columns=["error", "n_ipu"])
)
cpu_gpu = (
    pd.concat(
        [
            pd.read_json(f, lines=True)
            for f in [
                "data/20240119_sweep_cpu.jsonl",
                "data/20240111_sweep_a10g.jsonl",
                "data/20240114_sweep_a100.jsonl",
            ]
        ]
    )
    .reset_index(drop=True)
    .pipe(lambda d: d[d.method != "empty"])
    .pipe(lambda d: d[d.duration.apply(len) != 0])
    .pipe(
        lambda d: d.assign(
            kernel=d.apply(gpu_kernel_name, axis=1),
            store_k_twice=d.store_k_twice.apply(lambda x: not np.isnan(x) and bool(x)),
            inner_reps=1,
        )
    )
    .drop(columns=["method", "gather_matmul", "error"])
)
df = (
    pd.concat([ipu, cpu_gpu])
    .reset_index(drop=True)
    .pipe(
        lambda d: d.assign(
            duration=d.duration.apply(np.mean),
            duration_stderr=d.duration.apply(lambda x: np.std(x) / len(x) ** 0.5),
        )
    )
    .pipe(lambda d: d[sorted(d.columns)])
)
print(f"Rows: {len(df)}")
print("Columns:", list(df.columns))
print("Kernels:", sorted(df.kernel.unique()))
df.to_json("results.json")
