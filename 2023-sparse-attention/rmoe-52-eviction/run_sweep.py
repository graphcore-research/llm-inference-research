import os
from typing import Iterable

import torch

import llminference as L
import llminference.experiments as xp

torch.set_num_threads(32)


def _sparsities() -> Iterable[xp.Sparsity]:
    return (
        xp.Sparsity(
            "eviction",
            k=k,
            local_k=int(0.25 * k),
            importance=L.eviction_attention.Importance.Settings(
                metric=metric,
                metric_threshold=threshold,
                reduction=reduction,
                reduction_decay=decay,
            ),
        )
        for k in [128, 256, 512, 1024]
        for reduction in ["mean", "sum", "decaymax"]
        for metric in ["weight", "log_weight", "threshold", "global_threshold"]
        for threshold in ([1, 1 / 2] if "threshold" in metric else [None])
        for decay in ([1000, 200] if reduction == "decaymax" else [None])
        # Only "mean" makes sense for "log_weight"
        if not (metric == "log_weight" and reduction != "mean")
        # All decays are the same when using "threshold"
        if not ("threshold" in metric and decay == 200)
        # These runs failed last time
        if (reduction == "decaymax") or ("threshold" in metric)
    )


configs = [
    xp.Experiment(
        name="RMOE-52-eviction-v3",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=5),
    )
    for scale in ["1.4b"]
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        # xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
# for config in configs:
#     print(config.sparsity.__dict__)
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
