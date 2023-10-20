import os
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    for score in ["sparse_q", "low_rank"]:
        for local_ratio in [0, 0.25, 0.5, 0.75, 1.0]:
            k, rank = (32, 16) if score == "sparse_q" else (128, 32)
            yield xp.Sparsity(
                "ann", k=k, local_k=int(local_ratio * k), score=score, rank=rank
            )
    for strategy in ["sum_weight", "lru"]:
        for local_ratio in [0, 0.25, 0.5, 0.75]:
            k = 256
            yield xp.Sparsity(
                "eviction", k=k, local_k=int(local_ratio * k), strategy=strategy
            )


configs = [
    xp.Experiment(
        name="RMOE-58-ann-local",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=batch_size),
    )
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for scale, batch_size in [("2.8b", 5)]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
# for config in configs:
#     print(config.task.name, config.model, config.sparsity.__dict__)
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
