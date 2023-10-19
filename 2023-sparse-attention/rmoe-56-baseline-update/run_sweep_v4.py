import os
from typing import Iterable

import llminference.ann_attention as ann
import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for k in [4, 16, 64]:
        yield xp.Sparsity("sparse_v", k=k, apply_after_softmax=False, add_avg=False)
    for k in [16, 32, 64]:
        for rank in [8, 16, 32, 64]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                score=ann.SparseQ.Settings(rank=rank),
            )
    for k in [64, 128, 256]:
        for rank in [16, 32, 64]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                score=ann.LowRank.Settings(rank=rank),
            )
    for k in [64, 128, 256, 512]:
        for strategy in ["sum_weight", "lru"]:
            yield xp.Sparsity("eviction", k=k, local_k=int(0.25 * k), strategy=strategy)


configs = [
    xp.Experiment(
        name="RMOE-56-baselines",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=batch_size),
    )
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for scale, batch_size in [("6.9b", 2)]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
# for config in configs:
#     print(config.task.name, config.model, config.sparsity.__dict__)
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
