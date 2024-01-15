import os
from typing import Iterable

import torch

import llminference.ann_attention as ann
import llminference.experiments as xp

torch.set_num_threads(32)


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for k in [16, 32, 64, 128, 256, 512]:
        yield xp.Sparsity("sparse_v", k=k, apply_after_softmax=False, add_avg=False)
        yield xp.Sparsity("eviction", k=k, local_k=int(0.25 * k), strategy="sum_weight")
        yield xp.Sparsity(
            "ann", k=k, local_k=int(0.25 * k), score=ann.SparseQ.Settings(rank=32)
        )
    for rank in [8, 16, 64]:
        k = 32
        yield xp.Sparsity(
            "ann", k=k, local_k=int(0.25 * k), score=ann.SparseQ.Settings(rank=rank)
        )


execution = xp.Execution.auto(batch_size=5)
execution.wandb = True
configs = [
    xp.Experiment(
        name="RMOE-62-repetition-task-test",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=execution,
    )
    for scale in ["1b", "1.4b", "2.8b"]
    for task in [
        xp.Task("repetition", shots=0, samples=500),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
# for config in configs:
#     print(config.sparsity.__dict__)
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
