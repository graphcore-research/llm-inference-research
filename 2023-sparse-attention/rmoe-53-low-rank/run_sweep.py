import os
from typing import Iterable

import torch

import llminference.experiments as xp

torch.set_num_threads(32)


def _sparsities() -> Iterable[xp.Sparsity]:
    yield from (
        xp.Sparsity(
            "ann_low_rank",
            k=k,
            local_k=int(0.25 * k),
            rank=rank,
            init=init,
        )
        for init in ["orthonormal", "normal"]
        for k in [64, 128, 256, 512]
        for rank in [8, 16, 32, 64]
    )


configs = [
    xp.Experiment(
        name="RMOE-53-low-rank-v4",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=10),
    )
    for scale in ["1.4b"]
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
# for config in configs:
#     print(config.sparsity.__dict__)
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
