import json
import os
from pathlib import Path
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
            method="max_components",
            init=None,
        )
        for k in [128, 256]
        for rank in [8, 16, 32, 64]
    )


dry_run = False
configs = [
    xp.Experiment(
        name="RMOE-55-sparse-compression",
        # task=xp.Task("squad", shots=1, samples=1000),
        task=xp.Task("cnn_dailymail", shots=0, samples=500),
        model="EleutherAI/pythia-1.4b",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=10),
    )
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
if dry_run:
    for config in configs:
        print(config.sparsity.__dict__)
else:
    # Path("result.json").write_text(json.dumps(xp.run_one(configs[0])))
    os.environ.update(WANDB_SILENT="true")
    xp.run_many(configs, n_workers=8)
