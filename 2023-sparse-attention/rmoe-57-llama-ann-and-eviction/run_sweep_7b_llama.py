import os
import itertools
from typing import Iterable

import torch

import llminference.experiments as xp

from llminference.ann_attention import LowRank, SparseQ

torch.set_num_threads(32)


def _sparsities() -> Iterable[xp.Sparsity]:
    yield from itertools.chain(
        (
            xp.Sparsity(
                "ann_llama",
                k=k,
                local_k=int(0.25 * k),
                score=LowRank.Settings(rank),
            )
            for k in [64, 128, 256, 512]
            for rank in [8, 16, 32, 64]
        ),
        (
            xp.Sparsity(
                "ann_llama",
                k=k,
                local_k=int(0.25 * k),
                score=SparseQ.Settings(rank),
            )
            for k in [64, 128, 256, 512]
            for rank in [8, 16, 32, 64]
        )
    )

configs = [
    xp.Experiment(
        name="RMOE-57-llama-ann-and-eviction",
        task=task,
        model=f"meta-llama/Llama-2-{model_scale}-hf",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=4),
    )
    for model_scale in ["7b"]
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
os.environ.update(WANDB_SILENT="true")

xp.run_many(configs)
