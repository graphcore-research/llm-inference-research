import os
from typing import Iterable

import torch

import llminference.experiments as xp

torch.set_num_threads(16)


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for k in [64, 128, 256]:
        yield xp.Sparsity(
            "eviction",
            k=k,
            local_k=int(0.25 * k),
            strategy="sum_weight",
        )
    for rank in [16, 32, 64]:
        for k in [32]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                reallocate_to_mean_value=True,
                score="sparse_q",
                rank=rank,
            )


configs = [
    xp.Experiment(
        name="RMOE-63-perplexity-v1",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=10),
    )
    for task in [
        xp.Task("wikitext_bpc", shots=0, samples=200),
    ]
    for scale in ["410m", "1b", "1.4b"]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs, n_workers=4)
