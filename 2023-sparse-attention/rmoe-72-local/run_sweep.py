import os
from typing import Iterable

import torch

import llminference.experiments as xp

torch.set_num_threads(32)


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for k in [128, 256, 512]:
        for initial_k in [0, 4, 16, 64]:
            yield xp.Sparsity("local", k=k, initial_k=initial_k)


configs = [
    xp.Experiment(
        name="RMOE-72-local",
        task=xp.Task("wikitext_bpc", shots=0, samples=100),
        model=model,
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=batch_size),
    )
    for model, batch_size in [
        # ("EleutherAI/pythia-1.4b", 10),
        ("meta-llama/Llama-2-7b-hf", 2),
    ]
    for sparsity in _sparsities()
    if "pythia" in model or "dense" in sparsity.name or sparsity.k == 256
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs, n_workers=5)
