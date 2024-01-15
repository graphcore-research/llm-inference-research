import os
import sys
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    for k in [32, 64, 128, 256]:
        for rank in [16, 32, 64]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                reallocate_to_mean_value="L2",
                score="sparse_q",
                rank=rank,
            )


task = [
    xp.Task("squad", shots=1, samples=1000),
    xp.Task("wikitext_bpc", shots=0, samples=100),
    xp.Task("triviaqa", shots=0, samples=1000),
][int(sys.argv[1])]
configs = [
    xp.Experiment(
        name="RMOE-64-sparse-q-L2",
        task=task,
        model=model,
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=b),
    )
    for model, b in [
        ("meta-llama/Llama-2-7b-hf", 1),
        ("EleutherAI/pythia-6.9b", 1),
        ("EleutherAI/pythia-1.4b", 5),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
