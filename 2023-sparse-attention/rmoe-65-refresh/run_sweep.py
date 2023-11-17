import os
import sys
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for k in [64, 128, 256, 512]:
        yield xp.Sparsity(
            "eviction",
            k=k,
            local_k=int(0.25 * k),
            strategy="sum_weight",
        )
    for k in [32, 64, 128, 256]:
        for rank in [16, 32, 64]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                reallocate_to_mean_value=True,
                score="sparse_q",
                rank=rank,
            )


task = [
    xp.Task("squad", shots=1, samples=1000),
    xp.Task("cnn_dailymail", shots=0, samples=500),
    xp.Task("wikitext_bpc", shots=0, samples=500),
    xp.Task("triviaqa", shots=0, samples=1000),
][int(sys.argv[1])]
configs = [
    xp.Experiment(
        name="RMOE-65-refresh",
        task=task,
        model=model,
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=b),
    )
    for model, b in [
        ("meta-llama/Llama-2-7b-hf", 1),
        ("EleutherAI/pythia-6.9b", 1),
        ("EleutherAI/pythia-2.8b", 5),  # note: b=5 is too large!
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
