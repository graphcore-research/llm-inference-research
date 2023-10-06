import os
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for k in [32, 64, 128, 256, 512, 1024]:
        yield xp.Sparsity("eviction", k=k, local_k=int(0.25 * k))
        yield xp.Sparsity("sparse_v", k=k, apply_after_softmax=True, add_avg=False)


configs = [
    xp.Experiment(
        name="RMOE-51-baselines",
        task=task,
        model="EleutherAI/pythia-6.9b",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=2),
    )
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        xp.Task("triviaqa", shots=0, samples=1000),
        xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
