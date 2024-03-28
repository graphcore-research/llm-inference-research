import os
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for compression in [4, 8]:
        yield xp.Sparsity(
            "ann",
            rank=int(128 / compression),
            k=0.5 / compression,
            min_k=128,
            local_k=0.25,
            reallocate_to_mean_value=True,
            score="sparse_q",
        )
        yield xp.Sparsity(
            "eviction",
            k=1 / compression,
            min_k=128,
            local_k=0.25,
            strategy="sum_weight",
        )
        yield xp.Sparsity(
            "local",
            k=1 / compression,
            min_k=128,
            initial_k=16,
            apply_after_softmax=False,
        )


configs = [
    xp.Experiment(
        name="RMOE-91-needle-v8",
        task=xp.Task("needle", shots=0, samples=17, confusion_contexts=0),
        model="togethercomputer/LLaMA-2-7B-32K",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=1),
    )
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
