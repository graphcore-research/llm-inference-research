import os
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    for rank in [16, 32, 64]:
        for k in [32, 128, 512]:
            for add_remainder in ["zeros"]:
                yield xp.Sparsity(
                    "ann",
                    k=k,
                    local_k=int(0.25 * k),
                    add_remainder=add_remainder,
                    score="sparse_q",
                    rank=rank,
                )


configs = [
    xp.Experiment(
        name="RMOE-59-ann-reallocation",
        task=task,
        model=model,
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=1),
    )
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        # xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for model in ["meta-llama/Llama-2-7b-hf", "EleutherAI/pythia-6.9b"]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
