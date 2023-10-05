import os
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for k in [32, 64, 128, 256, 512, 1024]:
        for local_ratio in [0.25, 0.5]:
            yield xp.Sparsity("eviction", k=k, local_k=int(local_ratio * k))
        for apply_after_softmax, add_avg in [
            (False, False),
            (True, False),
            (True, True),
        ]:
            yield xp.Sparsity(
                "sparse_v",
                k=k,
                apply_after_softmax=apply_after_softmax,
                add_avg=add_avg,
            )


configs = [
    xp.Experiment(
        name="RMOE-51-baselines",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=5 if scale in {"2.8b"} else 10),
    )
    for scale in ["1b", "1.4b", "2.8b"]
    for task in [
        xp.Task("squad", shots=1, samples=1000),
        xp.Task("triviaqa", shots=0, samples=1000),
        xp.Task("cnn_dailymail", shots=0, samples=500),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
os.makedirs("./wandb-sweep", exist_ok=True)
os.environ.update(WANDB_SILENT="true", WANDB_DIR="./wandb-sweep")
xp.run_many(configs)
