import os
import sys
from typing import Iterable

import llminference.experiments as xp

device_idx = int(sys.argv[1])

devices = [2, 3]


def _sparsities() -> Iterable[xp.Sparsity]:
    # Dense
    yield xp.Sparsity("dense")

    # Sparse-Q
    for reallocate_to_mean_value in [True, False]:
        k = 128
        for rank in [8, 16, 32, 64]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                reallocate_to_mean_value=reallocate_to_mean_value,
                score="sparse_q",
                rank=rank,
            )

    # Eviction + Local
    for k in [192, 256, 384, 512, 768]:
        yield xp.Sparsity("eviction", k=k, local_k=int(0.25 * k), strategy="sum_weight")
        yield xp.Sparsity("local", k=k, initial_k=16, apply_after_softmax=False)

    # Sparse-V
    for k in [2, 8, 32, 128, 256]:
        yield xp.Sparsity(
            "sparse_v", k=k, apply_after_softmax=True, reallocate_to_mean=False
        )


tasks = [
    xp.Task("wikitext_bpc", shots=0, samples=500, confusion_contexts=0),
    xp.Task("squad", shots=1, samples=4000, confusion_contexts=7),
    xp.Task("triviaqa", shots=0, samples=2992, confusion_contexts=0),
    xp.Task("cnn_dailymail", shots=0, samples=500, confusion_contexts=0),
    xp.Task("repetition", shots=0, samples=1000, confusion_contexts=0),
]

model = "google/gemma-7b"

execution = xp.Execution(
    device=f"cuda:{devices[device_idx]}",
    dtype="float16",
    batch_size=1,
    pipeline_stages=1,
    wandb=True,
)

configs = [
    xp.Experiment(
        name="RMOE-94-gemma-sweep-v1",
        task=task,
        model=model,
        sparsity=sparsity,
        execution=execution,
    )
    for task in tasks
    for sparsity in _sparsities()
]

# Allocate every nth experiment to each GPU
configs = configs[device_idx :: len(devices)]

print(f"Running {len(configs)} experiments on GPU no {devices[device_idx]}")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()

os.environ.update(WANDB_SILENT="true")

xp.run_many(configs)
