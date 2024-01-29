import os
import sys

from typing import Iterable
import llminference.experiments as xp

device_no = int(sys.argv[1])


def _sparsities() -> Iterable[xp.Sparsity]:
    # Dense
    yield xp.Sparsity("dense")

    # Sparse-Q
    for rank in [16, 32, 64]:
        for k in [64, 128]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                reallocate_to_mean_value=False,
                score="sparse_q",
                rank=rank,
            )

    # Eviction + Local
    for k in [128, 192, 256, 384, 512, 768]:
        yield xp.Sparsity("eviction", k=k, local_k=int(0.25 * k), strategy="sum_weight")
        yield xp.Sparsity("local", k=k, initial_k=16)

    # Sparse-V
    for k in [2, 8, 32, 128, 256]:
        yield xp.Sparsity("sparse_v", k=k)


tasks = [
    xp.Task("wikitext_bpc", shots=0, samples=500, confusion_contexts=0),
    xp.Task("squad", shots=1, samples=4000, confusion_contexts=7),
    xp.Task("triviaqa", shots=0, samples=2992, confusion_contexts=0),
    xp.Task("cnn_dailymail", shots=0, samples=500, confusion_contexts=0),
    xp.Task("repetition", shots=0, samples=1000, confusion_contexts=0),
]

model = "mistralai/Mistral-7B-v0.1"
execution = xp.Execution(
    device=f"cuda:{device_no}",
    dtype="float16",
    batch_size=1,
    pipeline_stages=1,
    wandb=True,
)


configs = [
    xp.Experiment(
        name="RMOE-74-mistral-full-sweep",
        task=task,
        model=model,
        sparsity=sparsity,
        execution=execution,
    )
    for task in tasks
    for sparsity in _sparsities()
]

# Allocate every 4th experiment to each GPU
configs = configs[device_no::4]

print(f"Running {len(configs)} experiments on GPU no {device_no}")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()

os.environ.update(WANDB_SILENT="true")

xp.run_many(configs)
