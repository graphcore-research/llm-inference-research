import os
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    k = 128
    yield xp.Sparsity(
        "ann",
        k=k,
        local_k=int(0.25 * k),
        reallocate_to_mean_value=True,
        score="sparse_q",
        rank=8,
    )


models = [
    ("meta-llama/Llama-2-7b-hf", 1),
    ("EleutherAI/pythia-6.9b", 1),
    ("EleutherAI/pythia-2.8b", 2),
    ("EleutherAI/pythia-1.4b", 5),
]
tasks = [
    xp.Task("squad", shots=1, samples=4000, confusion_contexts=7),
    xp.Task("repetition", shots=0, samples=1000, confusion_contexts=0),
    xp.Task("triviaqa", shots=0, samples=2992, confusion_contexts=0),
    xp.Task("cnn_dailymail", shots=0, samples=500, confusion_contexts=0),
    xp.Task("wikitext_bpc", shots=0, samples=500, confusion_contexts=0),
]

configs = [
    xp.Experiment(
        name="RMOE-68-lower-r",
        task=task,
        model=model,
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=batch_size),
    )
    for model, batch_size in models
    for task in tasks
    for sparsity in _sparsities()
]
configs = configs[int(os.environ["CUDA_VISIBLE_DEVICES"]) :: 4]

print(f"Running {len(configs)} experiments -> {set(c.name for c in configs)}")
for config in configs:
    print(config.model, config.task.name, config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
