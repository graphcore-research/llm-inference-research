import argparse
import os
from typing import Iterable

import llminference.experiments as xp


def _sparsities() -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for rank in [16, 32, 64]:
        for k in [64, 128]:
            yield xp.Sparsity(
                "ann",
                k=k,
                local_k=int(0.25 * k),
                reallocate_to_mean_value=True,
                score="sparse_q",
                rank=rank,
            )
    for k in [128, 192, 256, 384, 512, 768]:
        yield xp.Sparsity(
            "eviction",
            k=k,
            local_k=int(0.25 * k),
            strategy="sum_weight",
        )

tasks = [
    xp.Task("wikitext_bpc", shots=0, samples=500),
    xp.Task("repetition", shots=0, samples=1000),
    xp.Task("squad", shots=1, samples=4000),
    xp.Task("triviaqa", shots=0, samples=2992),
    xp.Task("cnn_dailymail", shots=0, samples=500),
]
models = [
    ("meta-llama/Llama-2-7b-hf", 1),
    ("EleutherAI/pythia-6.9b", 1),
    ("EleutherAI/pythia-2.8b", 2),
    ("EleutherAI/pythia-1.4b", 5),
]
task_name_to_task = {task.name: task for task in tasks}
model_name_to_tuple = {name.split("/")[1]: (name, b) for name, b in models}
parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=task_name_to_task.keys(), nargs="+")
parser.add_argument("--model", choices=model_name_to_tuple.keys(), nargs="+")
args = parser.parse_args()
tasks = [task_name_to_task[t] for t in args.task]
models = [model_name_to_tuple[m] for m in args.model]

configs = [
    xp.Experiment(
        name="RMOE-68-workshop-sweep",
        task=task,
        model=model,
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=batch_size),
    )
    for model, batch_size in models
    for task in tasks
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
