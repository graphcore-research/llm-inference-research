import argparse
import os

import llminference.experiments as xp

tasks = [
    xp.Task("squad", shots=1, samples=4000),
    xp.Task("triviaqa", shots=0, samples=2992),
]
models = [
    ("meta-llama/Llama-2-7b-hf", 1),
]
local_ks = [128, 192, 256, 384, 512, 768]
parser = argparse.ArgumentParser()
parser.add_argument("-k", choices=local_ks, type=int, nargs="+")
args = parser.parse_args()
local_ks = args.k

configs = [
    xp.Experiment(
        name="RMOE-68-workshop-sweep",
        task=task,
        model=model,
        sparsity=xp.Sparsity("local", k=k, initial_k=16),
        execution=xp.Execution.auto(batch_size=batch_size),
    )
    for model, batch_size in models
    for task in tasks
    for k in local_ks
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(config.task.name, config.model, config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
