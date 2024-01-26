import os
from typing import Iterable

import transformers

import llminference.experiments as xp

model = "lmsys/vicuna-7b-v1.5-16k"

tasks = [  # "mean_length" computed from the dataset
    {"confusion_contexts": 63, "mean_length": 12187},
    {"confusion_contexts": 31, "mean_length": 6134},
    {"confusion_contexts": 15, "mean_length": 3102},
    {"confusion_contexts": 7, "mean_length": 1590},
    {"confusion_contexts": 3, "mean_length": 830},
]

config = transformers.AutoConfig.from_pretrained(model)
head_size = config.hidden_size // config.num_attention_heads


def _sparsities(mean_length: int) -> Iterable[xp.Sparsity]:
    yield xp.Sparsity("dense")
    for compression_ratio in [1 / 4]:  # , 1 / 2
        eviction_k = int(mean_length * compression_ratio)
        yield xp.Sparsity(
            "eviction",
            k=eviction_k,
            local_k=int(0.25 * eviction_k),
            strategy="sum_weight",
        )
        # ann_rank = int(head_size * compression_ratio)
        # ann_k = int(mean_length * compression_ratio) // 2
        ann_rank = head_size // 4
        ann_k = int(mean_length * (compression_ratio - ann_rank / head_size / 2))
        yield xp.Sparsity(
            "ann",
            rank=ann_rank,
            k=ann_k,
            local_k=int(0.25 * ann_k),
            reallocate_to_mean_value=True,
            score="sparse_q",
        )


configs = [
    xp.Experiment(
        name="RMOE-78-long-context-v3",
        task=xp.Task(
            "squad_train",
            shots=1,
            samples=1000,
            confusion_contexts=t["confusion_contexts"],
        ),
        model=model,
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=1),
    )
    for t in tasks
    for sparsity in _sparsities(t["mean_length"])
]
print(f"Running {len(configs)} experiments")
for config in configs:
    print(f"confusion_contexts: {config.task.confusion_contexts}", config.sparsity)
print()
os.environ.update(WANDB_SILENT="true")
assert os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:cudaMallocAsync"
xp.run_many(configs)
