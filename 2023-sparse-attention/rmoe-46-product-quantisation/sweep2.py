import os
from typing import Iterable

import torch

import llminference.experiments as xp
from llminference.pq_attention import IndexSettings

torch.set_num_threads(32)


def _sparsities() -> Iterable[xp.Sparsity]:
    yield from (
        xp.Sparsity(
            "pq",
            k=k,
            local_k=int(0.25 * k),
            index=IndexSettings(n_centroids=n_centroids, n_subvectors=n_subvectors, dot_product_search=dot_product_search),
        )
        for dot_product_search in [True]
        for k in [32, 16]
        for n_subvectors in [2, 4, 8, 16, 32]
        for n_centroids in [10, 20, 50, 100, 200]
    )


configs = [
    xp.Experiment(
        name="RMOE-46-pq-attention",
        task=task,
        model=f"EleutherAI/pythia-{scale}",
        sparsity=sparsity,
        execution=xp.Execution.auto(batch_size=5),
    )
    for scale in ["1.4b"]
    for task in [
        xp.Task("squad", shots=1, samples=500),
    ]
    for sparsity in _sparsities()
]
print(f"Running {len(configs)} experiments")
# for config in configs:
#     print(config.sparsity.__dict__)
os.environ.update(WANDB_SILENT="true")
xp.run_many(configs)
