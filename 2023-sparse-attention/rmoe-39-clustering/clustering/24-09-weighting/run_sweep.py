import argparse
from functools import partial
from pathlib import Path
from typing import *

import torch

import llminference as L

CACHE = "/net/group/research/lukar/llminference-cache"


def run(
    dataset: str, model_scale: str, sparsity: str, k: Optional[int]
) -> Dict[str, Any]:
    # Dataset
    if dataset == "triviaqa":
        data = L.qa.TriviaQA.data(context="wiki")
        shots = 0
        examples = [L.qa.add_zero_shot_prompt(data[i]) for i in range(400)]
    if dataset == "squad":
        data = L.qa.SQuAD.data()
        shots = 1
        examples = [L.qa.add_few_shot_prompt(data[i], k=shots) for i in range(400)]
    if dataset == "cnn_dailymail":
        data = L.summarisation.CnnDailymail.data()
        shots = 0
        examples = [data[i] for i in range(200)]
    evaluate = (
        L.summarisation.evaluate
        if dataset == "cnn_dailymail"
        else partial(L.qa.evaluate, open_book=True)
    )

    # Base model
    batch_size = {"160m": 40, "410m": 25, "1b": 20, "1.4b": 20, "2.8b": 10}[model_scale]
    adapter = L.Adapter.from_pretrained(f"EleutherAI/pythia-{model_scale}")

    # Technique
    local_ratio = None
    use_cache = True
    if sparsity == "none":
        context = L.eval_adapter.null_model_context
    elif "eviction" in sparsity:
        local_ratio = 0.25
        use_cache = False
        adapter.model = L.eviction_attention.convert_gptneox(
            adapter.model,
            L.eviction_attention.Settings(k=k, local_k=int(local_ratio * k)),
        )
        context = L.eviction_attention.generation_context
    elif "clustering" in sparsity:
        weigh_keys = "weighted" in sparsity
        local_ratio = 0.25
        adapter.model = L.cluster_attention.convert_gptneox(
            adapter.model,
            L.cluster_attention.Settings(
                k=k, local_k=int(local_ratio * k), weigh_keys=weigh_keys
            ),
        )
        context = L.cluster_attention.generation_context

    else:
        context = L.eval_adapter.patch_for_model(
            "torch.nn.functional.softmax",
            L.sparse_attention.sparse_softmax_fixed_k,
            k=k,
            apply_after_softmax=("_after" in sparsity),
            add_avg=(sparsity == "sparsev_after_avg"),
        )

    # Evaluation
    return dict(
        dataset=dataset,
        shots=shots,
        model_scale=model_scale,
        sparsity=sparsity,
        k=k,
        local_ratio=local_ratio,
        results=list(
            evaluate(
                adapter,
                examples,
                batch_size=batch_size,
                generation_context=context,
                use_cache=use_cache,
                cache_dir=CACHE,
            )
        ),
    )


if __name__ == "__main__":
    dataset = "squad"

    def _sparsities() -> Iterable[Dict[str, Any]]:
        for model_scale in ["1b"]:
            yield dict(model_scale=model_scale, sparsity="none", k=None)
            for sparsity in ["eviction", "clustering", "clustering-weighted"]:
                for k in [128, 256, 512]:
                    if not (k > 256 and sparsity.startswith("sparsev")):
                        yield dict(model_scale=model_scale, sparsity=sparsity, k=k)

    settings = [dict(dataset=dataset, **s) for s in _sparsities()]
    out = Path(f"data/{dataset}.jsonl")
    print(out, len(settings), settings)
    L.utility.run_multiprocess_sweep(run, settings, out, n_workers=2)
