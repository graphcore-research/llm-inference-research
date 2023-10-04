import argparse
from functools import partial
from pathlib import Path
from typing import *

import torch

import llminference as L

CACHE = "/net/group/research/lukar/llminference-cache"


def run(
    dataset: str,
    model_scale: str,
    k: Optional[int],
    cluster_evicted: bool,
    cluster_weight: Optional[float],
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
    use_cache = False
    local_ratio = 0.25
    adapter.model = L.eviction_attention.convert_gptneox(
        adapter.model,
        L.eviction_attention.Settings(
            k=k,
            local_k=int(local_ratio * k),
            cluster=cluster_evicted,
            cluster_weight=cluster_weight,
        ),
    )
    context = L.eviction_attention.generation_context

    # Evaluation
    return dict(
        dataset=dataset,
        shots=shots,
        model_scale=model_scale,
        k=k,
        cluster_evicted=cluster_evicted,
        cluster_weight=cluster_weight,
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

    def _settings() -> Iterable[Dict[str, Any]]:
        for model_scale in ["1b"]:
            for k in [128, 256, 512, 1024]:
                for cluster_evicted in [True, False]:
                    if not cluster_evicted:
                        yield dict(
                            model_scale=model_scale,
                            k=k,
                            cluster_evicted=cluster_evicted,
                            cluster_weight=None,
                        )
                    else:
                        for cluster_weight in [0.25, 0.5, 0.75, 0.9]:
                            yield dict(
                                model_scale=model_scale,
                                k=k,
                                cluster_evicted=cluster_evicted,
                                cluster_weight=cluster_weight,
                            )

    settings = [dict(dataset=dataset, **s) for s in _settings()]
    out = Path(f"data/{dataset}.jsonl")
    print(out, len(settings), settings)
    L.utility.run_multiprocess_sweep(run, settings, out, n_workers=2)
