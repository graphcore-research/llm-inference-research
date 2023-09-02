import argparse
from functools import partial
from pathlib import Path
from typing import *

import torch

import llminference as L

torch.set_num_threads(16)
CACHE = "/net/group/research/douglaso/llminference-cache"


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
    elif sparsity == "eviction":
        local_ratio = 0.25
        use_cache = False
        adapter.model = L.eviction_attention.convert_gptneox(
            adapter.model,
            L.eviction_attention.Settings(k=k, local_k=int(local_ratio * k)),
        )
        context = L.eviction_attention.generation_context
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["squad", "cnn_dailymail", "triviaqa"])
    parser.add_argument("--model-scale", choices=["160m", "410m", "1b", "1.4b", "2.8b"])
    args = vars(parser.parse_args())

    def _sparsities() -> Iterable[Dict[str, Any]]:
        yield dict(sparsity="none", k=None)
        for sparsity in [
            "eviction",
            "sparsev_before",
            "sparsev_after",
            "sparsev_after_avg",
        ]:
            for k in [32, 64, 128, 256, 512, 1024]:
                if not (k > 256 and sparsity.startswith("sparsev")):
                    yield dict(sparsity=sparsity, k=k)

    settings = [dict(**args, **s) for s in _sparsities()]
    out = Path(f"data/sweep_v1/{args['dataset']}_{args['model_scale']}.jsonl")
    print(out, len(settings), settings)
    L.utility.run_multiprocess_sweep(run, settings, out, n_workers=2)
