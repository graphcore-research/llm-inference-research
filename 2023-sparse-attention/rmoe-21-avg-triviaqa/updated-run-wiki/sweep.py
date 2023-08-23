import unittest.mock as um
from functools import partial
from pathlib import Path

import llminference as L

CACHE_DIR = "/net/group/research/lukar/cache/triviaqa-wiki/"


def run_baseline(model, examples, open_book, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    results = list(
        L.qa.evaluate(
            adapter,
            examples,
            batch_size=batch_size,
            open_book=open_book,
            cache_dir=CACHE_DIR,
        )
    )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, acc=acc, open_book=open_book)


def run_experiment(model, examples, k, add_avg, open_book, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    sparse_softmax = L.sparse_attention.sparse_softmax_fixed_k

    with um.patch(
        "torch.nn.functional.softmax", partial(sparse_softmax, k=k, add_avg=add_avg)
    ):
        results = list(
            L.qa.evaluate(
                adapter,
                examples,
                batch_size=batch_size,
                open_book=open_book,
                cache_dir=CACHE_DIR,
            )
        )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, acc=acc, k=k, add_avg=add_avg, open_book=open_book)


if __name__ == "__main__":
    data = L.qa.TriviaQA.data(context="wiki")
    n_examples = 400
    examples = [L.qa.add_zero_shot_prompt(data[i]) for i in range(n_examples)]

    models = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
    ]

    # Run dense baselines
    settings = [
        dict(model=model, examples=examples, open_book=True) for model in models
    ]
    dest = Path(__file__).parent / "results/triviaqa_baseline_wiki.jsonl"
    L.utility.run_multiprocess_sweep(run_baseline, settings, dest, n_workers=1)

    # Run sparse experiments
    ks = [8, 16, 32, 64, 128]

    settings = [
        dict(
            model=model,
            examples=examples,
            k=k,
            add_avg=add_avg,
            open_book=open_book,
        )
        for model in models
        for k in ks
        for add_avg in [False, True]
        for open_book in [True]
    ]

    dest = Path(__file__).parent / "results/triviaqa_add_avg_wiki.jsonl"
    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)
