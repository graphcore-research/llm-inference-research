import unittest.mock as um
from functools import partial
from pathlib import Path

import llminference as L


def run_experiment(model, examples, k, add_avg, open_book, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    sparse_softmax = L.sparse_attention.sparse_softmax_fixed_k

    with um.patch(
        "torch.nn.functional.softmax", partial(sparse_softmax, k=k, add_avg=add_avg)
    ):
        results = list(
            L.qa.evaluate(adapter, examples, batch_size=batch_size, open_book=open_book)
        )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, acc=acc, k=k, add_avg=add_avg, open_book=open_book)


if __name__ == "__main__":
    data = L.qa.TriviaQA.data()
    n_examples = 400
    examples = [
        L.qa.add_zero_shot_prompt(data[i], L.qa.TriviaQA.DEFAULT_PROMPT)
        for i in range(n_examples)
    ]

    models = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
    ]

    # Run sparse experiments
    ks = [16, 32, 64, 128, 256]

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

    dest = Path(__file__).parent / "results/triviaqa_add_avg.jsonl"
    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)
