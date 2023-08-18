import unittest.mock as um
from functools import partial
from typing import cast
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from torch import Tensor

import llminference as L


def run_experiment(model, examples, open_book, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    acc = L.eval_utils.evaluate_qa_task(
        adapter, examples, batch_size, open_book=open_book, use_cache=open_book
    )
    return dict(model=model, acc=acc, open_book=open_book)


if __name__ == "__main__":
    ds = datasets.load_from_disk("/nethome/lukar/datasets/triviaqa_2k/")["validation"]
    examples = [L.eval_utils.format_triviaqa(x) for x in ds]

    models = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        # "pythia-1.4b",
        "pythia-2.8b",
    ]

    settings = [
        dict(model=model, examples=examples, open_book=open_book)
        for model in models
        for open_book in [True, False]
    ]

    dest = Path(__file__).parent / "results/triviaqa_sparse_v_baseline.jsonl"
    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)
