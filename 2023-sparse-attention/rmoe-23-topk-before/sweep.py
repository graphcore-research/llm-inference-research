import unittest.mock as um
from functools import partial
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

import llminference as L

_softmax = torch.nn.functional.softmax


def sparse_softmax_fixed_k_before(x: Tensor, k: int, dim: int = -1) -> Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output.

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        k (int): Number of attention scores to keep
        dim (int, optional): Assumed dim = -1

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    if k < x.shape[-1]:
        kth_val = -torch.kthvalue(-x, k, keepdim=True).values
        mask = x >= kth_val
        x = mask * x + ~mask * -1e9

    y = _softmax(x, dim=-1)
    return y


def run_experiment(model, examples, k, topk_before, open_book, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    if topk_before:
        sparse_softmax = sparse_softmax_fixed_k_before
    else:
        sparse_softmax = L.sparse_attention.sparse_softmax_fixed_k

    with um.patch("torch.nn.functional.softmax", partial(sparse_softmax, k=k)):
        results = list(
            L.qa.evaluate(adapter, examples, batch_size=batch_size, open_book=open_book)
        )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, acc=acc, k=k, topk_before=topk_before, open_book=open_book)


def run_baseline(model, examples, open_book, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    results = list(
        L.qa.evaluate(adapter, examples, batch_size=batch_size, open_book=open_book)
    )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, acc=acc, open_book=open_book)


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

    # Run baseline, generate cache
    settings = [
        dict(
            model=model,
            examples=examples,
            open_book=open_book,
        )
        for model in models
        for open_book in [True, False]
    ]
    dest = Path(__file__).parent / "results/triviaqa_baseline.jsonl"
    L.utility.run_multiprocess_sweep(run_baseline, settings, dest, n_workers=1)

    # Run sparse experiments
    ks = [8, 16, 32, 64, 128]

    settings = [
        dict(
            model=model,
            examples=examples,
            k=k,
            topk_before=topk_before,
            open_book=open_book,
        )
        for model in models
        for k in ks
        for topk_before in [False, True]
        for open_book in [True]
    ]

    dest = Path(__file__).parent / "results/triviaqa_topk_before.jsonl"
    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)
