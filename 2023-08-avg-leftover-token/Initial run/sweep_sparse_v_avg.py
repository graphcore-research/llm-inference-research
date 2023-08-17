import unittest.mock as um
from functools import partial
from typing import cast
from pathlib import Path

import datasets
import torch
import torch.nn.functional as F
from torch import Tensor

import llminference as L

_softmax = F.softmax


def sparse_softmax_with_avgs(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Applies softmax accross last dimension, keeping top k
    elements of the output and averaging the rest.

    Args:
        x (Tensor): shape (batch_size, num_heads, q_len, k_len)
        k (int): Number of attention scores to keep
        dim (int, optional): Assumed dim = -1

    Returns:
        Tensor: shape (batch_size, num_heads, q_len, k_len)
    """
    assert dim == -1
    y = _softmax(x, dim=-1)
    if k >= x.shape[-1]:
        return y

    kth_val = -torch.kthvalue(-y, k, keepdim=True).values
    topk_mask = y >= kth_val
    topk = topk_mask * y
    rest = ~topk_mask * (1 - topk.sum(dim=-1, keepdim=True)) / (x.shape[-1] - k)
    return cast(Tensor, topk + rest)


def run_experiment(model, examples, k, use_avg, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    if use_avg:
        sparse_softmax = sparse_softmax_with_avgs
    else:
        sparse_softmax = L.sparse_attention.sparse_softmax_fixed_k

    with um.patch("torch.nn.functional.softmax", partial(sparse_softmax, k=k)):
        acc = L.eval_utils.evaluate_qa_task(
            adapter, examples, batch_size, open_book=True, use_cache=True
        )
    return dict(model=model, open_book_acc=acc, k=k, use_avg=use_avg)


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
    ks = [8, 16, 32, 128]

    settings = [
        dict(model=model, examples=examples, k=k, use_avg=use_avg)
        for model in models
        for k in ks
        for use_avg in [True, False]
    ]

    dest = Path(__file__).parent / "results/triviaqa_sparse_v_with_avg.jsonl"
    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)
