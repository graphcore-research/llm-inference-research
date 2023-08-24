"""Sparse-V experiments where for layers i < n/2 choose k1 = k + dk
and for layers i >= n/2 choose k2 = k - dk"""

import unittest.mock as um
from functools import partialmethod
from pathlib import Path

import llminference as L
from llminference.sparse_attention import number_attention_layers


def run_experiment(model, examples, k, dk_rel, open_book, cache_dir, batch_size=4):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    m = adapter.model.gpt_neox
    num_layers = len(m.layers)
    sparse_attention = L.sparse_attention.sparse_attn
    dk = round(k * dk_rel)
    k_per_layer = [
        k + dk if i < (num_layers // 2) else k - dk for i in range(num_layers)
    ]
    print(k_per_layer)

    with number_attention_layers(m), um.patch(
        "transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention._attn",
        partialmethod(sparse_attention, k_per_layer=k_per_layer),
    ):
        results = list(
            L.qa.evaluate(
                adapter,
                examples,
                batch_size=batch_size,
                open_book=open_book,
                cache_dir=cache_dir,
            )
        )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, acc=acc, k=k, dk_rel=dk_rel)


if __name__ == "__main__":
    models = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
    ]

    # Run sparse experiments
    ks = [8, 16, 32, 64, 128]
    dks_rel = [0.0, 0.25, 0.5, 0.75]

    # TriviaQA, zero-shot
    cache_dir = "/net/group/research/lukar/cache/triviaqa-wiki/"
    data = L.qa.TriviaQA.data(context="wiki")
    n_examples = 400
    examples = [L.qa.add_zero_shot_prompt(data[i]) for i in range(n_examples)]

    settings = [
        dict(
            model=model,
            examples=examples,
            k=k,
            dk_rel=dk_rel,
            open_book=True,
            cache_dir=cache_dir,
        )
        for model in models
        for k in ks
        for dk_rel in dks_rel
    ]

    dest = Path(__file__).parent / "results/triviaqa_wiki_vary_k_half.jsonl"
    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)
    
    # SQuAD
    cache_dir = "/net/group/research/lukar/cache/squad/"
    data = L.qa.SQuAD.data()
    n_examples = 400
    examples = [L.qa.add_few_shot_prompt(data[i], k=1) for i in range(n_examples)]
    
    settings = [
        dict(
            model=model,
            examples=examples,
            k=k,
            dk_rel=dk_rel,
            open_book=True,
            cache_dir=cache_dir,
        )
        for model in models
        for k in ks
        for dk_rel in dks_rel
    ]

    dest = Path(__file__).parent / "results/squad_vary_k_half.jsonl"
    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)