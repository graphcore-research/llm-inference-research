from lm_eval import evaluator, tasks
import llminference as L
import llminference.sparse_attention as sa
import unittest.mock as um
from functools import partial
from pathlib import Path


def evaluate_out_comparison(model, proportion, batch_size, limit):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model)
    outcompare_data = L.outcompare.Dataset.load("data/" + model + ".json")
    with um.patch(
        "torch.nn.functional.softmax",
        partial(L.sparse_attention.sparse_softmax, proportion=proportion),
    ):
        results = L.outcompare.evaluate(
            adapter.model, outcompare_data, batch_size=batch_size, limit=limit
        )

    return dict(model=model, proportion=proportion, **results)


def evaluate_task(model, task, softmax_func, batch_size, limit, **func_kwargs):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    with um.patch(
        "torch.nn.functional.softmax",
        partial(softmax_func, **func_kwargs),
    ):
        results = evaluator.evaluate(adapter, tasks.get_task_dict([task]), limit=limit)[
            "results"
        ][task]

    return dict(model=model, **func_kwargs, **results)


models = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    # "pythia-1.4b",
    "pythia-2.8b",
    # "pythia-6.9b",
    # "pythia-12b",
]
# ks = [4, 8, 16, 32, 256]
k_mins = [1, 2, 4]
ps = [0.05, 0.1, 0.2, 0.5]

# softmax_func = sa.sparse_softmax_fixed_k
softmax_func = sa.sparse_softmax_fixed_p

settings = [
    dict(
        model=model,
        task="lambada_openai",
        softmax_func=softmax_func,
        batch_size=32,
        limit=None,
        p=p,
        k_min=k_min,
    )
    for model in models
    for p in ps
    for k_min in k_mins
]


dest = Path(
    f"/nethome/lukar/code/research-llm-inference/out/{softmax_func.__name__}.jsonl"
)
L.utility.run_multiprocess_sweep(evaluate_task, settings, dest, n_workers=4)
