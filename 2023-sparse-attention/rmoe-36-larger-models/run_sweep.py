"""A big recap sweep."""

import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from typing import *

import torch

import llminference as L


@dataclass
class Data:
    dataset: str
    examples: List[Dict[str, Any]]

    def evaluate(
        self,
        adapter: L.Adapter,
        context: L.eval_adapter.ModelContext,
        batch_size: int,
        desc: str,
    ) -> Iterable[Dict[str, Any]]:
        evaluate = (
            L.summarisation.evaluate
            if self.dataset == "cnn_dailymail"
            else partial(L.qa.evaluate, open_book=True)
        )
        return evaluate(
            adapter,
            self.examples,
            batch_size=batch_size,
            generation_context=context,
            use_cache=False,
            desc=desc,
        )


def get_data(dataset: str, shots: int) -> Data:
    if dataset == "triviaqa":
        data = L.qa.TriviaQA.data(context="wiki")
        examples = [L.qa.add_few_shot_prompt(data[i], k=shots) for i in range(1000)]
    if dataset == "squad":
        data = L.qa.SQuAD.data()
        examples = [L.qa.add_few_shot_prompt(data[i], k=shots) for i in range(1000)]
    if dataset == "cnn_dailymail":
        assert shots == 0
        data = L.summarisation.CnnDailymail.data()
        examples = [data[i] for i in range(500)]
    return Data(dataset=dataset, examples=examples)


@contextmanager
def technique(
    sparsity: str, k: Optional[int], local_ratio: Optional[float]
) -> Iterator[L.eval_adapter.ModelContext]:
    if sparsity == "none":
        yield L.eval_adapter.null_model_context
    elif sparsity == "eviction":
        original_model = adapter.model
        adapter.model = L.eviction_attention.convert_gptneox(
            original_model,
            L.eviction_attention.Settings(k=k, local_k=int(local_ratio * k)),
        )
        try:
            yield L.eviction_attention.generation_context
        finally:
            adapter.model = original_model
    else:
        yield L.eval_adapter.patch_for_model(
            "torch.nn.functional.softmax",
            L.sparse_attention.sparse_softmax_fixed_k,
            k=k,
            apply_after_softmax=("_after" in sparsity),
            add_avg=(sparsity == "sparsev_after_avg"),
        )


model_scales = ["6.9b", "2.8b", "1b"]
datasets = ["squad", "triviaqa", "cnn_dailymail"]


def _sparsities() -> Iterable[Dict[str, Any]]:
    yield dict(sparsity="none", k=None, local_ratio=None)
    for sparsity in [
        "eviction",
        "sparsev_before",
        "sparsev_after",
        "sparsev_after_avg",
    ]:
        local_ratio = 0.25 if sparsity == "eviction" else None
        for k in [32, 64, 128, 256, 512, 1024]:
            if not (k > 256 and sparsity.startswith("sparsev")):
                yield dict(sparsity=sparsity, k=k, local_ratio=local_ratio)


# model_scales = ["6.9b"]
# datasets = ["squad"]
# _sparsities = lambda: [dict(sparsity="eviction", k=512, local_ratio=0.25)]


with L.utility.jsonlines_writer("data/sweep.jsonl") as log:
    for model_scale in model_scales:
        adapter = L.Adapter.from_pretrained(
            f"EleutherAI/pythia-{model_scale}", dtype=torch.half
        )
        batch_size = {"1b": 10, "2.8b": 4, "6.9b": 2}[model_scale]

        for dataset in datasets:
            shots = 1 if dataset == "squad" else 0
            data = get_data(dataset, shots)

            for sparsity in _sparsities():
                with technique(**sparsity) as context:
                    adapter.model.cuda()
                    settings = dict(
                        dataset=dataset,
                        shots=shots,
                        model_scale=model_scale,
                        **sparsity,
                    )
                    t0 = time.time()
                    results = list(
                        data.evaluate(adapter, context, batch_size, desc=str(settings))
                    )
                    duration = time.time() - t0
                    log(dict(**settings, results=results, _duration=duration))
                    adapter.model.cpu()
