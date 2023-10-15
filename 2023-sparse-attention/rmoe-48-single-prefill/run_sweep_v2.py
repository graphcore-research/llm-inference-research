"""Sweeping with & without combined prefill."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import *

import llminference as L


@dataclass
class Data:
    dataset: str
    examples: List[Dict[str, Any]]

    def __post_init__(self) -> None:
        self.evaluate = partial(
            L.summarisation.evaluate
            if self.dataset == "cnn_dailymail"
            else L.qa.evaluate,
            examples=self.examples,
            use_cache=False,
        )

    @classmethod
    def get(cls, dataset: str, shots: int) -> "Data":
        if dataset == "triviaqa":
            data = L.qa.TriviaQA.data(context="wiki")
            examples = [L.qa.add_few_shot_prompt(data[i], k=shots) for i in range(400)]
        if dataset == "squad":
            data = L.qa.SQuAD.data()
            examples = [L.qa.add_few_shot_prompt(data[i], k=shots) for i in range(400)]
        if dataset == "cnn_dailymail":
            assert shots == 0
            data = L.summarisation.CnnDailymail.data()
            examples = [data[i] for i in range(200)]
        return cls(dataset=dataset, examples=examples)


@contextmanager
def technique(
    adapter: L.Adapter,
    sparsity: str,
    k: Optional[int],
    local_ratio: Optional[float],
    sparse_generation_only: bool,
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
            generation_only=sparse_generation_only,
        )


def run(
    dataset: str,
    model_scale: str,
    sparsity: Dict[str, Any],
    combine_context_and_prompt: bool,
) -> Dict[str, Any]:
    shots = 1 if dataset == "squad" else 0
    data = Data.get(dataset, shots)
    adapter = L.Adapter.from_pretrained(f"EleutherAI/pythia-{model_scale}")
    with technique(adapter, **sparsity) as context:
        results = list(
            data.evaluate(
                adapter=adapter,
                generation_context=context,
                batch_size=5,
                combine_context_and_prompt=combine_context_and_prompt,
            )
        )
    return dict(
        dataset=dataset,
        shots=shots,
        model_scale=model_scale,
        **sparsity,
        combine_context_and_prompt=combine_context_and_prompt,
        results=results,
    )


if __name__ == "__main__":
    model_scales = ["2.8b"]
    datasets = ["squad", "triviaqa", "cnn_dailymail"]

    def _sparsities() -> Iterable[Dict[str, Any]]:
        yield dict(
            sparsity="none", k=None, local_ratio=None, sparse_generation_only=False
        )
        for sparsity in ["sparsev_before", "sparsev_after", "sparsev_after_avg"]:
            for sparse_generation_only in [False, True]:
                yield dict(
                    sparsity=sparsity,
                    k=64,
                    local_ratio=None,
                    sparse_generation_only=sparse_generation_only,
                )

    settings = [
        dict(
            dataset=dataset,
            model_scale=model_scale,
            sparsity=sparsity,
            combine_context_and_prompt=combine_context_and_prompt,
        )
        for model_scale in model_scales
        for dataset in datasets
        for sparsity in _sparsities()
        for combine_context_and_prompt in [False]
    ]
    with L.utility.jsonlines_writer("data/sweep_v2c.jsonl") as write:
        for setting in settings:
            write(run(**setting))
    # L.utility.run_multiprocess_sweep(
    #     run, settings, Path("data/sweep_v2c.jsonl"), n_workers=16
    # )
