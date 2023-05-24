"""Run a quantisation settings sweep over a range of settings.

Incompatible with commit `03342c46`.
"""

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import lm_eval.evaluator
import transformers

import llminference as L


def lm_eval_evaluate(
    model: transformers.PreTrainedModel, task: str, limit: Optional[int]
) -> Dict[str, float]:
    batch_size = min(
        limit or L.Adapter.DEFAULT_BATCH_SIZE, L.Adapter.DEFAULT_BATCH_SIZE
    )
    out = lm_eval.evaluator.evaluate(
        L.Adapter.from_model(model, batch_size=batch_size),
        lm_eval.tasks.get_task_dict([task]),
        limit=limit,
    )
    return {
        f"{task}:{metric}": value
        for task, d in out["results"].items()
        for metric, value in d.items()
    }


def outcompare_evaluate(
    model: transformers.PreTrainedModel,
    dataset: L.outcompare.Dataset,
    limit: Optional[int],
) -> Dict[str, float]:
    return {
        f"outcompare:{metric}": value
        for metric, value in L.outcompare.evaluate(
            model,
            dataset,
            batch_size=min(limit or 32, 32),
            limit=limit,
        ).items()
    }


def evaluate(
    short_model_name: str,
    format: Optional[L.quantisation.Format],
    tasks: Dict[str, Optional[int]],
    quantisation_mode: str,
) -> Dict[str, Any]:
    dataset = L.outcompare.Dataset.load(f"data/{short_model_name}.json")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        dataset.model, trust_remote_code=True
    )
    summary = dict(
        model=dataset.model,
        parameters=sum(p.nelement() for p in model.parameters()),
        quantisation_mode=quantisation_mode,
    )
    if quantisation_mode != "none":
        quantised_bytes = L.quantisation.quantise_model(
            model, format, mode=quantisation_mode
        )
        summary.update(
            quantised_bytes=quantised_bytes,
            exponent_bits=format.exponent_bits,
            mantissa_bits=format.mantissa_bits,
        )
    for task, limit in tasks.items():
        t0 = time.time()
        if task == "outcompare":
            summary.update(outcompare_evaluate(model, dataset, limit=limit))
        else:
            summary.update(lm_eval_evaluate(model, task, limit=limit))
        summary[f"{task}:time"] = time.time() - t0
    return summary


if __name__ == "__main__":

    def models() -> Iterable[str]:
        # yield from ["pythia-70m", "pythia-160m"]  # Test
        yield from (
            f"pythia-{s}" for s in ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
        )
        yield from (f"opt-{s}" for s in ["125m", "1.3b", "2.7b", "6.7b"])

    # tasks = dict(outcompare=3)  # Test
    tasks = dict(outcompare=300, lambada_openai=None, arc_easy=None)

    settings = []
    settings += [
        dict(short_model_name=model, format=None, quantisation_mode="none", tasks=tasks)
        for model in models()
    ]
    settings += [
        dict(
            short_model_name=model,
            format=L.quantisation.Format.parse(format),
            tasks=tasks,
            quantisation_mode=mode,
        )
        for model in models()
        for format in ["E0M7", "E2M5", "E3M4", "E4M3"]
        for mode in ["tensor", "input", "output", "inout"]
    ]
    L.utility.run_multiprocess_sweep(
        evaluate, settings, Path("out/sweep_quantisation_mpt.jsonl"), n_workers=8
    )
