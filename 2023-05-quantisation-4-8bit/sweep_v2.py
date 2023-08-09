"""Run a quantisation settings sweep."""

import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import transformers

import llminference as L
import llminference.quantisation as Q


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


def evaluate(model_name: str, format_: Q.TensorFormat) -> Dict[str, Any]:
    dataset = L.outcompare.Dataset.load(f"data/{model_name}.json")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        dataset.model, trust_remote_code=True
    )
    unembedding_format = Q.channel_scaling_format(Q.parse("E2M5"), per="output")
    spec = [
        ("embed_tokens|embed_out", unembedding_format),
        (".*", format_),
    ]
    quantised_bytes = Q.quantise_model(model, spec)
    summary = dict(
        model=dataset.model,
        parameters=sum(p.nelement() for p in model.parameters()),
        quantised_bytes=quantised_bytes,
        format_name=str(format_),
        format=dataclasses.asdict(format_),
        unembedding_format_name=str(unembedding_format),
        unembedding_format=dataclasses.asdict(unembedding_format),
    )
    t0 = time.time()
    summary.update(outcompare_evaluate(model, dataset, limit=300))
    summary[f"outcompare:time"] = time.time() - t0
    return summary


if __name__ == "__main__":

    def models() -> Iterable[str]:
        # yield "pythia-70m"  # TEST
        # yield from (f"pythia-{s}" for s in ["410m", "1.4b", "2.8b", "6.9b"])
        # yield from (f"pythia-{s}" for s in ["2.8b", "6.9b"])
        yield from (f"opt-{s}" for s in ["125m", "1.3b", "2.7b"])

    def element_formats() -> Iterable[Q.ScalarFormat]:
        for bits in range(4, 9):
            yield Q.IntFormat(bits)
            yield Q.FPFormat(2, bits - 3)
            yield Q.nf_approx(bits)
            yield Q.lut_function(lambda n: n * n.abs(), bits, "quad")

    def formats() -> Iterable[Q.TensorFormat]:
        for efmt in element_formats():
            yield Q.channel_scaling_format(efmt, per="output")
            # yield Q.channel_scaling_format(efmt, per="inout-prod")
            # yield Q.channel_scaling_format(efmt, per="inout-min")
            yield Q.group_scaling_format(efmt, grouping="input", group_size=32)
            yield Q.group_scaling_format(efmt, grouping="input", group_size=64)

    settings = [
        dict(model_name=model, format_=format_)
        for model in models()
        for format_ in formats()
    ]
    L.utility.run_multiprocess_sweep(
        evaluate, settings, Path("out/sweep_bits_v2.jsonl"), n_workers=4
    )
