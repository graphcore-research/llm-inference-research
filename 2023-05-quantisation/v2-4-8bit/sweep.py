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
            batch_size=min(limit or 64, 64),
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
        yield "pythia-1.4b"
        yield "opt-1.3b"

    def formats() -> Iterable[Q.TensorFormat]:
        for efmt in [
            Q.parse("E0M4"),  # 5 bit
            Q.parse("E2M2"),
            Q.parse("E0M5"),  # 6 bit
            Q.parse("E2M3"),
            Q.parse("E0M6"),  # 7 bit
            Q.parse("E2M4"),
            Q.NF4,  # 4 bit
            Q.nf_approx(4),
            Q.lut_function(lambda x: x, bits=4, name="linear"),
            Q.lut_function(lambda x: x * x.abs(), bits=4, name="quad"),
        ]:
            yield Q.channel_scaling_format(efmt, per="output")
            yield Q.channel_scaling_format(efmt, per="inout")
            for group_size in [4, 8, 16, 32, 64]:
                yield Q.group_scaling_format(
                    efmt, grouping="input", group_size=group_size
                )

    settings = [
        dict(model_name=model, format_=format_)
        for model in models()
        for format_ in formats()
    ]
    L.utility.run_multiprocess_sweep(
        evaluate, settings, Path("out/sweep_nf4.jsonl"), n_workers=16
    )
