"""Run a quantisation settings sweep."""

import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

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


def evaluate(
    model_name: str, unembedding_format: Optional[Q.Format], format_: Q.Format
) -> Dict[str, Any]:
    dataset = L.outcompare.Dataset.load(f"data/{model_name}.json")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        dataset.model, trust_remote_code=True
    )
    formats = []
    if unembedding_format:
        formats.append(("embed_tokens|embed_out", unembedding_format))
    formats.append((".*", format_))
    quantised_bytes = Q.quantise_model(model, formats)

    summary = dict(
        model=dataset.model,
        parameters=sum(p.nelement() for p in model.parameters()),
        quantised_bytes=quantised_bytes,
        format_name=str(format_),
        format=dataclasses.asdict(format_),
    )
    if unembedding_format:
        summary.update(
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

    def formats() -> Iterable[Tuple[Q.TensorFormat, Q.TensorFormat]]:
        # FP16 baseline
        yield (None, Q.FP16)
        # FP8 baselines
        yield (None, Q.channel_scaling_format(Q.parse("E2M5"), per="output"))
        yield (None, Q.channel_scaling_format(Q.parse("E2M5"), per="inout"))
        # 4-bit
        for _4bit in [Q.parse("E0M3"), Q.parse("E2M1"), Q.parse("E3M0")]:
            for embed_format in [
                None,
                Q.channel_scaling_format(Q.parse("E2M5"), per="output"),
            ]:
                yield (embed_format, Q.channel_scaling_format(_4bit, per="output"))
                yield (embed_format, Q.channel_scaling_format(_4bit, per="inout"))
                for group_size in [2, 4, 8, 16, 32, 64]:
                    for scale_format in [
                        Q.FP16,
                        Q.parse("E5M2"),
                        Q.tensor_scaling_format(
                            Q.ExpCeilFormat(4, 0), Q.ExpCeilFormat(8, 0)
                        ),
                    ]:
                        yield (
                            embed_format,
                            Q.group_scaling_format(
                                _4bit,
                                grouping="input",
                                group_size=group_size,
                                scale_format=scale_format,
                            ),
                        )

    settings = [
        dict(model_name=model, unembedding_format=unembedding_format, format_=format_)
        for model in models()
        for unembedding_format, format_ in formats()
    ]
    L.utility.run_multiprocess_sweep(
        evaluate, settings, Path("out/sweep_4bit.jsonl"), n_workers=16
    )
