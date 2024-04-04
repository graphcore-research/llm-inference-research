# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

import unittest.mock as um

import torch
import transformers

from llminference.eval_adapter import Adapter
from llminference.tasks import needle


def test_data() -> None:
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    data = list(needle.Dataset.data(tokenizer, [128, 256], 3))
    assert len(data) == 6
    {(x["length"], x["depth"]) for x in data} == {
        (128, 0.0),
        (128, 0.5),
        (128, 1.0),
        (256, 0.0),
        (256, 0.5),
        (256, 1.0),
    }
    for example in data:
        assert needle.NEEDLE_FORMAT in tokenizer.decode(example["prompt"])


def test_greedy_sample() -> None:
    adapter = Adapter.from_pretrained("EleutherAI/pythia-70m")
    input_ids = adapter.tokenizer("I am going to write a").input_ids
    expected = adapter.tokenizer.decode(
        adapter.model.generate(torch.tensor(input_ids)[None], max_new_tokens=3)[0, -3:]
    )
    actual = adapter.tokenizer.decode(
        needle.greedy_sample(
            adapter.model, input_ids, max_generated_tokens=3, prefill_chunk_length=4
        )
    )
    assert actual == expected


def test_evaluate() -> None:
    adapter = um.Mock()
    tokenizer = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    adapter.tokenizer = tokenizer
    with um.patch("llminference.tasks.needle.greedy_sample") as m:
        m.side_effect = [
            tokenizer(s, add_special_tokens=False).input_ids
            for s in [
                " " + needle.REFERENCE_OUTPUT,
                " hop along to Central Park for a Reuben.",
                "   " + needle.REFERENCE_OUTPUT + ". Then,",
            ]
        ]
        results = list(
            needle.evaluate(
                adapter,
                [dict(id=n, length=n, depth=n, prompt=None) for n in range(3)],
            )
        )
    assert [r["match"] for r in results] == [True, False, True]
    assert [r["id"] for r in results] == list(range(3))
