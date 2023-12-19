# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import unittest.mock as um
from typing import Any, List

from llminference.eval_adapter import Adapter
from llminference.tasks import summarisation


def test_cnn_dailymail() -> None:
    data = summarisation.CnnDailymail.data()
    assert len(data) > 100
    assert set(data.features) == {"id", "context", "reference"}


def test_evaluate() -> None:
    adapter = Adapter.from_pretrained("EleutherAI/pythia-70m")
    examples = [
        dict(
            id="1",
            context="rouge is french for red, but the rest is ignored",
            reference="rouge is french for red",
        ),
        dict(
            id="2",
            context="blah blah blah",
            reference="please listen carefully to what I have to say",
        ),
    ]

    def mock_sample(
        self: Adapter, ctxs_batch: List[str], prompts_batch: List[str], **kwargs: Any
    ) -> List[List[int]]:
        # Return the context
        return [self.tok_encode(q) for q in ctxs_batch]

    with um.patch.object(Adapter, "greedy_sample", mock_sample):
        results = list(
            summarisation.evaluate(adapter, examples, batch_size=2, use_cache=False)
        )

    prefill_lengths = [
        len(adapter.tok_encode(examples[i]["context"] + summarisation.DEFAULT_PROMPT))
        for i in range(len(examples))
    ]
    reference_lengths = [
        len(adapter.tok_encode(examples[i]["reference"])) for i in range(len(examples))
    ]
    # Note that the first output is truncated (to the tokenized length of 'reference')
    assert results == [
        dict(
            id="1",
            output="rouge is french for red",
            rougeL=1.0,
            prefill_length=prefill_lengths[0],
            reference_length=reference_lengths[0],
        ),
        dict(
            id="2",
            output="blah blah blah",
            rougeL=0.0,
            prefill_length=prefill_lengths[1],
            reference_length=reference_lengths[1],
        ),
    ]
