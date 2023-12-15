# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import itertools as it
import unittest.mock as um
from typing import Any, List

import pytest

from llminference.eval_adapter import Adapter
from llminference.tasks import qa


def test_datasets() -> None:
    for cls, kwargs in [
        (qa.TriviaQA, dict(context="search")),
        (qa.TriviaQA, dict(context="wiki")),
        (qa.SQuAD, {}),
    ]:
        data = cls.data(**kwargs)  # type:ignore[attr-defined]
        assert len(data) > 100
        assert set(data.features) == {
            "id",
            "question",
            "answers",
            "context",
            "examples",
        }
        assert set(map(len, data["examples"])) == {5}
        average_context_chars = sum(map(len, data["context"])) / len(data)
        assert 4000 <= average_context_chars <= 8000


def test_add_prompt() -> None:
    datum = dict(
        id="#0",
        question="Dummy qu",
        context="Dummy context",
        answers=["A1", "A2"],
        examples=[
            dict(question="Eg 1", answers=["long answer 1", "shortans1"]),
            dict(question="Eg 2", answers=["shortans2", "long answer 2"]),
        ],
    )
    # Zero shot
    example = qa.add_few_shot_prompt(datum, k=0, prompt_template="Q: {question}\nA:")
    assert example.pop("prompt") == "Q: Dummy qu\nA:"
    assert example == datum

    # Two shot
    example = qa.add_few_shot_prompt(datum, k=2, prompt_template="Q: {question}\nA:")
    assert example["prompt"] == "\n".join(
        [
            "Q: Eg 1\nA: shortans1",
            "Q: Eg 2\nA: shortans2",
            "Q: Dummy qu\nA:",
        ]
    )

    # Too-many-shot
    with pytest.raises(ValueError) as error:
        qa.add_few_shot_prompt(datum, k=3, prompt_template="Q: {question}\nA:")
    assert "k=3" in str(error)
    assert "2 examples" in str(error)


def test_evaluate_prediction() -> None:
    assert qa.evaluate_prediction("bicycle", ["car", "bicycle"])
    assert qa.evaluate_prediction(" bicycle", ["car", "bicycle"])
    assert qa.evaluate_prediction(
        "\n\n 'the BICYCLE is a fine...", ["car", "a Bicycle."]
    ), "all forms of LHS & RHS normalisation"
    assert not qa.evaluate_prediction("my bicycle", ["car", "bicycle"])
    assert not qa.evaluate_prediction("bicycle", ["car", "your bicycle"])
    assert not qa.evaluate_prediction("bicycle", ["car", "'.'"]), "rhs normalised to ''"


def test_evaluate() -> None:
    adapter = Adapter.from_pretrained("EleutherAI/pythia-70m")
    example_ctxs = ["context"] * 4
    example_prompts = ["plane", "boat", "bicycle", "car"]
    example_answers = [
        ["plane", "jet plane"],
        ["submarine"],
        ["Bicycle"],
        ["tank", "car"],
    ]
    examples = [
        {"id": i, "context": ctx, "prompt": prompt, "answers": answers}
        for i, ctx, prompt, answers in zip(
            it.count(), example_ctxs, example_prompts, example_answers
        )
    ]

    def mock_sample(
        self: Adapter, ctxs_batch: List[str], prompts_batch: List[str], **kwargs: Any
    ) -> List[List[int]]:
        # Just return the prompt
        assert all(c == "context\n" for c in ctxs_batch), str(ctxs_batch)
        return [self.tok_encode(q) for q in prompts_batch]

    prefill_lengths = [
        len(adapter.tok_encode(example_ctxs[i] + example_prompts[i]))
        for i in range(len(examples))
    ]
    with um.patch.object(Adapter, "greedy_sample", mock_sample):
        assert list(qa.evaluate(adapter, examples, 2)) == [
            dict(id=0, output="plane", match=True, prefill_length=prefill_lengths[0]),
            dict(id=1, output="boat", match=False, prefill_length=prefill_lengths[1]),
            dict(id=2, output="bicycle", match=True, prefill_length=prefill_lengths[2]),
            dict(id=3, output="car", match=True, prefill_length=prefill_lengths[3]),
        ]
