import itertools as it
import unittest.mock as um
from typing import Any, List

import pytest

from .. import qa
from ..eval_adapter import Adapter


def test_datasets() -> None:
    for cls, kwargs in [
        (qa.TriviaQA, dict(context="search")),
        (qa.TriviaQA, dict(context="wiki")),
        (qa.SQuAD, {}),
    ]:
        data = cls.data(**kwargs)  # type:ignore[attr-defined]
        assert len(data) > 100
        assert set(data.features) == {
            "question_id",
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
        question_id="#0",
        question="Dummy qu",
        context="Dummy context",
        answers=["A1", "A2"],
        examples=[
            dict(question="Eg 1", answers=["long answer 1", "shortans1"]),
            dict(question="Eg 2", answers=["shortans2", "long answer 2"]),
        ],
    )
    # Zero shot
    example = qa.add_zero_shot_prompt(datum, "\nQ: {question}\nA:")
    assert example.pop("prompt") == "\nQ: Dummy qu\nA:"
    assert example == datum

    # Two shot
    example = qa.add_few_shot_prompt(datum, k=2, prompt_template="\nQ: {question}\nA:")
    assert example["prompt"] == "".join(
        [
            "\nQ: Eg 1\nA: shortans1",
            "\nQ: Eg 2\nA: shortans2",
            "\nQ: Dummy qu\nA:",
        ]
    )

    # Too-many-shot
    with pytest.raises(ValueError) as error:
        qa.add_few_shot_prompt(datum, k=3)
    assert "k=3" in str(error)
    assert "2 examples" in str(error)


def test_evaluate_prediction() -> None:
    out = "\n\n BICYCLE!"
    answers1 = ["Bicycle", "car"]
    answers2 = ["plane", "boat", "\nbicycle"]
    assert qa.evaluate_prediction(out, answers1)
    assert not qa.evaluate_prediction(out, answers2)


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
        {"question_id": i, "context": ctx, "prompt": prompt, "answers": answers}
        for i, ctx, prompt, answers in zip(
            it.count(), example_ctxs, example_prompts, example_answers
        )
    ]

    def mock_sample(
        self: Adapter, ctxs_batch: List[str], prompts_batch: List[str], **kwargs: Any
    ) -> List[List[int]]:
        # Just return the prompt
        return [self.tok_encode(q) for q in prompts_batch]

    with um.patch.object(Adapter, "greedy_sample", mock_sample):
        assert list(qa.evaluate(adapter, examples, 2)) == [
            dict(question_id=0, output="plane", match=True),
            dict(question_id=1, output="boat", match=False),
            dict(question_id=2, output="bicycle", match=True),
            dict(question_id=3, output="car", match=True),
        ]
