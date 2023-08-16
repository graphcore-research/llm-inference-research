import itertools as it
import unittest.mock as um
from typing import Any, List

from .. import qa
from ..eval_adapter import Adapter


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
