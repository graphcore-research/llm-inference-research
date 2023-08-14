import unittest.mock as um
from typing import Any, List

from .. import eval_utils
from ..eval_adapter import Adapter


def test_evaluate_prediction() -> None:
    out = "\n\n BICYCLE!"
    answers1 = ["Bicycle", "car"]
    answers2 = ["plane", "boat", "\nbicycle"]
    assert eval_utils.evaluate_prediction(out, answers1)
    assert not eval_utils.evaluate_prediction(out, answers2)


def test_evaluate_qa_task() -> None:
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
        {"context": ctx, "prompt": prompt, "answers": answers}
        for ctx, prompt, answers in zip(example_ctxs, example_prompts, example_answers)
    ]

    def mock_sample(
        self: Adapter, ctxs_batch: List[str], prompts_batch: List[str], **kwargs: Any
    ) -> List[List[int]]:
        # Just return the question
        return [self.tok_encode(q) for q in prompts_batch]

    with um.patch.object(Adapter, "greedy_sample", mock_sample):
        assert eval_utils.evaluate_qa_task(adapter, examples, 2) == 0.75
