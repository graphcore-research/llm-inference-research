# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import unittest.mock as um
from typing import Any, List

from llminference.eval_adapter import Adapter
from llminference.tasks import repetition


def test_split_on_whitespace() -> None:
    text = "This is a\nshort\n\ntext."
    context_length = 5
    contexts = ["This is", " a\nshort", "\n\ntext."]
    ids = [0, 7, 15]
    assert list(zip(ids, contexts)) == list(
        repetition.split_on_whitespace(text, context_length)
    )


def test_split_into_prompt_and_reference() -> None:
    context = "He was walking down the street while looking at his phone."
    prompt_length = 3
    reference_length = 10
    expected = [
        (0, "He was", " walking down"),
        (len("He was walking down"), " the", " street while"),
        (
            len("He was walking down the street while"),
            " looking",
            " at his phone.",
        ),
    ]
    assert (
        list(
            repetition.split_into_prompt_and_reference(
                context, prompt_length, reference_length
            )
        )
        == expected
    )


def test_generate_examples() -> None:
    s1 = "This is a short sentence."
    s2 = " And another one to create a paragraph."
    text = s1 + s2
    examples = list(
        repetition.generate_examples(
            text, context_length=25, prompt_length=3, reference_length=5
        )
    )
    expected = [
        dict(
            id=0,
            context_id=0,
            context="This is a short sentence.",
            prompt="This",
            reference=" is a",
        ),
        dict(
            id=9,
            context_id=0,
            context="This is a short sentence.",
            prompt=" short",
            reference=" sentence.",
        ),
        dict(
            id=len(s1),
            context_id=len(s1),
            context=" And another one to create",
            prompt=" And",
            reference=" another",
        ),
        dict(
            id=len(s1) + 12,
            context_id=len(s1),
            context=" And another one to create",
            prompt=" one",
            reference=" to create",
        ),
    ]
    assert examples == expected


def test_shakespeare() -> None:
    data = repetition.Shakespeare.data(
        context_length=6000, prompt_length=128, reference_length=256
    )

    assert len(data) > 2000

    assert set(data.features) == {"id", "context_id", "context", "prompt", "reference"}

    assert len(set(data["context"])) == len(set(data["context_id"]))
    assert len(set(data["id"])) == len(data)  # ids are unique

    for example in data:
        assert 6000 + 32 >= len(example["context"]) >= 6000
        assert 128 + 32 >= len(example["prompt"]) >= 128
        assert 256 + 32 >= len(example["reference"]) >= 256
        assert example["prompt"] + example["reference"] in example["context"]


def test_evaluate_match_length() -> None:
    reference = "This is a sentence."
    generation = "This is b sentence. And is longer."
    assert repetition.evaluate_match_length(generation, reference) == dict(
        match_length_char=8, reference_length_char=19
    )

    # Check if it works for perfect match
    generation = "This is a sentence."
    assert repetition.evaluate_match_length(generation, reference) == dict(
        match_length_char=len(reference), reference_length_char=len(reference)
    )

    # Check leading space
    reference = "\nThis is a sentence."
    generation = "  This is b sentence. And is longer."
    assert repetition.evaluate_match_length(generation, reference) == dict(
        match_length_char=8, reference_length_char=len(reference) - 1
    )


def test_evaluate() -> None:
    adapter = Adapter.from_pretrained("EleutherAI/pythia-70m")
    examples = [
        {
            "id": 7,
            "context_id": 0,
            "context": "To be, or not to be: that is the question.",
            "prompt": " or not",
            "reference": " to be:",
        },
        {
            "id": 21,
            "context_id": 0,
            "context": "To be, or not to be: that is the question.",
            "prompt": " that is",
            "reference": " the question.",
        },
    ]

    def mock_sample(
        self: Adapter, ctxs: List[str], prompts: List[str], **kwargs: Any
    ) -> List[List[int]]:
        return [
            self.tok_encode(s)
            for s in [" to see. That is not the question.", " the question."]
        ]

    with um.patch.object(Adapter, "greedy_sample", mock_sample):
        results = list(repetition.evaluate(adapter, examples, batch_size=2))

        # Note - the first output is truncated to tokenised length of the reference
        assert results == [
            dict(
                id=7,
                context_id=0,
                output=" to see.",
                prefill_length=len(
                    adapter.tok_encode(
                        "To be, or not to be: that is the question.\n or not"
                    )
                ),
                match_length_char=3,
                reference_length_char=6,
            ),
            dict(
                id=21,
                context_id=0,
                output=" the question.",
                prefill_length=len(
                    adapter.tok_encode(
                        "To be, or not to be: that is the question.\n that is"
                    )
                ),
                match_length_char=13,
                reference_length_char=13,
            ),
        ]
