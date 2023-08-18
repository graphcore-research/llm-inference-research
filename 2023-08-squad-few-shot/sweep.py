import contextlib
import json
import unittest.mock as um
from functools import partial
from pathlib import Path
from typing import *

import torch

import llminference as L

# Generic
CACHE = "/net/group/research/douglaso/llminference-cache"
torch.set_num_threads(16)

# Script
adapter = L.Adapter.from_pretrained("EleutherAI/pythia-1b")
batch_size = 10
confusion_contexts = 3


@contextlib.contextmanager
def maybe_sparse_softmax(k: Optional[int]) -> Iterator[None]:
    if k is not None:
        with um.patch(
            "torch.nn.functional.softmax",
            partial(L.sparse_attention.sparse_softmax_fixed_k, k=k),
        ):
            yield
    else:
        yield


def evaluate(
    dataset: str,
    confusion_contexts: int,
    shots: int,
    prompt_template: str,
    sparse_softmax_k: Optional[int],
) -> List[Dict[str, Any]]:
    settings = dict(
        dataset=dataset,
        confusion_contexts=confusion_contexts,
        shots=shots,
        prompt_template=prompt_template,
        sparse_softmax_k=sparse_softmax_k,
    )
    if dataset == "squad":
        data = L.qa.SQuAD.data(confusion_contexts=confusion_contexts)
    if dataset == "trivia_qa":
        assert confusion_contexts == 0
        data = L.qa.TriviaQA.data()
    examples = [
        L.qa.add_few_shot_prompt(
            data[i],
            k=shots,
            prompt_template=prompt_template,
            answer_template=" {answer}",
        )
        for i in range(400)
    ]
    with maybe_sparse_softmax(sparse_softmax_k):
        return [
            dict(**settings, **result)
            for result in L.qa.evaluate(
                adapter,
                examples,
                batch_size=batch_size,
                open_book=True,
                cache_dir=CACHE,
                desc=f"Evaluating {settings}",
            )
        ]


# dataset = "trivia_qa"
# confusion_contexts = 0
dataset = "squad"
# confusion_contexts = 0
# confusion_contexts = 1
confusion_contexts = 3
# confusion_contexts = 5
# confusion_contexts = 7

with open(f"{dataset}{confusion_contexts}.jsonl", "w") as outf:
    for prompt_template in [
        "\nQuestion: {question}\nSingle-word answer:",
        "\nQuestion: {question}\nAnswer:",
        "\nQ: {question}\nA:",
    ]:
        for shots in [0, 1]:
            for sparse_softmax_k in [None, 16, 128]:
                for result in evaluate(
                    dataset,
                    confusion_contexts,
                    shots,
                    prompt_template,
                    sparse_softmax_k,
                ):
                    print(json.dumps(result), file=outf, flush=True)
