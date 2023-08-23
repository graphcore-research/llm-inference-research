import json
import sys
from typing import *

import datasets
import torch

import llminference as L

torch.set_num_threads(16)


def preprocess(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    chars_min, chars_max = 4000, 8000
    filtered_contexts = [
        c
        for c in d["search_results"]["search_context"]
        if chars_min <= len(c) <= chars_max
    ]
    if filtered_contexts:
        # Take the first context doc that has an in-range length
        return dict(
            question_id=d["question_id"],
            question=d["question"],
            answers=d["answer"]["aliases"],
            context=filtered_contexts[0],
        )


def add_prompt(d: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    return dict(**d, prompt=prompt_template.format(**d))


prompt_templates = [
    "\nQuestion: {question}\nAnswer:",
    "\nQuestion: {question}\nAnswer: ",
    "\nQuestion:\n{question}\nAnswer:\n",
    "\nQuestion: {question}\nSingle-word answer:",
    "\nQuestion: {question}\nShort answer:",
    "\nQuestion: {question} The answer is",
    "\nQ: {question}\nA:",
]
n_examples = 400

# Script

adapter = L.Adapter.from_pretrained("EleutherAI/pythia-2.8b")
data = L.utility.map_and_filter(
    datasets.load_dataset("trivia_qa", name="rc")["validation"], preprocess
).shuffle(487523)
with open("results.jsonl", "w") as out:
    for prompt_template in prompt_templates:
        for open_book in [True, False]:
            print(f"prompt={prompt_template!r}, open_book={open_book}", file=sys.stderr)
            results = L.eval_utils.evaluate_qa_task(
                adapter,
                [add_prompt(data[i], prompt_template) for i in range(n_examples)],
                batch_size=5,
                open_book=open_book,
                use_cache=True,
                cache_dir="/net/group/research/douglaso/llminference-cache",
            )
            for result in results:
                print(
                    json.dumps(
                        dict(
                            prompt_template=prompt_template,
                            open_book=open_book,
                            **result,
                        )
                    ),
                    file=out,
                    flush=True,
                )
