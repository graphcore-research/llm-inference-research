import unittest.mock as um
from functools import partial
from pathlib import Path
from typing import Iterator

import datasets
from datasets import Dataset

import llminference as L
import llminference.sparse_attention as sa

CACHE_DIR = "/net/group/research/lukar/cache/"
DATASET_DIR = "/nethome/lukar/datasets/triviaqa_2k/"


def batch_data(dataset: Dataset, batch_size: int) -> Iterator[dict]:
    l = len(dataset)
    for i in range(0, l, batch_size):
        yield dataset[i : min(i + batch_size, l)]


def format_example(example: dict) -> dict:
    context = example["entity_pages"]["wiki_context"][0]
    question = f"Question: {example['question']}\nAnswer:"
    answers = [" " + answer for answer in example["answer"]["aliases"]]
    return dict(context=context, question=question, answers=answers)


def evaluate_task_baseline(model, batch_size, limit):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    triviaqa2k = datasets.load_from_disk(DATASET_DIR)["validation"]
    if limit is not None:
        triviaqa2k = triviaqa2k.select(range(limit))
    triviaqa2k_formatted = triviaqa2k.map(
        format_example, remove_columns=triviaqa2k.column_names
    )
    closed_correct = []
    open_correct = []
    for batch in batch_data(triviaqa2k_formatted, batch_size=batch_size):
        ctx_batch = batch["context"]
        questions_batch = batch["question"]
        answers_batch = batch["answers"]
        answers_batch_tok = [
            adapter.tokenizer(answers)["input_ids"] for answers in answers_batch
        ]
        max_len = max(
            [len(answer) for answers in answers_batch_tok for answer in answers]
        )

        # Closed-book
        out = adapter.greedy_sample(
            [""] * len(ctx_batch), questions_batch, max_len, use_cache=False
        )
        closed_correct.extend(
            list(map(L.eval_adapter.evaluate_prediction, out, answers_batch_tok))
        )

        # Open-book
        out = adapter.greedy_sample(
            ctx_batch,
            questions_batch,
            max_len,
            use_cache=True,
            cache_dir=f"{CACHE_DIR}{model}",
        )
        open_correct.extend(
            list(map(L.eval_adapter.evaluate_prediction, out, answers_batch_tok))
        )
    return dict(
        model=model,
        closed_book_acc=sum(closed_correct) / len(closed_correct),
        open_book_acc=sum(open_correct) / len(open_correct),
    )


def evaluate_task_sparse(model, batch_size, limit, **func_kwargs):
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    triviaqa2k = datasets.load_from_disk(DATASET_DIR)["validation"]
    if limit is not None:
        triviaqa2k = triviaqa2k.select(range(limit))

    triviaqa2k_formatted = triviaqa2k.map(
        format_example, remove_columns=triviaqa2k.column_names
    )

    with um.patch(
        "torch.nn.functional.softmax",
        partial(sa.sparse_softmax_fixed_k, **func_kwargs),
    ):
        correct = []
        for batch in batch_data(triviaqa2k_formatted, batch_size=batch_size):
            ctx_batch = batch["context"]
            questions_batch = batch["question"]
            answers_batch = batch["answers"]
            answers_batch_tok = [
                adapter.tokenizer(answers)["input_ids"] for answers in answers_batch
            ]
            max_len = max(
                [len(answer) for answers in answers_batch_tok for answer in answers]
            )

            # Open-book
            out = adapter.greedy_sample(
                ctx_batch,
                questions_batch,
                max_len,
                use_cache=True,
                cache_dir=f"{CACHE_DIR}{model}",
            )
            correct.extend(
                list(map(L.eval_adapter.evaluate_prediction, out, answers_batch_tok))
            )

        open_book_acc = sum(correct) / len(correct)
        return dict(model=model, open_book_acc=open_book_acc, **func_kwargs)


models = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    # "pythia-1.4b",
    "pythia-2.8b",
]

ks = [4, 8, 16, 32, 64, 128, 256]

settings = [
    dict(
        model=model,
        batch_size=16,
        limit=None,
        k=k,
    )
    for model in models
    for k in ks
]

dest = Path("out/triviaqa2k_sparse_softmax.jsonl")
L.utility.run_multiprocess_sweep(evaluate_task_sparse, settings, dest, n_workers=1)
