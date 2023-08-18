"""Methods for evaluation on question answering tasks.

For example:

    data = qa.TriviaQA.data()
    results = list(qa.evaluate(
        adapter,
        [qa.add_zero_shot_prompt(data[i]) for i in range(10)],
        batch_size=5,
        open_book=False,
    ))
    print(results[0])  # => {"question_id": ..., "output": ..., "match": True|False}

    em_accuracy = sum(r["match"] for r in results) / len(results)
    print(em_accuracy)
"""

import collections
from functools import partial
from math import ceil
from typing import Any, Iterable, List, Optional, Tuple

import datasets
import numpy as np
import regex as re
from tqdm import tqdm

from .eval_adapter import Adapter
from .utility import AnyDict, batches, map_full_batch

CACHE_DIR = "/net/group/research/lukar/cache/"


class TriviaQA:
    @staticmethod
    def make_example(row: AnyDict) -> AnyDict:
        return dict(
            question_id=row["question_id"],
            question=row["question"],
            answers=row["answer"]["aliases"],
        )

    @classmethod
    def preprocess(
        cls,
        rows: List[AnyDict],
        examples: int = 5,
        chars_range: Tuple[int, int] = (4000, 8000),
        seed: int = 2751584,
    ) -> Iterable[AnyDict]:
        """Generate a few-shot generative evaluation dataset from TriviaQA.

        Yields: {"question_id": str,
                 "question": str,
                 "answers": [str],
                 "context": str,
                 "examples": [{"question_id": str, "question": str, "answers": [str]}]}
        """
        rng = np.random.RandomState(seed)
        for row in rows:
            # Take the first context doc that has an in-range length
            filtered_contexts = [
                c
                for c in row["search_results"]["search_context"]
                if chars_range[0] <= len(c) <= chars_range[1]
            ]
            if filtered_contexts:
                # Ensure we never use this question as a few-shot example
                # (note that each question_id can appear twice in TriviaQA, hence +2)
                few_shot_examples = [
                    cls.make_example(rows[i])
                    for i in rng.choice(len(rows), examples + 2)
                    if rows[i]["question_id"] != row["question_id"]
                ][:examples]
                yield dict(
                    **cls.make_example(row),
                    context=filtered_contexts[0],
                    examples=few_shot_examples,
                )

    @classmethod
    def data(
        cls, shuffle_seed: int = 487523, **preprocess_args: Any
    ) -> datasets.Dataset:
        return map_full_batch(
            datasets.load_dataset("trivia_qa", name="rc")["validation"],
            partial(cls.preprocess, **preprocess_args),
        ).shuffle(shuffle_seed)


class SQuAD:
    @staticmethod
    def format_context(row: AnyDict) -> str:
        return f"Title: {row['title'].replace('_', ' ')}. Background: {row['context']}"

    @staticmethod
    def make_example(row: AnyDict) -> AnyDict:
        return dict(
            question_id=row["id"],
            question=row["question"],
            answers=row["answers"]["text"],
        )

    @classmethod
    def preprocess(
        cls,
        rows: List[AnyDict],
        confusion_contexts: int = 7,
        examples: int = 5,
        seed: int = 82735235,
    ) -> Iterable[AnyDict]:
        """Generate a few-shot generative evaluation dataset from SQuAD.

        Yields: {"question_id": str,
                 "context": str,
                 "question": str,
                 "answers": [str],
                 "examples": [{"question_id": str, "question": str, "answers": [str]}]}
        """
        rng = np.random.RandomState(seed)
        context_to_examples = collections.defaultdict(list)
        for row in rows:
            context_to_examples[row["context"]].append(cls.make_example(row))
        all_examples = [e for exs in context_to_examples.values() for e in exs]
        all_contexts = list({cls.format_context(row) for row in rows})

        for row in rows:
            # Find `confusion_contexts` other contexts, shuffle & concatenate
            contexts = [cls.format_context(row)] + [
                all_contexts[i]
                for i in rng.choice(len(all_contexts), confusion_contexts)
            ]
            rng.shuffle(contexts)
            full_context = (
                "\n".join(contexts)
                + f"\nFrom what you've just read about {row['title'].replace('_', ' ')}"
                + ", please answer the following questions."
            )
            # Find `examples` other example questions, starting with the same
            # context, but then broadening the search if there aren't enough.
            same_context_examples = [
                c
                for c in context_to_examples[row["context"]]
                if c["question_id"] != row["id"]
            ]
            rng.shuffle(same_context_examples)  # type:ignore[arg-type]
            other_examples = [
                all_examples[i]
                for i in rng.choice(len(all_examples), examples + 1)
                if all_examples[i]["question_id"] != row["id"]
            ]
            yield dict(
                **cls.make_example(row),
                context=full_context,
                examples=(same_context_examples + other_examples)[:examples],
            )

    @classmethod
    def data(
        cls, shuffle_seed: int = 945237, **preprocess_args: Any
    ) -> datasets.Dataset:
        return map_full_batch(
            datasets.load_dataset("squad")["validation"],
            partial(cls.preprocess, **preprocess_args),
        ).shuffle(shuffle_seed)


DEFAULT_ZERO_SHOT_PROMPT = "\nQuestion: {question}\nSingle-word answer:"
DEFAULT_FEW_SHOT_PROMPT = "\nQuestion: {question}\nAnswer:"


def add_few_shot_prompt(
    datum: AnyDict,
    k: int,
    prompt_template: Optional[str] = None,
    answer_template: str = " {answer}",
) -> AnyDict:
    """Add a key "prompt" to the returned dictionary, following `prompt_template`.

    Args:
        datum (AnyDict): Single example from the dataset, containing:
            {"question_id": str,
             "question": str,
             "answers": List[str],
             "examples": List[{"question": str, "answers": [str]}]}
        k (int): Number of k-shot examples to provide.
        prompt_template (str): String template e.g. "Q: {question}, A:". Default:
        (DEFAULT_ZERO_SHOT_PROMPT if k == 0 else DEFAULT_FEW_SHOT_PROMPT).
        answer_template (str): String template e.g. " {answer}", used for k-shot
        examples only.

    Returns:
        {**datum, "prompt": str}: As `datum` but with a formatted "prompt".
    """
    if prompt_template is None:
        prompt_template = (
            DEFAULT_ZERO_SHOT_PROMPT if k == 0 else DEFAULT_FEW_SHOT_PROMPT
        )
    examples = datum.get("examples", [])
    if len(examples) < k:
        raise ValueError(
            f"Cannot form a (k={k})-shot prompt for question {datum['question_id']}"
            f", which only provides {len(examples)} examples"
        )
    prompt = "".join(
        [
            prompt_template.format(**eg)
            + answer_template.format(answer=min(eg["answers"], key=len))
            for eg in examples[:k]
        ]
        + [prompt_template.format(**datum)]
    )
    return dict(**datum, prompt=prompt)


def add_zero_shot_prompt(
    datum: AnyDict, prompt_template: Optional[str] = None
) -> AnyDict:
    """Add a key "prompt" to the returned dictionary, following `prompt_template`."""
    return add_few_shot_prompt(
        datum, k=0, prompt_template=prompt_template, answer_template=""
    )


def evaluate_prediction(
    out: str,
    answers: List[str],
    leading_space_pattern: str = r"""^[\s"']+""",
) -> bool:
    out = re.sub(leading_space_pattern, "", out)
    return any(out.lower().startswith(answer.lower()) for answer in answers)


def evaluate(
    adapter: Adapter,
    examples: List[AnyDict],
    batch_size: int,
    output_token_limit: int = 30,
    output_spare_tokens: int = 5,
    open_book: bool = False,
    use_cache: bool = True,
    cache_dir: str = CACHE_DIR,
    desc: Optional[str] = None,
) -> Iterable[AnyDict]:
    """Evaluate a generic QA task consisting of a list of examples, each a
    dictionary with keys "context", "prompt", and "answers".

    Args:
        adapter (L.Adapter): Adapter wrapper for the LM model
        examples (List[AnyDict]): QA dataset containing:
          {context: str, prompt: str, question_id: str, answers: List[str]}
        batch_size (int): Batch examples for greedy sample steps
        output_token_limit (int, optional): Defaults to 30.
        output_spare_tokens (int, optional): Defaults to 5.
        open_book (bool, optional): Prepend context to the prompt.
        Defaults to False.
        use_cache (bool, optional): Use cached context when sampling.
        Defaults to True.
        cache_dir (str, optional): Context cache path.
        Defaults to CACHE_DIR.

    Yields:
        {"question_id": int, "output": str, "match": bool}: One for each input
    """
    for batch in tqdm(
        batches(examples, batch_size),
        desc=desc or f"Evaluating {adapter.model.name_or_path}",
        total=ceil(len(examples) / batch_size),
    ):
        max_answer_tokens = max(
            len(adapter.tok_encode(a)) for x in batch for a in x["answers"]
        )
        out_ids_batch = adapter.greedy_sample(
            [x["context"] if open_book else "\n" for x in batch],
            [x["prompt"] for x in batch],
            num_generated_tokens=min(
                output_token_limit, output_spare_tokens + max_answer_tokens
            ),
            use_cache=use_cache and open_book,
            cache_dir=cache_dir,
        )
        for x, ids in zip(batch, out_ids_batch):
            output = adapter.tok_decode(ids)
            yield dict(
                question_id=x["question_id"],
                output=output,
                match=evaluate_prediction(output, x["answers"]),
            )
