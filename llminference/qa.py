"""Methods for evaluation on question answering tasks.

For example:

    data = qa.TriviaQA.data()
    results = list(qa.evaluate(
        adapter,
        [qa.add_zero_shot_prompt(data[i], qa.TriviaQA.DEFAULT_PROMPT)
         for i in range(10)],
        batch_size=5,
        open_book=False,
    ))
    print(results[0])  # => {"question_id": ..., "output": ..., "match": True|False}

    em_accuracy = sum(r["match"] for r in results) / len(results)
    print(em_accuracy)
"""

from functools import partial
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import datasets
import regex as re
from tqdm import tqdm

from .eval_adapter import Adapter
from .utility import AnyDict, batches, map_and_filter

CACHE_DIR = "/net/group/research/lukar/cache/"


class TriviaQA:
    DEFAULT_PROMPT = "\nQuestion: {question}\nSingle-word answer:"
    DEFAULT_CHARS_RANGE = (4000, 8000)

    @staticmethod
    def preprocess(
        d: Dict[str, Any], chars_range: Tuple[int, int] = DEFAULT_CHARS_RANGE
    ) -> Optional[Dict[str, Any]]:
        filtered_contexts = [
            c
            for c in d["search_results"]["search_context"]
            if chars_range[0] <= len(c) <= chars_range[1]
        ]
        if filtered_contexts:
            # Take the first context doc that has an in-range length
            return dict(
                question_id=d["question_id"],
                question=d["question"],
                answers=d["answer"]["aliases"],
                context=filtered_contexts[0],
            )

    @classmethod
    def data(
        cls, chars_range: Tuple[int, int] = DEFAULT_CHARS_RANGE, seed: int = 487523
    ) -> datasets.Dataset:
        return map_and_filter(
            datasets.load_dataset("trivia_qa", name="rc")["validation"],
            partial(cls.preprocess, chars_range=chars_range),
        ).shuffle(seed)


def add_zero_shot_prompt(d: Dict[str, Any], prompt_template: str) -> Dict[str, Any]:
    """Add a key "prompt" to the returned dictionary, following `prompt_template`."""
    return dict(**d, prompt=prompt_template.format(**d))


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
