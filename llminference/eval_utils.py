"""Basic methods for prediction accuracy eval"""

from math import ceil
from typing import List

import regex as re
from tqdm import tqdm

from .eval_adapter import Adapter
from .utility import AnyDict, batches

CACHE_DIR = "/net/group/research/lukar/cache/"


def format_triviaqa(example: AnyDict) -> AnyDict:
    context = example["entity_pages"]["wiki_context"][0]
    prompt = f"Question: {example['question']}\nAnswer:"
    answers = example["answer"]["aliases"]
    return dict(context=context, prompt=prompt, answers=answers)


def evaluate_prediction(
    out: str,
    answers: List[str],
    leading_space_pattern: str = r"""^[\s"']+""",
) -> bool:
    out = re.sub(leading_space_pattern, "", out)
    return any(out.lower().startswith(answer.lower()) for answer in answers)


def evaluate_qa_task(
    adapter: Adapter,
    examples: List[AnyDict],
    batch_size: int,
    output_token_limit: int = 30,
    output_spare_tokens: int = 5,
    open_book: bool = False,
    use_cache: bool = True,
    cache_dir: str = CACHE_DIR,
) -> float:
    """Evaluate a generic QA task consisting of a list of examples, each a
    dictionary with keys "context", "prompt", and "answers".

    Args:
        adapter (L.Adapter): Adapter wrapper for the LM model
        examples (List[AnyDict]): QA dataset with "context", "prompt", and
        "answers" keys (context/prompt: str, answers: List[str])
        batch_size (int): Batch examples for greedy sample steps
        output_token_limit (int, optional): Defaults to 30.
        output_spare_tokens (int, optional): Defaults to 5.
        open_book (bool, optional): Prepend context to the prompt.
        Defaults to False.
        use_cache (bool, optional): Use cached context when sampling.
        Defaults to True.
        cache_dir (str, optional): Context cache path.
        Defaults to CACHE_DIR.

    Returns:
        float: Prediction accuracy
    """
    correct = []
    for batch in tqdm(
        batches(examples, batch_size),
        desc=f"Evaluating {adapter.model.name_or_path}",
        total=ceil(len(examples) / batch_size),
    ):
        ctxs_batch = [x["context"] if open_book else "" for x in batch]
        prompts_batch = [x["prompt"] for x in batch]
        answers_batch = [x["answers"] for x in batch]
        max_answer_tokens = max(
            len(adapter.tok_encode(a)) for answers in answers_batch for a in answers
        )
        out_ids_batch = adapter.greedy_sample(
            ctxs_batch,
            prompts_batch,
            num_generated_tokens=min(
                output_token_limit, output_spare_tokens + max_answer_tokens
            ),
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        out_batch = [adapter.tok_decode(o) for o in out_ids_batch]
        correct.extend(list(map(evaluate_prediction, out_batch, answers_batch)))
    return sum(correct) / len(correct)
