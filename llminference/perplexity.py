"""Method for evaluating models for perplexity.

For example:

    adapter = ...
    warmup = 200
    num_examples = 10

    data = perplexity.WikiText.data(warmup)
    results = list(perplexity.evaluate(
        adapter,
        [data[i] for i in range(num_examples)],
        batch_size=5,
    ))
    print(results[0]["perplexity"])  # => 1.234
    print(results[9]["perplexity"])  # => 5.678
"""
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Union

import datasets
import torch
from torch import Tensor
from tqdm import tqdm

from . import utility
from .eval_adapter import Adapter, ModelContext
from .utility import AnyDict


class WikiText:
    """Filtered version of wikitext-2-v1 (training set) from HuggingFace"""

    @staticmethod
    def preprocess(d: Dict[str, Any], warmup: int) -> Optional[Dict[str, Any]]:
        """Generate a summarisation example from wikitext-2-v1.

        Yields: {"prefill": str, "reference": str}
        """
        text = d["text"]
        while warmup < len(text):
            if text[warmup].isspace():
                return dict(
                    prefill=text[:warmup],
                    reference=text[warmup:],
                )
            warmup += 1

    @classmethod
    def data(cls, shuffle_seed: int = 2353669, warmup: int = 200) -> datasets.Dataset:
        return utility.map_and_filter(
            datasets.load_dataset("wikitext", "wikitext-2-v1")["train"],
            partial(cls.preprocess, warmup=warmup),
        ).shuffle(shuffle_seed)


def calc_perplexity(logits: Tensor) -> Tensor:
    """Calculates the perplexity of an tensor of (normalised) logits.

    Values equal to exact zero are assumed to be padding and ignored.
    Expects a tensor of shape (..., sequence_length), and reduces across the
    sequence dimension.
    """
    num_non_zero = (logits != 0).sum(-1)
    mean = logits.sum(-1) / num_non_zero
    return torch.exp(-mean)


def evaluate(
    adapter: Adapter,
    examples: List[AnyDict],
    batch_size: int,
    generation_context: Optional[ModelContext] = None,
    progress: Union[bool, str] = True,
) -> Iterable[AnyDict]:
    """Evaluate perplexity with respect to a list of example sequences.

    Args:
        adapter (L.Adapter): Adapter wrapper for the LM model.
        examples (List[AnyDict]): Summarisation dataset containing:
        {prefill: str, reference: str}.
        batch_size (int): Batch examples for greedy sample steps.
        generation_context (ModelContext, optional): Override model
        during generation.
        progress (bool|str, optional): enable tqdm (True) and set description
        (str), or disable (False).

    Yields:
        {perplexity: int, prefill_length: int, reference_length: int}: For each input.
    """
    for batch in tqdm(
        list(utility.batches(examples, batch_size)),
        desc=(
            progress
            if isinstance(progress, str)
            else f"Evaluating {adapter.model.name_or_path}"
        ),
        disable=progress is False,
    ):
        prefill_lengths = [len(adapter.tok_encode(b["prefill"])) for b in batch]
        reference_lengths = [len(adapter.tok_encode(b["reference"])) for b in batch]
        logit_batch = adapter.forced_sample(
            [b["prefill"] for b in batch],
            [b["reference"] for b in batch],
            generation_context=generation_context,
        )
        perplexities = calc_perplexity(logit_batch)
        for p, pr_len, g_len in zip(perplexities, prefill_lengths, reference_lengths):
            yield dict(
                perplexity=p.item(),
                prefill_length=pr_len,
                reference_length=g_len,
            )
