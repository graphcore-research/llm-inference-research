# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

"""Implements the artificial needle-in-a-haystack task.

Uses the setup described in the Gemini 1.5 report, Google.
"""

import re
from typing import Any, Dict, Iterable, List, Union

import datasets
import numpy as np
import torch
import transformers
from tqdm import tqdm

from .. import eval_adapter

SEP_FORMAT = ".\n"
PREFIX = "Articles by Paul Graham:\n"
NEEDLE_FORMAT = (
    "The best thing to do in San Francisco is to "
    "eat a sandwich and sit in Dolores Park on a sunny day.\n"
)
QUERY_FORMAT = (
    "\n\nWhat is the best thing to do in San Francisco?"
    ' "The best thing to do in San Francisco is to'
)
REFERENCE_OUTPUT = "eat a sandwich and sit in Dolores Park on a sunny day"
MATCH_PATTERN = r"^\s*" + REFERENCE_OUTPUT


def get_default_lengths(max_length: int, step: int = 1024) -> List[int]:
    """A linear range of context lengths to evaluate."""
    # Provide 128 tokens of space for the needle, query and output
    return [x - 128 for x in range(step, max_length + 1, step)]


class Dataset:
    @staticmethod
    def data(
        tokenizer: transformers.PreTrainedTokenizerBase,
        lengths: List[int],
        depth_steps: int,
        seed: int = 4852341,
    ) -> Iterable[Dict[str, Any]]:
        def tokenize(s: str) -> List[int]:
            tokens: List[int] = tokenizer(s, add_special_tokens=False).input_ids
            return tokens

        rng = np.random.RandomState(seed)

        articles = datasets.load_dataset(
            "Mindofmachine/paul_graham_and_sam_altman_articles",
            revision="e63d4aa9a59da7f95a766af6437ee3e465e1f360",
        )["train"]["content"]
        rng.shuffle(articles)
        character_limit = int(1 + np.mean([len(t) for t in tokenizer.vocab])) * max(
            lengths
        )
        context_tokens = tokenize("\n\n".join(articles)[:character_limit])
        if len(context_tokens) < max(lengths):
            raise ValueError(
                "Insufficient tokens in dataset"
                f" (expected: {max(lengths)}, actual: {len(context_tokens)})"
            )
        prefix_tokens = tokenize(PREFIX)

        for length in lengths:
            for depth_ratio in np.linspace(0, 1, depth_steps):
                depth_index = int(depth_ratio * length)
                needle_tokens = tokenize(
                    (SEP_FORMAT if depth_index else "") + NEEDLE_FORMAT
                )
                question_tokens = tokenize(SEP_FORMAT + QUERY_FORMAT)
                prompt = (
                    [tokenizer.bos_token_id]
                    + prefix_tokens
                    + context_tokens[:depth_index]
                    + needle_tokens
                    + context_tokens[depth_index:length]
                    + question_tokens
                )
                yield dict(
                    id=f"{length}-{depth_index}",
                    length=length,
                    depth=float(depth_ratio),
                    prompt=prompt,
                )


def greedy_sample(
    model: transformers.PreTrainedModel,
    prompt: List[int],
    max_generated_tokens: int,
    prefill_chunk_length: int,
) -> List[int]:
    """Unbatched, chunked-prefill generation"""
    context = getattr(model, "generation_context", eval_adapter.null_model_context)
    with torch.no_grad(), context(model) as model:
        input_ids = torch.zeros(
            len(prompt) + max_generated_tokens, dtype=torch.long, device=model.device
        )
        input_ids[: len(prompt)] = torch.tensor(prompt, device=input_ids.device)
        position_ids = torch.arange(input_ids.shape[0], device=model.device)
        attention_mask = torch.ones(
            input_ids.shape[0], dtype=torch.long, device=model.device
        )
        past_key_values = None
        idx = 0
        while True:
            end = max(idx + 1, min(len(prompt), idx + prefill_chunk_length))
            if end >= input_ids.shape[0]:
                break
            out = model(
                input_ids=input_ids[None, idx:end],
                position_ids=position_ids[None, idx:end],
                attention_mask=attention_mask[None, :end],
                past_key_values=past_key_values,
            )
            if end >= len(prompt):
                if not out.logits.isfinite().all():
                    raise ValueError("Output logits are not finite")
                input_ids[end] = out.logits[0, -1].argmax()
            past_key_values = out.past_key_values
            idx = end
        return input_ids[-max_generated_tokens:].tolist()


def evaluate(
    adapter: eval_adapter.Adapter,
    examples: Iterable[Dict[str, Any]],
    prefill_chunk_length: int = 256,
    batch_size: int = 1,
    progress: Union[bool, str] = True,
) -> Iterable[Dict[str, Any]]:
    """Evaluate needle-in-a-haystack examples, returning un-aggregated results.

    prefill_chunk_length -- how many tokens of prefill to compute in one go;
                            this should impact only memory, not model output

    batch_size -- must be set to 1; included for compatibility with qa.evaluate

    Yields:
      {"id": str, "length": int, "depth": float, "output": str, "match": bool}
      (One for each input)
    """
    assert batch_size == 1, "batching is not implemented in needle.evaluate()"
    for example in tqdm(
        list(examples),
        desc=(
            progress
            if isinstance(progress, str)
            else f"Evaluating {adapter.model.name_or_path}"
        ),
        disable=progress is False,
    ):
        generate_tokens = 5 + len(adapter.tokenizer.tokenize(REFERENCE_OUTPUT))
        output = adapter.tokenizer.decode(
            greedy_sample(
                adapter.model,
                example["prompt"],
                max_generated_tokens=generate_tokens,
                prefill_chunk_length=prefill_chunk_length,
            )
        )
        yield dict(
            id=example["id"],
            length=example["length"],
            depth=example["depth"],
            output=output,
            match=bool(re.match(MATCH_PATTERN, output)),
        )
