import unittest.mock as um
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import lm_eval.evaluator
import torch

from .. import eval_adapter


def test_eval_adapter() -> None:
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    out = lm_eval.evaluator.evaluate(
        adapter, lm_eval.tasks.get_task_dict(["wikitext"]), limit=1
    )
    assert 1 < out["results"]["wikitext"]["word_perplexity"] < 200


def test_get_cache_str() -> None:
    adapter1 = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    adapter2 = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    adapter3 = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m-deduped")
    s1 = "This is some text"
    s2 = "This is some text"
    s3 = "This is different text"
    # Same model, same text
    assert adapter1._get_cache_str(s1) == adapter2._get_cache_str(s1)
    assert adapter1._get_cache_str(s1) == adapter1._get_cache_str(s2)
    # Same model, different text
    assert adapter1._get_cache_str(s1) != adapter1._get_cache_str(s3)
    # Different model, same text
    assert adapter1._get_cache_str(s1) != adapter3._get_cache_str(s1)


def test_prefill_with_cache() -> None:
    s = ["How are you", "She was walking down the street"]
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    adapter.model.double()
    dir_path = "cache/"
    with um.patch("torch.save") as mock_save:
        out = adapter.prefill_with_cache(s, max_context_length=64, dir_path=dir_path)
    assert mock_save.call_count == 2
    for i, (call_args, ctx) in enumerate(zip(mock_save.call_args_list, s)):
        args = call_args.args
        inp = adapter.tokenizer(ctx, return_tensors="pt")
        pkv = adapter._kv_to_tensor(adapter.model(**inp).past_key_values)
        torch.testing.assert_close(out[i], pkv)
        torch.testing.assert_close(args[0], pkv)
        assert args[1] == Path(dir_path, adapter._get_cache_str(ctx) + ".pt")


@contextmanager
def torch_load_save_to_memory() -> Iterator[None]:
    cache = {}

    def save_to_dict(t: torch.Tensor, path: Path) -> None:
        cache[path] = t

    def load_from_dict(path: Path) -> torch.Tensor:
        return cache[path]

    def exists_in_dict(path: Path) -> bool:
        return path in cache

    with um.patch("torch.save", save_to_dict), um.patch(
        "torch.load", load_from_dict
    ), um.patch("pathlib.Path.exists", exists_in_dict):
        yield


def test_greedy_sample() -> None:
    ctxs = ["How are you", "She was walking down the street"]
    questions = [" doing", " and"]

    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    adapter.model.double()
    adapter.model.config.max_position_embeddings = 10
    num_generated_tokens = 1
    max_prompt_and_generated_tokens = 6
    max_ctx = (
        adapter.model.config.max_position_embeddings - max_prompt_and_generated_tokens
    )
    assert max_ctx < len(adapter.tok_encode(ctxs[1]))

    def run_reference(ctx: str, question: str) -> torch.Tensor:
        ctx_ids = adapter.tok_encode(ctx)[-max_ctx:]
        input_ids = ctx_ids + adapter.tok_encode(question)
        out: torch.Tensor = adapter.model.generate(
            torch.tensor(input_ids)[None],
            max_length=len(input_ids) + num_generated_tokens,
        )
        return out[:, -num_generated_tokens:]

    out_expected = torch.cat(
        [run_reference(ctx, question) for ctx, question in zip(ctxs, questions)],
        dim=0,
    )
    for use_cache in [False, True]:
        with torch_load_save_to_memory():
            out = adapter.greedy_sample(
                ctxs,
                questions,
                num_generated_tokens,
                max_prompt_and_generated_tokens=max_prompt_and_generated_tokens,
                use_cache=use_cache,
            )
            assert out.shape == (2, 1)
            torch.testing.assert_close(
                out,
                out_expected,
                msg=f"greedy_sample mismatch when use_cache={use_cache}",
            )
