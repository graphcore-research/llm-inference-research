import unittest.mock as um
from pathlib import Path

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


def test_generate_kv_cache() -> None:
    s = ["How are you", "She was walking down the street"]
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    adapter.model.double()
    dir_path = "cache/"
    with um.patch("torch.save") as mock_save:
        adapter.generate_kv_cache(s, dir_path=dir_path)
    assert mock_save.call_count == 2
    for call_args, ctx in zip(mock_save.call_args_list, s):
        args = call_args.args
        inp = adapter.tokenizer(ctx, return_tensors="pt")
        pkv = adapter._kv_to_tensor(adapter.model(**inp).past_key_values)
        torch.testing.assert_close(args[0], pkv)
        assert args[1] == Path(dir_path, adapter._get_cache_str(ctx) + ".pt")


def test_greedy_sample() -> None:
    ctxs = ["How are you", "She was walking down the street"]
    questions = [" doing", " and"]

    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    adapter.model.double()
    num_new_tokens = 1

    inps = [
        adapter.tokenizer(ctx + question, return_tensors="pt")
        for ctx, question in zip(ctxs, questions)
    ]
    out_expected = torch.cat(
        [
            adapter.model.generate(
                **inp, max_length=inp["input_ids"].shape[1] + num_new_tokens
            )[:, -num_new_tokens:]
            for inp in inps
        ],
        dim=0,
    )

    # Test without caching
    out_no_cache = adapter.greedy_sample(
        ctxs, questions, num_new_tokens, use_cache=False
    )
    assert out_no_cache.shape == (2, 1)
    torch.testing.assert_close(out_no_cache, out_expected)

    # Test with caching
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
        out_cache = adapter.greedy_sample(
            ctxs, questions, num_new_tokens, use_cache=True, cache_dir="cache/"
        )
        assert out_cache.shape == (2, 1)
        torch.testing.assert_close(out_cache, out_expected)
