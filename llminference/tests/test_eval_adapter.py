import collections
import unittest.mock as um
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Optional, cast

import lm_eval.evaluator
import torch
from torch import Tensor

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
    n1 = 1024
    n2 = 768
    # Same model, same text, same length
    assert adapter1._get_cache_str(s1, n1) == adapter2._get_cache_str(s1, n1)
    assert adapter1._get_cache_str(s1, n1) == adapter1._get_cache_str(s2, n1)
    # Same model, different text, same length
    assert adapter1._get_cache_str(s1, n1) != adapter1._get_cache_str(s3, n1)
    # Different model, same text, same length
    assert adapter1._get_cache_str(s1, n1) != adapter3._get_cache_str(s1, n1)
    # Same model, same text, different length
    assert adapter1._get_cache_str(s1, n1) != adapter1._get_cache_str(s1, n2)


def test_prefill_with_cache() -> None:
    s = ["How are you", "She was walking down the street"]
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    adapter.model.double()
    dir_path = "cache/"
    with um.patch("torch.save") as mock_save:
        pkv, sequence_lens = adapter.prefill_with_cache(
            s, max_context_length=64, dir_path=dir_path
        )
    assert mock_save.call_count == 2
    for i, (call_args, ctx) in enumerate(zip(mock_save.call_args_list, s)):
        args = call_args.args
        inp = adapter.tokenizer(ctx, return_tensors="pt")
        pkv_reference = adapter.model(**inp).past_key_values
        assert sequence_lens[i] == inp["input_ids"].shape[1]
        torch.testing.assert_close(
            adapter._kv_to_tensor(pkv)[:, :, i, None, :, : sequence_lens[i], :],
            adapter._kv_to_tensor(pkv_reference),
        )
        torch.testing.assert_close(args[0], adapter._kv_to_tensor(pkv_reference))
        assert args[1] == Path(dir_path, adapter._get_cache_str(ctx, 64) + ".pt")


@contextmanager
def torch_load_save_to_memory(
    cache: Optional[Dict[Path, Tensor]] = None
) -> Iterator[None]:
    cache_dict = {} if cache is None else cache

    def save_to_dict(t: Tensor, path: Path) -> None:
        cache_dict[path] = t

    def load_from_dict(path: Path) -> Tensor:
        return cache_dict[path]

    def exists_in_dict(path: Path) -> bool:
        return path in cache_dict

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

    def run_reference(ctx: str, question: str) -> Tensor:
        ctx_ids = adapter.tok_encode(ctx)[-max_ctx:]
        input_ids = ctx_ids + adapter.tok_encode(question)
        out: Tensor = adapter.model.generate(
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


def test_greedy_sample_generation_context() -> None:
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    context = ["You think greedy sampling is wrong because greed is wrong."]
    prompt = [" I think"]

    @contextmanager
    def prune_model_context(model: eval_adapter.Model) -> eval_adapter.Model:
        original_params = {k: v.clone() for k, v in model.state_dict().items()}
        for p in model.parameters():
            p.data *= (p.abs().max() / 1000) < p.abs()
        yield model
        model.load_state_dict(original_params)

    outputs: Dict[str, Dict[str, str]] = collections.defaultdict(dict)
    caches: Dict[str, Dict[Path, Tensor]] = collections.defaultdict(dict)
    for name, generation_context in [
        ("default", eval_adapter.null_model_context),
        ("pruned", prune_model_context),
    ]:
        with torch_load_save_to_memory(caches[name]):
            for use_cache in [False, True]:
                # Run twice to see what the cache-hit behaviour is
                for run in range(2):
                    out = adapter.greedy_sample(
                        context,
                        prompt,
                        num_generated_tokens=8,
                        generation_context=generation_context,
                        use_cache=use_cache,
                    )
                    outputs[name][
                        f"cache_{run}" if use_cache else f"nocache_{run}"
                    ] = adapter.tok_decode(out[0])

    # Pruning gives different outputs
    assert outputs["default"]["nocache_0"] != outputs["pruned"]["nocache_0"]

    # Default always gives the same output, regardless of caching
    assert all(
        v == outputs["default"]["nocache_0"] for v in outputs["default"].values()
    )
    # Pruning always gives the same output, regardless of caching
    assert all(v == outputs["pruned"]["nocache_0"] for v in outputs["pruned"].values())

    # The cached state is the same
    for k in caches["default"]:
        assert torch.equal(caches["default"][k], caches["pruned"][k])


def dummy_fn() -> int:
    return 100


def test_patch_for_model() -> None:
    model = cast(eval_adapter.Model, None)
    assert dummy_fn() == 100
    with eval_adapter.patch_for_model(f"{__name__}.dummy_fn", lambda a: a, a=200)(
        model
    ):
        assert dummy_fn() == 200
    assert dummy_fn() == 100
