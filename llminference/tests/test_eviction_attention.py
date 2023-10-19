import pytest
import torch
from torch import tensor

from .. import eval_adapter
from .. import eviction_attention as ea


@pytest.mark.parametrize("strategy", ["sum_weight", "lru"])
def test_eviction_strategy(strategy: str) -> None:
    eviction = ea.Eviction(
        ea.Settings(k=3, local_k=2, strategy=strategy),
        (1, 1, 10),
        device=torch.device("cpu"),
    )
    assert eviction.mask.all(), "nothing evicted"

    # Step 1
    eviction.update(
        tensor([0.1, 0.7, 0.2])[None, None, None],
        tensor([2, 1, 0])[None, None],
    )
    assert eviction.mask.all(), "nothing evicted"

    # Step 2
    eviction.update(
        tensor([0.1, 0.0, 0.0, 0.0, 0.9, 0.0])[None, None, None],
        tensor([4, 3, 2, 1, 0, -1])[None, None],
    )
    # Last 2 tokens (excluding m) are local
    # Sum-weight: best (non-local) score is index 1 (0.7)
    # LRU: most recently used (non-local) is index 1 (used in step 1)
    assert torch.equal(eviction.mask[0, 0, :6].long(), tensor([0, 1, 0, 1, 1, 0]))


def test_convert_gptneox() -> None:
    settings = ea.Settings(k=16, local_k=8, strategy="lru")
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-160m")
    eviction_adapter = eval_adapter.Adapter(
        ea.convert_gptneox(adapter.model, settings),
        adapter.tokenizer,
        adapter.batch_size,
    )
    for layer in eviction_adapter.model.gpt_neox.layers:
        layer.attention.eviction_masks = []

    context = " predict parrot" * 10 + " noise" * 10
    prompt = " predict"

    dense_out = adapter.greedy_sample(
        [context],
        [prompt],
        num_generated_tokens=4,
        use_cache=False,
        combine_context_and_prompt=False,
    )
    eviction_out = eviction_adapter.greedy_sample(
        [context],
        [prompt],
        num_generated_tokens=4,
        use_cache=False,
        combine_context_and_prompt=False,
    )

    # Check the masks don't exceed `k+1`
    # Note: +1 for the current query's (key, value)
    for layer in eviction_adapter.model.gpt_neox.layers:
        assert layer.attention.eviction_masks
        for mask in layer.attention.eviction_masks:
            assert (mask.sum(-1) == settings.k + 1).all()

    # This is a slightly risky (possibly unstable) test
    #  - dense will have conditioned to follow "predict" with "parrot"
    #  - eviction will not have seen lots of "predict parrot"
    assert adapter.tok_decode(dense_out[0]).startswith(" parrot")
    assert not adapter.tok_decode(eviction_out[0]).startswith(" parrot")
