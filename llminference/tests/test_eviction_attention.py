from functools import partial

import numpy as np
import torch
from torch import tensor

from .. import eval_adapter
from .. import eviction_attention as ea


def test_eviction() -> None:
    eviction = ea.Eviction(
        ea.Settings(k=3, local_k=2), (1, 1, 10), device=torch.device("cpu")
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
    # Last 2 tokens (excluding m) are local, best non-local score is 0.7 (index 1)
    assert torch.equal(eviction.mask[0, 0, :6].long(), tensor([0, 1, 0, 1, 1, 0]))


def test_eviction_neox() -> None:
    settings = ea.Settings(k=16, local_k=8)
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-160m")
    adapter.model = ea.convert_gptneox(adapter.model, settings)

    context = " predict parrot" * 10 + " noise" * 10
    prompt = " predict"

    dense_out = adapter.greedy_sample(
        [context], [prompt], num_generated_tokens=4, use_cache=False
    )
    eviction_out = adapter.greedy_sample(
        [context],
        [prompt],
        num_generated_tokens=4,
        use_cache=False,
        combine_context_and_prompt=False,
        generation_context=partial(ea.generation_context, keep_history=True),
    )

    # Check the actual unmasked k-value, allowing some exceptions for when the kth value
    # is not unique, but this should not be very common.
    actual_k = []
    for layer in adapter.model.gpt_neox.layers:
        mask = layer.attention.history[0].mask
        actual_k.extend(mask.sum(-1).flatten().tolist())
    assert np.quantile(actual_k, 0.95) == settings.k

    # This is a slightly risky (possibly unstable) test
    #  - dense will have conditioned to follow "predict" with "parrot"
    #  - eviction will not have seen lots of "predict parrot", falls back to "noise"
    assert adapter.tok_decode(dense_out[0]).startswith(" parrot")
    assert adapter.tok_decode(eviction_out[0]).startswith(" noise")
