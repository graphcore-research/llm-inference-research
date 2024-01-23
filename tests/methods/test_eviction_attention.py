# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pytest
import torch
from torch import tensor
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig

from llminference import eval_adapter
from llminference.methods import eviction_attention as ea


@pytest.mark.parametrize("strategy", ["sum_weight", "lru"])
def test_eviction_strategy(strategy: str) -> None:
    eviction = ea.EvictionMask(
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


def test_gptneox_with_eviction() -> None:
    module = ea.GPTNeoXAttentionWithEviction(
        GPTNeoXConfig(hidden_size=128, num_attention_heads=4),
        ea.Settings(k=8, local_k=2, strategy="sum_weight"),
    )
    # Prefill
    _, layer_past, _ = module(
        torch.randn(13, 19, 128),
        attention_mask=torch.zeros(13, 1, 19, 19),
        position_ids=torch.arange(19)[None],
        use_cache=True,
        output_attentions=True,
    )
    # Generation (with eviction)
    module.eviction.enable(True)
    output, _, weights = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        layer_past=layer_past,
        use_cache=True,
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 8 + 1).all(), "sparse attention"


def test_llama_with_eviction() -> None:
    module = ea.LlamaAttentionWithEviction(
        LlamaConfig(hidden_size=128, num_attention_heads=4, num_key_value_heads=4),
        ea.Settings(k=8, local_k=2, strategy="lru"),
    )
    # Prefill
    _, _, past_key_value = module(
        torch.randn(13, 19, 128),
        attention_mask=torch.zeros(13, 1, 19, 19),
        position_ids=torch.arange(19)[None],
        use_cache=True,
        output_attentions=True,
    )
    # Generation (with eviction)
    module.eviction.enable(True)
    output, weights, _ = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        past_key_value=past_key_value,
        use_cache=True,
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 8 + 1).all(), "sparse attention"


@pytest.mark.parametrize("strategy", ["sum_weight", "lru"])
def test_mistral_with_eviction(strategy: str) -> None:
    module = ea.MistralAttentionWithEviction(
        MistralConfig(hidden_size=128, num_attention_heads=16, num_key_value_heads=4),
        ea.Settings(k=8, local_k=2, strategy=strategy),
    )
    # Prefill
    _, _, past_key_value = module(
        torch.randn(13, 19, 128),
        attention_mask=torch.zeros(13, 1, 19, 19),
        position_ids=torch.arange(19)[None],
        use_cache=True,
        output_attentions=True,
    )
    # Generation (with eviction)
    module.eviction.enable(True)
    output, weights, _ = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        past_key_value=past_key_value,
        use_cache=True,
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 8 + 1).all(), "sparse attention"

    # Check that non-zero weights within KV groups match
    weights_grouped = (weights != 0).unflatten(dim=1, sizes=(4, 4))
    assert weights_grouped.unique(dim=2).size(dim=2) == 1
    # But across groups are different
    assert weights_grouped.unique(dim=1).size(dim=1) == 4


def test_convert_gptneox() -> None:
    settings = ea.Settings(k=16, local_k=8, strategy="lru")
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-160m")
    eviction_adapter = eval_adapter.Adapter(
        ea.convert(adapter.model, settings), adapter.tokenizer, adapter.batch_size
    )
    for layer in eviction_adapter.model.gpt_neox.layers:
        layer.attention.eviction.debug_masks = []

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
        assert layer.attention.eviction.debug_masks
        for mask in layer.attention.eviction.debug_masks:
            assert (mask.sum(-1) == settings.k + 1).all()

    # This is a slightly risky (possibly unstable) test
    #  - dense will have conditioned to follow "predict" with "parrot"
    #  - eviction will not have seen lots of "predict parrot"
    assert adapter.tok_decode(dense_out[0]).startswith(" parrot")
    assert not adapter.tok_decode(eviction_out[0]).startswith(" parrot")
