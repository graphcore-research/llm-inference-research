# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import cast

import pytest
import torch
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig

from llminference import eval_adapter
from llminference.methods import ann_attention as ann


def test_gather() -> None:
    table = torch.arange(20).reshape(2, 5, 2)
    indices = torch.tensor([[0, 0, 4, 4, 2], [1, 1, 1, 1, 1]])[:, :, None]
    expected = torch.tensor(
        [
            [[0, 1], [0, 1], [8, 9], [8, 9], [4, 5]],
            [[12, 13], [12, 13], [12, 13], [12, 13], [12, 13]],
        ]
    )
    torch.testing.assert_close(ann.gather(table, 1, indices), expected)
    torch.testing.assert_close(ann.gather(table, -2, indices), expected)


def test_ann_module() -> None:
    batch, n_heads, n_key, head_size = (3, 5, 7, 16)
    query = torch.randn((batch, n_heads, 1, head_size))
    key = torch.randn((batch, n_heads, n_key, head_size))
    value = torch.randn((batch, n_heads, n_key, head_size))

    # Mask out some leading token positions (different across the batch)
    mask = (torch.tensor([2, 0, 5])[:, None] <= torch.arange(n_key))[
        :, None, None, :
    ].broadcast_to(batch, n_heads, 1, n_key)
    key.masked_fill_(~mask[:, :, 0, :, None], 1e9)
    value.masked_fill_(~mask[:, :, 0, :, None], float("inf"))

    for score, reallocate_to_mean_value in [
        (ann.LowRank.Settings(8), False),
        (ann.SparseQ.Settings(6), True),
    ]:
        module = ann.AnnAttention(
            ann.Settings(
                k=3,
                local_k=1,
                reallocate_to_mean_value=reallocate_to_mean_value,
                score=cast(ann.ScoreSettings, score),
            ),
            n_heads,
            head_size,
        )
        output, weights = module(query, key, value, mask.float().log())
        assert not torch.isinf(output).any()
        assert output.shape == (batch, n_heads, 1, head_size)
        assert weights.shape == (batch, n_heads, 1, n_key)
        # At most 4 nonzeros per (batch, head), but may be fewer (due to the mask)
        assert ((weights != 0).sum(-1) == mask.sum(-1).minimum(torch.tensor(4))).all()
        assert (weights[..., -1] != 0).all(), "local_k"
        if reallocate_to_mean_value:
            assert (weights.sum(-1) <= 1.00001).all()
        else:
            torch.testing.assert_close(weights.sum(-1), torch.ones((batch, n_heads, 1)))


# Test GQA
def test_sparse_q_gqa() -> None:
    module = ann.SparseQ(ann.SparseQ.Settings(rank=1))
    # Should select dim 0 for first KV group, and dim 1 for second KV group
    query = torch.tensor([[1, 10], [-20, 2], [30, 3], [4, 40]], dtype=torch.float)[
        None, :, None, :
    ].reshape(1, 2, 2, 1, 2)
    key = torch.tensor([[1, 10], [20, 2]], dtype=torch.float)[None, :, None, :].reshape(
        1, 2, 1, 1, 2
    )
    scale = torch.sqrt(torch.tensor([1 / 11, 20 / 22, 3 / 33, 40 / 44]) * 2)[
        None, :, None, None
    ]
    expected = (torch.tensor([1, -20, 6, 80])[None, :, None, None] / scale).reshape(
        1, 2, 2, 1, 1
    )
    out = module(query, key)
    torch.testing.assert_close(out, expected)


def test_low_rank_gqa() -> None:
    module = ann.LowRank(ann.LowRank.Settings(rank=8), n_kv_heads=4, head_size=16)
    assert module.weight.shape == (4, 1, 16, 8)

    # Same query and keys across heads
    query = torch.randn(1, 1, 1, 1, 16).expand(1, 4, 2, 1, 16)
    key = torch.randn(1, 1, 1, 5, 16).expand(1, 4, 1, 5, 16)

    out = module(query, key)
    assert out.shape == (1, 4, 2, 1, 5)

    # Same within KV groups (same weight projection)
    out.unique(dim=2).size(dim=2) == 1
    # Different across KV groups (different weight projection)
    out.unique(dim=1).size(dim=1) == 4


def test_gptneox_with_ann() -> None:
    module = ann.GPTNeoXAttentionWithANN(
        GPTNeoXConfig(hidden_size=128, num_attention_heads=4),
        ann.Settings(
            k=8, local_k=2, reallocate_to_mean_value=True, score="sparse_q", rank=12
        ),
    )
    output, _, weights = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        layer_past=(torch.randn(13, 4, 19, 32), torch.randn(13, 4, 19, 32)),
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 9).all(), "sparse attention"


def test_llama_with_ann() -> None:
    module = ann.LlamaAttentionWithANN(
        LlamaConfig(hidden_size=128, num_attention_heads=4, num_key_value_heads=4),
        ann.Settings(
            k=8, local_k=2, reallocate_to_mean_value=True, score="sparse_q", rank=12
        ),
    )
    output, weights, _ = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        past_key_value=(torch.randn(13, 4, 19, 32), torch.randn(13, 4, 19, 32)),
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 9).all(), "sparse attention"


@pytest.mark.parametrize("score", ["low_rank", "sparse_q"])
def test_mistral_with_ann(score: str) -> None:
    module = ann.MistralAttentionWithANN(
        MistralConfig(hidden_size=128, num_attention_heads=16, num_key_value_heads=4),
        ann.Settings(
            k=8, local_k=2, reallocate_to_mean_value=True, score=score, rank=4
        ),
    )

    output, weights, _ = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        # KV values should be same within groups
        past_key_value=(torch.randn(13, 4, 19, 8), torch.randn(13, 4, 19, 8)),
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 9).all(), "sparse attention"

    # Check that non-zero weights within KV groups match
    weights_grouped = (weights != 0).unflatten(dim=1, sizes=(4, 4))
    assert weights_grouped.unique(dim=2).size(dim=2) == 1
    # But across groups are different
    assert weights_grouped.unique(dim=1).size(dim=1) == 4


def test_convert_gptneox() -> None:
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-160m")

    # Note: the head size of pythia-160m is 64, so this should be exact
    # (due to orthonormal init)
    ann_model = ann.convert(
        adapter.model,
        ann.Settings(
            k=8,
            local_k=2,
            reallocate_to_mean_value=False,
            score=ann.LowRank.Settings(64),
        ),
    )
    for layer in ann_model.gpt_neox.layers:
        layer.attention.ann.debug_indices = []

    # Run a simple test case
    converted = eval_adapter.Adapter(ann_model, adapter.tokenizer, adapter.batch_size)
    context = "The answer is 42. The answer is 42. The answer"
    assert (
        adapter.tokenizer.decode(adapter.greedy_sample([context], [""], 3)[0])
        == " is 42."
    )
    assert (
        converted.tokenizer.decode(converted.greedy_sample([context], [""], 3)[0])
        == " is 42."
    )

    # Check that the masks all match the expected k
    for layer in ann_model.gpt_neox.layers:
        assert layer.attention.ann.debug_indices
        for indices in layer.attention.ann.debug_indices:
            assert indices.shape[-1] == 9
