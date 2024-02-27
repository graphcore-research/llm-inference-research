# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import torch.nn.functional as F
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig

from llminference import eval_adapter
from llminference.methods import sparse_attention as sa

# Note: these tests use `finfo(float16).min` for masking, even though they are
# run with float32 inputs. This matches gpt_neox's masking behaviour (the softmax
# is always run in float32, but the mask uses finfo(float16).min when the model
# is in half precision.)


def test_topk_mask() -> None:
    nan = torch.nan
    x = torch.tensor(
        [
            [0, 10, 5, 6, nan],
            [2, 2, 2, 2, 2],
            [nan, nan, nan, nan, nan],
        ]
    )
    mask0 = sa.topk_mask(x, k=2)
    assert torch.equal(torch.sum(mask0, -1), torch.tensor([2, 2, 2]))
    # Only the first row is determinate
    assert torch.equal(mask0[0], torch.tensor([0, 1, 0, 1, 0]))

    mask1 = sa.topk_mask(x, k=1, dim=0)
    assert torch.equal(
        mask1,
        torch.tensor(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
            ]
        ),
    )


def test_causal_index() -> None:
    m = torch.finfo(torch.float16).min
    mask = torch.tensor(
        [
            [m, 0, m, m],
            [m, 0, 0, m],
            [m, 0, 0, 0],
        ]
    )
    expected = torch.tensor(
        [
            [-1, 0, -1, -1],
            [-1, 1, 0, -1],
            [-1, 2, 1, 0],
        ]
    )
    assert torch.equal(expected, sa.causal_index(mask))
    assert torch.equal(expected[None, None], sa.causal_index(mask[None, None]))


def test_sparse_softmax() -> None:
    k = 2
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )[None, :, None, None, :]
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, False, True, True],
            [True, False, False, True],
        ]
    )[None, :, None, None, :]
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax(x, k, apply_after_softmax=True, reallocate_to_mean=False)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_gqa() -> None:
    k = 2
    x = torch.tensor(
        [[10, 10, 0, 0], [0, 1, 1, 98], [99, 1, 0, 0], [0, 10, 10, 0]],
        dtype=torch.float32,
    )[None, :, None, :].unflatten(
        dim=1, sizes=(2, 2)
    )  # type:ignore[no-untyped-call]
    mask = ~torch.tensor(
        [
            [False, True, False, True],
            [False, True, False, True],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )[None, :, None, :].unflatten(
        dim=1, sizes=(2, 2)
    )  # type:ignore[no-untyped-call]
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax(x, k, apply_after_softmax=True, reallocate_to_mean=False)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_large_k() -> None:
    k = 8
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )[None, :, None, None, :]
    expected = F.softmax(x, dim=-1)
    out = sa.sparse_softmax(x, k, apply_after_softmax=True, reallocate_to_mean=False)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_reallocation() -> None:
    k = 2
    m = torch.finfo(torch.float16).min
    x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4, m])
    s = F.softmax(x, dim=-1)
    avg = (s[0] + s[1] + s[3]) / 3
    expected = torch.tensor([avg, avg, s[2], avg, s[4], 0])
    out = sa.sparse_softmax(
        x[None, None, None, None], k, apply_after_softmax=True, reallocate_to_mean=True
    ).squeeze()
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_before_softmax() -> None:
    k = 2
    m = torch.finfo(torch.float16).min
    x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4, m])
    expected = F.softmax(torch.tensor([m, m, 0.5, m, 0.4, m]), dim=-1)
    out = sa.sparse_softmax(
        x[None, None, None, None],
        k,
        apply_after_softmax=False,
        reallocate_to_mean=False,
    ).squeeze()
    torch.testing.assert_close(out, expected)


def test_local_softmax() -> None:
    m = torch.finfo(torch.float16).min
    x = torch.tensor(
        [
            [m, 8, m, m, m],
            [m, 2, 5, m, m],
            [m, 14, 12, 7, m],
            [m, 4, 15, 9, 3],
        ],
        dtype=torch.float32,
    )[None, :, None, None, :]
    # k=2
    x_masked = torch.tensor(
        [
            [m, 8, m, m, m],
            [m, 2, 5, m, m],
            [m, m, 12, 7, m],
            [m, m, m, 9, 3],
        ],
        dtype=torch.float32,
    )[None, :, None, None, :]
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, initial_k=0, apply_after_softmax=False),
        F.softmax(x_masked, dim=-1),
    )
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, initial_k=0, apply_after_softmax=True),
        F.softmax(x, dim=-1) * (x_masked != m),
    )
    # k=2, initial_k=1
    x_masked = torch.tensor(
        [
            [m, 8, m, m, m],
            [m, 2, 5, m, m],
            [m, 14, m, 7, m],
            [m, 4, m, m, 3],
        ],
        dtype=torch.float32,
    )[None, :, None, None, :]
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, initial_k=1, apply_after_softmax=False),
        F.softmax(x_masked, dim=-1),
    )
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, initial_k=1, apply_after_softmax=True),
        F.softmax(x, dim=-1) * (x_masked != m),
    )


def test_gptneox_with_sa() -> None:
    # Check sparse softmax
    module = sa.GPTNeoXSparseAttention(
        GPTNeoXConfig(hidden_size=128, num_attention_heads=4),
        sa.SparseSettings(k=8, apply_after_softmax=True, reallocate_to_mean=False),
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
    assert ((weights != 0).sum(-1) == 8).all(), "sparse attention"

    # Check local softmax
    module = sa.GPTNeoXSparseAttention(
        GPTNeoXConfig(hidden_size=128, num_attention_heads=4),
        sa.LocalSettings(k=8, initial_k=2, apply_after_softmax=True),
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
    mask = torch.zeros(13, 4, 1, 20, dtype=torch.bool)
    mask[..., :2] = True
    mask[..., -6:] = True
    assert ((weights != 0) == mask).all()


def test_llama_with_sa() -> None:
    module = sa.LlamaSparseAttention(
        LlamaConfig(hidden_size=128, num_attention_heads=4, num_key_value_heads=4),
        sa.SparseSettings(k=8, apply_after_softmax=True, reallocate_to_mean=False),
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
    assert ((weights != 0).sum(-1) == 8).all(), "sparse attention"


def test_mistral_with_sa() -> None:
    module = sa.MistralSparseAttention(
        MistralConfig(hidden_size=128, num_attention_heads=16, num_key_value_heads=4),
        sa.SparseSettings(k=8, apply_after_softmax=True, reallocate_to_mean=False),
    )
    output, weights, _ = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        past_key_value=(torch.randn(13, 4, 19, 8), torch.randn(13, 4, 19, 8)),
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 8).all(), "sparse attention"

    # Check that non-zero weights within KV groups match
    weights_grouped = (weights != 0).unflatten(dim=1, sizes=(4, 4))
    assert weights_grouped.unique(dim=2).size(dim=2) == 1
    # But across groups are different
    assert weights_grouped.unique(dim=1).size(dim=1) == 4


def test_convert_gptneox() -> None:
    # NOTE: Last test breaks for pythia-160m, not for 70m or 410m
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")

    sa_model = sa.convert(
        adapter.model,
        sa.SparseSettings(
            k=8,
            apply_after_softmax=True,
            reallocate_to_mean=False,
        ),
    )

    for layer in sa_model.gpt_neox.layers:
        layer.attention.sparse_attention.debug_masks = []

    # Run a simple test case
    converted = eval_adapter.Adapter(sa_model, adapter.tokenizer, adapter.batch_size)
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
    for layer in sa_model.gpt_neox.layers:
        assert layer.attention.sparse_attention.debug_masks
        for mask in layer.attention.sparse_attention.debug_masks:
            assert (mask.sum(-1) == 8).all()
