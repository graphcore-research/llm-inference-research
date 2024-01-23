# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import unittest.mock as um
from functools import partialmethod

import torch
import torch.nn.functional as F

from llminference.eval_adapter import Adapter
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


def test_sparse_softmax_fixed_k() -> None:
    k = 2
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )[None, :, None, :]
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, False, True, True],
            [True, False, False, True],
        ]
    )[None, :, None, :]
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax_fixed_k(x, dim=-1, k=k, generation_only=False)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_gqa() -> None:
    k = 2
    x = torch.tensor(
        [[10, 10, 0, 0], [0, 1, 1, 98], [99, 1, 0, 0], [0, 10, 10, 0]],
        dtype=torch.float32,
    )[None, :, None, :]
    mask = ~torch.tensor(
        [
            [False, True, False, True],
            [False, True, False, True],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )[None, :, None, :]
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax_fixed_k(x, k=k, apply_after_softmax=True, kv_group_size=2)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_large_k() -> None:
    k = 8
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    expected = F.softmax(x, dim=-1)
    out = sa.sparse_softmax_fixed_k(x, dim=-1, k=k, generation_only=False)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_add_avg() -> None:
    k = 2
    m = torch.finfo(torch.float16).min
    x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4, m])
    s = F.softmax(x, dim=-1)
    avg = (s[0] + s[1] + s[3]) / 3
    expected = torch.tensor([avg, avg, s[2], avg, s[4], 0])
    out = sa.sparse_softmax_fixed_k(x[None, None, None], k, add_avg=True).squeeze()
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_out_weights() -> None:
    k = 2
    m = torch.finfo(torch.float16).min
    x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4, m])
    out_weights = torch.tensor([1.0, 100.0, 1.0, 1.0, 1.0, 1.0])
    s = F.softmax(x, dim=-1)
    expected = torch.tensor([0, s[1], s[2], 0, 0, 0])
    out = sa.sparse_softmax_fixed_k(
        x[None, None, None], k, out_weights=out_weights
    ).squeeze()
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_apply_before_softmax() -> None:
    k = 2
    m = torch.finfo(torch.float16).min
    x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4, m])
    expected = F.softmax(torch.tensor([m, m, 0.5, m, 0.4, m]), dim=-1)
    out = sa.sparse_softmax_fixed_k(x[None], k, apply_after_softmax=False)[0]
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_p_half() -> None:
    p = 0.5
    k_min = 1
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, False],
            [False, True, True, True],
            [True, False, False, True],
        ]
    )
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax_fixed_p(x, p=p, k_min=k_min, dim=-1)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_p_full() -> None:
    p = 1
    k_min = 1
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, False, True, False],
            [False, False, True, False],
            [False, False, False, False],
        ]
    )
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax_fixed_p(x, p=p, k_min=k_min, dim=-1)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_p_small() -> None:
    p = 0.1
    k_min = 3
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [False, True, False, False],
            [False, False, True, False],
            [False, False, True, False],
            [False, False, False, True],
        ]
    )
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax_fixed_p(x, p=p, k_min=k_min, dim=-1)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_p_batched() -> None:
    p = 0.5
    k_min = 1
    x = torch.tensor(
        [
            [[[12, 10], [9, 6]], [[11, 8], [13, 5]]],
            [[[2, 14], [15, 0]], [[4, 3], [7, 1]]],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [[[False, True], [False, True]], [[False, True], [False, True]]],
            [[[True, False], [False, True]], [[False, True], [False, True]]],
        ]
    )

    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax_fixed_p(x, p=p, k_min=k_min, dim=-1)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_p_k_min_large() -> None:
    p = 0.5
    k_min = 8
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    expected = F.softmax(x, dim=-1)
    out = sa.sparse_softmax_fixed_p(x, p=p, k_min=k_min, dim=-1)
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
    )
    # k=2
    x_masked = torch.tensor(
        [
            [m, 8, m, m, m],
            [m, 2, 5, m, m],
            [m, m, 12, 7, m],
            [m, m, m, 9, 3],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, apply_after_softmax=False),
        F.softmax(x_masked, dim=-1),
    )
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, apply_after_softmax=True),
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
    )
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, initial_k=1, apply_after_softmax=False),
        F.softmax(x_masked, dim=-1),
    )
    torch.testing.assert_close(
        sa.local_softmax(x, k=2, initial_k=1, apply_after_softmax=True),
        F.softmax(x, dim=-1) * (x_masked != m),
    )


def test_sparse_attn_vary_k_per_layer() -> None:
    adapter = Adapter.from_pretrained("EleutherAI/pythia-70m")
    inp = adapter.tokenizer("She", return_tensors="pt")
    model = adapter.model.gpt_neox
    k_per_layer = [i + 1 for i in range(len(model.layers))]

    with sa.number_attention_layers(model) as m, um.patch(
        "transformers.models.gpt_neox.modeling_gpt_neox.GPTNeoXAttention._attn",
        partialmethod(sa.sparse_attn, k_per_layer=k_per_layer),
    ), um.patch(
        "llminference.methods.sparse_attention.sparse_softmax_fixed_k",
        wraps=sa.sparse_softmax_fixed_k,
    ) as mock_softmax:
        m(**inp)
    call_ks = [c.kwargs["k"] for c in mock_softmax.call_args_list]
    assert k_per_layer == call_ks
