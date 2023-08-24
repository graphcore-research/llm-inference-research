import torch
import torch.nn.functional as F

from .. import sparse_attention as sa


def test_sparse_softmax_fixed_k() -> None:
    k = 2
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, False, True, True],
            [True, False, False, True],
        ]
    )
    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.sparse_softmax_fixed_k(x, dim=-1, k=k)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_large_k() -> None:
    k = 8
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    expected = F.softmax(x, dim=-1)
    out = sa.sparse_softmax_fixed_k(x, dim=-1, k=k)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_add_avg() -> None:
    k = 2
    x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4])
    s = F.softmax(x, dim=-1)
    avg = (s[0] + s[1] + s[3]) / 3
    expected = torch.tensor([avg, avg, s[2], avg, s[4]])
    out = sa.sparse_softmax_fixed_k(x, k, add_avg=True)
    torch.testing.assert_close(out, expected)


def test_sparse_softmax_fixed_k_out_weights() -> None:
    k = 2
    x = torch.tensor([0.3, 0.1, 0.5, 0.2, 0.4])
    out_weights = torch.tensor([1.0, 100.0, 1.0, 1.0, 1.0])
    s = F.softmax(x, dim=-1)
    expected = torch.tensor([0, s[1], s[2], 0, 0])
    out = sa.sparse_softmax_fixed_k(x, k, out_weights=out_weights)
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


def test_local_softmax_pre() -> None:
    context_len = 2
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    x_masked = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [-1e9, 12, 7, 10], [-1e9, -1e9, 9, 3]],
        dtype=torch.float32,
    )
    expected = F.softmax(x_masked, dim=-1)
    out = sa.local_softmax(x, context_len=context_len, apply_after_softmax=False)
    torch.testing.assert_close(out, expected)


def test_local_softmax_post() -> None:
    context_len = 2
    x = torch.tensor(
        [[8, 0, 11, 13], [2, 5, 1, 6], [14, 12, 7, 10], [4, 15, 9, 3]],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, False],
            [True, False, False, False],
            [True, True, False, False],
        ]
    )

    expected = torch.masked_fill(F.softmax(x, dim=-1), mask=mask, value=0.0)
    out = sa.local_softmax(x, context_len=context_len, apply_after_softmax=True)
    torch.testing.assert_close(out, expected)
