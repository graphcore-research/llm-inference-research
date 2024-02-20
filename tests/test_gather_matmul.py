# Copyright (c) 2024 Graphcore Ltd. All rights reserved.

from typing import Tuple

import pytest
import torch

import gather_matmul as G
from sparq_benchmark import gather


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("group_shape", [(2,), (3, 2)])
def test_gather_inner_bmv(group_shape: Tuple[int, ...]) -> None:
    torch.manual_seed(100)
    a = torch.randn(*group_shape, 1, 8, device="cuda", dtype=torch.float16)
    b = torch.randn(*group_shape, 8, 10, device="cuda", dtype=torch.float16)
    i = torch.randint(0, 8, size=group_shape + (4,), device="cuda")

    expected = gather(a, -1, i[..., None, :]) @ gather(b, -2, i[..., :, None])
    actual = G.gather_inner_bmv(a, b, i, chunk=4)
    assert actual.shape == group_shape + (1, 10)
    # Note: tolerance is set empirically, may be too tight
    torch.testing.assert_close(actual, expected, atol=5e-3 * actual.std(), rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("group_shape", [(2,), (3, 2)])
def test_gather_inner_matrix_only_bmv(group_shape: Tuple[int, ...]) -> None:
    torch.manual_seed(200)
    a = torch.randn(*group_shape, 1, 4, device="cuda", dtype=torch.float16)
    b = torch.randn(*group_shape, 8, 10, device="cuda", dtype=torch.float16)
    i = torch.randint(0, 8, size=group_shape + (4,), device="cuda")

    expected = a @ gather(b, -2, i[..., :, None])
    actual = G.gather_inner_matrix_only_bmv(a, b, i, chunk=4)
    assert actual.shape == group_shape + (1, 10)
    # Note: tolerance is set empirically, may be too tight
    torch.testing.assert_close(actual, expected, atol=5e-3 * actual.std(), rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
@pytest.mark.parametrize("group_shape", [(2,), (3, 2)])
def test_gather_outer_bmv(group_shape: Tuple[int, ...]) -> None:
    torch.manual_seed(300)
    a = torch.randn(*group_shape, 1, 8, device="cuda", dtype=torch.float16)
    b = torch.randn(*group_shape, 8, 20, device="cuda", dtype=torch.float16)
    i = torch.randint(0, 20, size=group_shape + (9,), device="cuda")

    expected = a @ gather(b, -1, i[..., None, :])
    actual = G.gather_outer_bmv(a, b, i, chunk=4)
    assert actual.shape == group_shape + (1, 9)
    # Note: tolerance is set empirically, may be too tight
    torch.testing.assert_close(actual, expected, atol=5e-3 * actual.std(), rtol=0)
