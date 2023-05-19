from typing import cast

import torch

from .. import quantisation


def test_int_format() -> None:
    fmt = quantisation.Format.parse("E0M7")
    assert fmt.bits == 8
    assert fmt.max_absolute_value == 127
    assert set(fmt.quantise(torch.linspace(-150, 150, steps=1000)).tolist()) == set(
        range(-127, 128)
    )
    assert set(
        quantisation.Format.parse("E0M3")
        .quantise(torch.linspace(-10, 10, steps=100))
        .tolist()
    ) == set(range(-7, 8))


def test_fp_format() -> None:
    fmt = cast(quantisation.FPFormat, quantisation.Format.parse("E2M1"))
    assert fmt.bits == 4
    assert fmt.max_absolute_value == 3
    assert fmt.min_absolute_normal == 0.5
    assert fmt.min_absolute_subnormal == 0.25
    assert set(fmt.quantise(torch.linspace(-4, 4, steps=100)).tolist()) == {
        sx for x in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3] for sx in [x, -x]
    }
    assert set(
        quantisation.Format.parse("E3M0")
        .quantise(torch.linspace(-10, 10, steps=1000))
        .abs()
        .tolist()
    ) == {0, 0.125, 0.25, 0.5, 1, 2, 4, 8}
