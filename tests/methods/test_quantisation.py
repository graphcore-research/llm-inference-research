# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import dataclasses
import json
from typing import cast

import torch
import torch.nn.functional as F

from llminference.methods import quantisation


def test_int_format() -> None:
    fmt = quantisation.parse("E0M7")
    assert fmt.bits == 8
    assert fmt.max_absolute_value == 127
    assert set(fmt.quantise(torch.linspace(-150, 150, steps=1000)).tolist()) == set(
        range(-127, 128)
    )
    assert set(
        quantisation.parse("E0M3").quantise(torch.linspace(-10, 10, steps=100)).tolist()
    ) == set(range(-7, 8))


def test_fp_format() -> None:
    fmt = cast(quantisation.FPFormat, quantisation.parse("E2M1"))
    assert fmt.bits == 4
    assert fmt.max_absolute_value == 3
    assert fmt.min_absolute_normal == 0.5
    assert fmt.min_absolute_subnormal == 0.25
    assert set(fmt.quantise(torch.linspace(-4, 4, steps=100)).tolist()) == {
        sx for x in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3] for sx in [x, -x]
    }
    assert set(
        quantisation.parse("E3M0")
        .quantise(torch.linspace(-10, 10, steps=1000))
        .abs()
        .tolist()
    ) == {0, 0.125, 0.25, 0.5, 1, 2, 4, 8}


def test_ieee_fp16() -> None:
    assert quantisation.FP16.max_absolute_value == 65504
    assert torch.equal(
        quantisation.FP16.quantise(
            torch.tensor([2**-25 * 0.99, 2**-25 * 1.01, -1, 1e5])
        ),
        torch.tensor([0, 2**-24, -1, 65504.0]),
    )


def test_lut_format() -> None:
    fmt = quantisation.LUTFormat((-1, -0.125, 0.125, 1), "fours")
    assert str(fmt) == "LUT2[fours]"
    assert fmt.bits == 2
    assert fmt.max_absolute_value == 1
    assert torch.equal(
        fmt.quantise(torch.tensor([0.8, 0.6, -0.001, -1.2])),
        torch.tensor([1, 1, -0.125, -1]),
    )


def test_scalar_formats() -> None:
    for fmt in [
        quantisation.FP16,
        quantisation.FP32,
        quantisation.NF4,
        quantisation.nf_approx(5),
        quantisation.parse("E0M3"),
        quantisation.parse("E2M2"),
    ]:
        assert 0 < fmt.max_absolute_value
        assert 1 <= fmt.bits <= 32
        assert 600 <= fmt.count_bits((20, 30))

        x = torch.linspace(-20, 20, steps=100).view(2, 1, 50)
        qx = fmt.quantise(x)
        assert qx.shape == x.shape
        assert torch.all(qx <= fmt.max_absolute_value)

        assert json.loads(json.dumps(dataclasses.asdict(fmt)))


def test_linear_scaling_format() -> None:
    torch.manual_seed(23875)
    tensor = torch.randn((10, 20))
    e3m4 = quantisation.parse("E3M4")

    # 1. Per-tensor scaling
    per_tensor = quantisation.tensor_scaling_format(e3m4)
    assert per_tensor.count_bits(tensor.shape) == 8 * 200 + 16
    per_tensor_mse = F.mse_loss(per_tensor.quantise(tensor), tensor)

    # 2. Per-output channel scaling
    per_output_channel = quantisation.channel_scaling_format(e3m4, per="output")
    assert per_output_channel.count_bits(tensor.shape) == 8 * 200 + 16 * 10
    per_output_channel_mse = F.mse_loss(per_output_channel.quantise(tensor), tensor)
    assert per_output_channel_mse < per_tensor_mse  # could be ==, but unlikely

    # 3. Input-group scaling
    input_group = quantisation.group_scaling_format(
        e3m4, grouping="input", group_size=5
    )
    assert input_group.count_bits(tensor.shape) == 8 * 200 + 16 * 10 * (20 // 5)
    input_group_mse = F.mse_loss(input_group.quantise(tensor), tensor)
    assert input_group_mse < per_output_channel_mse  # could be ==, but unlikely

    # 4. Others
    for format_ in [
        quantisation.channel_scaling_format(e3m4, per="output"),
        quantisation.channel_scaling_format(e3m4, per="inout-prod"),
        quantisation.channel_scaling_format(e3m4, per="inout-min"),
        quantisation.channel_scaling_format(
            e3m4, per="input", scale_format=quantisation.tensor_scaling_format(e3m4)
        ),
    ]:
        quantised = format_.quantise(tensor)
        assert quantised.shape == tensor.shape
        assert torch.all(~torch.isnan(quantised))
        assert format_.count_bits(tensor.shape) > 8 * tensor.nelement()
