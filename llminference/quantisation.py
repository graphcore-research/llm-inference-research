"""Utilities for "fake quantisation"."""

import re
from dataclasses import dataclass
from typing import Optional, Union, cast

import torch
from torch import Tensor, nn


@dataclass
class Format:
    exponent_bits: int
    mantissa_bits: int

    @property
    def max_absolute_value(self) -> float:
        raise NotImplementedError

    def quantise(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    def __str__(self) -> str:
        return f"E{self.exponent_bits}M{self.mantissa_bits}"

    @staticmethod
    def parse(value: str) -> Union["FPFormat", "IntFormat"]:
        m = re.match(r"^E(\d+)M(\d+)$", value)
        if not m:
            raise ValueError(f"Couldn't parse {value!r}")
        exponent_bits = int(m.group(1))
        mantissa_bits = int(m.group(2))
        if exponent_bits == 0:
            return IntFormat(0, mantissa_bits)
        if exponent_bits >= 2:
            return FPFormat(exponent_bits, mantissa_bits)
        raise ValueError(f"No format {value!r} available (note: E1M6 == E0M7)")


@dataclass
class FPFormat(Format):
    """Note that this format does not reserve an exponent code for specials.

    For exponent e : [0, 2^E - 1], mantissa m : [0, 2^M - 1], the represented value is:

        2^(e - 2^(E-1))) * (1 + m / 2^M)   if e != 0  (normal)
        2^(1 - 2^(E-1))) * (m / 2^M)       if e == 0  (subnormal)
    """

    def __post_init__(self) -> None:
        assert self.exponent_bits >= 2, "FPFormat requires at least 2 exponent bits"

    @property
    def max_absolute_value(self) -> float:
        max_exponent = 2 ** (self.exponent_bits - 1) - 1
        return cast(float, 2**max_exponent * (2 - 2**-self.mantissa_bits))

    @property
    def min_absolute_normal(self) -> float:
        min_exponent = 1 - 2 ** (self.exponent_bits - 1)
        return cast(float, 2**min_exponent)

    @property
    def min_absolute_subnormal(self) -> float:
        return self.min_absolute_normal * 2.0**-self.mantissa_bits

    def quantise(self, x: Tensor) -> Tensor:
        absmax = self.max_absolute_value
        downscale = 2.0 ** (127 - 2 ** (self.exponent_bits - 1))
        mask = 2 ** (23 - self.mantissa_bits) - 1

        q = x.to(torch.float32)
        q = torch.clip(x, -absmax, absmax)
        q /= downscale
        q = ((q.view(torch.int32) + (mask >> 1)) & ~mask).view(torch.float32)
        q *= downscale
        return q.to(x.dtype)


@dataclass
class IntFormat(Format):
    def __post_init__(self) -> None:
        assert self.exponent_bits == 0, "IntFormat has no exponent bits"

    @property
    def max_absolute_value(self) -> float:
        return 2.0**self.mantissa_bits - 1

    def quantise(self, x: Tensor) -> Tensor:
        return torch.clip(
            torch.round(x), -self.max_absolute_value, self.max_absolute_value
        )


def quantisation_scale(
    value: Tensor, format: Format, mode: str, global_bias: Optional[int] = None
) -> Tensor:
    """Get a scaling tensor to apply to quantise a given tensor.

    value -- shape (output_channels, input_channels)

    mode -- "global|tensor|input|output|inout"

    global_bias -- only used when mode="global"
    """
    if mode == "global":
        assert global_bias
        return torch.tensor(2**global_bias)
    absvalue = value.abs()
    if mode == "tensor":
        return absvalue.max() / format.max_absolute_value
    if mode == "input":
        return cast(
            Tensor, absvalue.max(dim=0, keepdim=True).values / format.max_absolute_value
        )
    if mode == "output":
        return cast(
            Tensor, absvalue.max(dim=1, keepdim=True).values / format.max_absolute_value
        )
    if mode == "inout":
        return (
            torch.sqrt(
                absvalue.max(dim=0, keepdim=True).values
                * absvalue.max(dim=1, keepdim=True).values
            )
            / format.max_absolute_value
        )
    raise ValueError(f"Unexpected quantisation_scale mode={mode}")


def quantisation_bits(
    value: Tensor, format: Format, mode: str, scaling_format_bits: int = 16
) -> int:
    """Count the number of bits in a quantised representation of a tensor."""
    count = format.bits * value.nelement()
    if mode == "tensor":
        count += scaling_format_bits
    if mode in ["input", "inout"]:
        count += scaling_format_bits * value.shape[1]
    if mode in ["output", "inout"]:
        count += scaling_format_bits * value.shape[0]
    return count


def quantise_model(
    model: nn.Module,
    format: Format,
    mode: str,
    global_bias: Optional[int] = None,
    unquantised_format_bits: int = 16,
) -> float:
    """In-place quantise a model, returning the size of the quantised model (bytes)."""
    bitcount = 0
    for name, p in model.named_parameters():
        if p.ndim == 1:
            # Don't quantise biases/element-scales
            bitcount += p.nelement() * unquantised_format_bits
        elif p.ndim == 2:
            scale = quantisation_scale(
                p.data, format, mode=mode, global_bias=global_bias
            )
            p.data = format.quantise(p.data / scale) * scale
            bitcount += quantisation_bits(
                p.data, format, mode=mode, scaling_format_bits=unquantised_format_bits
            )
        else:
            raise ValueError(
                f"Don't know how to quantise a rank-{p.ndim} tensor ({name})"
            )
    return bitcount / 8
