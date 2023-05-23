"""Utilities for "fake quantisation"."""

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, cast

import torch
from torch import Tensor, nn

Shape = Tuple[int, ...]


class TensorFormat:
    """Quantisation formats for tensors."""

    def quantise(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError

    def count_bits(self, shape: Shape) -> int:
        raise NotImplementedError


@dataclass
class Format(TensorFormat):
    """Elementwise scalar formats."""

    exponent_bits: int
    mantissa_bits: int

    @property
    def max_absolute_value(self) -> float:
        raise NotImplementedError

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    def count_bits(self, shape: Shape) -> int:
        return self.bits * math.prod(shape)

    def __str__(self) -> str:
        return f"E{self.exponent_bits}M{self.mantissa_bits}"


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
class IEEEFormat(Format):
    def __post_init__(self) -> None:
        assert (self.exponent_bits, self.mantissa_bits) in [(5, 10), (8, 23)]

    @property
    def max_absolute_value(self) -> float:
        max_exponent = 2 ** (self.exponent_bits - 1) - 1
        return cast(float, (2 - 2**-self.mantissa_bits) * 2**max_exponent)

    def quantise(self, x: Tensor) -> Tensor:
        assert x.dtype == torch.float32
        if self.exponent_bits == 8:
            return x
        return (
            torch.clip(x, -self.max_absolute_value, self.max_absolute_value)
            .half()
            .float()
        )


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


@dataclass
class ExpCeilFormat(Format):
    def __post_init__(self) -> None:
        assert self.mantissa_bits == 0, "ExpCeilFormat has no exponent bits"

    @property
    def bits(self) -> int:
        return self.exponent_bits  # override to remove sign bit

    @property
    def exponent_bias(self) -> float:
        return 2.0 ** (self.exponent_bits - 1) - 1

    @property
    def max_absolute_value(self) -> float:
        return cast(float, 2 ** (2**self.exponent_bits - 1 - self.exponent_bias))

    def quantise(self, x: Tensor) -> Tensor:
        assert torch.all(x >= 0), f"{type(self)} does not support negative numbers"
        y: Tensor = 2 ** torch.clip(
            torch.ceil(torch.log2(x)),
            -self.exponent_bias,
            2**self.exponent_bits - 1 - self.exponent_bias,
        )
        return y


FP32 = IEEEFormat(8, 23)
FP16 = IEEEFormat(5, 10)


def parse(value: str) -> Format:
    if value == "FP16":
        return FP16
    if value == "FP32":
        return FP32
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
class LinearScalingFormat(TensorFormat):
    """Specifies a scheme for quantising tensors.

    group_shapes -- list of groupings (size of groups in each dimension)
                    e.g. [(1, 8)]                input-groups of size 8
                         [(2, 2)]                square groups of 2x2 (4 elements)
                         [(1, None)]             per-output-channel scaling
                         [(None, 1), (None, 1)]  sqrt(input*output)-channel scaling
    """

    GroupShape = Tuple[Optional[int], ...]

    element_format: Format
    group_shapes: Sequence[GroupShape]
    scale_format: TensorFormat

    @staticmethod
    def _group_shape_for(tensor_shape: Shape, group_shape: GroupShape) -> Shape:
        assert len(tensor_shape) == len(group_shape), f"{tensor_shape} vs {group_shape}"
        return tuple((t if g is None else g) for t, g in zip(tensor_shape, group_shape))

    def count_bits(self, shape: Shape) -> int:
        """Count the number of bits in the quantised representation."""
        count = self.element_format.count_bits(shape)
        for group_shape in self.group_shapes:
            count += self.scale_format.count_bits(
                tuple(
                    t // g
                    for t, g in zip(shape, self._group_shape_for(shape, group_shape))
                )
            )
        return count

    @staticmethod
    def _group_scale_for(absratio: Tensor, group_shape: Shape) -> Tensor:
        full_grouped_shape = tuple(
            s
            for size, group_size in zip(absratio.shape, group_shape)
            for s in [size // group_size, group_size]
        )
        grouped_absratio = absratio.reshape(full_grouped_shape)
        for dim in range(1, len(full_grouped_shape), 2):
            grouped_absratio = grouped_absratio.max(dim=dim, keepdim=True).values
        return grouped_absratio.broadcast_to(full_grouped_shape).reshape(absratio.shape)

    def scale_for(self, tensor: Tensor) -> Tensor:
        """Get the quantised scaling tensor to apply to quantise a given tensor."""
        absratio = tensor.abs() / self.element_format.max_absolute_value
        scale = torch.ones_like(absratio)
        for group_shape in self.group_shapes:
            scale *= self.scale_format.quantise(
                self._group_scale_for(
                    absratio, self._group_shape_for(absratio.shape, group_shape)
                )
            )
        return torch.pow(scale, 1 / len(self.group_shapes))

    def quantise(self, tensor: Tensor) -> Tensor:
        """Quantise a tensor under the scheme."""
        scale = self.scale_for(tensor)
        return self.element_format.quantise(tensor / scale) * scale


def tensor_scaling_format(
    element_format: Format, scale_format: Format = FP16
) -> LinearScalingFormat:
    """A per-(2D)-tensor scaling format."""
    return LinearScalingFormat(element_format, [(None, None)], scale_format)


def channel_scaling_format(
    element_format: Format, per: str, scale_format: TensorFormat = FP16
) -> LinearScalingFormat:
    """A per-channel scaling format.

    per -- "input|output|inout"
    """
    groups = cast(
        Sequence[LinearScalingFormat.GroupShape],
        dict(input=[(None, 1)], output=[(1, None)], inout=[(None, 1), (1, None)])[per],
    )
    return LinearScalingFormat(element_format, groups, scale_format)


def group_scaling_format(
    element_format: Format,
    grouping: str,
    group_size: int,
    scale_format: TensorFormat = FP16,
) -> LinearScalingFormat:
    """A simplified grouped scaling format: only 1D groups & no "inout" product.

    groups -- "input|output"
    """
    return LinearScalingFormat(
        element_format,
        dict(input=[(1, group_size)], output=[(group_size, 1)])[grouping],
        scale_format,
    )


def quantise_model(
    model: nn.Module,
    formats: Sequence[Tuple[str, TensorFormat]] = [],
    vector_format: TensorFormat = FP16,
) -> float:
    """In-place quantise a model, returning the size of the quantised model (bytes).

    formats -- [(str -- PATTERN, TensorFormat)] -- for each parameter, use the first
               format for which NAME matches PATTERN
    """
    bitcount = 0
    for name, p in model.named_parameters():
        if p.ndim == 1:
            format_ = vector_format
        else:
            for pattern, pattern_format in formats:
                if re.search(pattern, name):
                    format_ = pattern_format
                    break
            else:
                raise ValueError(f"No format matched tensor {name!r}")
        p.data = format_.quantise(p.data)
        bitcount += format_.count_bits(p.data.shape)
    return bitcount / 8
