from typing import Protocol

from torch import Tensor


class TopK(Protocol):
    @staticmethod
    def __call__(xs: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]: ...
