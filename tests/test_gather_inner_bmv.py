import torch

from sparq_benchmark import gather
from gather_inner_bmv import gather_inner_bmv


def test_gather_inner_bmv() -> None:
    torch.manual_seed(42)
    a = torch.randn(2, 1, 8, device="cuda", dtype=torch.float16)
    b = torch.randn(2, 8, 10, device="cuda", dtype=torch.float16)
    i = torch.tensor([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=torch.long, device="cuda")

    expected = gather(a, 2, i[:, None, :]) @ gather(b, 1, i[:, :, None])
    actual = gather_inner_bmv(a, b, i, chunk=4)
    torch.testing.assert_close(actual, expected, atol=0, rtol=1e-2)
