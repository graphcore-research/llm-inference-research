import torch

import sparq_benchmark as B


def test_attn() -> None:
    torch.manual_seed(435352)
    batch_size, n_head, head_dim, sequence_length = 2, 6, 32, 128
    Q = torch.randn((batch_size, n_head, 1, head_dim))
    K = torch.randn((batch_size, n_head, sequence_length, head_dim))
    V = torch.randn((batch_size, n_head, sequence_length, head_dim))
    expected = torch.nn.functional.scaled_dot_product_attention(Q, K, V)

    actual = B.attn(Q, K, V)
    torch.testing.assert_close(actual, expected)

    actual = B.sparq_attn(
        Q, K, K, V, V_mean=V.mean(-2, keepdim=True), r=4, k=sequence_length
    )
    torch.testing.assert_close(actual, expected)
