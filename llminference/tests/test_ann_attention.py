from typing import cast

import torch
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.llama.configuration_llama import LlamaConfig

from .. import ann_attention as ann
from .. import eval_adapter


def test_gather() -> None:
    table = torch.arange(20).reshape(2, 5, 2)
    indices = torch.tensor([[0, 0, 4, 4, 2], [1, 1, 1, 1, 1]])[:, :, None]
    expected = torch.tensor(
        [
            [[0, 1], [0, 1], [8, 9], [8, 9], [4, 5]],
            [[12, 13], [12, 13], [12, 13], [12, 13], [12, 13]],
        ]
    )
    torch.testing.assert_close(ann.gather(table, 1, indices), expected)
    torch.testing.assert_close(ann.gather(table, -2, indices), expected)


def test_ann_module() -> None:
    batch, n_heads, n_key, head_size = (3, 5, 7, 16)
    query = torch.randn((batch, n_heads, 1, head_size))
    key = torch.randn((batch, n_heads, n_key, head_size))
    value = torch.randn((batch, n_heads, n_key, head_size))

    # Mask out some leading token positions (different across the batch)
    mask = (torch.tensor([2, 0, 5])[:, None] <= torch.arange(n_key))[
        :, None, None, :
    ].broadcast_to(batch, n_heads, 1, n_key)
    key.masked_fill_(~mask[:, :, 0, :, None], 1e9)
    value.masked_fill_(~mask[:, :, 0, :, None], float("inf"))

    for score, reallocate_to_mean_value in [
        (ann.LowRank.Settings(8), False),
        (ann.SparseQ.Settings(6), True),
    ]:
        module = ann.ANN(
            ann.Settings(
                k=4,
                local_k=1,
                reallocate_to_mean_value=reallocate_to_mean_value,
                score=cast(ann.ScoreSettings, score),
            ),
            n_heads,
            head_size,
        )
        output, weights = module(query, key, value, mask.float().log())
        assert not torch.isinf(output).any()
        assert output.shape == (batch, n_heads, 1, head_size)
        assert weights.shape == (batch, n_heads, 1, n_key)
        # At most 4 nonzeros per (batch, head), but may be fewer (due to the mask)
        assert ((weights != 0).sum(-1) == mask.sum(-1).minimum(torch.tensor(4))).all()
        assert (weights[..., -1] != 0).all(), "local_k"
        if reallocate_to_mean_value:
            assert (weights.sum(-1) <= 1.00001).all()
        else:
            torch.testing.assert_close(weights.sum(-1), torch.ones((batch, n_heads, 1)))


def test_gptneox_with_ann() -> None:
    module = ann.GPTNeoXAttentionWithANN(
        GPTNeoXConfig(hidden_size=128, num_attention_heads=4),
        ann.Settings(
            k=8, local_k=2, reallocate_to_mean_value=True, score="sparse_q", rank=12
        ),
    )
    output, _, weights = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        layer_past=(torch.randn(13, 4, 19, 32), torch.randn(13, 4, 19, 32)),
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 8).all(), "sparse attention"


def test_llama_with_ann() -> None:
    module = ann.LlamaAttentionWithANN(
        LlamaConfig(hidden_size=128, num_attention_heads=4, num_key_value_heads=4),
        ann.Settings(
            k=8, local_k=2, reallocate_to_mean_value=True, score="sparse_q", rank=12
        ),
    )
    output, weights, _ = module(
        torch.randn(13, 1, 128),
        attention_mask=torch.zeros(13, 1, 1, 20),
        position_ids=torch.tensor([19])[None],
        past_key_value=(torch.randn(13, 4, 19, 32), torch.randn(13, 4, 19, 32)),
        output_attentions=True,
    )
    assert output.shape == (13, 1, 128)
    assert ((-1e3 <= output) & (output <= 1e3)).all(), "'reasonable' outputs"
    assert ((weights != 0).sum(-1) == 8).all(), "sparse attention"


def test_convert_gptneox() -> None:
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-160m")

    # Note: the head size of pythia-160m is 64, so this should be exact
    # (due to orthonormal init)
    ann_model = ann.convert(
        adapter.model,
        ann.Settings(
            k=8,
            local_k=2,
            reallocate_to_mean_value=False,
            score=ann.LowRank.Settings(64),
        ),
    )
    for layer in ann_model.gpt_neox.layers:
        layer.attention.ann.log_indices = []

    # Run a simple test case
    converted = eval_adapter.Adapter(ann_model, adapter.tokenizer, adapter.batch_size)
    context = "The answer is 42. The answer is 42. The answer"
    assert (
        adapter.tokenizer.decode(adapter.greedy_sample([context], [""], 3)[0])
        == " is 42."
    )
    assert (
        converted.tokenizer.decode(converted.greedy_sample([context], [""], 3)[0])
        == " is 42."
    )

    # Check that the masks all match the expected k
    for layer in ann_model.gpt_neox.layers:
        assert layer.attention.ann.log_indices
        for indices in layer.attention.ann.log_indices:
            assert indices.shape[-1] == 8
