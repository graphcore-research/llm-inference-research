from typing import cast

import torch

from .. import ann_attention as ann
from .. import eval_adapter


def test_ann_module() -> None:
    batch, n_heads, n_key, head_size = (3, 5, 7, 16)
    query = torch.randn((batch, n_heads, 1, head_size))
    key = torch.randn((batch, n_heads, n_key, head_size))
    value = torch.randn((batch, n_heads, n_key, head_size))
    # Mask contains leading `fmin`, different across the batch
    logmask = torch.finfo(torch.float32).min * (
        torch.arange(n_key)[None, None, None, :]
        < torch.tensor([2, 0, 5])[:, None, None, None]
    ).broadcast_to(batch, n_heads, 1, n_key)
    for score, add_remainder in [
        (ann.LowRank.Settings(8), False),
        (ann.SparseQ.Settings(6), True),
    ]:
        module = ann.ANN(
            ann.Settings(
                k=4,
                local_k=1,
                add_remainder=add_remainder,
                score=cast(ann.ScoreSettings, score),
            ),
            n_heads,
            head_size,
        )
        q, k, v, m = module(query, key, value, logmask)
        assert torch.equal(q, query)
        assert k.shape == (batch, n_heads, 4 + add_remainder, head_size)
        assert v.shape == (batch, n_heads, 4 + add_remainder, head_size)
        assert m.shape == (batch, n_heads, 1, 4 + add_remainder)
        assert torch.isin(key[:, :, -1, :].flatten(), k.flatten()).all(), "local_k"


def test_convert_gptneox() -> None:
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-160m")

    # Note: the head size of pythia-160m is 64, so this should be exact
    # (due to orthonormal init)
    ann_model = ann.convert(
        adapter.model,
        ann.Settings(
            k=8, local_k=2, add_remainder=False, score=ann.LowRank.Settings(64)
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
