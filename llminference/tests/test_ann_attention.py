from typing import cast

import torch

from .. import ann_attention as ann
from .. import eval_adapter


def test_ann() -> None:
    batch, n_heads, n_key, head_size = (3, 5, 7, 16)
    query = torch.randn((batch, n_heads, 1, head_size))
    key = torch.randn((batch, n_heads, n_key, head_size))
    # Mask contains leading fmins, different across the batch
    logmask = torch.finfo(torch.float32).min * (
        torch.arange(n_key)[None, None, None, :]
        < torch.tensor([2, 0, 5])[:, None, None, None]
    )
    for score in (ann.LowRank.Settings(8), ann.SparseQ.Settings(6)):
        module = ann.ANN(
            ann.Settings(k=4, local_k=1, score=cast(ann.ScoreSettings, score)),
            n_heads,
            head_size,
        )
        ann_mask = module(query, key, logmask).to(torch.long)
        assert ann_mask.shape == (batch, n_heads, 1, n_key)
        assert ((ann_mask * logmask) == 0).all(), "already-masked"
        assert (ann_mask.sum(-1) <= 4).all(), "max k"
        assert ann_mask[..., -1:].all(), "local_k"


def test_convert_gptneox() -> None:
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-160m")

    # Note: the head size of pythia-160m is 64, so this should be exact
    # (due to orthonormal init)
    ann_model = ann.convert_gptneox(
        adapter.model, ann.Settings(k=8, local_k=2, score=ann.LowRank.Settings(64))
    )
    for layer in ann_model.gpt_neox.layers:
        layer.attention.ann_masks = []

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
        assert layer.attention.ann_masks
        for mask in layer.attention.ann_masks:
            assert (mask.sum(-1) == 8).all()
