# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from math import log2

import numpy as np
import torch

from llminference.eval_adapter import Adapter
from llminference.tasks import bpc


def test_wikitext() -> None:
    data = bpc.WikiText.data(prefill_len=6000, reference_len=400)
    assert len(data) >= 20_000
    assert 6000 <= np.mean([len(d["prefill"]) for d in data]) <= 6050
    assert 400 <= np.mean([len(d["reference"]) for d in data]) <= 405
    assert set(data.features) == {"prefill", "reference"}


def test_wikitext_filter() -> None:
    s = "example of a\ntest string !"  # length = 26

    d = bpc.WikiText.preprocess(dict(page=s), prefill_len=0, reference_len=26 - 7)
    assert d == dict(prefill="example", reference=" of a\ntest string !")

    d = bpc.WikiText.preprocess(dict(page=s), prefill_len=0, reference_len=24 - 7)
    assert d == dict(prefill="example", reference=" of a\ntest string")

    d = bpc.WikiText.preprocess(dict(page=s), prefill_len=3, reference_len=24 - 7)
    assert d == dict(prefill="example", reference=" of a\ntest string")

    d = bpc.WikiText.preprocess(dict(page=s), prefill_len=10, reference_len=24 - 10)
    assert d == dict(prefill="example of", reference=" a\ntest string")

    d = bpc.WikiText.preprocess(dict(page=s), prefill_len=11, reference_len=24 - 12)
    assert d == dict(prefill="example of a", reference="\ntest string")

    d = bpc.WikiText.preprocess(dict(page=s), prefill_len=13, reference_len=24 - 17)
    assert d == dict(prefill="example of a\ntest", reference=" string")

    d = bpc.WikiText.preprocess(dict(page=s), prefill_len=1, reference_len=1)
    assert d == dict(prefill="example", reference=" of")

    d = bpc.WikiText.preprocess(dict(page="qwerty"), prefill_len=2, reference_len=10)
    assert d is None

    d = bpc.WikiText.preprocess(dict(page="qwerty "), prefill_len=2, reference_len=1)
    assert d == dict(prefill="qwerty", reference=" ")

    d = bpc.WikiText.preprocess(dict(page=" qwerty "), prefill_len=2, reference_len=1)
    assert d == dict(prefill=" qwerty", reference=" ")

    d = bpc.WikiText.preprocess(dict(page="    "), prefill_len=1, reference_len=2)
    assert d == dict(prefill=" ", reference="  ")


def test_calc_bpc() -> None:
    nll = -torch.tensor([0.8, 0.2]).log()
    p = bpc.calc_bpc(nll, torch.tensor([3]))
    expected_p = -1 / 3 * (log2(0.8) + log2(0.2))
    np.testing.assert_approx_equal(p.item(), expected_p)

    # Can handle padding & batching
    padded_nll = torch.cat([nll, torch.zeros(10)])
    logits = torch.stack([torch.randn(12), padded_nll])
    p = bpc.calc_bpc(logits, torch.tensor([20, 3]))
    np.testing.assert_approx_equal(p[1].item(), expected_p)


def test_evaluate() -> None:
    adapter = Adapter.from_pretrained("EleutherAI/pythia-70m")
    _examples = [
        (  # Simple, longer passage
            "Compared to the preprocessed version of Penn Treebank (PTB),"
            " WikiText-2 is over 2 times larger and WikiText-103 is over 110 times"
            " larger. As it is composed of full articles, the dataset"
            " is well suited for models that can take advantage of long term"
            " dependencies. The WikiText dataset also features a far larger vocabulary"
            " and retains the original case, punctuation and numbers - all of which"
            " are removed in PTB."
        ),
        (  # Simple, shorter passage
            "Compared to the preprocessed version of Penn Treebank (PTB),"
            " WikiText-2 is over 2 times larger and WikiText-103 is over 110 times"
            " larger. As it is composed of full articles, the dataset"
            " is well suited for models that can take advantage of long term"
            " dependencies."
        ),
        (  # Harder (jumbled-up), longer passage
            "The markup language called wikitext! also known as wiki wiki or"
            " software to format a software. (Note the lowercase spelling of where"
            " wikicode, consists of the syntax and copied and pasted used by the"
            " MediaWiki new code. In addition to presentation? some HTML elements"
            " are also terms.) Generally, coding can be keywords, without writing"
            " allowed for wikitext formatting. "
        ),
    ]

    prefill_lens = [1, 20, 50]
    examples = {
        w: [{"prefill": e[:w], "reference": e[w:]} for e in _examples]
        for w in prefill_lens
    }
    prefill_lengths = {
        w: [len(adapter.tok_encode(item["prefill"])) for item in e]
        for w, e in examples.items()
    }
    reference_lengths = {
        w: [len(adapter.tok_encode(item["reference"])) for item in e]
        for w, e in examples.items()
    }
    bpcs = {
        w: list(bpc.evaluate(adapter, examples[w], batch_size=2)) for w in prefill_lens
    }

    # Check correct prefill and reference length
    for w, item in bpcs.items():
        for d, c_len, g_len in zip(item, prefill_lengths[w], reference_lengths[w]):
            assert d["prefill_length"] == c_len
            assert d["reference_length"] == g_len

    p = {w: [s["bpc"] for s in scores] for w, scores in bpcs.items()}

    # All bpcs must be > 0 and less than log2(vocab_size)
    assert 0 < p[1][0] < 16
    assert 0 < p[20][0] < 16
    assert 0 < p[50][0] < 16
    assert 0 < p[1][1] < 16
    assert 0 < p[20][1] < 16
    assert 0 < p[50][1] < 16
    assert 0 < p[1][2] < 16
    assert 0 < p[20][2] < 16
    assert 0 < p[50][2] < 16

    # Simple text much lower bpc than harder text
    assert p[1][0] + 0.2 < p[1][2]
    assert p[20][0] + 0.2 < p[20][2]
    assert p[50][0] + 0.2 < p[50][2]

    # bpc decreases with prefill_len
    assert p[1][0] > p[20][0] > p[50][0]
    assert p[1][1] > p[20][1] > p[50][1]
    assert p[1][2] > p[20][2] > p[50][2]

    # Longer text -> lower bpc (across prefill_len values)
    assert p[1][0] < p[1][1]
    assert p[20][0] < p[20][1]
    assert p[50][0] < p[50][1]
