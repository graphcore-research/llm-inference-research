from math import exp, log

import numpy as np
import torch

from .. import perplexity
from ..eval_adapter import Adapter


def test_wikitext() -> None:
    data = perplexity.WikiText.data(warmup=200)
    assert len(data) > 36_700 * 0.35  # ~40% of the original 36,700 rows are long enough
    assert 200 <= np.mean([len(d["prefill"]) for d in data]) <= 205
    assert set(data.features) == {"prefill", "reference"}


def test_calc_perplexity() -> None:
    normalised_logits = torch.tensor([0.8, 0.2]).log()
    p = perplexity.calc_perplexity(normalised_logits)
    expected_p = exp(-0.5 * (log(0.8) + log(0.2)))
    assert p == expected_p

    # Can handle padding & batching
    padded_logits = torch.cat([normalised_logits, torch.zeros(10)])
    logits = torch.stack([torch.randn(12), padded_logits])
    p = perplexity.calc_perplexity(logits)
    assert p[1] == expected_p


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

    warmups = [1, 20, 50]
    examples = {
        w: [{"prefill": e[:w], "reference": e[w:]} for e in _examples] for w in warmups
    }
    prefill_lengths = {
        w: [len(adapter.tok_encode(item["prefill"])) for item in e]
        for w, e in examples.items()
    }
    reference_lengths = {
        w: [len(adapter.tok_encode(item["reference"])) for item in e]
        for w, e in examples.items()
    }
    perplexities = {
        w: list(perplexity.evaluate(adapter, examples[w], batch_size=2))
        for w in warmups
    }

    # Check correct prefill and reference length
    for w, item in perplexities.items():
        for d, c_len, g_len in zip(item, prefill_lengths[w], reference_lengths[w]):
            assert d["prefill_length"] == c_len
            assert d["reference_length"] == g_len

    p = {w: [s["perplexity"] for s in scores] for w, scores in perplexities.items()}

    # All perplexities must be > 1 and less than vocab size
    assert 1 < p[1][0] < 50_000
    assert 1 < p[20][0] < 50_000
    assert 1 < p[50][0] < 50_000
    assert 1 < p[1][1] < 50_000
    assert 1 < p[20][1] < 50_000
    assert 1 < p[50][1] < 50_000
    assert 1 < p[1][2] < 50_000
    assert 1 < p[20][2] < 50_000
    assert 1 < p[50][2] < 50_000

    # Simple text much lower perplexity than harder text
    assert p[1][0] * 2 < p[1][2]
    assert p[20][0] * 2 < p[20][2]
    assert p[50][0] * 2 < p[50][2]

    # Perplexity decreases with warmup
    assert p[1][0] > p[20][0] > p[50][0]
    assert p[1][1] > p[20][1] > p[50][1]
    assert p[1][2] > p[20][2] > p[50][2]

    # Longer text -> lower perplexity (across warmup values)
    assert p[1][0] < p[1][1]
    assert p[20][0] < p[20][1]
    assert p[50][0] < p[50][1]
