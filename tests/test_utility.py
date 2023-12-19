# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import contextlib
import io
import json
import unittest.mock as um
from pathlib import Path
from typing import Any, Dict

import datasets
import pytest

from llminference import utility


def test_batches() -> None:
    assert list(utility.batches("abcdefg", 3)) == [list("abc"), list("def"), list("g")]


def _fake_task(value: int) -> Dict[str, Any]:
    if value == 5:
        raise ValueError("High five!")
    return dict(result=value * 10)


def test_jsonlines_writer() -> None:
    with pytest.raises(FileExistsError), um.patch(
        "pathlib.Path.exists", lambda _: True
    ):
        with utility.jsonlines_writer("existing.jsonl") as _:
            pass

    buf = io.StringIO()
    with um.patch("pathlib.Path.exists", lambda _: False), um.patch(
        "pathlib.Path.mkdir"
    ), um.patch("pathlib.Path.open", lambda *_: contextlib.nullcontext(buf)):
        with utility.jsonlines_writer("new.jsonl") as writer:
            writer(dict(n=1, result=True))
            writer(dict(n=2, result=False))
    assert [json.loads(line) for line in buf.getvalue().rstrip("\n").split("\n")] == [
        dict(n=1, result=True),
        dict(n=2, result=False),
    ]


def test_multiprocess_sweep(tmp_path: Path) -> None:
    utility.run_multiprocess_sweep(
        _fake_task,
        [dict(value=n) for n in range(9)],
        tmp_path / "results.jsonl",
        n_workers=4,
    )
    results = list(
        map(json.loads, (tmp_path / "results.jsonl").read_text().splitlines())
    )
    (bad_result,) = (r for r in results if "_error" in r)
    assert "High five!" in bad_result["_error"]
    assert bad_result["_args"]["value"] == "5"
    assert {r["result"] for r in results if "_error" not in r} == {
        10 * i for i in range(9) if i != 5
    }


def test_map_full_batch() -> None:
    out = utility.map_full_batch(
        datasets.Dataset.from_list([dict(input=n) for n in range(5)]),
        lambda rows: (dict(output=x["input"]) for x in rows[::-1]),
    )
    assert list(out.features) == ["output"]
    assert list(out["output"]) == [4, 3, 2, 1, 0]


def test_map_and_filter() -> None:
    out = utility.map_and_filter(
        datasets.Dataset.from_list([dict(input=n) for n in range(10)]),
        lambda x: dict(output=100 * x["input"]) if x["input"] % 3 == 0 else None,
    )
    assert list(out.features) == ["output"]
    assert list(out["output"]) == [0, 300, 600, 900]
