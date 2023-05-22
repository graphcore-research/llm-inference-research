import json
from pathlib import Path
from typing import Any, Dict

from .. import utility


def test_batches() -> None:
    assert list(utility.batches("abcdefg", 3)) == [list("abc"), list("def"), list("g")]


def _fake_task(value: int) -> Dict[str, Any]:
    if value == 5:
        raise ValueError("High five!")
    return dict(result=value * 10)


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
    assert bad_result["_args"]["value"] == 5
    assert {r["result"] for r in results if "_error" not in r} == {
        10 * i for i in range(9) if i != 5
    }
