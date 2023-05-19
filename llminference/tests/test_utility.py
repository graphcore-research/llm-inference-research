from .. import utility


def test_batches() -> None:
    assert list(utility.batches("abcdefg", 3)) == [list("abc"), list("def"), list("g")]
