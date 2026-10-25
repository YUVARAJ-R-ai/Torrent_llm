"""The self-check script, run against a real in-process pool."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pool_check  # noqa: E402
from tests.test_pool import Pool, wait_for  # noqa: E402


class CharTokens:
    pass


def run(url):
    lines: list[str] = []
    ok = pool_check.check(url, out=lines.append)
    return ok, "\n".join(lines)


def test_an_unreachable_pool_says_so_plainly():
    ok, text = run("http://127.0.0.1:1")

    assert not ok and "Could not reach" in text


def test_a_working_pool_passes_or_fails_only_on_the_answer(model_factory, num_layers, monkeypatch):
    monkeypatch.setattr(pool_check, "CHAT", False)  # the tiny model has no chat template
    # The tiny model cannot say "Paris", so the check must report exactly that and
    # nothing about the pool's own health.
    pool = Pool(model_factory, num_layers)
    try:
        for name in "abcd":
            pool.add(name)
        wait_for(lambda: pool.settled([2, 2]), what="two copies of each stage")

        ok, text = run(pool.peers["a"].url)
    finally:
        pool.close()

    assert not ok
    assert "does not mention Paris" in text
    assert "every device gave the same answer" in text  # the model is deterministic
    assert "all five answered" in text
    assert "health healthy" in text


@pytest.mark.parametrize("flag", ["--help"])
def test_the_script_describes_itself(flag, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["pool_check.py", flag])

    with pytest.raises(SystemExit) as exit_info:
        pool_check.main()

    assert exit_info.value.code == 0
    assert "same question" in capsys.readouterr().out
