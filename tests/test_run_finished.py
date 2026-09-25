"""A run closed by PR-6's finish: runs/<run>/finished_by.json (blink.train.finished).

tools/p7_finish.py branches the flagship's final cooldown from runs/long as runs/long-final and closes
runs/long, whose checkpoints and logs stay as they are. `blink train` and `blink supervise` then refuse
any command that would write runs/long again (the P6 v2 driver's printed resume command, say), while a
branch from it, which only reads it, still runs.
"""

import json

import pytest

from blink import cli
from blink.train import finished

MARKER = {"by": "tools/p7_finish.py (PR-6)", "at_step": 123_457, "branch": "long-final", "time": "t"}
RESUME = ["train", "--config", "configs/long.toml", "--run", "long", "--data", "D:/nowhere", "--resume"]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    run = tmp_path / "runs" / "long"
    run.mkdir(parents=True)
    (run / finished.MARKER).write_text(json.dumps(MARKER), encoding="utf-8")
    return tmp_path


def test_a_closed_run_says_who_closed_it_at_which_step_and_what_carries_it_on(home):
    why = finished.refusal(home / "runs" / "long")
    assert "tools/p7_finish.py (PR-6)" in why and "123,457" in why and "runs/long-final" in why
    assert finished.MARKER in why and finished.refusal(home / "runs" / "long-final") is None


def test_an_unreadable_marker_still_closes_the_run(home):
    (home / "runs" / "long" / finished.MARKER).write_text("{torn", encoding="utf-8")
    assert "runs/long" in finished.refusal(home / "runs" / "long")


def test_supervise_refuses_to_resume_a_closed_run(home, capsys):
    assert cli.main(["supervise", "--", *RESUME]) == 2
    err = capsys.readouterr().err
    assert "runs/long" in err and "long-final" in err and finished.MARKER in err


def test_supervise_still_branches_from_a_closed_run(home, capsys):
    branch = ["train", "--run", "long", "--data", "D:/nowhere", "--from-step", "123457", "--preview-steps",
              "30865", "--preview-name", "long-final"]  # fmt: skip
    assert cli.main(["supervise", "--dry-run", "--", *branch]) == 0
    assert "long-final" in capsys.readouterr().out


def test_train_refuses_to_resume_a_closed_run(home, capsys):
    pytest.importorskip("torch")
    assert cli.main(RESUME) == 2
    assert finished.MARKER in capsys.readouterr().err
