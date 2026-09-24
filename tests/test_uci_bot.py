"""blink-uci under lichess-bot: pinned weights and the P7 CPU budget (plan P9).

lichess-bot starts a fresh blink-uci for every game, for days, from a selector (`ship`) whose file
can be overwritten. `--sha` makes each engine hash its weights file before the UCI handshake and
exit 2 on a mismatch, so the bot plays the evaluated model or no game at all. The G5 casual smoke
runs while the long training run is live, so its engine keeps to the plan's side-process budget:
one torch thread at below-normal priority (`--threads 1 --priority below_normal`).
"""

import hashlib
import io
import os
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from blink import uci

ROOT = Path(__file__).resolve().parent.parent

WEIGHTS = b"blink weights"
SHA = hashlib.sha256(WEIGHTS).hexdigest()
HANDSHAKE = "uci\nquit\n"


def run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = uci.main(argv, stdin=io.StringIO(HANDSHAKE), stdout=out)
    return code, out.getvalue()


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "blink.pt"
    path.write_bytes(WEIGHTS)
    return path


@pytest.mark.torch
def test_blink_uci_starts_when_its_weights_hash_to_the_pinned_sha(weights):
    code, out = run(["--model", str(weights), "--sha", SHA])
    assert code == 0 and "uciok" in out


@pytest.mark.torch
def test_blink_uci_refuses_weights_whose_sha256_is_not_the_pinned_one(weights, capsys):
    code, out = run(["--model", str(weights), "--sha", "0" * 64])
    assert code == 2 and out == ""  # nothing reaches lichess-bot, so its engine check fails at startup
    err = capsys.readouterr().err
    assert "refusing to start" in err and SHA[:12] in err


@pytest.mark.torch
def test_blink_uci_refuses_a_pinned_model_whose_weights_file_is_missing(tmp_path, capsys):
    code, out = run(["--model", str(tmp_path / "gone.pt"), "--sha", SHA])
    assert code == 2 and out == "" and "no weights file" in capsys.readouterr().err


def test_a_sha_pin_needs_a_blink_weights_file_not_a_random_network(capsys):
    code, out = run(["--random", "--sha", SHA])
    assert code == 2 and out == "" and "--sha" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["abcdef1", "G" * 64, SHA.upper() + "0"])
def test_the_sha_flag_takes_only_a_full_lowercase_sha256(bad):
    with pytest.raises(SystemExit):
        uci.build_parser().parse_args(["--sha", bad])


# ---------------------------------------------------------------- the P7 CPU budget


def test_blink_uci_applies_its_cpu_limits_before_anything_else(monkeypatch):
    calls = []
    monkeypatch.setattr(uci, "limit_cpu", lambda threads, priority: calls.append((threads, priority)))
    code, out = run(["--random", "--threads", "1", "--priority", "below_normal"])
    assert code == 0 and "uciok" in out and calls == [(1, "below_normal")]
    calls.clear()
    run(["--random"])
    assert calls == [(None, "normal")]  # no limits unless asked: the rated bot uses the whole PC


@pytest.mark.parametrize("argv", [["--threads", "0"], ["--priority", "high"]])
def test_blink_uci_refuses_a_zero_thread_count_or_an_unknown_priority(argv):
    with pytest.raises(SystemExit):
        uci.build_parser().parse_args(argv)


@pytest.mark.torch
def test_the_cpu_limits_leave_torch_one_thread_at_below_normal_priority():
    probe = (
        "from blink import uci; uci.limit_cpu(1, 'below_normal'); import torch, psutil; "
        "print(torch.get_num_threads(), torch.get_num_interop_threads(), psutil.Process().nice())"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    done = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=120,
        check=True,
    )
    below_normal = psutil.BELOW_NORMAL_PRIORITY_CLASS if sys.platform == "win32" else 10
    assert done.stdout.split() == ["1", "1", str(below_normal)]
