"""The dm:<size>[:ema] selector: parsing, weights, and routing in eval puzzles, blink-uci and match."""

import csv
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from blink import cli, paths, uci
from blink.commands import play as play_command
from blink.eval import puzzles
from blink.play import agents, factory
from blink.reference import registry

REPO = Path(__file__).resolve().parent.parent
SCHOLAR = {"PuzzleId": "s1", "Rating": "650", "PGN": "1. e4 e5 2. Bc4 Nc6 3. Qh5", "Moves": "g8f6 h5f7"}
REAL_WEIGHTS = paths.home() / "dm" / "9M-params.npz"


def write_puzzles(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SCHOLAR))
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.fixture
def fake_dm(monkeypatch, tmp_path) -> list[tuple[str, str]]:
    """BLINK_HOME with a (dummy) converted 9M file, and a loader that returns a material player."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    (tmp_path / "dm").mkdir()
    (tmp_path / "dm" / "9M-params.npz").write_bytes(b"")
    loaded: list[tuple[str, str]] = []

    def load_agent(selector: str, device: str = "cuda", sink=None) -> agents.Agent:
        loaded.append((selector, device))
        return agents.MaterialAgent(name=registry.parse(selector).name)

    monkeypatch.setattr(registry, "load_agent", load_agent)
    return loaded


def test_dm_selectors_name_a_size_and_a_parameter_set():
    assert registry.parse("dm:9M") == registry.DmSelector("9M", "params")
    assert registry.parse("dm:9M:ema") == registry.DmSelector("9M", "params_ema")
    assert (registry.parse("dm:9M").name, registry.parse("dm:9M:ema").name) == ("DM-9M", "DM-9M-ema")


@pytest.mark.parametrize("bad", ["dm", "dm:", "dm:9m", "dm:270M", "dm:9M:raw", "dm:9M:ema:x"])
def test_malformed_dm_selectors_are_refused(bad):
    with pytest.raises(ValueError, match=r"dm:9M\[:ema\]"):
        registry.parse(bad)


def test_only_dm_selectors_are_routed_to_the_registry():
    assert registry.is_dm("dm:9M") and registry.is_dm("dm:9M:ema") and registry.is_dm("dm:bogus")
    others = ("ship", "run:dm", "release:dm", "random", "random-net", "dmx:9M", r"D:\dm\blink.pt")
    assert not any(registry.is_dm(selector) for selector in others)


def test_converted_weights_live_under_blink_home_dm(monkeypatch, tmp_path):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert registry.weights_path("dm:9M") == tmp_path / "dm" / "9M-params.npz"
    assert registry.weights_path("dm:9M:ema") == tmp_path / "dm" / "9M-params_ema.npz"


def test_missing_weights_fail_fast_and_name_the_converter(monkeypatch, tmp_path):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    with pytest.raises(factory.ModelUnavailable, match="tools/dm_convert.py"):
        registry.check_available("dm:9M:ema")


def test_the_registry_imports_no_torch():
    modules = (
        "blink.reference.registry, blink.reference.gauntlet, blink.commands.evaluate, "
        "blink.commands.play, blink.uci"
    )
    code = f"import sys, {modules}; print('torch' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**os.environ, "PYTHONPATH": str(REPO)},
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_eval_puzzles_runs_a_dm_selector_once_in_action_value_mode(fake_dm, tmp_path, capsys):
    source = write_puzzles(tmp_path / "p.csv", [SCHOLAR])
    out = tmp_path / "out"
    args = ["eval", "puzzles", "--set", str(source), "--model", "dm:9M", "--device", "cpu", "--out", str(out)]
    assert cli.main(args) == 0
    assert fake_dm == [("dm:9M", "cpu")]
    assert sorted(p.name for p in out.iterdir()) == [
        "puzzles_p_dm_9M_action-value.csv",
        "puzzles_p_dm_9M_action-value.json",
    ]
    summary = json.loads((out / "puzzles_p_dm_9M_action-value.json").read_text(encoding="utf-8"))
    assert (summary["mode"], summary["n"], summary["illegal_moves"]) == ("action-value", 1, 0)
    assert "DM-9M (dm:9M)" in capsys.readouterr().out


def test_eval_puzzles_refuses_a_dm_selector_without_weights(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    source = write_puzzles(tmp_path / "p.csv", [SCHOLAR])
    assert cli.main(["eval", "puzzles", "--set", str(source), "--model", "dm:9M"]) == 2
    assert "dm_convert" in capsys.readouterr().err


def test_blink_uci_plays_a_dm_selector_under_its_own_name(fake_dm):
    out = io.StringIO()
    script = "uci\nisready\nposition startpos moves e2e4\ngo movetime 100\nquit\n"
    assert uci.main(["--model", "dm:9M", "--device", "cpu"], io.StringIO(script), out) == 0
    lines = out.getvalue().splitlines()
    assert "id name DM-9M" in lines and lines[-1].startswith("bestmove ")
    assert fake_dm == [("dm:9M", "cpu")]


def test_blink_uci_refuses_a_dm_selector_without_weights(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert uci.main(["--model", "dm:9M:ema"], io.StringIO("uci\nquit\n"), io.StringIO()) == 2
    assert "tools/dm_convert.py" in capsys.readouterr().err


def test_blink_match_and_gauntlet_accept_a_dm_side(fake_dm, monkeypatch, tmp_path, capsys):
    assert play_command.side_agent("dm:9M", "policy", "cpu", 0, 0.0).name == "DM-9M"
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "empty"))
    assert cli.main(["gauntlet", "--model", "dm:9M"]) == 2  # no weights: refused before fastchess starts
    assert "dm_convert" in capsys.readouterr().err


@pytest.mark.local
@pytest.mark.torch
@pytest.mark.skipif(
    not (REAL_WEIGHTS.is_file() and puzzles.resolve_set("dm10k").is_file()), reason="needs the converted 9M"
)
def test_the_real_dm_9m_scores_dm10k_puzzles_end_to_end(tmp_path, capsys):
    args = ["eval", "puzzles", "--set", "dm10k", "--limit", "3", "--model", "dm:9M", "--device", "cpu"]
    assert cli.main([*args, "--out", str(tmp_path)]) == 0
    summary = json.loads((tmp_path / "puzzles_dm10k_dm_9M_action-value.json").read_text(encoding="utf-8"))
    assert (summary["n"], summary["illegal_moves"]) == (3, 0)
