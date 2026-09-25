"""`blink supervise --bench-size` for a branch: a preview, the size-m rung, the flagship's final cooldown.

A branch command names no --config (it trains the parent checkpoint's config), so the benchmark row is
chosen from that config's micro-batch pin and compile mode, as for the run it was cut from: a size-m
branch of the flagship is policed at M's micro-256 compiled row, never at the faster 512 row.
"""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from blink import cli  # noqa: E402
from blink.model.config import config_to_dict, load_config  # noqa: E402

pytestmark = pytest.mark.torch
REAL_BENCH = Path(r"D:\blink\eval\bench.json")
M_ROWS = [  # bench.json's compiled M rows (2026-09-24): 256 fits and is pinned, 512 is faster
    {"size": "m", "micro": 256, "compile": "inductor", "samples_per_s": 2695.2782738630303},
    {"size": "m", "micro": 512, "compile": "inductor", "samples_per_s": 2803.0459090360405},
    {"size": "m", "micro": 256, "compile": "off", "samples_per_s": 1747.9505045625124},
]
FLAGSHIP_FLOOR = "floor 2,291 samples/s (85% of 2,695, size m in bench.json)"
BRANCH = ["train", "--run", "long", "--data", "D:/blink/data/v1", "--from-step", "47301"]
SIZE_M = [*BRANCH, "--preview-steps", "11825", "--preview-name", "size-m"]


def _bench(tmp_path: Path, rows=M_ROWS) -> Path:
    path = tmp_path / "bench.json"
    ok = {"oom": False, "error": None, "spilled": False}
    path.write_text(json.dumps({"throughput": [{**ok, **row} for row in rows]}), encoding="utf-8")
    return path


def _parent_checkpoint(home: Path, repo_root: Path, step: int = 47_301) -> dict:
    """runs/long/ckpt_<step>.pt holding the flagship's config, as the trainer saves it."""
    config = config_to_dict(load_config(repo_root / "configs" / "long.toml"))
    run = home / "runs" / "long"
    run.mkdir(parents=True)
    torch.save({"config": config, "step": step}, run / f"ckpt_{step:09d}.pt")
    return config


def _supervise(bench: Path, train_args: list[str]) -> int:
    argv = ["supervise", "--dry-run", "--bench", str(bench), "--bench-size", "m", "--", *train_args]
    return cli.main(argv)


def test_the_flagship_itself_is_policed_at_m_s_pinned_256_row(tmp_path, monkeypatch, capsys, repo_root):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    flagship = ["train", "--config", str(repo_root / "configs" / "long.toml"), "--run", "long"]
    assert _supervise(_bench(tmp_path), flagship) == 0
    assert FLAGSHIP_FLOOR in capsys.readouterr().out


def test_a_size_m_branch_is_policed_at_the_parent_checkpoint_s_pin_and_mode(
    tmp_path, monkeypatch, capsys, repo_root
):
    """Without --config the rule used to fall back to the fastest M row in any mode (2,803 at 512)."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    _parent_checkpoint(tmp_path, repo_root)
    assert _supervise(_bench(tmp_path), SIZE_M) == 0
    assert FLAGSHIP_FLOOR in capsys.readouterr().out


def test_a_resumed_branch_reads_its_own_config_json_not_the_parent_checkpoint(
    tmp_path, monkeypatch, capsys, repo_root
):
    """The parent may have pruned its branch point by then (blink.commands.train reads the same file)."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    config = config_to_dict(load_config(repo_root / "configs" / "long.toml"))
    branch = tmp_path / "runs" / "size-m"
    branch.mkdir(parents=True)
    (branch / "config.json").write_text(json.dumps({"config": config}), encoding="utf-8")
    assert _supervise(_bench(tmp_path), [*SIZE_M, "--resume"]) == 0
    assert FLAGSHIP_FLOOR in capsys.readouterr().out


def test_a_branch_whose_parent_checkpoint_is_missing_is_refused_not_policed_at_a_guess(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert _supervise(_bench(tmp_path), SIZE_M) == 2
    assert "47301" in capsys.readouterr().err.replace(",", "")


@pytest.mark.local
@pytest.mark.skipif(not REAL_BENCH.is_file(), reason="needs D:/blink/eval/bench.json (read only)")
def test_against_the_real_bench_json_a_size_m_branch_gets_the_flagship_s_2695_row(
    tmp_path, monkeypatch, capsys, repo_root
):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    _parent_checkpoint(tmp_path, repo_root)
    assert _supervise(REAL_BENCH, SIZE_M) == 0
    assert FLAGSHIP_FLOOR in capsys.readouterr().out
