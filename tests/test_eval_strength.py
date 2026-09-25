"""PR-6's strength check (`blink eval strength`): the run's latest EMA checkpoint on the first 2,000
DeepMind puzzles beside DeepMind 9M, the training hours so far, the trend, and PR-3's due notice.

The scorer here is a fake that writes the per-puzzle CSVs `blink eval puzzles` writes; the real one's
command line, environment and priority are checked without running it.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from blink import cli
from blink.eval import puzzles, strength

REAL_DM = Path(r"D:\blink\eval\puzzles\puzzles_dm10k_dm_9M_action-value.csv")
IDS = [f"p{i:02d}" for i in range(10)]
DM = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]  # DM-9M solves 8 of the 10
VALUE = [1, 1, 1, 1, 1, 0, 0, 0, 1, 0]  # Blink value mode: 6, one DM missed
POLICY = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def _write_results(path: Path, ids, correct) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["puzzle_id", "rating", "band", "correct", "illegal"])
        writer.writeheader()
        writer.writerows(
            {"puzzle_id": pid, "rating": 1500, "band": "1500-2000", "correct": c, "illegal": 0}
            for pid, c in zip(ids, correct, strict=True)
        )


def _metrics(run_dir: Path, last: int, every: int = 1000, seconds: float = 360.0, paused_at=None) -> None:
    """Rows every `every` steps, `seconds` apart (10 h per 100,000 steps), one trainer process each side
    of an 8-hour pause at `paused_at`."""
    rows, now = [], 1_000_000.0
    for step in range(every, last + 1, every):
        now += seconds + (8 * 3600.0 if step == paused_at else 0.0)
        session = 2.0 if paused_at is not None and step >= paused_at else 1.0
        rows.append({"step": step, "time": now, "phase": "train", "session": session})
    (run_dir / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("BLINK_HOME", str(home))
    run = home / "runs" / "long"
    run.mkdir(parents=True)
    (run / "ckpt_000100000.pt").write_bytes(b"")
    (run / "ckpt_000120000.pt.tmp").write_bytes(b"")  # being written: never scored
    _metrics(run, 130_000, paused_at=110_000)
    _write_results(home / "eval" / "puzzles" / "puzzles_dm10k_dm_9M_action-value.csv", IDS, DM)
    return home


class FakeScorer:
    """Writes the two CSVs `blink eval puzzles --mode both` writes; says which checkpoint it loaded."""

    def __init__(self, home: Path, value=VALUE, policy=POLICY, loaded: int | None = 100_000) -> None:
        self.home, self.value, self.policy, self.loaded = home, value, policy, loaded
        self.calls: list[tuple] = []

    def __call__(self, run: str, out_dir: Path, limit: int) -> Path | None:
        self.calls.append((run, out_dir, limit))
        for mode, correct in (("value", self.value), ("policy", self.policy)):
            _write_results(strength.results_csv(out_dir, run, mode), IDS[:limit], correct[:limit])
        return None if self.loaded is None else self.home / "runs" / run / f"ckpt_{self.loaded:09d}.pt"


def test_the_real_scorer_runs_the_v0_probe_s_command_on_the_cpu_at_below_normal_priority(home):
    out = home / "eval" / "puzzles" / "checks" / "long-100000"
    argv = strength.scorer_argv("long", out, 2000)
    probe = ["eval", "puzzles", "--model", "run:long:ema", "--mode", "both", "--device", "cpu"]
    assert argv[1:] == ["-m", "blink.cli", *probe, "--limit", "2000", "--epsilon", "0", "--out", str(out)]
    env = strength.scorer_env({"OMP_NUM_THREADS": "12", "CUDA_VISIBLE_DEVICES": "0"})
    assert env["CUDA_VISIBLE_DEVICES"] == "" and env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "4"
    assert strength.scorer_env({"OMP_NUM_THREADS": "2"})["OMP_NUM_THREADS"] == "2"  # never more than asked
    if sys.platform == "win32":
        assert strength.priority_kwargs() == {"creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS}


def test_a_check_records_both_modes_with_wilson_intervals_and_the_pairing_with_dm_9m(home):
    row = strength.run_check("long", home, FakeScorer(home), limit=10, now=lambda: 5.0)
    assert (row["kind"], row["run"], row["step"], row["puzzles"], row["time"]) == (
        "strength",
        "long",
        100_000,
        10,
        5.0,
    )
    assert row["value"]["correct"] == 6 and row["value"]["accuracy"] == 0.6
    assert row["value"]["wilson95"] == list(puzzles.wilson(6, 10))
    assert row["value"]["paired_with_dm9m"] == {"both": 5, "only_blink": 1, "only_dm": 3, "neither": 1}
    assert row["policy"]["paired_with_dm9m"] == {"both": 3, "only_blink": 0, "only_dm": 5, "neither": 2}
    assert row["dm9m"] == {"correct": 8, "n": 10, "accuracy": 0.8, "wilson95": list(puzzles.wilson(8, 10))}
    assert row["out"] == str(home / "eval" / "puzzles" / "checks" / "long-100000")
    saved = [
        json.loads(line) for line in (home / "eval" / "strength_checks.jsonl").read_text("utf-8").splitlines()
    ]
    assert saved == [row]


def test_training_hours_count_up_to_the_scored_step_and_leave_out_the_pause(home):
    """100 rows of 6 minutes to step 100,000 is 9.9 h counted (the first row opens the log), and the
    8-hour pause before step 110,000 lies beyond the scored step anyway; at 120,000 it drops out."""
    row = strength.run_check("long", home, FakeScorer(home), limit=10)
    assert row["hours"] == pytest.approx(99 * 360.0 / 3600)
    assert strength.training_hours(home / "runs" / "long", 130_000) == pytest.approx(128 * 360.0 / 3600)


def test_the_pairing_refuses_blink_results_on_other_puzzles_than_dm_s():
    with pytest.raises(ValueError, match="same puzzles"):
        strength.paired({"a": 1, "b": 0}, {"a": 1, "c": 1})


def test_dm_9m_s_accuracy_comes_from_its_csv_on_the_first_n_puzzles(home):
    assert strength.dm_results(home, 5) == dict(zip(IDS[:5], DM[:5], strict=True))
    assert strength.summary(strength.dm_results(home, 10))["accuracy"] == 0.8


@pytest.mark.local
@pytest.mark.skipif(not REAL_DM.is_file(), reason="needs D:/blink/eval/puzzles (read only)")
def test_dm_9m_solves_86_6_percent_of_the_first_2000_real_puzzles():
    """EVAL.md PR-6's line: DeepMind 9M's 86.6% on the same 2,000 puzzles, from its per-puzzle CSV."""
    dm = strength.dm_results(REAL_DM.parents[2], 2000)
    assert len(dm) == 2000 and strength.summary(dm)["correct"] == 1732
    assert strength.summary(dm)["accuracy"] == pytest.approx(0.866)


def test_a_checkpoint_saved_while_scoring_moves_the_check_to_the_step_the_scorer_loaded(home):
    (home / "runs" / "long" / "ckpt_000110000.pt").write_bytes(b"")  # the latest when the check starts
    fake = FakeScorer(home, loaded=120_000)  # a newer one landed before the scorer loaded

    def scorer(run, out_dir, limit):
        (home / "runs" / "long" / "ckpt_000120000.pt").write_bytes(b"")
        return fake(run, out_dir, limit)

    row = strength.run_check("long", home, scorer, limit=10)
    checks = home / "eval" / "puzzles" / "checks"
    assert fake.calls[0][1] == checks / "long-110000"  # the latest when the check started
    assert row["step"] == 120_000 and row["out"] == str(checks / "long-120000")
    assert (checks / "long-120000").is_dir() and not (checks / "long-110000").exists()


def test_without_a_weights_line_a_moved_latest_checkpoint_is_refused(home):
    def racing(run, out_dir, limit):
        FakeScorer(home)(run, out_dir, limit)
        (home / "runs" / "long" / "ckpt_000120000.pt").write_bytes(b"")
        return None

    with pytest.raises(RuntimeError, match="check again"):
        strength.run_check("long", home, racing, limit=10)
    assert not (home / "eval" / "strength_checks.jsonl").exists()


def test_a_step_already_checked_is_not_scored_again_unless_asked(home):
    scorer = FakeScorer(home)
    assert strength.run_check("long", home, scorer, limit=10) is not None
    assert strength.run_check("long", home, scorer, limit=10) is None and len(scorer.calls) == 1
    assert strength.run_check("long", home, scorer, limit=10, again=True)["step"] == 100_000


def test_the_trend_lists_every_check_of_the_run_with_the_dm_9m_line(home):
    strength.run_check("long", home, FakeScorer(home), limit=10)
    (home / "runs" / "long" / "ckpt_000130000.pt").write_bytes(b"")
    strength.run_check("long", home, FakeScorer(home, value=DM, loaded=130_000), limit=10)
    lines = strength.trend(strength.read_checks(home), "long")
    assert "first 10 DeepMind puzzles" in lines[0]
    assert any("100,000" in line and "60.0%" in line for line in lines)
    assert any("130,000" in line and "80.0%" in line for line in lines)
    assert "DM-9M" in lines[-1] and "80.0%" in lines[-1]


def test_pr3_parity_and_soak_are_due_after_24_training_hours_until_their_record(home, capsys):
    rows = strength.read_checks(home)
    assert not strength.pr3_due(rows, "long", 23.9) and strength.pr3_due(rows, "long", 24.0)
    lines = strength.pr3_lines("long", 331_758, 24.2)
    text = "\n".join(lines)
    assert text.startswith("PR-3 parity and soak are due")
    assert (
        "bench parity --model run:long:ema --positions 20000 --precision bf16 --compile --device cuda" in text
    )
    assert "bench play --sizes configs/long.toml --rows 219 --concurrency 5,4,3,2 --iters 5000" in text
    assert "eval strength --run long --record-pr3" in text
    parity = home / "eval" / "parity" / "run_long_ema-bf16-compile.json"
    parity.parent.mkdir(parents=True)
    parity.write_text(json.dumps({"model": "run:long:ema", "value_choice_agreement": 0.995}), "utf-8")
    assert cli.main(["eval", "strength", "--run", "long", "--record-pr3", str(parity)]) == 0
    assert not strength.pr3_due(strength.read_checks(home), "long", 30.0)
    other = home / "eval" / "parity" / "other.json"
    other.write_text(json.dumps({"model": "run:size-m:ema"}), "utf-8")
    assert cli.main(["eval", "strength", "--run", "long", "--record-pr3", str(other)]) == 2


def test_the_command_scores_prints_the_trend_and_says_when_pr3_is_due(home, monkeypatch, capsys):
    _metrics(home / "runs" / "long", 300_000)  # 29.9 training hours by step 300,000
    (home / "runs" / "long" / "ckpt_000300000.pt").write_bytes(b"")
    monkeypatch.setattr(strength, "child_scorer", lambda h: FakeScorer(h, loaded=300_000))
    assert cli.main(["eval", "strength", "--run", "long", "--limit", "10"]) == 0
    out = capsys.readouterr().out
    assert "300,000" in out and "DM-9M" in out and "PR-3 parity and soak are due" in out
    assert cli.main(["eval", "strength", "--show", "--last", "1"]) == 0
    shown = capsys.readouterr().out
    assert "300,000" in shown and "PR-3" not in shown
    assert cli.main(["eval", "strength"]) == 2  # scoring needs --run


def test_the_puzzle_command_names_the_checkpoint_a_run_selector_loads(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    from blink.play import factory
    from blink.play.oracles import RandomLogitEvaluator

    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    (tmp_path / "runs" / "long").mkdir(parents=True)
    (tmp_path / "runs" / "long" / "ckpt_000000500.pt").write_bytes(b"")
    source = tmp_path / "set.csv"
    source.write_text("PuzzleId,Rating,PGN,Moves\ns1,650,1. e4 e5 2. Bc4 Nc6 3. Qh5,g8f6 h5f7\n", "utf-8")
    monkeypatch.setattr(factory, "load_evaluator", lambda *a, **k: RandomLogitEvaluator())
    argv = ["eval", "puzzles", "--set", str(source), "--model", "run:long:ema", "--device", "cpu"]
    assert cli.main([*argv, "--epsilon", "0", "--out", str(tmp_path / "out")]) == 0
    weights = strength.WEIGHTS.findall(capsys.readouterr().out)
    assert weights == [(str(tmp_path / "runs" / "long" / "ckpt_000000500.pt"), "ema")]
