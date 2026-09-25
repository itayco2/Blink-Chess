import json
import threading
import time
from pathlib import Path

import pytest

from blink import cli, heartbeat
from blink.train import status

SPEED_CASES = json.loads(
    (Path(__file__).parent / "fixtures" / "dashboard_speed_cases.json").read_text(encoding="utf-8")
)["cases"]


def _run(tmp_path, name="run"):
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True)
    return run_dir


def test_heartbeat_marks_a_run_live_within_30_s(tmp_path):
    run_dir = _run(tmp_path)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 5, "steps": 10}, now=1_000.0)
    assert status.run_status(run_dir, now=1_000.0).live
    assert status.run_status(run_dir, now=1_029.0).live
    assert not status.run_status(run_dir, now=1_031.0).live
    assert status.LIVE_WITHIN_S == 30


def test_a_run_without_a_heartbeat_is_not_live(tmp_path):
    report = status.run_status(_run(tmp_path), now=5.0)
    assert not report.live and report.state == "unknown" and report.heartbeat_age_s is None


def test_a_finished_run_is_not_live_and_exits_zero(tmp_path):
    run_dir = _run(tmp_path)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "finished", "step": 10, "steps": 10}, now=1.0)
    report = status.run_status(run_dir, now=2.0)
    assert not report.live
    assert status.exit_code(report) == 0


def test_a_stale_or_crashed_run_exits_nonzero(tmp_path):
    stale = _run(tmp_path, "stale")
    heartbeat.write(stale / "heartbeat.json", {"state": "running", "step": 3}, now=1.0)
    assert status.exit_code(status.run_status(stale, now=100.0)) == 1
    crashed = _run(tmp_path, "crashed")
    heartbeat.write(crashed / "heartbeat.json", {"state": "crashed", "error": "x"}, now=1.0)
    assert status.exit_code(status.run_status(crashed, now=2.0)) == 1


def test_a_nan_loss_makes_a_live_run_exit_nonzero(tmp_path):
    run_dir = _run(tmp_path)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 50}, now=1.0)
    (run_dir / "metrics.jsonl").write_text('{"step": 50, "loss_policy": NaN}\n', encoding="utf-8")
    report = status.run_status(run_dir, now=2.0)
    assert report.live and status.exit_code(report) == 1


def test_the_last_jsonl_record_skips_a_torn_line(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"step": 1}\n{"step": 50}\n{"step": 10', encoding="utf-8")
    assert status.last_jsonl_record(path) == {"step": 50}
    assert status.last_jsonl_record(tmp_path / "missing.jsonl") is None


@pytest.mark.parametrize("name", ["..", ".", "../x", "a/b", "a\\b", "", "x" * 65, "C:", " run"])
def test_run_names_that_could_escape_the_runs_directory_are_rejected(name):
    assert not status.valid_run_name(name)


def test_ordinary_run_names_are_accepted():
    for name in ("skeleton", "s10m", "a01-seed.1", "long_2026"):
        assert status.valid_run_name(name)


def test_list_runs_reports_every_run_directory_newest_heartbeat_first(tmp_path):
    for name, t in (("old", 10.0), ("new", 20.0)):
        heartbeat.write(_run(tmp_path, name) / "heartbeat.json", {"state": "finished"}, now=t)
    (tmp_path / "not-a-run.txt").write_text("x", encoding="utf-8")
    assert [r.name for r in status.list_runs(tmp_path, now=30.0)] == ["new", "old"]


def test_the_status_text_names_the_state_step_and_last_numbers(tmp_path):
    run_dir = _run(tmp_path)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 50, "steps": 100}, now=1.0)
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"step": 50, "loss_policy": 3.2}) + "\n", encoding="utf-8"
    )
    text = status.format_status(status.run_status(run_dir, now=2.0))
    assert "LIVE" in text and "50/100" in text and "3.2" in text


def test_a_running_trainer_is_live_before_its_first_step(tmp_path):
    pytest.importorskip("torch")
    from train_helpers import fixture_records, tiny_train_config

    from blink.train import loop

    release = threading.Event()
    records = fixture_records()

    def gated(start_step):
        release.wait(timeout=60)
        while True:
            yield records[:16]

    cfg = tiny_train_config(steps=5, warmup_steps=1, batch_size=16)
    spec = loop.RunSpec(run_dir=tmp_path / "live", world="0123456789ab", device="cpu")
    worker = threading.Thread(
        target=loop.train, args=(cfg, spec, gated, records), kwargs={"log": lambda _: None}
    )
    worker.start()
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not status.run_status(spec.run_dir).live:
            time.sleep(0.05)
        assert status.run_status(spec.run_dir).live
    finally:
        release.set()
        worker.join(timeout=120)
    assert status.run_status(spec.run_dir).state == "finished"


def test_the_status_text_shows_the_phase_clip_and_vaa_with_a_failed_check(tmp_path):
    run_dir = _run(tmp_path)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 60, "steps": 100}, now=1.0)
    metrics = {"step": 60, "loss_policy": 3.2, "phase": "eval", "clip": 12.5, "clip_frac": 0.02}
    (run_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")
    failed = {"step": 5, "check": "5%", "vaa": 0.31, "ema_vaa": 0.33, "vaa_set": "full"}
    failed |= {"vaa_check_failed": True, "check_failure": {"check": "5%"}}
    (run_dir / "evals.jsonl").write_text(json.dumps(failed) + "\n", encoding="utf-8")
    text = status.format_status(status.run_status(run_dir, now=2.0))
    assert "[eval]" in text and "clip 12.5" in text
    assert "VAA 0.31 (ema 0.33, full)" in text and "check 5% FAILED" in text


def test_a_skipped_check_says_skipped_with_its_reason_never_passed(tmp_path):
    """P6 v2 skips the flagship's 5% check (vaa_reference is "" until the guard): the trainer's log says
    so, and `blink status` must say the same."""
    run_dir = _run(tmp_path)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 60, "steps": 100}, now=1.0)
    row = {"step": 5, "check": "5%", "vaa": 0.31, "ema_vaa": 0.33, "vaa_set": "full"}
    row |= {"check_skipped": "no reference run"}
    (run_dir / "evals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    text = status.format_status(status.run_status(run_dir, now=2.0))
    assert "check 5% skipped (no reference run)" in text and "passed" not in text


def test_the_status_text_shows_a_subset_eval_s_ema_vaa(tmp_path):
    run_dir = _run(tmp_path)
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 60, "steps": 100}, now=1.0)
    row = {"step": 40, "top1": 0.3, "ema_vaa": 0.41, "vaa_set": "subset", "vaa_n": 2000}
    (run_dir / "evals.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    text = status.format_status(status.run_status(run_dir, now=2.0))
    assert "VAA ema 0.41 (subset of 2000)" in text and "check" not in text


@pytest.mark.parametrize("case", SPEED_CASES, ids=[case["name"] for case in SPEED_CASES])
def test_the_speed_warning_gives_every_shared_dashboard_case(case):
    check = status.speed_check(case["rows"])
    expect = case["expect"]
    assert check.warn is expect["warn"]
    assert check.slow_s == pytest.approx(expect["slow_s"])
    for field in ("reference", "rate"):
        got, want = getattr(check, field), expect[field]
        assert got is None if want is None else got == pytest.approx(want), field


def test_the_shared_cases_cover_both_verdicts_and_every_skipped_phase():
    assert {case["expect"]["warn"] for case in SPEED_CASES} == {True, False}
    phases = {row.get("phase") for case in SPEED_CASES for row in case["rows"]}
    assert {"train", "eval", "ckpt", None} <= phases


def test_a_nan_or_boolean_rate_is_skipped_like_a_missing_one():
    rows = [{"step": 50, "samples_per_s": 1000.0, "time": 0.0}]
    rows += [{"step": 100, "samples_per_s": value, "time": 400.0} for value in (float("nan"), True, "900")]
    check = status.speed_check(rows)
    assert (check.rate, check.reference, check.slow_s) == (1000.0, 1000.0, 0.0)


def test_blink_status_prints_the_speed_warning_but_keeps_its_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    run_dir = _run(tmp_path / "runs", "spill")
    heartbeat.write(run_dir / "heartbeat.json", {"state": "running", "step": 1050, "steps": 5000})
    warn = next(case for case in SPEED_CASES if case["name"] == "slow for five minutes of wall time warns")
    lines = "".join(json.dumps({**row, "loss_policy": 3.0, "loss_value": 4.0}) + "\n" for row in warn["rows"])
    (run_dir / "metrics.jsonl").write_text(lines + '{"step": 99', encoding="utf-8")  # plus a torn line
    assert cli.main(["status", "--run", "spill"]) == 0
    out = capsys.readouterr().out
    assert "WARN" in out and "600 samples/s" in out and "1,000" in out and "5.0 min" in out
    (run_dir / "metrics.jsonl").write_text(lines.splitlines(keepends=True)[0], encoding="utf-8")
    assert cli.main(["status", "--run", "spill"]) == 0
    assert "WARN" not in capsys.readouterr().out
