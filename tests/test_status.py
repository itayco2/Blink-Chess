import json
import threading
import time

import pytest

from blink import heartbeat
from blink.train import status


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
