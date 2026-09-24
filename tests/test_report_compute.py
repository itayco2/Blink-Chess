"""results/compute.json: GPU-hours and GPU-board kWh measured from each run's own telemetry."""

import json

import pytest

from blink.report import compute


def _run(root, name, rows, batch_size=256, device="cuda", nvsmi=None, branched_from=None):
    run = root / name
    run.mkdir(parents=True)
    config = {"run": name, "device": device, "config": {"batch_size": batch_size}}
    if branched_from is not None:
        config["branched_from"] = branched_from
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if rows is not None:
        text = "".join(json.dumps(r) + "\n" for r in rows)
        (run / "metrics.jsonl").write_text(text, encoding="utf-8")
    if nvsmi is not None:
        (run / compute.NVSMI_LOG).write_text(nvsmi, encoding="utf-8")
    return run


def _rows(power=None):
    """Steps 1, 50 and 100 at 256 rows per step: windows of 1, 49 and 50 steps."""
    rows = [
        {"step": 1, "samples_per_s": 256.0},
        {"step": 50, "samples_per_s": 49 * 256 / 10.0},
        {"step": 100, "samples_per_s": 50 * 256 / 20.0},
    ]
    if power is not None:
        rows = [{**r, compute.POWER_FIELD: power} for r in rows]
    return rows


def test_gpu_hours_come_from_the_metrics_windows(tmp_path):
    """Each row closes (step - previous step) steps of batch_size rows at samples_per_s: 1 + 10 + 20 s."""
    run = _run(tmp_path, "a", _rows())
    result = compute.run_compute(run)
    assert result.steps == 100
    assert result.gpu_hours == pytest.approx(31.0 / 3600)
    assert result.kwh is None and result.kwh_source == "none"


def test_kwh_uses_the_metrics_gpu_power_when_every_window_has_it(tmp_path):
    run = _run(tmp_path, "a", _rows(power=200.0))
    result = compute.run_compute(run)
    assert result.kwh_source == "metrics"
    assert result.kwh == pytest.approx(31.0 * 200.0 / 3.6e6)


def test_kwh_falls_back_to_the_nvidia_smi_log(tmp_path):
    log = (
        "timestamp, power.draw [W]\n"
        "2026/09/24 18:00:00.000, 200.00 W\n"
        "2026/09/24 18:00:10.000, 220.00 W\n"
        "2026/09/24 18:00:20.000, [N/A]\n"
        "2026/09/24 18:00:30.000, 220.00 W\n"
        "2026/09/24 19:00:30.000, 220.00 W\n"
    )
    run = _run(tmp_path, "a", _rows(), nvsmi=log)
    result = compute.run_compute(run)
    assert result.kwh_source == "nvidia-smi"
    # 10 s at a mean of 210 W; the N/A sample is dropped (20 s to the next at 220 W); the hour gap is skipped
    assert result.kwh == pytest.approx((10 * 210 + 20 * 220) / 3.6e6)


def test_a_partial_power_log_in_metrics_is_not_trusted(tmp_path):
    rows = _rows()
    rows[2] = {**rows[2], compute.POWER_FIELD: 210.0}
    run = _run(tmp_path, "a", rows)
    assert compute.run_compute(run).kwh is None


def test_cpu_runs_and_folders_without_metrics_are_skipped_with_a_reason(tmp_path):
    _run(tmp_path, "cpu-run", _rows(), device="cpu")
    _run(tmp_path, "empty", None)
    (tmp_path / "loose-file.txt").write_text("x", encoding="utf-8")
    report = compute.project_compute(tmp_path, flagship="long", now="2026-10-07T12:00:00+03:00")
    assert report["runs"] == []
    reasons = {s["run"]: s["reason"] for s in report["skipped"]}
    assert set(reasons) == {"cpu-run", "empty"}
    assert "cpu" in reasons["cpu-run"] and "metrics.jsonl" in reasons["empty"]


def test_compute_json_names_the_flagship_the_total_and_the_kwh_coverage(tmp_path):
    _run(tmp_path, "long", _rows(power=200.0))
    _run(tmp_path, "sweep-s", _rows())
    report = compute.project_compute(tmp_path, flagship="long", now="2026-10-07T12:00:00+03:00")
    hours = 31.0 / 3600
    assert report["flagship"] == "long"
    assert report["flagship_gpu_hours"] == pytest.approx(hours)
    assert report["total_gpu_hours"] == pytest.approx(2 * hours)
    assert report["gpu_board_kwh"] == pytest.approx(31.0 * 200.0 / 3.6e6)
    assert report["kwh_coverage"] == pytest.approx(0.5)
    assert report["generated_at"] == "2026-10-07T12:00:00+03:00"
    assert [r["run"] for r in report["runs"]] == ["long", "sweep-s"]


def test_a_missing_flagship_leaves_its_hours_empty_rather_than_zero(tmp_path):
    _run(tmp_path, "sweep-s", _rows())
    report = compute.project_compute(tmp_path, flagship="long", now="t")
    assert report["flagship_gpu_hours"] is None and report["flagship_kwh"] is None
    assert report["gpu_board_kwh"] is None and report["kwh_coverage"] == 0.0


def test_named_runs_limit_the_total(tmp_path):
    _run(tmp_path, "long", _rows())
    _run(tmp_path, "scratch", _rows())
    report = compute.project_compute(tmp_path, flagship="long", names=["long"], now="t")
    assert [r["run"] for r in report["runs"]] == ["long"]


def test_torn_and_duplicate_metric_lines_count_each_step_once(tmp_path):
    run = _run(tmp_path, "a", _rows())
    with open(run / "metrics.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"step": 100, "samples_per_s": 50 * 256 / 20.0}) + "\n")
        handle.write('{"step": 150, "samp')
    assert compute.run_compute(run).gpu_hours == pytest.approx(31.0 / 3600)


def test_compute_json_round_trips_through_write_and_read(tmp_path):
    _run(tmp_path, "long", _rows(power=200.0))
    report = compute.project_compute(tmp_path, flagship="long", now="t")
    out = compute.write_compute(report, tmp_path / "results" / "compute.json")
    assert compute.read_compute(out) == report


def test_read_compute_refuses_another_schema_version(tmp_path):
    path = tmp_path / "compute.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        compute.read_compute(path)


def test_blink_report_compute_writes_results_compute_json(tmp_path, capsys):
    from blink import cli

    runs = tmp_path / "runs"
    _run(runs, "long", _rows(power=200.0))
    out = tmp_path / "compute.json"
    code = cli.main(["report", "compute", "--runs-root", str(runs), "--flagship", "long", "--out", str(out)])
    assert code == 0
    assert compute.read_compute(out)["flagship"] == "long"
    assert "GPU-h" in capsys.readouterr().out


def test_an_unusable_power_log_or_config_is_skipped_not_fatal(tmp_path):
    run = _run(tmp_path, "a", _rows(), nvsmi="index, memory.used [MiB]\n0, 100 MiB\n")
    assert compute.run_compute(run).kwh is None
    other = tmp_path / "b"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"device": "cuda", "config": {}}), encoding="utf-8")
    (other / "metrics.jsonl").write_text('{"step": 1, "samples_per_s": null}\n', encoding="utf-8")
    assert "batch_size" in compute.run_compute(other).reason


def _steady(first: int, last: int, rate: float = 2048.0) -> list[dict]:
    return [{"step": s, "samples_per_s": rate} for s in range(first, last + 1, 50)]


def test_a_branched_run_bills_only_the_steps_it_trained(tmp_path):
    """A preview cooldown loads ckpt_<N>.pt and counts on from step N: its first window starts at N, not 0."""
    _run(tmp_path, "long", _steady(50, 36_000), batch_size=1024)
    parent_ckpt = str(tmp_path / "long" / "ckpt_000010800.pt")
    preview = _run(
        tmp_path, "long-preview", _steady(10_850, 11_800), batch_size=1024, branched_from=parent_ckpt
    )
    result = compute.run_compute(preview)
    assert result.gpu_hours == pytest.approx(1_000 * 1024 / 2048 / 3600)
    report = compute.project_compute(tmp_path, flagship="long", now="t")
    hours = {r["run"]: r["gpu_hours"] for r in report["runs"]}
    assert hours["long-preview"] == pytest.approx(0.139, abs=1e-3)
    assert report["total_gpu_hours"] == pytest.approx((36_000 + 1_000) * 1024 / 2048 / 3600)


def test_the_start_step_comes_from_the_branch_checkpoint_name(tmp_path):
    assert compute.start_step({}) == 0 and compute.start_step({"branched_from": None}) == 0
    assert compute.start_step({"branched_from": "D:/blink/runs/long/ckpt_000010800.pt"}) == 10_800
    assert compute.start_step({"branched_from": r"C:\runs\long\ckpt_000000030.pt"}) == 30
    with pytest.raises(ValueError, match="branch"):
        compute.start_step({"branched_from": "runs/long/weights.pt"})
    windows = compute.windows([{"step": 150, "samples_per_s": 256.0}], 256, start=100)
    assert windows == [(50.0, None)]


def test_a_branch_whose_checkpoint_name_is_unreadable_is_skipped_not_overbilled(tmp_path):
    run = _run(tmp_path, "odd", _rows(), branched_from="somewhere/weights.pt")
    skipped = compute.run_compute(run)
    assert isinstance(skipped, compute.Skipped) and "branch" in skipped.reason
