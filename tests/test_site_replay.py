"""`blink site replay`: the frozen training replay under site/replay/, one row per 2,000 steps (P10)."""

import json
import re
from pathlib import Path

import pytest

from blink import cli

REPO = Path(__file__).resolve().parent.parent
DASHBOARD_CARDS = re.compile(r'data-chart="([a-z0-9]+)"')


def _rows(steps: list[int], **fields) -> list[dict]:
    return [{"step": step, **fields} for step in steps]


def test_decimation_keeps_the_first_row_one_row_per_2000_steps_and_the_last():
    from blink.site import replay

    rows = _rows([1, *range(50, 6101, 50)])
    assert [r["step"] for r in replay.decimate(rows)] == [1, 2000, 4000, 6000, 6100]
    assert [r["step"] for r in replay.decimate(_rows([0, 1500, 2100, 2500, 3900, 4100]))] == [0, 2100, 4100]
    assert replay.decimate([]) == []
    assert [r["step"] for r in replay.decimate(_rows([7]))] == [7]


def test_decimation_keeps_a_row_per_bucket_even_when_steps_skip_a_whole_bucket():
    from blink.site import replay

    kept = [r["step"] for r in replay.decimate(_rows([0, 1999, 6500, 6600]), every=2000)]
    assert kept == [0, 6500, 6600]


def test_rows_with_nan_or_infinity_become_null_and_torn_lines_are_skipped(tmp_path):
    from blink.site import replay

    path = tmp_path / "metrics.jsonl"
    path.write_text(
        '{"step": 1, "loss_policy": NaN, "lr": Infinity}\n{"step": 2, "loss_policy": 3.5}\n{"step": 3, "lo',
        encoding="utf-8",
    )
    rows = replay.read_rows(path)
    assert rows == [{"step": 1, "loss_policy": None, "lr": None}, {"step": 2, "loss_policy": 3.5}]
    assert all(json.loads(json.dumps(row, allow_nan=False)) == row for row in rows)


def _fake_run(root: Path, name: str = "demo", steps: int = 5000) -> Path:
    run = root / "runs" / name
    run.mkdir(parents=True)
    config = {"run": name, "world": "abc123def456", "parameters": 339_456, "config": {"steps": steps}}
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    metrics = [
        {"step": s, "loss_policy": 7.5 - s / 1000, "loss_value": 4.8} for s in [1, *range(50, steps + 1, 50)]
    ]
    evals = [{"step": s, "top1": s / steps / 3, "ema_top1": s / steps / 3} for s in range(0, steps + 1, 250)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in metrics), encoding="utf-8")
    (run / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in evals), encoding="utf-8")
    return run


def test_site_replay_writes_decimated_metrics_evals_and_a_run_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    _fake_run(tmp_path)
    out = tmp_path / "replay"
    assert cli.main(["site", "replay", "--run", "demo", "--out", str(out)]) == 0
    metrics = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    evals = [json.loads(line) for line in (out / "evals.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["step"] for r in metrics] == [1, 2000, 4000, 5000]
    assert [r["step"] for r in evals] == [0, 2000, 4000, 5000]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run == {
        "run": "demo",
        "world": "abc123def456",
        "parameters": 339_456,
        "steps": 5000,
        "every": 2000,
        "metrics_rows": 4,
        "evals_rows": 4,
        "source_metrics_rows": 101,
        "source_evals_rows": 21,
    }
    assert "ok: 4 metrics rows and 4 eval rows" in capsys.readouterr().out
    assert b"\r\n" not in (out / "metrics.jsonl").read_bytes()


def test_site_replay_refuses_a_run_name_that_could_leave_the_runs_root(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    assert cli.main(["site", "replay", "--run", "../demo", "--out", str(tmp_path / "o")]) == 2
    assert "run name" in capsys.readouterr().err
    assert cli.main(["site", "replay", "--run", "missing", "--out", str(tmp_path / "o")]) == 2
    assert "metrics.jsonl" in capsys.readouterr().err


def test_the_tracked_replay_is_decimated_and_names_its_run():
    from blink.site import replay

    folder = REPO / "site" / "replay"
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    for name in ("metrics.jsonl", "evals.jsonl"):
        steps = [
            json.loads(line)["step"] for line in (folder / name).read_text(encoding="utf-8").splitlines()
        ]
        assert steps == sorted(steps) and len(steps) == run[f"{name.split('.')[0]}_rows"]
        buckets = [step // run["every"] for step in steps[:-1]]
        assert len(buckets) == len(set(buckets)), f"{name} keeps more than one row per {run['every']} steps"
    assert run["every"] == replay.EVERY == 2000
    assert run["run"] and re.fullmatch(r"[0-9a-f]{12}", run["world"])


def test_the_replay_page_draws_the_six_dashboard_cards_from_the_three_static_files():
    live = (REPO / "blink" / "dashboard" / "live.html").read_text(encoding="utf-8")
    page = (REPO / "site" / "replay" / "index.html").read_text(encoding="utf-8")
    script = (REPO / "site" / "replay" / "replay.js").read_text(encoding="utf-8")
    assert DASHBOARD_CARDS.findall(page) == DASHBOARD_CARDS.findall(live)
    for name in ("run.json", "metrics.jsonl", "evals.jsonl"):
        assert f'"{name}"' in script
    assert "/api/" not in script, "the replay is static: no dashboard server behind it"


@pytest.mark.parametrize("every", [0, -5])
def test_decimation_refuses_a_step_count_below_one(every):
    from blink.site import replay

    with pytest.raises(ValueError, match="every"):
        replay.decimate(_rows([1, 2]), every=every)
