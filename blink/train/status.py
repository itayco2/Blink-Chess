"""Run status from the files a run writes (torch-free: the dashboard and `blink status` use it).

A run is LIVE when its heartbeat says "running" and is at most 30 s old. `exit_code` is 0 for a live
healthy run or a finished one, and 1 for a stale, crashed or NaN run.
"""

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink import heartbeat

LIVE_WITHIN_S = 30
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TAIL_BYTES = 65536


def valid_run_name(name: str) -> bool:
    """Letters, digits, '_', '-', '.', starting alphanumeric: never a path, drive or parent reference."""
    return bool(RUN_NAME.fullmatch(name))


def last_jsonl_record(path: Path) -> dict[str, Any] | None:
    """The last complete JSON line of a file (a torn final line from a live writer is skipped)."""
    try:
        with open(path, "rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - TAIL_BYTES))
            chunk = handle.read()
    except OSError:
        return None
    for line in reversed(chunk.split(b"\n")[:-1]):
        try:
            return json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
    return None


@dataclass(frozen=True)
class RunStatus:
    name: str
    live: bool
    state: str
    step: int | None
    steps: int | None
    heartbeat_age_s: float | None
    heartbeat_time: float | None
    last_metrics: dict[str, Any] | None
    last_eval: dict[str, Any] | None


def run_status(run_dir: Path, now: float | None = None) -> RunStatus:
    now = time.time() if now is None else now
    beat = heartbeat.read(run_dir / "heartbeat.json") or {}
    beat_time = beat.get("time")
    age = None if beat_time is None else now - float(beat_time)
    state = str(beat.get("state", "unknown"))
    return RunStatus(
        name=run_dir.name,
        live=state == "running" and age is not None and age <= LIVE_WITHIN_S,
        state=state,
        step=beat.get("step"),
        steps=beat.get("steps"),
        heartbeat_age_s=age,
        heartbeat_time=beat_time,
        last_metrics=last_jsonl_record(run_dir / "metrics.jsonl"),
        last_eval=last_jsonl_record(run_dir / "evals.jsonl"),
    )


def list_runs(runs_root: Path, now: float | None = None) -> list[RunStatus]:
    if not runs_root.is_dir():
        return []
    runs = [run_status(p, now) for p in runs_root.iterdir() if p.is_dir() and valid_run_name(p.name)]
    return sorted(runs, key=lambda r: r.heartbeat_time or 0.0, reverse=True)


def _has_nan(record: dict[str, Any] | None) -> bool:
    if not record:
        return False
    return any(isinstance(v, float) and not math.isfinite(v) for v in record.values())


def exit_code(report: RunStatus) -> int:
    if _has_nan(report.last_metrics):
        return 1
    return 0 if report.live or report.state == "finished" else 1


def format_status(report: RunStatus) -> str:
    badge = "LIVE" if report.live else report.state.upper()
    age = (
        "no heartbeat" if report.heartbeat_age_s is None else f"heartbeat {report.heartbeat_age_s:.0f} s ago"
    )
    lines = [f"{report.name}: {badge}, step {report.step}/{report.steps}, {age}"]
    if report.last_metrics:
        m = report.last_metrics
        lines.append(
            f"  metrics @ {m.get('step')}: policy CE {m.get('loss_policy')}, value CE {m.get('loss_value')}, "
            f"lr {m.get('lr')}, {m.get('samples_per_s')} samples/s"
        )
    if report.last_eval:
        e = report.last_eval
        lines.append(
            f"  eval @ {e.get('step')}: top-1 {e.get('top1')}, value CE {e.get('value_ce')}, "
            f"win% MAE {e.get('win_mae')}"
        )
    return "\n".join(lines)
