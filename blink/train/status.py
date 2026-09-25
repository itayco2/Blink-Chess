"""Run status from the files a run writes (torch-free: the dashboard and `blink status` use it).

A run is LIVE when its heartbeat says "running" and is at most 30 s old. `exit_code` is 0 for a live
healthy run, a finished one, or one paused by the user (blink.train.userpause) whose supervisor still
beats its heartbeat, and 1 for a stale, crashed or NaN run.

`speed_check` is the P4 speed WARN (a sysmem spill): train-phase samples/s more than 30% below the
median of the last 10 train rows that were not slow, for 5 minutes of wall time. It warns and never
stops anything; the supervisor's throughput stop rule is separate (15% below the benchmark for 10
minutes). live.html's speedCheck is the same rule line for line, and both are held to the cases in
tests/fixtures/dashboard_speed_cases.json.
"""

import json
import math
import re
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink import heartbeat
from blink.train.userpause import FLAG_NAME, PAUSED_USER

LIVE_WITHIN_S = 30
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TAIL_BYTES = 65536
SPEED_DROP = 0.30  # a train row this far below the reference is slow
SPEED_WINDOW_S = 300.0  # slow rows spanning this much wall time warn
SPEED_REFERENCE_ROWS = 10  # the reference: the median of the last 10 train rows that were not slow
QUIET_PHASES = frozenset({"eval", "ckpt"})  # windows whose samples/s is not a training rate


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


def _user_paused(report: RunStatus) -> bool:
    """Paused by the user with a fresh heartbeat: its supervisor waits for the flag, nothing stalled."""
    age = report.heartbeat_age_s
    return report.state == PAUSED_USER and age is not None and age <= LIVE_WITHIN_S


def active(report: RunStatus) -> bool:
    """Training now, or waiting out a user pause under a supervisor that still beats: what `blink status
    --live` (and the Blink Status button) shows."""
    return report.live or _user_paused(report)


def exit_code(report: RunStatus) -> int:
    if _has_nan(report.last_metrics):
        return 1
    return 0 if report.live or report.state == "finished" or _user_paused(report) else 1


def format_status(report: RunStatus) -> str:
    badge = "LIVE" if report.live else report.state.upper()
    age = (
        "no heartbeat" if report.heartbeat_age_s is None else f"heartbeat {report.heartbeat_age_s:.0f} s ago"
    )
    lines = [f"{report.name}: {badge}, step {report.step}/{report.steps}, {age}"]
    if report.state == PAUSED_USER:
        lines.append(f"  paused by the user: Resume Blink (deleting BLINK_HOME/{FLAG_NAME}) lets it continue")
    if report.last_metrics:
        m = report.last_metrics
        lines.append(
            f"  metrics @ {m.get('step')}: policy CE {m.get('loss_policy')}, value CE {m.get('loss_value')}, "
            f"lr {m.get('lr')}, {m.get('samples_per_s')} samples/s, clip {m.get('clip')} "
            f"(clipped {m.get('clip_frac')}) [{m.get('phase', 'train')}]"
        )
    if report.last_eval:
        lines.append(_format_eval(report.last_eval))
    return "\n".join(lines)


def _format_eval(e: dict[str, Any]) -> str:
    text = f"  eval @ {e.get('step')}: top-1 {e.get('top1')}, value CE {e.get('value_ce')}"
    text += f", win% MAE {e.get('win_mae')}"
    if "vaa" in e:
        text += f", VAA {e.get('vaa')} (ema {e.get('ema_vaa')}, {e.get('vaa_set', 'full')})"
    elif "ema_vaa" in e:
        text += f", VAA ema {e.get('ema_vaa')} ({e.get('vaa_set', 'subset')} of {e.get('vaa_n')})"
    if "check" in e:
        text += f", check {e['check']} {_check_verdict(e)}"
    return text


def _check_verdict(e: dict[str, Any]) -> str:
    """As the trainer's log says it (blink.train.evals): a skipped check (the 5% one under P6 v2, whose
    reference is set only by the guard) never reads as passed."""
    if "vaa_check_failed" in e:
        return "FAILED"
    if "check_skipped" in e:
        return f"skipped ({e['check_skipped']})"
    return "passed"


# ---------------------------------------------------------------- the speed WARN (P4)


@dataclass(frozen=True)
class SpeedCheck:
    warn: bool
    reference: float | None  # samples/s the drop is measured against
    rate: float | None  # the latest train-phase samples/s
    slow_s: float  # wall time from the first to the latest row of the current slow stretch


def _number(x: Any) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


def _upper_median(xs: Sequence[float]) -> float:
    """The element at len // 2 of the sorted values: the JS mirror's median, exactly."""
    return sorted(xs)[len(xs) // 2]


def _train_rates(rows: Sequence[dict[str, Any]]) -> Iterator[tuple[float, float]]:
    """(time, samples/s) of every row that is not an eval or checkpoint window and has both numbers."""
    for row in rows:
        rate, at = row.get("samples_per_s"), row.get("time")
        if row.get("phase") not in QUIET_PHASES and _number(rate) and _number(at):
            yield float(at), float(rate)


def speed_check(rows: Sequence[dict[str, Any]]) -> SpeedCheck:
    """Slow rows never move the reference, so a long spill keeps warning; a fast row ends the stretch."""
    fast: list[float] = []
    reference = rate = since = last = None
    for at, rate in _train_rates(rows):
        if reference is not None and rate < (1.0 - SPEED_DROP) * reference:
            since = at if since is None else since
            last = at
            continue
        since = last = None
        fast.append(rate)
        reference = _upper_median(fast[-SPEED_REFERENCE_ROWS:])
    slow_s = 0.0 if since is None else last - since
    return SpeedCheck(slow_s >= SPEED_WINDOW_S, reference, rate, slow_s)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Every complete JSON row of a log (a torn last line from a live writer is skipped)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    rows = []
    for line in text.split("\n")[:-1]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def speed_warning(check: SpeedCheck) -> str | None:
    if not check.warn:
        return None
    return (
        f"  WARN: {check.rate:,.0f} samples/s, over {100 * SPEED_DROP:.0f}% below {check.reference:,.0f} "
        f"for {check.slow_s / 60:.1f} min of training (a sysmem spill?)"
    )
