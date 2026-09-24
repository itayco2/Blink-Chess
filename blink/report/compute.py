"""results/compute.json: GPU-hours and GPU-board kWh, measured from each run's own telemetry.

GPU time. Every metrics.jsonl row closes a window of (step - previous step) optimizer steps of
batch_size rows each (roots plus children), logged with that window's samples_per_s, so the window
lasted steps * batch_size / samples_per_s seconds; evaluation and checkpoint time inside a window is
included, because the GPU was held for it. A resume rewinds the log to its checkpoint, so steps a dead
attempt computed and threw away are not counted (the published hours slightly undercount, never
overcount). A branch (a preview cooldown) loads its parent's ckpt_<N>.pt and counts on from step N, so
its first window starts at N, read from config.json "branched_from": the parent's first N steps are
billed once, to the parent. Runs trained on the CPU, folders without a metrics.jsonl and a branch whose
checkpoint name gives no step are listed as skipped.

Energy (GPU board only, never the whole PC). When every window carries gpu_power_w (the mean board
power over that window), energy is the sum of seconds times watts. Otherwise the run's nvidia-smi log
(runs/NAME/nvidia-smi.csv, `nvidia-smi --query-gpu=timestamp,power.draw --format=csv -l 10`) is
integrated with the trapezoid rule, skipping gaps longer than 120 s. A run with neither has no kWh,
and kwh_coverage says what share of the GPU-hours the published kWh covers.
"""

import csv
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath

from blink.train.atomic import write_text_atomic

SCHEMA_VERSION = 1
POWER_FIELD = "gpu_power_w"
NVSMI_LOG = "nvidia-smi.csv"
MAX_GAP_S = 120.0
JOULES_PER_KWH = 3.6e6
SCOPE = "training runs under BLINK_HOME/runs, from each run's own telemetry; GPU board energy only"
_TIME_FORMATS = ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S")
_CHECKPOINT = re.compile(r"^ckpt_(\d+)\.pt$")  # blink.train.checkpoint.PATTERN (that module imports torch)


@dataclass(frozen=True)
class RunCompute:
    run: str
    steps: int
    gpu_hours: float
    kwh: float | None
    kwh_source: str  # "metrics" | "nvidia-smi" | "none"


@dataclass(frozen=True)
class Skipped:
    run: str
    reason: str


def read_metrics(path: Path) -> list[dict]:
    """Complete rows with a step and a positive samples_per_s, one per step (the last wins), in step order."""
    by_step: dict[int, dict] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines(keepends=True):
        if not line.endswith("\n"):
            continue  # a torn last line from a live writer
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("step"), int):
            continue
        rate = row.get("samples_per_s")
        if isinstance(rate, int | float) and rate > 0:
            by_step[row["step"]] = row
    return [by_step[step] for step in sorted(by_step)]


def start_step(config: dict) -> int:
    """The step a run's counter started from: 0, or N for a branch from a ckpt_<N>.pt."""
    source = config.get("branched_from")
    if not source:
        return 0
    match = _CHECKPOINT.match(PureWindowsPath(str(source)).name)  # the name, whichever separator
    if match is None:
        raise ValueError(f"branched_from {source!r} is not a ckpt_<step>.pt, so the branch step is unknown")
    return int(match.group(1))


def windows(rows: list[dict], batch_size: int, start: int = 0) -> list[tuple[float, float | None]]:
    """(seconds, mean board watts or None) for every logged window after the run's start step."""
    out, previous = [], start
    for row in (r for r in rows if r["step"] > start):
        steps = row["step"] - previous
        previous = row["step"]
        power = row.get(POWER_FIELD)
        out.append((steps * batch_size / row["samples_per_s"], float(power) if power is not None else None))
    return out


def _parse_time(text: str) -> datetime | None:
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
    return None


def _parse_watts(text: str) -> float | None:
    try:
        return float(text.strip().removesuffix("W").strip())
    except ValueError:
        return None  # "[N/A]" or a torn line


def _power_samples(path: Path) -> list[tuple[datetime, float]]:
    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle, skipinitialspace=True))
    if not rows:
        return []
    header = [cell.strip().lower() for cell in rows[0]]
    t_col = next((i for i, name in enumerate(header) if name.startswith("timestamp")), None)
    p_col = next((i for i, name in enumerate(header) if name.startswith("power.draw")), None)
    if t_col is None or p_col is None:
        return []  # not a timestamp,power.draw log
    samples = []
    for row in rows[1:]:
        if len(row) <= max(t_col, p_col):
            continue
        when, watts = _parse_time(row[t_col]), _parse_watts(row[p_col])
        if when is not None and watts is not None:
            samples.append((when, watts))
    return sorted(samples)


def nvsmi_kwh(path: Path) -> float | None:
    """Trapezoid-integrated board energy from an nvidia-smi CSV log; None when it has fewer than 2 samples."""
    samples = _power_samples(path)
    if len(samples) < 2:
        return None
    joules = 0.0
    for (t0, w0), (t1, w1) in zip(samples, samples[1:], strict=False):
        gap = (t1 - t0).total_seconds()
        if 0 < gap <= MAX_GAP_S:
            joules += gap * (w0 + w1) / 2
    return joules / JOULES_PER_KWH


def _kwh(run_dir: Path, spans: list[tuple[float, float | None]]) -> tuple[float | None, str]:
    if spans and all(watts is not None for _, watts in spans):
        return sum(s * w for s, w in spans) / JOULES_PER_KWH, "metrics"
    log = run_dir / NVSMI_LOG
    if log.is_file():
        kwh = nvsmi_kwh(log)
        if kwh is not None:
            return kwh, "nvidia-smi"
    return None, "none"


def run_compute(run_dir: Path) -> RunCompute | Skipped:
    run_dir = Path(run_dir)
    config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.jsonl"
    if not config_path.is_file():
        return Skipped(run_dir.name, "no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("device", "cuda") != "cuda":
        return Skipped(run_dir.name, f"trained on {config.get('device')}, not the GPU")
    if not metrics_path.is_file():
        return Skipped(run_dir.name, "no metrics.jsonl")
    batch_size = config.get("config", {}).get("batch_size")
    if not batch_size:
        return Skipped(run_dir.name, "config.json has no config.batch_size")
    try:
        start = start_step(config)
    except ValueError as exc:
        return Skipped(run_dir.name, str(exc))
    rows = read_metrics(metrics_path)
    if not rows:
        return Skipped(run_dir.name, "metrics.jsonl has no complete rows")
    spans = windows(rows, int(batch_size), start)
    kwh, source = _kwh(run_dir, spans)
    return RunCompute(run_dir.name, rows[-1]["step"], sum(s for s, _ in spans) / 3600, kwh, source)


def _run_dirs(runs_root: Path, names: list[str] | None) -> list[Path]:
    if names is not None:
        return [Path(runs_root) / name for name in names]
    return sorted(p for p in Path(runs_root).iterdir() if p.is_dir())


def project_compute(
    runs_root: Path, flagship: str, names: list[str] | None = None, now: str | None = None
) -> dict:
    """The compute.json payload over every run under runs_root (or only `names`)."""
    counted: list[RunCompute] = []
    skipped: list[Skipped] = []
    for run_dir in _run_dirs(runs_root, names):
        result = run_compute(run_dir)
        (counted if isinstance(result, RunCompute) else skipped).append(result)
    total = sum(r.gpu_hours for r in counted)
    powered = [r for r in counted if r.kwh is not None]
    covered = sum(r.gpu_hours for r in powered)
    flag = next((r for r in counted if r.run == flagship), None)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now or datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": SCOPE,
        "flagship": flagship,
        "flagship_gpu_hours": flag.gpu_hours if flag else None,
        "flagship_kwh": flag.kwh if flag else None,
        "total_gpu_hours": total,
        "gpu_board_kwh": sum(r.kwh for r in powered) if powered else None,
        "kwh_gpu_hours": covered,
        "kwh_coverage": covered / total if total > 0 else 0.0,
        "runs": [asdict(r) for r in counted],
        "skipped": [asdict(s) for s in skipped],
    }


def write_compute(report: dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return out


def read_compute(path: Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"compute.json schema {data.get('schema_version')!r}, expected {SCHEMA_VERSION}")
    return data


def summary(report: dict) -> str:
    def hours(value: float | None) -> str:
        return "-" if value is None else f"{value:.3f}"

    kwh = report["gpu_board_kwh"]
    return (
        f"flagship {report['flagship']}: {hours(report['flagship_gpu_hours'])} GPU-h; "
        f"total {hours(report['total_gpu_hours'])} GPU-h over {len(report['runs'])} runs "
        f"({len(report['skipped'])} skipped); GPU-board kWh "
        f"{'-' if kwh is None else f'{kwh:.2f}'} covering {100 * report['kwh_coverage']:.0f}% of the GPU-h"
    )
