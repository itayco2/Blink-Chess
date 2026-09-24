"""`blink site replay --run NAME`: a run's training curves, frozen into static files for Pages (P10).

The live dashboard tails metrics.jsonl and evals.jsonl from a local server; Pages has no server, so the
replay page (site/replay/) reads three static files instead: metrics.jsonl and evals.jsonl decimated to
one row per 2,000 steps (the first row, the first row at or past each multiple of 2,000, and the last
row), and run.json naming the run, its WORLD and how many rows were kept of how many. Non-finite
numbers (a NaN loss) become null, so every line is strict JSON for the browser's JSON.parse.
"""

import json
import math
import re
from pathlib import Path

EVERY = 2000
FILES = ("metrics.jsonl", "evals.jsonl")
RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class ReplayError(RuntimeError):
    pass


def _finite(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def read_rows(path: Path) -> list[dict]:
    """Every complete JSON line of a run log, non-finite floats as None; a torn last line is skipped."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "step" in row:
            rows.append({key: _finite(value) for key, value in row.items()})
    return rows


def decimate(rows: list[dict], every: int = EVERY) -> list[dict]:
    """The first row, the first row in each later block of `every` steps, and the last row."""
    if every < 1:
        raise ValueError(f"every must be at least 1 step, got {every}")
    kept: list[dict] = []
    for row in rows:
        if not kept or row["step"] // every > kept[-1]["step"] // every:
            kept.append(row)
    if rows and kept[-1] is not rows[-1]:
        kept.append(rows[-1])
    return kept


def _write_jsonl(rows: list[dict], path: Path) -> None:
    text = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8", newline="\n")


def run_dir(runs_root: Path, name: str) -> Path:
    if not RUN_NAME.fullmatch(name):
        raise ReplayError(f"refused run name {name!r}: letters, digits, '_', '.', '-' only")
    folder = runs_root / name
    if not (folder / FILES[0]).is_file():
        raise ReplayError(f"no {FILES[0]} in {folder}")
    return folder


def build(run: Path, out: Path, every: int = EVERY) -> dict:
    """Write out/{metrics,evals}.jsonl decimated, and out/run.json; returns the run.json content."""
    config = (
        json.loads((run / "config.json").read_text(encoding="utf-8"))
        if (run / "config.json").is_file()
        else {}
    )
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "run": config.get("run", run.name),
        "world": config.get("world"),
        "parameters": config.get("parameters"),
        "steps": config.get("config", {}).get("steps"),
        "every": every,
    }
    counts = {}
    for name in FILES:
        source = read_rows(run / name) if (run / name).is_file() else []
        kept = decimate(source, every)
        _write_jsonl(kept, out / name)
        stem = name.split(".")[0]
        counts[f"{stem}_rows"], counts[f"source_{stem}_rows"] = len(kept), len(source)
    summary = {
        **summary,
        "metrics_rows": counts["metrics_rows"],
        "evals_rows": counts["evals_rows"],
        "source_metrics_rows": counts["source_metrics_rows"],
        "source_evals_rows": counts["source_evals_rows"],
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary
