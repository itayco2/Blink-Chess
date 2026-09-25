"""A run closed for good: runs/<run>/finished_by.json, which `blink train` and `blink supervise` honour.

PR-6's finish (tools/p7_finish.py) branches the flagship's final cooldown from runs/long as
runs/long-final. runs/long keeps its checkpoints and logs, but it must never train again: resumed beside
long-final (say by the resume command the P6 v2 driver printed), the two would share the GPU. So the
finish writes this marker into runs/long, and every command that would write that run refuses while it
is there. A branch from the run only reads it and still runs. A marker that cannot be read still closes
the run; removing the file by hand reopens it. Torch-free.
"""

import json
from pathlib import Path
from typing import Any

MARKER = "finished_by.json"


def finished_by(run_dir: Path) -> dict[str, Any] | None:
    """The marker's record; {} when it is there but unreadable; None when the run is open."""
    path = Path(run_dir) / MARKER
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def refusal(run_dir: Path) -> str | None:
    """Why nothing may write this run again, or None when it is open."""
    record = finished_by(run_dir)
    if record is None:
        return None
    name = Path(run_dir).name
    by = record.get("by") or "a finish"
    step = record.get("at_step")
    at = f" at step {step:,}" if isinstance(step, int) else ""
    carried = f"; runs/{record['branch']} carries it on" if record.get("branch") else ""
    return (
        f"runs/{name} was closed by {by}{at}{carried} ({name}/{MARKER}): it never trains again; "
        f"remove that file only to reopen it on purpose"
    )
