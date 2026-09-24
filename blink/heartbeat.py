"""Liveness files. A long job writes one every few seconds; watchers read its age.

Windows has no pgrep, and os.kill(pid, 0) terminates the process there (PF22), so a
heartbeat file is the only liveness signal Blink trusts.
"""

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPLACE_RETRIES = 5


def write(
    path: Path,
    payload: Mapping[str, Any],
    now: float | None = None,
    retry_sleep: float = 0.05,
) -> None:
    """Atomically replace `path` with the payload plus a `time` field. Never mutates payload.

    On Windows, os.replace fails with PermissionError while another process holds the target open
    without delete-sharing (a tail, an editor, an antivirus scan), so the replace is retried.
    """
    record = {**payload, "time": time.time() if now is None else now}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record), encoding="utf-8")
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(retry_sleep)


def beat_once(path: Path, payload: Mapping[str, Any], retry_sleep: float = 0.05) -> bool:
    """One heartbeat. A locked file costs one missed beat, never the job that is beating."""
    try:
        write(path, payload, retry_sleep=retry_sleep)
        return True
    except OSError as exc:
        print(f"heartbeat: missed a beat on {path}: {exc}", flush=True)
        return False


def read(path: Path) -> dict[str, Any] | None:
    """The heartbeat record, or None if it is missing, half-written or momentarily locked."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def age_seconds(path: Path, now: float | None = None) -> float | None:
    """Seconds since the heartbeat's own timestamp, or None if there is no readable heartbeat."""
    record = read(path)
    if record is None or "time" not in record:
        return None
    return (time.time() if now is None else now) - float(record["time"])


def probe(path: Path, minutes: float, interval: float = 10.0) -> None:
    """Write a heartbeat every `interval` seconds for `minutes`. Used to prove detached jobs survive."""
    start = time.time()
    beat = 0
    while time.time() - start < minutes * 60:
        beat_once(path, {"kind": "probe", "beat": beat, "pid": os.getpid()})
        beat += 1
        time.sleep(interval)
    beat_once(path, {"kind": "probe", "beat": beat, "pid": os.getpid(), "done": True})
