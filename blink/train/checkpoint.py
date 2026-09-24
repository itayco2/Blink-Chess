"""Atomic, numerically ordered checkpoints: ckpt_<step:09d>.pt, written as .tmp then os.replace.

A crash between the write and the replace leaves the .tmp behind and the previous checkpoint intact;
readers never list .tmp files, and the next save sweeps them. Loading uses weights_only=True, so a
checkpoint can hold tensors and plain Python values but never arbitrary pickled objects.
"""

import os
import re
import time
from pathlib import Path
from typing import Any

import torch

PATTERN = re.compile(r"^ckpt_(\d+)\.pt$")
REPLACE_RETRIES = 5
RETRY_SLEEP_S = 0.2


def checkpoint_name(step: int) -> str:
    return f"ckpt_{step:09d}.pt"


def step_of(path: Path) -> int:
    match = PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"{path} is not a checkpoint name")
    return int(match.group(1))


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Complete checkpoints in step order (numeric, so step 1,000,000,000 sorts after 999,999,999)."""
    if not Path(run_dir).is_dir():
        return []
    found = [p for p in Path(run_dir).iterdir() if p.is_file() and PATTERN.match(p.name)]
    return sorted(found, key=step_of)


def latest_checkpoint(run_dir: Path) -> Path | None:
    found = list_checkpoints(run_dir)
    return found[-1] if found else None


def _replace_with_retry(tmp: Path, path: Path) -> None:
    """Windows refuses os.replace while a scanner or reader holds the target, so retry briefly."""
    for attempt in range(REPLACE_RETRIES):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == REPLACE_RETRIES - 1:
                raise
            time.sleep(RETRY_SLEEP_S)


def save_checkpoint(run_dir: Path, step: int, state: dict[str, Any], keep_last: int = 3) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale in run_dir.glob("ckpt_*.pt.tmp"):
        stale.unlink(missing_ok=True)
    path = run_dir / checkpoint_name(step)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    _replace_with_retry(tmp, path)
    for old in list_checkpoints(run_dir)[:-keep_last]:
        old.unlink(missing_ok=True)
    return path


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)
