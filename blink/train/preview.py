"""The preview cooldown: a branch that cools a checkpoint of a live run down to LR 0.

`blink train --run long --preview-cooldown 3h --from-step N` reads runs/long/ckpt_<N>.pt and trains a
new run, runs/long-preview, from it: the same data stream from step N on, with the learning rate
falling from its step-N value to zero along the same 1 - sqrt curve over the preview's steps. The
main run's directory is only read. The preview's step count comes from the duration and the main
run's measured training throughput (the median samples/s of its "train"-phase metrics rows), so the
branch-a-cooldown idea (arXiv 2410.05192) costs a fixed amount of GPU time. `--preview-steps K` gives
the length as exactly K steps instead (a branch that must match another run's schedule step for step),
and `--preview-name BRANCH` writes runs/BRANCH instead of runs/long-preview.
"""

import dataclasses
import json
import re
import statistics
from pathlib import Path
from typing import Any

from blink.model.config import TrainConfig
from blink.train.schedule import cooldown_start

SUFFIX = "-preview"
DURATION = re.compile(r"^(\d+(?:\.\d+)?)\s*([hms])$")
SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0}


def parse_duration(text: str) -> float:
    """'3h', '90m', '2.5h' or '600s' -> seconds."""
    match = DURATION.match(text.strip().lower())
    if match is None:
        raise ValueError(f"bad duration {text!r}: use a number with h, m or s, like 3h or 90m")
    seconds = float(match.group(1)) * SECONDS[match.group(2)]
    if seconds <= 0:
        raise ValueError(f"the duration must be positive, got {text!r}")
    return seconds


def preview_name(run: str) -> str:
    return run + SUFFIX


def training_rate(metrics: list[dict[str, Any]]) -> float:
    """Median samples/s over training-phase rows (the first row, which holds start-up, is skipped)."""
    rates = [
        row["samples_per_s"]
        for row in metrics
        if row.get("step", 0) > 1 and row.get("phase", "train") == "train" and row.get("samples_per_s")
    ]
    if not rates:
        raise ValueError("the run has no training-phase metrics rows to measure its throughput from")
    return statistics.median(rates)


def read_metrics(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir.name} has no metrics.jsonl to measure its throughput from")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preview_steps(seconds: float, samples_per_s: float, batch_size: int) -> int:
    steps = int(seconds * samples_per_s / batch_size)
    if steps < 1:
        raise ValueError(f"{seconds:.0f} s at {samples_per_s:.0f} samples/s is less than one step")
    return steps


def preview_config(cfg: TrainConfig, from_step: int, steps: int) -> TrainConfig:
    """The main config with a 1 - sqrt cooldown that starts exactly at from_step and lasts `steps`."""
    if from_step <= cfg.warmup_steps:
        raise ValueError(f"--from-step {from_step} is inside the {cfg.warmup_steps}-step warmup")
    total = from_step + steps
    preview = dataclasses.replace(cfg, steps=total, cooldown_frac=steps / total, film=False)
    if cooldown_start(total, preview.cooldown_frac) != from_step:
        raise ValueError(f"the preview cooldown would not start at step {from_step}")
    return preview
