"""Post-hoc arm metrics: games10k_top1 and the mate rates of a finished run's final checkpoint.

Arms a01-a05, a11 and a12 ran from a commit whose checks did not score games10k or the mateset, yet
a07 and a08 are judged against the a01-a03 noise floor of exactly these metrics. `score_run` scores a
finished run's final checkpoint the way its last check row scores a run today (blink.train.checksets:
the raw weights write the plain keys, the EMA the ema_ keys, in the run's own chunk size, so the two
routes give the same numbers) and writes posthoc.json beside evals.jsonl, which it never rewrites.
blink.train.sweep.final_metrics merges the record over the last evals row when both are of the same
step, so decide() reads one row either way.

Only a finished run is scored: its latest checkpoint must be at the planned last step (config.json),
since that is the checkpoint the final check row describes. Scoring is idempotent: a record of the
same checkpoint and the same input files (path and sha1) is kept as it is unless forced, and a new
games10k or mateset file is scored again. Torch is imported only when weights are scored, so the sweep
can read records torch-free.
"""

import hashlib
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink.train.atomic import write_text_atomic

POSTHOC = "posthoc.json"
METRICS = ("games10k_top1", "mate_preserving", "shortest_mate")  # what the arms predating them lack

Log = Callable[[str], None]


class NotFinished(ValueError):
    """The run has no checkpoint at its planned last step, so there is no final model to score."""


@dataclass(frozen=True)
class Weights:
    raw: Any  # torch.nn.Module, in eval mode
    ema: Any
    step: int
    micro: int  # the micro-batch the run trained with: its check rows ran in chunks of twice this


# ---------------------------------------------------------------- records (torch-free)


def read(run_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(run_dir) / POSTHOC).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def merge(row: Mapping[str, Any], record: Mapping[str, Any] | None) -> dict[str, Any]:
    """The final evals row with a post-hoc record's metrics over it, when both describe the same step
    (a record of an older checkpoint describes another model and is left out)."""
    if not record or not row or record.get("step") != row.get("step"):
        return dict(row)
    return {**row, **record.get("metrics", {}), "posthoc": record.get("checkpoint")}


def pack_of(run_dir: Path) -> Path | None:
    """The pack a run trained on (config.json's data dir), whose mateset.npz it is scored on."""
    try:
        record = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    folder = (record.get("data") or {}).get("dir")
    return Path(folder) if folder else None


def fingerprint(path: Path | None) -> dict[str, str] | None:
    """{"path", "sha1"} of an input file, or None when there is none: a changed file is scored again."""
    if path is None or not Path(path).is_file():
        return None
    return {"path": str(Path(path)), "sha1": hashlib.sha1(Path(path).read_bytes()).hexdigest()}


def summary(name: str, record: Mapping[str, Any]) -> str:
    metrics = record.get("metrics", {})
    parts = [f"{name}: step {record['step']:,} ({record['checkpoint']})"]
    for key in METRICS:
        if key in metrics:
            parts.append(f"{key} {metrics[key]:.4f} (ema {metrics['ema_' + key]:.4f})")
        else:
            parts.append(f"{key} not scored")
    return ", ".join(parts)


# ---------------------------------------------------------------- scoring


def planned_steps(run_dir: Path) -> int:
    path = Path(run_dir) / "config.json"
    if not path.is_file():
        raise NotFinished(f"{Path(run_dir).name} has no config.json: it never started")
    return int(json.loads(path.read_text(encoding="utf-8"))["config"]["steps"])


def final_checkpoint(run_dir: Path) -> Path:
    """The checkpoint at the run's planned last step; NotFinished for a run that has not reached it."""
    from blink.train.checkpoint import latest_checkpoint, step_of

    run_dir = Path(run_dir)
    latest = latest_checkpoint(run_dir)
    if latest is None:
        raise NotFinished(f"{run_dir.name} has no checkpoint in {run_dir}")
    steps = planned_steps(run_dir)
    if step_of(latest) < steps:
        raise NotFinished(
            f"{run_dir.name}: its latest checkpoint is step {step_of(latest):,} of {steps:,}; "
            "only a finished run is scored"
        )
    return latest


def load_weights(path: Path, device: str) -> Weights:
    """The raw and EMA models of one training checkpoint, in eval mode on `device`."""
    import torch

    from blink.model.config import config_from_dict
    from blink.model.transformer import BlinkNet
    from blink.train.checkpoint import load_checkpoint

    state = load_checkpoint(path, map_location="cpu")
    missing = [key for key in ("model", "ema", "config", "step") if key not in state]
    if missing:
        raise ValueError(f"{Path(path).name} is not a training checkpoint: it has no {', '.join(missing)}")
    cfg = config_from_dict(state["config"])
    models = []
    for key in ("model", "ema"):
        model = BlinkNet(cfg.model)
        model.load_state_dict(state[key])
        models.append(model.to(torch.device(device)).eval())
    micro = int(state.get("micro_batch") or cfg.batch_size)
    return Weights(models[0], models[1], int(state["step"]), micro)


def _score(
    checkpoint: Path, inputs: dict, games10k: Path | None, mateset: Path | None, device: str, log: Log
):
    import torch

    from blink.train import checksets, evals

    started = time.perf_counter()
    weights = load_weights(checkpoint, device)
    sets = checksets.for_run(games10k, mateset)
    chunk = evals.check_chunk(weights.micro)
    metrics = checksets.score(weights.raw, weights.ema, torch.device(device), sets, log, chunk)
    return {
        "step": weights.step,
        "checkpoint": checkpoint.name,
        "inputs": inputs,
        "device": device,
        "metrics": metrics,
        "score_s": time.perf_counter() - started,
        "scored": time.time(),
    }


def score_run(
    run_dir: Path,
    games10k: Path | None,
    mateset: Path | None,
    device: str = "cuda",
    log: Log = print,
    force: bool = False,
) -> dict[str, Any]:
    """Score a finished run's final checkpoint on games10k and the mateset into its posthoc.json.

    NotFinished for a run without its final checkpoint; FileNotFoundError when neither file exists."""
    run_dir = Path(run_dir)
    checkpoint = final_checkpoint(run_dir)
    inputs = {
        "checkpoint": checkpoint.name,
        "games10k": fingerprint(games10k),
        "mateset": fingerprint(mateset),
    }
    if inputs["games10k"] is None and inputs["mateset"] is None:
        raise FileNotFoundError(
            f"nothing to score for {run_dir.name}: no games10k at {games10k} and no mateset at {mateset}"
        )
    kept = read(run_dir)
    if not force and kept is not None and kept.get("inputs") == inputs:
        log(f"{run_dir.name}: already scored from {checkpoint.name} (see {POSTHOC}; --force scores again)")
        return kept
    record = _score(checkpoint, inputs, games10k, mateset, device, log)
    write_text_atomic(run_dir / POSTHOC, json.dumps(record, indent=2) + "\n")
    log(f"{summary(run_dir.name, record)} in {record['score_s']:.1f} s -> {run_dir / POSTHOC}")
    return record
