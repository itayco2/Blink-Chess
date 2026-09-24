"""What a checkpoint holds and how a run comes back from one.

A checkpoint carries the model, EMA, optimizer, RNG states, schedule, step, world, config, the clip
state (auto clips measure over the warmup), the LR scale in force, and which steps are kept for good.
`restore` continues a run from its own latest checkpoint; `branch` starts a new run directory from
another run's checkpoint (the preview cooldown) and leaves that run untouched.
"""

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blink.model.config import config_to_dict
from blink.train import film, telemetry
from blink.train.checkpoint import latest_checkpoint, load_checkpoint
from blink.train.world import require_same_world

CHECKPOINT_FORMAT = 2


def rng_state(device: torch.device) -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    on_cuda = device.type == "cuda" and torch.cuda.is_initialized()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if on_cuda else [],
        "numpy": [name, keys.tolist(), int(pos), int(has_gauss), float(cached)],
        "python": random.getstate(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    random.setstate(state["python"])


def checkpoint_state(run) -> dict[str, Any]:
    cfg = run.cfg
    return {
        "format": CHECKPOINT_FORMAT,
        "step": run.step,
        "model": run.model.state_dict(),
        "ema": run.ema.state_dict(),
        "optimizer": run.optimizer.state_dict(),
        "schedule": {
            "kind": "wsd",
            "peak_lr": cfg.peak_lr,
            "warmup_steps": cfg.warmup_steps,
            "steps": cfg.steps,
            "cooldown_frac": cfg.cooldown_frac,
            "lr_scale": run.lr_scale,
        },
        "clip": run.clip.state_dict(),
        "lr_scale": run.lr_scale,
        "kept": list(run.kept),
        "kept_wall": run.last_kept_wall,
        "micro_batch": run.micro,
        "rng": rng_state(run.device),
        "world": run.spec.world,
        "config": config_to_dict(cfg),
    }


def _load_into(run, state: dict[str, Any]) -> None:
    require_same_world(found=state["world"], expected=run.spec.world)
    run.model.load_state_dict(state["model"])
    run.ema.load_state_dict(state["ema"])
    run.optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    run.step = int(state["step"])
    run.clip.load_state_dict(state.get("clip", {}), step=run.step)
    run.lr_scale = float(state.get("lr_scale", 1.0)) * run.spec.lr_scale
    if run.spec.lr_scale != 1.0:
        run.log(
            f"lr scale x{run.spec.lr_scale:g} from step {run.step} (the schedule runs at x{run.lr_scale:g})"
        )


def _history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if "check" in row]


def restore(run) -> None:
    """Continue the run from its own latest checkpoint; telemetry and film frames rewind with it."""
    run_dir = run.spec.run_dir
    path = latest_checkpoint(run_dir)
    if path is None:
        raise FileNotFoundError(f"--resume: no checkpoint in {run_dir}")
    state = load_checkpoint(path, map_location="cpu")
    if state["config"] != config_to_dict(run.cfg):
        changed = sorted(k for k, v in config_to_dict(run.cfg).items() if state["config"].get(k) != v)
        run.log(f"warning: config differs from the checkpoint in {changed}; resume will not be bitwise")
    _load_into(run, state)
    run.kept = [int(s) for s in state.get("kept", [])]
    run.last_kept_wall = float(state.get("kept_wall", time.time()))
    run.last_checkpoint = path
    for name in ("metrics.jsonl", "evals.jsonl"):
        telemetry.truncate_after(run_dir / name, run.step)
    film.drop_after(run_dir, run.step)
    run.check_history = _history(run_dir / "evals.jsonl")
    run.log(f"resumed from {path.name} at step {run.step}")


def branch(run, source: Path) -> None:
    """Start this (new) run from another run's checkpoint, e.g. a preview cooldown from step N."""
    if not source.is_file():
        raise FileNotFoundError(f"no checkpoint at {source}")
    _load_into(run, load_checkpoint(source, map_location="cpu"))
    run.log(f"branched from {source.parent.name}/{source.name} at step {run.step}")
