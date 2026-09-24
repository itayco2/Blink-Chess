"""The learning film's 21 frames, saved while the run trains.

Frame 1 is step 0 (the network at birth); frames 2-20 are EMA snapshots at
round(250 * (S_final / 250) ** (k / 19)) for k = 0..18, geometric so the fast early learning gets as
many frames as the slow late polish; frame 21 is the final weights (the EMA, with the raw weights
beside it for the section 1 checkpoint rule). Each frame is a slim weights file
runs/NAME/film/frame_<step:09d>.pt that `--model <path>` loads, and records the run's WORLD, so
`frames_world` can prove the whole film comes from one data universe.
"""

import os
import re
from pathlib import Path
from typing import Any

import torch

from blink.train.atomic import replace_with_retry

FIRST_EMA_STEP = 250
EMA_FRAMES = 19
PATTERN = re.compile(r"^frame_(\d+)\.pt$")
DIR = "film"


def frame_plan(s_final: int) -> list[tuple[int, str]]:
    """(step, kind) for every frame of a run planned to end at s_final, in step order."""
    ema = {round(FIRST_EMA_STEP * (s_final / FIRST_EMA_STEP) ** (k / EMA_FRAMES)) for k in range(EMA_FRAMES)}
    return [(0, "init"), *((s, "ema") for s in sorted(ema) if 0 < s < s_final), (s_final, "final")]


def frame_name(step: int) -> str:
    return f"frame_{step:09d}.pt"


def step_of(path: Path) -> int:
    match = PATTERN.match(Path(path).name)
    if match is None:
        raise ValueError(f"{path} is not a film frame name")
    return int(match.group(1))


def list_frames(run_dir: Path) -> list[Path]:
    folder = Path(run_dir) / DIR
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.is_file() and PATTERN.match(p.name)), key=step_of)


def save_frame(
    run_dir: Path, step: int, kind: str, weights: dict, world: str, config: dict, **extra: Any
) -> Path:
    folder = Path(run_dir) / DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / frame_name(step)
    tmp = path.with_name(path.name + ".tmp")
    state = {"step": step, "kind": kind, "world": world, "config": config, "model": weights, **extra}
    with open(tmp, "wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    replace_with_retry(tmp, path)
    return path


def frames_world(frames: list[Path]) -> str:
    """The one WORLD every frame was trained in; ValueError if they disagree or there are none."""
    worlds = {torch.load(p, map_location="cpu", weights_only=True)["world"] for p in frames}
    if len(worlds) != 1:
        raise ValueError(f"film frames come from {len(worlds)} worlds: {sorted(worlds)}")
    return worlds.pop()


def drop_after(run_dir: Path, step: int) -> None:
    """A resume rewinds to `step`: frames a dead attempt saved past it are removed, to be remade."""
    for path in list_frames(run_dir):
        if step_of(path) > step:
            path.unlink(missing_ok=True)
