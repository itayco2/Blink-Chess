"""The dm:<size>[:ema] model selector: DeepMind's released checkpoints as agents (plan E0 and E7).

    dm:9M       BLINK_HOME/dm/9M-params.npz       `params` at step 6.4M, what DeepMind's engines load
    dm:9M:ema   BLINK_HOME/dm/9M-params_ema.npz   `params_ema`, what DeepMind's notebook loads

tools/dm_convert.py writes both files once, in the throwaway JAX venv. A dm selector gives a
DeepMindAgent, not an Evaluator: DeepMind's model reads (FEN, move) rows, so it has one mode,
"action-value", instead of Blink's policy and value modes. `blink eval puzzles`, `blink-uci` (and so
`blink gauntlet`) and `blink match` route a dm selector here. This module imports no torch; the model
is imported only when an agent is loaded.
"""

from dataclasses import dataclass
from pathlib import Path

from blink import paths
from blink.play.agents import Agent
from blink.play.factory import ModelUnavailable

PREFIX = "dm"
SIZES = ("9M",)
EMA = "ema"
MODE = "action-value"
CONVERTER = "tools/dm_convert.py"
GRAMMAR = "dm:9M[:ema]"


@dataclass(frozen=True)
class DmSelector:
    size: str
    kind: str  # params | params_ema

    @property
    def name(self) -> str:
        return f"DM-{self.size}" + ("-ema" if self.kind == "params_ema" else "")


def is_dm(selector: str) -> bool:
    return selector.split(":", 1)[0] == PREFIX


def parse(selector: str) -> DmSelector:
    parts = selector.split(":")
    if (
        parts[0] != PREFIX
        or len(parts) not in (2, 3)
        or parts[1] not in SIZES
        or parts[2:] not in ([], [EMA])
    ):
        raise ValueError(
            f"bad DeepMind selector {selector!r}: use {GRAMMAR} (only {', '.join(SIZES)} is ported)"
        )
    return DmSelector(parts[1], "params_ema" if parts[2:] else "params")


def weights_path(selector: str | DmSelector) -> Path:
    parsed = parse(selector) if isinstance(selector, str) else selector
    return paths.home() / "dm" / f"{parsed.size}-{parsed.kind}.npz"


def check_available(selector: str) -> None:
    """Fail fast, before any UCI traffic or puzzle, when the selector is bad or its weights are absent."""
    try:
        path = weights_path(selector)
    except ValueError as exc:
        raise ModelUnavailable(str(exc)) from exc
    if not path.is_file():
        raise ModelUnavailable(
            f"--model {selector} needs {path}; convert DeepMind's checkpoint once with "
            f"{CONVERTER} (run it with the D: conversion venv, not the project venv)"
        )


def load_agent(selector: str, device: str = "cuda", sink=None) -> Agent:
    """The DeepMindAgent for a dm selector, on the ported model in fp32 on `device`."""
    check_available(selector)
    parsed = parse(selector)
    try:
        from blink.reference import agent
    except ImportError as exc:
        raise ModelUnavailable(f"--model {selector} needs torch (the train group): {exc}") from exc
    return agent.load_agent(weights_path(parsed), parsed.size, device=device, sink=sink, name=parsed.name)
