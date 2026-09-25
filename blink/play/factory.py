"""From a model selector and a mode to an agent.

Selectors follow the one grammar in blink/cli.py: run:<name>[:ema] | ship | release:<tag> | <path>.
The trained-weights loader lives in the train area (blink.model.loading) and is imported only when
a real model is asked for, so the harness and its tests run without torch. `random` (alias
`random-net`) selects the deterministic random-logit evaluator used to test the harness end to end.
"""

import functools
import importlib.util
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from blink.play import fastmode, rules
from blink.play.agents import Agent, PolicyAgent, ValueAgent
from blink.play.budget import DecisionRecord
from blink.play.evaluator import Evaluator
from blink.play.oracles import LADDER_CP_PER_POINT, MaterialEvaluator, RandomLogitEvaluator

RANDOM_SELECTORS = frozenset({"random", "random-net"})
MODES = ("policy", "value")
LOADER = "blink.model.loading"
MATERIAL_NAME = "Material"
MATERIAL = "material"
LEARNED_BASELINES = ("linear", "mlp")
BASELINE_PREFIX = "baseline:"


class ModelUnavailable(RuntimeError):
    """A real model was asked for but the train area's loader cannot be imported."""


def _loader_missing_message(selector: str, exc: BaseException) -> str:
    return (
        f"--model {selector} needs {LOADER} (built by the train area) and it cannot be imported: {exc}. "
        "Use --random (or --model random) to test the harness without a model."
    )


def check_available(selector: str) -> None:
    """Fail fast, before any UCI traffic, when a real model is asked for and the loader is absent."""
    if selector in RANDOM_SELECTORS:
        return
    try:
        spec = importlib.util.find_spec(LOADER)
    except (ModuleNotFoundError, ValueError) as exc:
        raise ModelUnavailable(_loader_missing_message(selector, exc)) from exc
    if spec is None:
        raise ModelUnavailable(_loader_missing_message(selector, ModuleNotFoundError(LOADER)))


def load_evaluator(
    selector: str,
    device: str = "cuda",
    seed: int = 0,
    precision: str = fastmode.DEFAULT_PRECISION,
    compile: bool = False,
) -> Evaluator:
    """A real model in the given play mode (fp32, no compile by default; blink.play.fastmode), or the
    random-logit evaluator, which has no precision or compile of its own. bf16 off CUDA is refused."""
    fastmode.check(precision, device)
    if selector in RANDOM_SELECTORS:
        return RandomLogitEvaluator(seed)
    try:
        from blink.model.loading import load_evaluator as load_model
    except ImportError as exc:
        raise ModelUnavailable(_loader_missing_message(selector, exc)) from exc
    return load_model(selector, device=device, precision=precision, compile=compile)


def material_agent(epsilon: float = rules.DEFAULT_EPSILON) -> ValueAgent:
    """Ladder rung 1: MaterialEvaluator behind the same ValueAgent (rules R1-R5) as every baseline and
    Blink's value mode. Torch-free, so `blink match --a material` never loads torch.

    It ranks children by material however far ahead it is (LADDER_CP_PER_POINT, exact value). With the
    test oracle's settings every lead past about +10 looked the same, and against random the rung let
    pieces go until 15 of 100 dev games were drawn by insufficient material.
    """
    evaluator = MaterialEvaluator(cp_per_point=LADDER_CP_PER_POINT, exact=True)
    return ValueAgent(evaluator, epsilon=epsilon, name=MATERIAL_NAME)


def is_baseline(side: str) -> bool:
    return side == MATERIAL or side in LEARNED_BASELINES or side.startswith(BASELINE_PREFIX)


def baseline_side(side: str, device: str, epsilon: float, seed: int) -> ValueAgent:
    """material, linear, mlp or baseline:<path>, in value mode with the match's epsilon (rule R4).

    The one constructor behind `blink match` and E6's rungs. A baseline's policy is flat, so its R4 ties
    are drawn by (seed, game, position), as MaterialAgent's were. Always taking the lowest vocab index,
    material shuffled its king while random repeated the position: 15 of 20 dev games drawn by threefold
    repetition, material 7 to 15 points up. The learned rungs import torch only when one plays.
    """
    if side == MATERIAL:
        agent = material_agent()
    else:
        from blink.baselines.evaluator import baseline_agent  # torch, only when a learned rung plays

        agent = baseline_agent(side.removeprefix(BASELINE_PREFIX), device=device)
    return replace(agent, epsilon=epsilon, tie_seed=seed)


def make_agent(
    mode: str,
    evaluator: Evaluator,
    epsilon: float = rules.DEFAULT_EPSILON,
    sink: Callable[[DecisionRecord], None] | None = None,
) -> Agent:
    if mode == "policy":
        return PolicyAgent(evaluator, sink=sink)
    if mode == "value":
        return ValueAgent(evaluator, epsilon=epsilon, sink=sink)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


class JsonlSink:
    """Appends one JSON line per decision; the file is opened per record so a crash loses nothing."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, record: DecisionRecord) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.as_dict()) + "\n")


def friendly(command: Callable[..., int]) -> Callable[..., int]:
    """Wrap a CLI command so a missing model loader prints one clear line and exits 2, not a traceback."""

    @functools.wraps(command)
    def run(*args, **kwargs) -> int:
        try:
            return command(*args, **kwargs)
        except ModelUnavailable as exc:
            print(f"blink: {exc}", file=sys.stderr)
            return 2

    return run
