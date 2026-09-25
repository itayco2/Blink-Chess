"""The Stockfish processes a side job may label on while Blink trains (PR-4 and plan P7).

P8's `eval all` labels on 5 Stockfish processes, one thread each: its idle-machine budget on the
i7-8700's 6 cores. Beside a live blink train, supervise or sweep process (the flagship trains at night,
a sweep between arms is about to start one) a side job gets 3 at most, so the trainer's data loader
keeps its cores. The process list is read the way `blink ops ps` reads it; torch-free.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

TRAINING_SF_PROCS = 3  # PR-4 and plan P7: at most 3 Stockfish processes beside training
TRAINING_COMMANDS = ("train", "supervise", "sweep")  # Blink commands that train or start training runs


def training_processes() -> list[list[str]]:
    """The Blink arguments of every live Blink process (blink.ops.launch, as `blink ops ps` lists them)."""
    from blink.ops import launch

    return [row["args"] for row in launch.blink_processes()]


def trains(args: Sequence[str]) -> bool:
    """Whether Blink arguments train or start training runs (a dry run starts nothing)."""
    return bool(args) and args[0] in TRAINING_COMMANDS and "--dry-run" not in args


def sf_procs_now(requested: int, processes: Callable[[], list[list[str]]] = training_processes) -> int:
    """The processes a job may label on now: at most TRAINING_SF_PROCS while any blink train, supervise
    or sweep process is live, else what was asked. A request of 3 or fewer never lists processes."""
    if requested <= TRAINING_SF_PROCS or not any(trains(args) for args in processes()):
        return requested
    return TRAINING_SF_PROCS


def budgeted(ctx: Any, log: Callable[[str], None], processes=training_processes) -> Any:
    """An eval context (blink.eval.orchestrate.EvalContext) with its sf_procs cut to what the machine
    allows now: `eval all` asks before every block, so a run that starts mid-suite is seen."""
    procs = sf_procs_now(ctx.sf_procs, processes)
    if procs == ctx.sf_procs:
        return ctx
    log(f"blink training is live: SF19 labels on {procs} Stockfish processes, not {ctx.sf_procs} (PR-4)")
    return replace(ctx, sf_procs=procs)
