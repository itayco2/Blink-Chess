"""PR-5's calibration (EVAL.md section 5): the flagship's steps from its true training rate.

`blink train calibrate --config configs/long.toml --steps 2000 [--write]` trains the real trainer on
that config for 2,000 steps under a throwaway run name, then computes

    R_true = the samples over the wall-clock seconds between consecutive metrics rows whose later
             row has phase "train", leaving out the first 500 steps (an interval counts only when its
             earlier row is at step 500 or later). Seconds come from the rows' `time` stamps and
             samples from their steps (steps x batch_size): never a row's samples_per_s, which leaves
             out each window's GPU drain, and never a bench row. An eval or a checkpoint is charged
             to the window after it, whose row is marked eval or ckpt, so it drops out.
    steps  = floor(120 x 3600 x R_true / 1024)   (1024 is the batch size)

A calibration that did not measure one uninterrupted, unshared run is refused, and --write then writes
nothing (the reason goes to calibration.json): when any counted interval took more than 3x the median
counted interval's seconds per step (the GPU was shared or the run stalled), or when a user pause or
resume, or a restart of the trainer, falls inside the counted span (two trainer processes' rows, told
apart by their session stamp, or a supervisor.json event). The same intervals, from the first row on and
leaving out any that spans a restart, give a run's training hours (PR-6 reports them).

and with --write puts that steps value into the config's [train] table, and nothing else. Every other
number that depends on steps follows from it when the run starts: the 5/25/30/50/100% check steps
(blink.train.vaa.check_steps), so the 30% preview's --from-step; the WSD cooldown start; and the 21 film
frames (blink.train.film.frame_plan). The command prints them. The calibration run has the film off
(its frames would land inside the measured windows) and no 5% reference (not needed in 2,000 steps).
"""

import dataclasses
import math
import re
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink.model.config import TrainConfig, config_to_dict, load_config
from blink.train.atomic import write_text_atomic
from blink.train.schedule import cooldown_start

T_LONG_HOURS = 120.0  # PR-5: fixed, with no adaptive shortening
SKIP_STEPS = 500  # PR-5: the first 500 steps (compile, warm-up of clocks and caches) are left out
CALIBRATION_STEPS = 2000
TRAIN_PHASE = "train"
PREFIX = "calib"
SLOW_FACTOR = 3.0  # a counted interval over 3x the median seconds per step refuses the calibration
SESSION = "session"  # blink.train.loop: the trainer process's stamp on each of its metrics rows
# supervisor.json events (blink.train.supervise) that stop or restart the trainer, by refusal kind
STOP_EVENTS = {
    **dict.fromkeys(("user_pause", "user_pause_exit", "user_pause_kill", "user_resume"), "pause"),
    **dict.fromkeys(("start", "crash", "rollback"), "restart"),
}


class Refused(ValueError):
    """A calibration that must not be used; `kind` is "slow interval", "restart" or "pause"."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind


@dataclass(frozen=True)
class TrueRate:
    samples_per_s: float
    samples: int
    seconds: float
    intervals: int  # consecutive row pairs counted
    first_step: int  # the earlier row of the first interval counted
    last_step: int


def _stamp(row: dict[str, Any]) -> float:
    stamp = row.get("time")
    if isinstance(stamp, bool) or not isinstance(stamp, int | float):
        raise ValueError(f"metrics row at step {row.get('step')} has no time stamp: R_true is read from them")
    return float(stamp)


@dataclass(frozen=True)
class Interval:
    earlier: int  # the earlier row's step
    later: int
    seconds: float  # between the two rows' time stamps
    restarted: bool  # two trainer processes wrote the rows: a pause, a crash restart or a relaunch between

    @property
    def steps(self) -> int:
        return self.later - self.earlier


def _restarted(earlier: dict[str, Any], later: dict[str, Any]) -> bool:
    """Rows of one trainer process share its session stamp; rows from before the stamp existed have none."""
    return earlier.get(SESSION) != later.get(SESSION)


def train_intervals(rows: Sequence[dict[str, Any]], skip_steps: int = 0) -> list[Interval]:
    """Consecutive metrics rows (file order) whose later row has phase "train" and whose earlier row is at
    step `skip_steps` or later: PR-5's intervals, timed by the rows' stamps."""
    out = []
    for earlier, later in zip(rows, rows[1:], strict=False):
        if int(earlier["step"]) < skip_steps or later.get("phase") != TRAIN_PHASE:
            continue
        steps, elapsed = int(later["step"]) - int(earlier["step"]), _stamp(later) - _stamp(earlier)
        if steps <= 0 or elapsed <= 0:
            raise ValueError(
                f"metrics rows at steps {earlier['step']} and {later['step']} do not move forward in steps "
                "and time: the log is not one run's rows in order"
            )
        out.append(Interval(int(earlier["step"]), int(later["step"]), elapsed, _restarted(earlier, later)))
    return out


def training_seconds(rows: Sequence[dict[str, Any]]) -> float:
    """A run's training seconds: its train-phase intervals from the first row on, leaving out any whose
    rows two trainer processes wrote (the time a pause, crash or relaunch took is not training)."""
    return sum(i.seconds for i in train_intervals(rows) if not i.restarted)


def _refuse_restarts(rows: Sequence[dict[str, Any]], first: int, last: int, events: Iterable[dict]) -> None:
    inside = [row for row in rows if first <= int(row["step"]) <= last]
    for earlier, later in zip(inside, inside[1:], strict=False):
        if _restarted(earlier, later):
            raise Refused(
                "restart",
                f"the metrics rows at steps {int(earlier['step']):,} and {int(later['step']):,} come from "
                "two trainer processes (a user pause, a crash restart or a relaunch between them): R_true "
                "is measured on one uninterrupted run",
            )
    opened, closed = _stamp(inside[0]), _stamp(inside[-1])
    for event in events:
        at, name = event.get("time"), event.get("event")
        if name in STOP_EVENTS and isinstance(at, int | float) and opened < at <= closed:
            raise Refused(
                STOP_EVENTS[name],
                f"supervisor.json records {name} at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(at))},"
                f" inside the counted span (steps {first:,}-{last:,}): R_true is measured on one "
                "uninterrupted run",
            )


def _refuse_slow(counted: Sequence[Interval]) -> None:
    per_step = [i.seconds / i.steps for i in counted]
    median = statistics.median(per_step)
    for interval, seconds in zip(counted, per_step, strict=True):
        if seconds > SLOW_FACTOR * median:
            raise Refused(
                "slow interval",
                f"steps {interval.earlier:,}-{interval.later:,} took {seconds:.4f} s a step, over "
                f"{SLOW_FACTOR:g}x the median {median:.4f} s of the counted intervals (the GPU was shared or "
                "the run stalled): calibrate again on an idle GPU",
            )


def true_rate(
    rows: Sequence[dict[str, Any]],
    batch_size: int,
    skip_steps: int = SKIP_STEPS,
    events: Iterable[dict[str, Any]] = (),
) -> TrueRate:
    """R_true over metrics.jsonl rows in file order, exactly as PR-5 defines it; Refused when the run was
    interrupted or shared inside the counted span (`events`: its supervisor.json events, if supervised)."""
    counted = train_intervals(rows, skip_steps)
    if not counted:
        raise ValueError(f"no train-phase interval after step {skip_steps}: calibrate for more steps")
    _refuse_restarts(rows, counted[0].earlier, counted[-1].later, events)
    _refuse_slow(counted)
    samples = sum(i.steps for i in counted) * batch_size
    seconds = sum(i.seconds for i in counted)
    return TrueRate(samples / seconds, samples, seconds, len(counted), counted[0].earlier, counted[-1].later)


def flagship_steps(samples_per_s: float, batch_size: int, hours: float = T_LONG_HOURS) -> int:
    """floor(hours x 3600 x R_true / batch_size): the steps that fill T_long at the true rate."""
    return math.floor(hours * 3600 * samples_per_s / batch_size)


def calibration_config(cfg: TrainConfig) -> TrainConfig:
    """The config the calibration run trains: the flagship's, with its film off and no 5% reference.

    Its steps (so its warmup and LR schedule) stay the flagship's; the run just stops early."""
    return dataclasses.replace(cfg, film=False, vaa_reference="")


def throwaway_name(config: Path, now: float | None = None) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return f"{PREFIX}-{Path(config).stem}-{stamp}"


def derived(steps: int, cooldown_frac: float) -> dict[str, Any]:
    """What follows from steps when the run starts: nothing of it is written by hand."""
    from blink.train import film, vaa

    # as the trainer runs them: {step: label}, so two fractions that round to one step share it
    checks = {label: step for step, label in sorted(vaa.check_steps(steps).items())}
    frames = film.frame_plan(steps)
    return {
        "checks": checks,
        "preview_from_step": checks.get("30%"),
        "cooldown_start": cooldown_start(steps, cooldown_frac),
        "film_frames": len(frames),
        "film_last_ema_step": max((step for step, kind in frames if kind == "ema"), default=None),
    }


def describe(rate: TrueRate, steps: int, batch_size: int, facts: dict[str, Any]) -> list[str]:
    checks = ", ".join(f"{label} {step:,}" for label, step in facts["checks"].items())
    preview = facts["preview_from_step"]
    lines = [
        f"R_true {rate.samples_per_s:,.2f} samples/s: {rate.samples:,} samples in {rate.seconds:,.1f} s over "
        f"{rate.intervals} train-phase intervals, steps {rate.first_step:,}-{rate.last_step:,}",
        f"steps = floor({T_LONG_HOURS:g} x 3600 x {rate.samples_per_s:,.2f} / {batch_size}) = {steps:,}",
        f"checks at {checks}",
    ]
    if preview is not None:
        lines.append(
            f"the 30% preview: --max-steps {preview}, then --preview-cooldown 3h --from-step {preview}"
        )
    last = facts["film_last_ema_step"]
    frames = f"{facts['film_frames']} film frames"
    frames += "" if last is None else f", the last EMA frame at step {last:,}"
    lines.append(f"cooldown from step {facts['cooldown_start']:,}; {frames}")
    return lines


def steps_note(samples_per_s: float, batch_size: int, run: str, now: float | None = None) -> str:
    """The comment --write leaves beside steps: the formula with its R_true, the run and the date."""
    day = time.strftime("%Y-%m-%d", time.localtime(now))
    return f"PR-5: floor({T_LONG_HOURS:g} x 3600 x {samples_per_s:.2f} / {batch_size}), run {run}, {day}"


STEPS_KEY = re.compile(r"^steps\s*=")


def with_steps(text: str, steps: int, note: str) -> str:
    """The TOML text with its [train] table's steps line replaced (comments, every other line and the
    line endings kept)."""
    lines = text.split("\n")
    table = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            table = stripped.split("#", 1)[0].strip()
        elif table == "[train]" and STEPS_KEY.match(stripped):
            ending = "\r" if line.endswith("\r") else ""
            lines[i] = f"{f'steps = {steps}':<23} # {note}{ending}"
            return "\n".join(lines)
    raise ValueError("no `steps = ...` line in the [train] table to replace")


def write_steps(path: Path, steps: int, note: str) -> int:
    """Set steps in a config file and check that nothing else changed; returns the old steps."""
    path = Path(path)
    before = load_config(path)
    text = path.read_bytes().decode("utf-8")  # untranslated, so its line endings survive
    write_text_atomic(path, with_steps(text, steps, note))
    after = load_config(path)
    expected = config_to_dict(dataclasses.replace(before, steps=steps))
    if config_to_dict(after) != expected:
        write_text_atomic(path, text)
        raise ValueError(f"writing steps into {path} changed more than steps; the file was restored")
    return before.steps
