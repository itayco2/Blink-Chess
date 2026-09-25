"""P6 v2's flagship choreography (EVAL.md PR-2), run detached: calibrate, leg 1, the size-m branch, the
guard, the N* record, then the flagship resumed with size-m as its reference, trained under PR-6.

Run it ONLY after P5 has ended (PR-2 was adopted on 2026-09-25), the recipe is frozen and EVAL.md v1 is
tagged, from the worktree the flagship trains in (it must hold the p6v2, p7prep and p7final merges).
Launch it through WMI so it outlives the agent session (PF38), for example:

    powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create
      -Arguments @{CommandLine='C:\\dev\\blink-chess\\.venv\\Scripts\\python.exe tools\\p7_v2_driver.py';
      CurrentDirectory='C:\\dev\\blink-run'}"

With no flags it branches size-m at PR-2's step 47,301 over 11,825 steps; rerun it with the same flags.
`--dry-run --rate R` prints the plan and every command and runs nothing. One instance runs at a time
(logs/p7v2.lock holds its pid and create time; a stale lock is replaced). Each step logs to
<home>/logs/p7v2-<step>.out|err; the state goes to p7v2.status.json (and p7v2.state.json).

The PC is Itay's during the day (PR-6): no step starts while BLINK_HOME/PAUSE is up (the Pause Blink
button), and a heartbeat or supervisor.json saying "paused: user" is alive, never a failure, and never
counted toward a timeout. The supervised steps wait out a pause inside `blink supervise` itself.

0. preflight: no blink train/supervise/sweep process runs; configs/sweep.toml's rule is "prior";
   configs/long.toml's vaa_sigma is sigma_EMA, the sample sd of a01-a03's 100% check EMA VAA (what the
   guard computes), to its printed precision; --preview-steps/--preview-name exist; `sweep choose` into a
   scratch file says N* = m. Nothing trains when any of these fails.
1. calibrate (PR-5): the endgame screen is stopped (PR-4: never during a calibration), then `blink
   train calibrate --config configs/long.toml --steps 2000 --write` measures R_true and writes
   long.toml's steps = floor(120 x 3600 x R_true / 1024). The driver reads R_true from its output
   and checks those steps; it fails clearly when the command is missing. The calibration runs with
   BLINK_PAUSE_RESUMER=1 (this driver reruns a paused one), so Pause Blink stops it at its next step and
   frees the GPU. A user pause inside the calibration (it exits 75, or its calibration.json or metrics
   rows show a pause or restart) is never used: the driver waits for Resume and calibrates again as a
   fresh run.
2. plan: size-m is PR-2's literal 59,126-step M run (adopted 2026-09-25: "branched from the flagship at
   step 47,301"), cooling down over round(0.2 x 59,126) = 11,825 steps from 47,301
   (blink.train.schedule), whatever R_true is. 6 h at R_true, floor(6 x 3600 x R_true / 1024) steps
   (as `blink sweep sizes` sizes a run), is only printed beside it: 59,126 at the bench's 2,803.05
   samples/s, 55,296 at a true ~2,621. `--rung-from-rate` uses that instead and `--rung-steps N` any
   other length; the choice goes to p7v2.state.json and eval/size_guard.json. The cooldown start must
   fall inside the flagship's stable phase and before its 5% check, and vaa_reference must be "" until
   the guard.
3. leg 1: `supervise -- train --config configs/long.toml --run long --data DATA --max-steps START`.
4. branch: `supervise -- train --run long --data DATA --from-step START --preview-steps COOLDOWN
   --preview-name size-m`; it inherits vaa_reference "" so its end check is skipped.
5. guard (PR-2 (3)): Delta = size-m's final EMA VAA - the mean final EMA VAA of a01-a03, all on the full
   valprobe. If Delta < -2 sigma_EMA (their sample sd), the flagship stays paused at START: runs/long's
   heartbeat says "paused: P7-VAA" (gate P7-VAA) and the driver stops. The numbers, and long.toml's
   vaa_sigma beside sigma_EMA, go to eval/size_guard.json either way.
6. `blink sweep choose` (rule prior) records the floor, VRAM and p99 facts in eval/sweep.json.
7. long.toml's vaa_reference = "size-m"; the flagship is relaunched detached (`ops launch --name
   p7-long -- supervise -- train ... --max-steps P --resume`) and the priority keeper restarted. P is the
   30% check (blink.train.vaa.check_steps), where the plan's P7 pauses the long run for the 3 GPU-h
   preview (size-m is its reference, PR-2 (2)). The driver is done only once runs/long's heartbeat says
   running (or "paused: user") past step START, polled for up to 10 minutes of unpaused time; otherwise
   it fails with the supervisor's state.

After the driver, PR-6 sets the flagship's length: it trains whenever Itay does not need the GPU (always
at night) and pauses when he asks (Pause Blink / Resume Blink; Blink Status shows the state). At each
pause, or at least daily, `blink eval strength --run long` scores the latest EMA checkpoint on the first
2,000 DeepMind puzzles beside DM-9M's 86.6% and the earlier checks; PR-3's parity and soak run at the
first check after 24 training hours (the command says when), not at the 30% preview. When Itay says the
level is good enough, `python tools\\p7_finish.py` re-plans the run: the final 20% 1-sqrt cooldown
branches from the current step as run long-final. long.toml's steps (PR-5's 120 h) stay the upper bound.

Stdlib only (with psutil): it imports nothing from the repo, so code changes cannot reach it mid-run
(tools/p7_machine.py and tools/p7_guard.py are loaded at its start). A rerun (after a reboot, say)
picks up where the files say it stopped: checkpoints, size-m's final row, the state file. It never
calibrates once leg 1 has a checkpoint, refuses a plan other than the recorded one from then on (rerun
it with the first launch's flags), and does nothing once the flagship was launched.
"""

import argparse
import json
import math
import re
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from p7_guard import PAUSED_VAA, check_sigma, final_full_row, guard, guard_verdict, mark_paused  # noqa: F401
from p7_machine import (  # noqa: F401 - the tests and p7_finish read these from here
    EXIT_USER_PAUSE,
    KEEPER,
    PAUSED_USER,
    RESUMER_ENV,
    Host,
    LockHeld,
    StepFailed,
    acquire_lock,
    checkpoint_name,
    checkpoint_steps,
    flag_path,
    parse_number,
    read_json,
    read_json_or_empty,
    read_rows,
    release_lock,
    run_state,
    set_train_string,
    train_literal,
    train_table,
    user_paused,
    wait_while_flagged,
    write_atomic,
)

EXIT_DONE, EXIT_FAILED, EXIT_REFUSED, EXIT_PAUSED = 0, 1, 2, 3
CALIBRATE = ("train", "calibrate")
BUSY = re.compile(r"blink\.cli\s+(train|supervise|sweep)\b")
RATE = re.compile(r"\bR[_ ]?true\b[^0-9\n]{0,24}([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
N_STAR = re.compile(r"^N\* = (\S+)", re.MULTILINE)
CALIB_RUN = re.compile(r"\bas run (\S+)")  # blink train calibrate's first line names its run
FIRST_CHECK = 0.05  # the flagship's first check (blink.train.vaa.CHECK_FRACS)
PREVIEW_CHECK = 0.30  # the check the long run pauses at for the 3 GPU-h preview (plan P7)
PR2_RUNG_STEPS = 59_126  # EVAL.md PR-2: size-m is a 59,126-step M run, branched at 47,301
PLAN_STEPS = ("batch", "long_steps", "rung_steps", "rung_cooldown", "rung_start", "first_check",
              "preview_step")  # the plan's numbers a rerun must keep  # fmt: skip
MAX_CALIBRATIONS = 20  # calibrations a row of user pauses may cost before the driver gives up
VERIFY_S = 600.0  # unpaused seconds the relaunched flagship has to train past the rung start
VERIFY_POLL_S = 10.0
ENDED = ("finished", "stopped", "paused")  # supervisor.json states after which nothing trains


@dataclass(frozen=True)
class Settings:
    repo: Path  # the worktree the flagship trains in (python -m blink.cli runs from here)
    python: Path
    home: Path  # BLINK_HOME
    data: Path
    config: str = "configs/long.toml"  # relative to repo, as blink commands take it
    sweep_config: str = "configs/sweep.toml"
    run: str = "long"
    branch: str = "size-m"
    size: str = "m"  # the prior's N*
    arms: tuple[str, ...] = ("a01", "a02", "a03")  # the noise floor's seeds
    calib_steps: int = 2000
    long_hours: float = 120.0  # T_long (PR-5)
    rung_hours: float | None = None  # None: configs/sweep.toml [sizes] hours
    rung_steps: int | None = PR2_RUNG_STEPS  # None: floor(rung_hours x 3600 x R_true / batch)
    sigma_factor: float = 2.0
    bench_size: str = "m"  # supervise's throughput benchmark; "" turns the rule off
    keeper: Path | None = None
    launch_name: str = "p7-long"
    rate: float | None = None  # a calibrated R_true to use instead of calibrating
    rate_eps: float = 0.0  # half the last printed digit of `rate`

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def runs(self) -> Path:
        return self.home / "runs"

    @property
    def config_path(self) -> Path:
        return self.repo / self.config

    def path(self, name: str) -> Path:
        return self.logs / f"p7v2{name}"


@dataclass(frozen=True)
class Plan:
    rate: float  # R_true, samples/s
    batch: int
    long_steps: int  # the flagship: floor(T_long x 3600 x R_true / batch)
    rung_steps: int  # size-m's total: PR-2's 59,126 unless the settings say otherwise
    rung_cooldown: int
    rung_start: int  # leg 1 stops here; the branch cools down from here
    first_check: int  # the flagship's 5% check
    preview_step: int  # the flagship's 30% check: the relaunch stops there for the preview
    rung_rule: str  # where rung_steps came from
    rung_hours: float  # M's rung in hours at R_true (configs/sweep.toml [sizes] hours)
    rate_rung_steps: int  # floor(rung_hours x 3600 x R_true / batch): the cross-check beside a fixed rung

    def hours(self, steps: int) -> float:
        return steps * self.batch / self.rate / 3600


# ---------------------------------------------------------------- arithmetic


def steps_for(hours: float, rate: float, batch: int) -> int:
    """Optimizer steps filling `hours` at `rate` samples/s: blink.train.sweep.steps_for, which is also
    PR-5's floor(T_long x 3600 x R_true / 1024)."""
    return int(hours * 3600 * rate // batch)


def cooldown_start(total: int, frac: float) -> int:
    """blink.train.schedule.cooldown_start: where a `total`-step WSD run starts cooling down."""
    return total - int(round(frac * total))


def check_step(frac: float, total: int) -> int:
    """blink.train.vaa.check_steps: the step of the check at `frac` of `total`."""
    return max(1, math.floor(frac * total + 0.5))


def rung_rule(rung_steps: int | None, rung_hours: float) -> str:
    if rung_steps is None:
        return f"{rung_hours:g} h at R_true"
    if rung_steps == PR2_RUNG_STEPS:
        return f"EVAL.md PR-2's {rung_steps:,} steps"
    return f"--rung-steps {rung_steps:,}"


def make_plan(
    rate: float, train: dict[str, Any], long_hours: float, rung_hours: float, rung_steps: int | None = None
) -> Plan:
    """The flagship's steps and size-m's rung, from the calibrated rate and the config's schedule.

    `rung_steps` (PR-2's literal 59,126, say) replaces the rung's length at `rung_hours` of R_true, which
    is then only a cross-check; its cooldown and start still follow from the schedule."""
    batch, frac = int(train["batch_size"]), float(train["cooldown_frac"])
    long_steps = steps_for(long_hours, rate, batch)
    derived = steps_for(rung_hours, rate, batch)
    rung = rung_steps if rung_steps is not None else derived
    start = cooldown_start(rung, frac)
    first = check_step(FIRST_CHECK, long_steps)
    if start <= int(train["warmup_steps"]):
        raise ValueError(
            f"the rung's cooldown start {start:,} is inside the {train['warmup_steps']}-step warmup"
        )
    if start >= first or start >= cooldown_start(long_steps, frac):
        raise ValueError(
            f"the rung's cooldown start {start:,} is not before the flagship's 5% check {first:,}"
        )
    return Plan(rate=rate, batch=batch, long_steps=long_steps, rung_steps=rung, rung_cooldown=rung - start,
                rung_start=start, first_check=first, preview_step=check_step(PREVIEW_CHECK, long_steps),
                rung_rule=rung_rule(rung_steps, rung_hours), rung_hours=rung_hours,
                rate_rung_steps=derived)  # fmt: skip


def read_rate(output: str) -> tuple[float, float] | None:
    """The last R_true a calibration printed, with its printed precision, or None."""
    found = RATE.findall(output)
    return parse_number(found[-1]) if found else None


def steps_agree(rate: float, eps: float, hours: float, batch: int, steps: int) -> bool:
    """Whether `steps` is floor(hours x 3600 x R / batch) for some R within the printed rate's precision."""
    return steps_for(hours, rate - eps, batch) <= steps <= steps_for(hours, rate + eps, batch)


def pinned_rate(rate: float, steps: int, hours: float, batch: int) -> float:
    """The printed rate moved onto the rates whose floor gives the written `steps`: the plan's flagship is
    then long.toml's to the step, and the rung the exact rate's (6 h is 1/20 of 120 h, so floor(6 h x R)
    is the same for every R that gives those steps)."""
    low = steps * batch / (hours * 3600)
    while steps_for(hours, low, batch) < steps:
        low = math.nextafter(low, math.inf)
    high = (steps + 1) * batch / (hours * 3600)
    while steps_for(hours, high, batch) > steps:
        high = math.nextafter(high, -math.inf)
    return min(max(rate, low), high)


# ---------------------------------------------------------------- state


def status(s: Settings, state: str, step: str, detail: str = "") -> None:
    record = {"state": state, "step": step, "detail": detail, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_atomic(s.path(".status.json"), json.dumps(record, indent=1) + "\n")


def load_state(s: Settings) -> dict[str, Any]:
    path = s.path(".state.json")
    return read_json(path) if path.is_file() else {}


def save_state(s: Settings, state: dict[str, Any]) -> None:
    write_atomic(s.path(".state.json"), json.dumps(state, indent=1) + "\n")


def begin(s: Settings, host, step: str, detail: str = "") -> None:
    """Start a step, once BLINK_HOME/PAUSE is gone: while it is up the status says so and nothing starts."""

    def waiting() -> None:
        status(s, PAUSED_USER, step, f"BLINK_HOME/{flag_path(s.home).name} is up: {step} starts once "
                                     "Resume Blink removes it")  # fmt: skip

    wait_while_flagged(s.home, host, waiting)
    status(s, "running", step, detail)


def busy_commands(lines: list[str]) -> list[str]:
    """The command lines of blink processes that hold or start GPU training."""
    return [line for line in lines if BUSY.search(line)]


def ensure_keeper(s: Settings, host) -> None:
    """The priority keeper exits after 30 idle minutes; start it unless one runs."""
    if s.keeper is not None and not any(KEEPER in line for line in host.command_lines()):
        host.start_keeper(s.keeper)


# ---------------------------------------------------------------- the steps


def supervise(s: Settings) -> list[str]:
    return ["supervise", *(["--bench-size", s.bench_size] if s.bench_size else []), "--"]


def leg_one_args(s: Settings, plan: Plan, resume: bool) -> list[str]:
    args = ["train", "--config", s.config, "--run", s.run, "--data", str(s.data)]
    return [*supervise(s), *args, "--max-steps", str(plan.rung_start), *(["--resume"] if resume else [])]


def branch_args(s: Settings, plan: Plan, resume: bool) -> list[str]:
    args = ["train", "--run", s.run, "--data", str(s.data), "--from-step", str(plan.rung_start)]
    args += ["--preview-steps", str(plan.rung_cooldown), "--preview-name", s.branch]
    return [*supervise(s), *args, *(["--resume"] if resume else [])]


def launch_args(s: Settings, plan: Plan | None = None) -> list[str]:
    """The flagship relaunched detached: to the 30% check with the plan (the preview pause), to the end
    without one (the resume after the preview)."""
    train = ["train", "--config", s.config, "--run", s.run, "--data", str(s.data)]
    stop = ["--max-steps", str(plan.preview_step)] if plan is not None else []
    return ["ops", "launch", "--name", s.launch_name, "--", *supervise(s), *train, *stop, "--resume"]


def preview_args(s: Settings, plan: Plan) -> list[str]:
    """Plan P7's 3 GPU-h preview cooldown from the 30% checkpoint (p7prep's configs/long.toml, step 2)."""
    return ["train", "--run", s.run, "--data", str(s.data), "--preview-cooldown", "3h",
            "--from-step", str(plan.preview_step)]  # fmt: skip


def after_launch(s: Settings, plan: Plan) -> str:
    """What follows the driver: PR-6's pauses, strength checks and finish; the 30% preview if reached."""
    blink = "python -m blink.cli"
    pr6 = (
        f"it trains under PR-6: pause it with Pause Blink, check it with `{blink} eval strength --run "
        f"{s.run}` at each pause, and when the level is good enough re-plan it with `python "
        "tools\\p7_finish.py`"
    )
    preview = f"`{blink} {' '.join(preview_args(s, plan))}`"
    return (
        f"{pr6}. If it reaches the 30% check, step {plan.preview_step:,}, it stops: run the 3 GPU-h preview "
        f"{preview}, then resume without --max-steps: `{blink} {' '.join(launch_args(s))}` (refused once "
        f"p7_finish closes runs/{s.run}: its finish is final)"
    )


def locked_plan(s: Settings, state: dict[str, Any], plan: Plan) -> dict[str, Any]:
    """The state with this plan, which must be the recorded one once leg 1 has a checkpoint: a rerun under
    other flags would otherwise move leg 1's stop and the branch point."""
    recorded = state.get("plan")
    if recorded and checkpoint_steps(s.runs / s.run):
        current = asdict(plan)
        differ = [key for key in PLAN_STEPS if recorded.get(key) != current[key]]
        if differ:
            was = f"size-m {recorded.get('rung_steps', 0):,} = {recorded.get('rung_start', 0):,} + "
            was += f"{recorded.get('rung_cooldown', 0):,} ({recorded.get('rung_rule', '?')})"
            raise StepFailed(
                "plan",
                f"runs/{s.run} was started under the recorded plan, {was}, flagship "
                f"{recorded.get('long_steps', 0):,}; this command's plan differs in {', '.join(differ)}: "
                "rerun with the first launch's --rung-steps/--rung-from-rate/--rung-hours/--long-hours",
            )
    return {**state, "plan": asdict(plan)}


def choose_n_star(s: Settings, host, step: str, extra: list[str]) -> str:
    code, out = host.run(step, ["sweep", "choose", "--config", s.sweep_config, *extra])
    found = N_STAR.findall(out)
    if code != 0 or not found or found[-1] != s.size:
        why = f"N* = {found[-1]}" if found else f"exit {code}"
        raise StepFailed(
            step, f"`sweep choose` under the prior gave {why}, not {s.size}: see p7v2-{step}.out"
        )
    return found[-1]


def preflight(s: Settings, host) -> None:
    begin(s, host, "preflight")
    busy = busy_commands(host.command_lines())
    if busy:
        raise StepFailed("preflight", f"a blink process holds the GPU, so nothing started: {busy[0][:200]}")
    rule = tomllib.loads((s.repo / s.sweep_config).read_text(encoding="utf-8")).get("choose", {}).get("rule")
    if rule != "prior":
        raise StepFailed("preflight", f"{s.sweep_config} rule is {rule!r}, not 'prior': P6 v2 is not merged")
    check_sigma(s)
    _, text = host.run("probe-branch", ["train", "--help"])
    if "--preview-steps" not in text or "--preview-name" not in text:
        raise StepFailed("preflight", "`blink train` has no --preview-steps/--preview-name: merge p7prep")
    choose_n_star(s, host, "preflight-choose", ["--sweep", str(s.path("-preflight-sweep.json"))])


def check_reference(s: Settings) -> None:
    """The calibration, leg 1 and the branch load vaa_reference at startup: it stays "" until the guard
    sets it, which happens only once the branch has its final row."""
    reference = train_table(s.config_path).get("vaa_reference", "")
    if not reference:
        return
    try:
        final_full_row(s.runs / s.branch)
    except (OSError, ValueError, KeyError):
        raise StepFailed("preflight", f"{s.config} vaa_reference is {reference!r} before the branch exists: "
                         'set it to "" (the guard sets it)') from None  # fmt: skip


def calibrate_supported(help_text: str) -> bool:
    """Whether `blink train calibrate --help` came from a CLI that has it (p7prep makes calibrate an
    optional positional of `train`, shown as {calibrate}); an older `train` prints its own help."""
    named = "{calibrate}" in help_text or "train calibrate" in help_text
    return named and "--write" in help_text


def calibration_run(s: Settings, out: str) -> Path | None:
    """The calibration's run directory, from the first line `blink train calibrate` prints."""
    named = CALIB_RUN.findall(out)
    return s.runs / named[0] if named else None


def refusal(run_dir: Path | None) -> dict[str, Any]:
    """Why blink train calibrate refused its run (its calibration.json), or {}."""
    return {} if run_dir is None else read_json_or_empty(run_dir / "calibration.json").get("refused") or {}


def calibration_paused(s: Settings, code: int, out: str) -> bool:
    """A user pause (or a restart) inside the calibration: it exited 75, or its run's calibration.json
    refused it for one, or its heartbeat or metrics rows show one (two trainer sessions)."""
    run_dir = calibration_run(s, out)
    if code == EXIT_USER_PAUSE or run_dir is None:
        return code == EXIT_USER_PAUSE
    sessions = {row.get("session") for row in read_rows(run_dir / "metrics.jsonl")}
    beat, _ = run_state(run_dir)
    paused = refusal(run_dir).get("kind") in ("pause", "restart")
    return paused or len(sessions) > 1 or beat.get("state") == PAUSED_USER


def run_calibration(s: Settings, host) -> tuple[float, float]:
    """PR-5's 2,000-step calibration through `blink train calibrate`; R_true from its output. One with a
    user pause inside is never used: a fresh one runs once the flag is gone."""
    code, text = host.run("probe-calibrate", [*CALIBRATE, "--help"])
    if code != 0 or not calibrate_supported(text):
        raise StepFailed(
            "calibrate",
            f"`blink train calibrate ... --write` is not in {s.repo} (p7prep builds it): merge it, or "
            "calibrate by hand and pass --rate R_true",
        )
    args = [*CALIBRATE, "--config", s.config, "--steps", str(s.calib_steps), "--write"]
    args += ["--data", str(s.data)] if "--data" in text else []
    for attempt in range(1, MAX_CALIBRATIONS + 1):
        begin(s, host, "calibrate", f"attempt {attempt}" if attempt > 1 else "")
        host.stop_endgame_screen()  # PR-4: the screen never runs during a calibration
        code, out = host.run("calibrate", args, env={RESUMER_ENV: "1"})  # this loop reruns a paused one
        if calibration_paused(s, code, out):
            why = "a user pause landed inside the calibration, which is never used: a fresh one runs"
            status(s, PAUSED_USER, "calibrate", f"{why} once Blink is resumed")
            continue
        if code != 0:
            why = refusal(calibration_run(s, out)).get("detail") or "see p7v2-calibrate.out and .err"
            raise StepFailed("calibrate", f"exit {code}: {why}")
        found = read_rate(out)
        if found is None:
            raise StepFailed("calibrate", "no R_true in p7v2-calibrate.out: pass --rate R_true from it")
        return found
    raise StepFailed("calibrate", f"{MAX_CALIBRATIONS} calibrations in a row had a user pause in them")


def calibrated(s: Settings, host, state: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """R_true: --rate, else the recorded calibration, else a new one (never once leg 1 has started)."""
    begin(s, host, "calibrate")
    if s.rate is not None:
        rate, eps = s.rate, s.rate_eps
    elif "rate" in state:
        rate, eps = float(state["rate"]), float(state.get("rate_eps", 0.0))
    elif checkpoint_steps(s.runs / s.run):
        raise StepFailed(
            "calibrate", f"runs/{s.run} has checkpoints but no recorded rate: pass --rate R_true"
        )
    else:
        rate, eps = run_calibration(s, host)
        state = {**state, "screen_stopped": True}
    train = train_table(s.config_path)
    batch, steps = int(train["batch_size"]), int(train["steps"])
    if not steps_agree(rate, eps, s.long_hours, batch, steps):
        why = f"{s.config} steps {steps:,} is not floor({s.long_hours:g} x 3600 x {rate:g} / {batch})"
        raise StepFailed("calibrate", f"{why}: the calibration did not write it, or another rate did")
    state = {**state, "rate": rate, "rate_eps": eps}
    save_state(s, state)  # recorded only once long.toml agrees: a rerun never trusts a half-done calibration
    return pinned_rate(rate, steps, s.long_hours, batch), state


def planned(s: Settings, rate: float) -> Plan:
    train = train_table(s.config_path)
    rung_hours = s.rung_hours
    if rung_hours is None:
        sizes = tomllib.loads((s.repo / s.sweep_config).read_text(encoding="utf-8")).get("sizes", {})
        rung_hours = float(sizes.get("hours", 6.0))
    try:
        return make_plan(rate, train, s.long_hours, rung_hours, s.rung_steps)
    except ValueError as exc:
        raise StepFailed("plan", str(exc)) from exc


def branch_done(s: Settings, plan: Plan) -> bool:
    run_dir = s.runs / s.branch
    if not (run_dir / "config.json").is_file():
        return False
    steps = int(read_json(run_dir / "config.json")["config"]["steps"])
    if steps != plan.rung_steps:
        raise StepFailed(
            "branch", f"runs/{s.branch} was branched for {steps:,} steps, the plan is {plan.rung_steps:,}"
        )
    try:
        final_full_row(run_dir)
    except ValueError:
        return False
    return True


def leg_one(s: Settings, host, plan: Plan) -> None:
    steps = checkpoint_steps(s.runs / s.run)
    if steps and steps[-1] >= plan.rung_start:
        return
    begin(s, host, "leg1", f"to step {plan.rung_start:,} ({plan.hours(plan.rung_start):.2f} h)")
    code, _ = host.run("leg1", leg_one_args(s, plan, resume=bool(steps)))
    if code != 0:
        raise StepFailed("leg1", f"supervise exit {code}: see p7v2-leg1.out and runs/{s.run}/supervisor.json")
    if plan.rung_start not in checkpoint_steps(s.runs / s.run):
        raise StepFailed("leg1", f"runs/{s.run} has no checkpoint at step {plan.rung_start:,}")


def branch(s: Settings, host, plan: Plan) -> None:
    if branch_done(s, plan):
        return
    if not (s.runs / s.run / checkpoint_name(plan.rung_start)).is_file():
        raise StepFailed("branch", f"runs/{s.run} no longer has {checkpoint_name(plan.rung_start)} to branch")
    hours = plan.hours(plan.rung_cooldown)
    begin(s, host, "branch", f"{plan.rung_cooldown:,} cooldown steps ({hours:.2f} h)")
    code, _ = host.run("branch", branch_args(s, plan, resume=bool(checkpoint_steps(s.runs / s.branch))))
    if code != 0:
        raise StepFailed(
            "branch", f"supervise exit {code}: see p7v2-branch.out and runs/{s.branch}/supervisor.json"
        )
    if not branch_done(s, plan):
        raise StepFailed(
            "branch", f"runs/{s.branch} has no full-valprobe EMA VAA at step {plan.rung_steps:,}"
        )


def run_guard(s: Settings, host, plan: Plan) -> dict[str, Any]:
    begin(s, host, "guard")
    try:
        verdict = guard(s)
    except (OSError, ValueError, KeyError) as exc:
        raise StepFailed("guard", str(exc)) from exc
    rung = {"steps": plan.rung_steps, "start": plan.rung_start, "cooldown": plan.rung_cooldown,
            "rule": plan.rung_rule, "rate_steps": plan.rate_rung_steps}  # fmt: skip
    record = {**verdict, "rung": rung, "vaa_sigma": check_sigma(s)}
    write_atomic(s.home / "eval" / "size_guard.json", json.dumps(record, indent=1) + "\n")
    return verdict


def relaunch(s: Settings, host, state: dict[str, Any], plan: Plan) -> float:
    """The flagship relaunched detached; returns when (the clock) it was launched."""
    begin(s, host, "launch")
    if train_table(s.config_path).get("vaa_reference", "") != s.branch:
        set_train_string(s.config_path, "vaa_reference", s.branch)
    launched_at = host.clock()
    code, out = host.run("launch", launch_args(s, plan))
    if code != 0:
        raise StepFailed("launch", f"ops launch exit {code}: see p7v2-launch.out and .err")
    launched = next((line for line in out.splitlines() if line.startswith("launched")), "launched")
    save_state(s, {**state, "launched": launched})
    ensure_keeper(s, host)
    return launched_at


def since(record: dict[str, Any], key: str, launched_at: float) -> dict[str, Any]:
    """A record written since the launch, else {}: leg 1's heartbeat and supervisor.json never count."""
    return record if float(record.get(key) or 0) >= launched_at - 1 else {}


def trains_past(beat: dict[str, Any], start: int) -> bool:
    step = beat.get("step")
    return beat.get("state") in ("running", PAUSED_USER) and isinstance(step, int) and step > start


def verify_resumed(s: Settings, host, plan: Plan, launched_at: float) -> str:
    """Done only once runs/long trains again: its heartbeat running (or paused by the user) past the rung
    start, within VERIFY_S seconds that were not paused by the user."""
    status(s, "running", "verify", f"waiting for runs/{s.run} to train past step {plan.rung_start:,}")
    counted, last = 0.0, host.clock()
    logs = f"runs/{s.run}/supervisor.json and logs/{s.launch_name}.err"
    while True:
        found = run_state(s.runs / s.run)
        beat, record = since(found[0], "time", launched_at), since(found[1], "started", launched_at)
        if trains_past(beat, plan.rung_start):
            return f"runs/{s.run} is {beat['state']} at step {beat['step']:,}"
        if record.get("state") in ENDED:
            raise StepFailed("verify", f"runs/{s.run}'s supervisor is {record.get('status')}: see {logs}")
        now = host.clock()
        counted += 0.0 if user_paused(s.home, beat, record) else now - last
        last = now
        if counted >= VERIFY_S:
            said = f"supervisor {record.get('status') or 'silent since the launch'}"
            where = f"heartbeat {beat.get('state', 'none')} at step {beat.get('step')}"
            why = f"did not train past step {plan.rung_start:,} in {VERIFY_S / 60:.0f} minutes"
            raise StepFailed("verify", f"runs/{s.run} {why}: {said}, {where}; see {logs}")
        host.sleep(VERIFY_POLL_S)


def rung_line(plan: Plan) -> str:
    rung = f"size-m {plan.rung_steps:,} = {plan.rung_start:,} + {plan.rung_cooldown:,} cooldown"
    if plan.rung_rule == rung_rule(None, plan.rung_hours):
        return f"{rung} ({plan.rung_rule})"
    return f"{rung} ({plan.rung_rule}; {plan.rung_hours:g} h at R_true would be {plan.rate_rung_steps:,})"


def describe(plan: Plan, verdict: dict[str, Any]) -> str:
    flagship = f"flagship {plan.long_steps:,} steps (5% check {plan.first_check:,})"
    rung = rung_line(plan)
    guarded = f"guard Delta {verdict['delta']:+.4f} vs {verdict['threshold']:+.4f}"
    sigma = f"sigma_EMA {verdict['sigma_ema']:.4f}"
    return f"R_true {plan.rate:,.2f} samples/s: {flagship}; {rung}; {guarded} ({sigma})"


SCREEN_NOTE = (
    " The endgame screen was stopped for the calibration: relaunch it (at most 3 Stockfish processes "
    "during P7, PR-4)."
)


def finished(s: Settings, state: dict[str, Any], plan: Plan, verdict: dict[str, Any], resumed: str) -> str:
    screen = SCREEN_NOTE if state.get("screen_stopped") else ""
    flagship = f"flagship resumed with vaa_reference {s.branch!r} ({resumed}); {after_launch(s, plan)}"
    return f"{flagship}. {describe(plan, verdict)}.{screen}"


def drive(s: Settings, host) -> int:
    state = load_state(s)
    if state.get("launched"):
        print(f"p7v2: the flagship was already launched ({state['launched']}); nothing to do")
        return EXIT_DONE
    try:
        preflight(s, host)
        check_reference(s)
        ensure_keeper(s, host)  # the calibration measures the rate the flagship will train at
        rate, state = calibrated(s, host, state)
        plan = planned(s, rate)
        save_state(s, state := locked_plan(s, state, plan))
        leg_one(s, host, plan)
        branch(s, host, plan)
        verdict = run_guard(s, host, plan)
        if not verdict["passed"]:
            detail = (
                f"P6 v2 guard failed, flagship paused at step {plan.rung_start:,}: {describe(plan, verdict)}"
            )
            mark_paused(s.runs / s.run, detail)
            status(s, "paused", "guard", detail)
            return EXIT_PAUSED
        begin(s, host, "choose")
        choose_n_star(s, host, "choose", [])
        resumed = verify_resumed(s, host, plan, relaunch(s, host, state, plan))
    except StepFailed as exc:
        status(s, "failed", exc.step, exc.detail)
        return EXIT_FAILED
    except Exception as exc:  # a detached driver must leave its reason behind, whatever it was
        step = read_json(s.path(".status.json")).get("step", "?") if s.path(".status.json").is_file() else "?"
        status(s, "failed", step, f"{type(exc).__name__}: {exc}")
        raise
    status(s, "done", "launch", finished(s, state, plan, verdict, resumed))
    return EXIT_DONE


# ---------------------------------------------------------------- command line


def dry_run(s: Settings) -> int:
    state = load_state(s)
    rate = s.rate if s.rate is not None else state.get("rate")
    if rate is None:
        print("p7v2 --dry-run needs --rate R_true (or a recorded calibration)", file=sys.stderr)
        return EXIT_REFUSED
    try:
        plan = planned(s, float(rate))
    except StepFailed as exc:
        print(f"p7v2: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    hours = {name: plan.hours(steps) for name, steps in (("long", plan.long_steps), ("leg1", plan.rung_start),
                                                             ("branch", plan.rung_cooldown))}  # fmt: skip
    print(f"plan at R_true {plan.rate:,.2f} samples/s (batch {plan.batch}):")
    print(f"  flagship {plan.long_steps:,} steps ({hours['long']:.1f} h); 5% check at {plan.first_check:,}, "
          f"30% check (the preview pause) at {plan.preview_step:,}")  # fmt: skip
    print(f"  {rung_line(plan)}")
    print(f"  leg 1 to {plan.rung_start:,} ({hours['leg1']:.2f} h), branch {plan.rung_cooldown:,} "
          f"({hours['branch']:.2f} h)")  # fmt: skip
    calibrate = [*CALIBRATE, "--config", s.config, "--steps", str(s.calib_steps), "--write"]
    for name, args in (("calibrate", calibrate), ("leg1", leg_one_args(s, plan, False)),
                       ("branch", branch_args(s, plan, False)), ("choose", ["sweep", "choose"]),
                       ("launch", launch_args(s, plan)), ("then the preview", preview_args(s, plan)),
                       ("then the resume", launch_args(s))):  # fmt: skip
        print(f"  {name}: python -m blink.cli {' '.join(args)}")
    print(f"  under PR-6: python -m blink.cli eval strength --run {s.run} at each pause; when the level is "
          "good enough, python tools\\p7_finish.py (the final 20% cooldown from that step)")  # fmt: skip
    return EXIT_DONE


def settings_from(argv: list[str] | None = None) -> tuple[Settings, bool]:
    p = argparse.ArgumentParser(prog="p7_v2_driver", description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--repo", default=r"C:\dev\blink-run", help="the worktree the flagship trains in")
    p.add_argument("--python", default=r"C:\dev\blink-chess\.venv\Scripts\python.exe")
    p.add_argument("--home", default=r"D:\blink", help="BLINK_HOME")
    p.add_argument("--data", help="the pack (default HOME/data/v1)")
    p.add_argument("--keeper", help="the priority keeper (default HOME/ops/keep_training_priority.ps1)")
    p.add_argument(
        "--rate", type=parse_number, help="a calibrated R_true (samples/s) to use instead of calibrating"
    )
    p.add_argument("--rung-hours", type=float, help="M's rung at R_true (default: sweep.toml [sizes] hours)")
    rung = p.add_mutually_exclusive_group()
    rung.add_argument("--rung-steps", type=int, default=PR2_RUNG_STEPS,
                      help=f"size-m's steps (default: PR-2's {PR2_RUNG_STEPS}, from 47301)")  # fmt: skip
    rung.add_argument("--rung-from-rate", action="store_true",
                      help="size-m is the rung hours at R_true (departs from PR-2)")  # fmt: skip
    p.add_argument("--long-hours", type=float, default=120.0, help="T_long (PR-5)")
    p.add_argument("--bench-size", default="m", help="supervise's throughput benchmark ('' turns it off)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the commands, run nothing")
    args = p.parse_args(argv)
    home = Path(args.home)
    rate, eps = args.rate if args.rate else (None, 0.0)
    s = Settings(
        repo=Path(args.repo),
        python=Path(args.python),
        home=home,
        data=Path(args.data) if args.data else home / "data" / "v1",
        keeper=Path(args.keeper) if args.keeper else home / "ops" / KEEPER,
        rate=rate,
        rate_eps=eps,
        rung_hours=args.rung_hours,
        rung_steps=None if args.rung_from_rate else args.rung_steps,
        long_hours=args.long_hours,
        bench_size=args.bench_size,
    )
    return s, args.dry_run


def main(argv: list[str] | None = None) -> int:
    s, is_dry = settings_from(argv)
    if is_dry:
        return dry_run(s)
    s.logs.mkdir(parents=True, exist_ok=True)
    try:
        lock = acquire_lock(s.path(".lock"))
    except LockHeld as exc:
        print(f"p7v2: {exc}; nothing started", file=sys.stderr)
        return EXIT_REFUSED
    try:
        return drive(s, Host(s.repo, s.python, s.home, s.logs))
    finally:
        release_lock(lock)


if __name__ == "__main__":
    sys.exit(main())
