"""P6 v2's flagship choreography (EVAL.md PR-2), run detached: calibrate, leg 1, the size-m branch, the
guard, the N* record, then the flagship resumed with size-m as its reference.

Run it ONLY after P5 has ended (PR-2 was adopted on 2026-09-25), the recipe is frozen and EVAL.md v1 is
tagged, from the worktree the flagship trains in (it must hold the p6v2 and p7prep merges). Launch it
through WMI so it outlives the agent session (PF38), for example:

    powershell -NoProfile -Command "Invoke-CimMethod -ClassName Win32_Process -MethodName Create
      -Arguments @{CommandLine='C:\\dev\\blink-chess\\.venv\\Scripts\\python.exe tools\\p7_v2_driver.py';
      CurrentDirectory='C:\\dev\\blink-run'}"

With no flags it branches size-m at PR-2's step 47,301 over 11,825 steps; rerun it with the same flags.
`--dry-run --rate R` prints the plan and every command and runs nothing. Each step logs to
<home>/logs/p7v2-<step>.out|err; the state goes to p7v2.status.json (and p7v2.state.json):

0. preflight: no blink train/supervise/sweep process runs; configs/sweep.toml's rule is "prior";
   --preview-steps/--preview-name exist; `sweep choose` into a scratch file says N* = m. Nothing
   trains when any of these fails.
1. calibrate (PR-5): the endgame screen is stopped (PR-4: never during a calibration), then `blink
   train calibrate --config configs/long.toml --steps 2000 --write` measures R_true and writes
   long.toml's steps = floor(120 x 3600 x R_true / 1024). The driver reads R_true from its output
   and checks those steps; it fails clearly when the command is missing.
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
   heartbeat says "paused: P7-VAA" (gate P7-VAA) and the driver stops. The numbers go to
   eval/size_guard.json either way.
6. `blink sweep choose` (rule prior) records the floor, VRAM and p99 facts in eval/sweep.json.
7. long.toml's vaa_reference = "size-m"; the flagship is relaunched detached (`ops launch --name
   p7-long -- supervise -- train ... --max-steps P --resume`) and the priority keeper restarted. P is the
   30% check (blink.train.vaa.check_steps), where the plan's P7 pauses the long run for the 3 GPU-h
   preview (size-m is its reference, PR-2 (2); PR-3's parity and soak run on its weights). The driver
   ends there: its done status names P and the two commands that follow, `train --run long --data DATA
   --preview-cooldown 3h --from-step P`, then the relaunch without --max-steps.

Stdlib only: it imports nothing from the repo, so code changes cannot reach it mid-run. A rerun (after a
reboot, say) picks up where the files say it stopped: checkpoints, size-m's final row, the state file.
It never calibrates once leg 1 has a checkpoint, refuses a plan other than the recorded one from then on
(rerun it with the first launch's flags), and does nothing once the flagship was launched.
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXIT_DONE, EXIT_FAILED, EXIT_REFUSED, EXIT_PAUSED = 0, 1, 2, 3
CALIBRATE = ("train", "calibrate")
BUSY = re.compile(r"blink\.cli\s+(train|supervise|sweep)\b")
CHECKPOINT = re.compile(r"^ckpt_(\d+)\.pt$")
RATE = re.compile(r"\bR[_ ]?true\b[^0-9\n]{0,24}([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
N_STAR = re.compile(r"^N\* = (\S+)", re.MULTILINE)
FIRST_CHECK = 0.05  # the flagship's first check (blink.train.vaa.CHECK_FRACS)
PREVIEW_CHECK = 0.30  # the check the long run pauses at for the 3 GPU-h preview (plan P7)
PR2_RUNG_STEPS = 59_126  # EVAL.md PR-2: size-m is a 59,126-step M run, branched at 47,301
PLAN_STEPS = ("batch", "long_steps", "rung_steps", "rung_cooldown", "rung_start", "first_check",
              "preview_step")  # the plan's numbers a rerun must keep  # fmt: skip
TOLERANCE = 1e-9  # float slack for a pre-registered comparison (blink.train.nstar)
PAUSED_VAA = "paused: P7-VAA"  # blink.train.supervise's pause status: gate P7-VAA
KEEPER = "keep_training_priority.ps1"
DETACHED = 0x00000008 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)


class StepFailed(RuntimeError):
    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step, self.detail = step, detail


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


def parse_number(text: str) -> tuple[float, float]:
    """'2,621.44' -> (2621.44, 0.005): the value and half its last printed digit."""
    digits = text.replace(",", "")
    decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
    return float(digits), 0.5 * 10.0**-decimals


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


# ---------------------------------------------------------------- files


def train_table(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """A config's [train] table with its `base` chain resolved (blink.model.config.read_tables)."""
    path = path.resolve()
    if path in seen:
        raise ValueError(f"config base cycle at {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    base = train_table(path.parent / str(data["base"]), (*seen, path)) if "base" in data else {}
    return {**base, **data.get("train", {})}


def set_train_string(path: Path, key: str, value: str) -> None:
    """Set [train] key = "value" in place, keeping comments and line endings; the file must then parse to
    exactly that one change."""
    with open(path, encoding="utf-8", newline="") as handle:
        text = handle.read()
    before = tomllib.loads(text)
    pattern = re.compile(rf'^([ \t]*{re.escape(key)}[ \t]*=[ \t]*)"[^"\r\n]*"', re.MULTILINE)
    new, count = pattern.subn(lambda m: m.group(1) + json.dumps(value), text, count=1)
    after = tomllib.loads(new)
    if count != 1 or after != {**before, "train": {**before.get("train", {}), key: value}}:
        raise ValueError(f"{path} has no single [train] {key} string to set")
    write_atomic(path, new)


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    for attempt in range(5):  # a reader holding the file blocks os.replace on Windows
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.2)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def checkpoint_steps(run_dir: Path) -> list[int]:
    if not run_dir.is_dir():
        return []
    return sorted(int(m.group(1)) for p in run_dir.iterdir() if (m := CHECKPOINT.match(p.name)))


def checkpoint_name(step: int) -> str:
    return f"ckpt_{step:09d}.pt"


def status(s: Settings, state: str, step: str, detail: str = "") -> None:
    record = {"state": state, "step": step, "detail": detail, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_atomic(s.path(".status.json"), json.dumps(record, indent=1) + "\n")


def load_state(s: Settings) -> dict[str, Any]:
    path = s.path(".state.json")
    return read_json(path) if path.is_file() else {}


def save_state(s: Settings, state: dict[str, Any]) -> None:
    write_atomic(s.path(".state.json"), json.dumps(state, indent=1) + "\n")


# ---------------------------------------------------------------- the guard


def guard_verdict(branch_vaa: float, arm_vaas: list[float], sigma_factor: float = 2.0) -> dict[str, Any]:
    """PR-2 (3): Delta = the branch's final EMA VAA - the arms' mean; it fails when Delta < -2 sigma_EMA,
    sigma_EMA being the arms' sample standard deviation."""
    mean, sigma = statistics.fmean(arm_vaas), statistics.stdev(arm_vaas)
    delta, threshold = branch_vaa - mean, -sigma_factor * sigma
    return {"mean": mean, "sigma_ema": sigma, "delta": delta, "threshold": threshold,
            "passed": delta >= threshold - TOLERANCE}  # fmt: skip


def final_full_row(run_dir: Path) -> dict[str, Any]:
    """A run's full-valprobe EMA VAA row at its last planned step."""
    planned = int(read_json(run_dir / "config.json")["config"]["steps"])
    rows = [r for r in read_rows(run_dir / "evals.jsonl") if r.get("vaa_set") == "full" and "ema_vaa" in r]
    if not rows or rows[-1]["step"] != planned:
        raise ValueError(f"{run_dir.name} has no full-valprobe EMA VAA at its last step {planned:,}")
    return rows[-1]


def guard(s: Settings) -> dict[str, Any]:
    """The guard's inputs from ablations.json, the arms' evals and the branch's, and its verdict."""
    arms = read_json(s.home / "eval" / "ablations.json").get("arms", {})
    rows = {}
    for name in s.arms:
        entry = arms.get(name) or {}
        if not str(entry.get("status", "")).startswith("finished") or not entry.get("run"):
            raise ValueError(f"arm {name} has not finished in ablations.json")
        rows[name] = final_full_row(s.runs / entry["run"])
    branch = final_full_row(s.runs / s.branch)
    probes = {row.get("vaa_n") for row in (*rows.values(), branch)}
    if len(probes) != 1:
        raise ValueError(f"the final rows scored different valprobes ({sorted(map(str, probes))} roots)")
    verdict = guard_verdict(branch["ema_vaa"], [rows[a]["ema_vaa"] for a in s.arms], s.sigma_factor)
    picked = {a: {"step": r["step"], "ema_vaa": r["ema_vaa"]} for a, r in rows.items()}
    return {
        "rule": "PR-2 (3): Delta = size-m final EMA VAA - mean final EMA VAA of a01-a03; pause if "
        f"Delta < -{s.sigma_factor:g} sigma_EMA (their sample sd); not equal GPU-hours, no scaling law",
        "branch": {"run": s.branch, "step": branch["step"], "ema_vaa": branch["ema_vaa"]},
        "arms": picked,
        "vaa_n": branch.get("vaa_n"),
        **verdict,
    }


def mark_paused(run_dir: Path, detail: str) -> None:
    """Gate P7-VAA as the supervisor sets it: `blink status` then reports the run as paused."""
    path = run_dir / "heartbeat.json"
    beat = read_json(path) if path.is_file() else {}
    record = {**beat, "state": "paused", "stopped": PAUSED_VAA, "detail": detail, "time": time.time()}
    write_atomic(path, json.dumps(record))


# ---------------------------------------------------------------- the machine


class Host:
    """The real machine: blink commands as children, process listings and detached starts."""

    def __init__(self, s: Settings) -> None:
        self.s = s
        self.env = {**os.environ, "BLINK_HOME": str(s.home), "PYTHONUTF8": "1"}

    def run(self, step: str, blink_args: list[str]) -> tuple[int, str]:
        out_path, err_path = self.s.path(f"-{step}.out"), self.s.path(f"-{step}.err")
        argv = [str(self.s.python), "-m", "blink.cli", *blink_args]
        with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
            code = subprocess.run(argv, cwd=self.s.repo, env=self.env, stdout=out, stderr=err).returncode
        return code, out_path.read_text(encoding="utf-8", errors="replace")

    def command_lines(self) -> list[str]:
        """Every python.exe and powershell.exe command line (Windows has no pgrep)."""
        query = (
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','powershell.exe' } | "
            "ForEach-Object { $_.CommandLine }"
        )
        out = subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True)
        return [line for line in out.stdout.splitlines() if line.strip()]

    def stop_endgame_screen(self) -> None:
        """Kill any `blink eval endgames` tree (its Stockfish children too); it resumes from its cache."""
        query = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            r"Where-Object { $_.CommandLine -match 'blink\.cli\s+eval\s+endgames' } | "
            "ForEach-Object { taskkill /PID $_.ProcessId /T /F }"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", query], capture_output=True, text=True)

    def start_keeper(self, script: Path) -> None:
        args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File"]
        subprocess.Popen([*args, str(script)], cwd=script.parent, creationflags=DETACHED)


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
    """What follows the driver: the flagship stops at the 30% check, the preview runs, the run resumes."""
    blink = "python -m blink.cli"
    return (
        f"it stops at the 30% check, step {plan.preview_step:,}: run the 3 GPU-h preview `{blink} "
        f"{' '.join(preview_args(s, plan))}` (PR-3's parity and soak on its weights), then resume without "
        f"--max-steps: `{blink} {' '.join(launch_args(s))}`"
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
    status(s, "running", "preflight")
    busy = busy_commands(host.command_lines())
    if busy:
        raise StepFailed("preflight", f"a blink process holds the GPU, so nothing started: {busy[0][:200]}")
    rule = tomllib.loads((s.repo / s.sweep_config).read_text(encoding="utf-8")).get("choose", {}).get("rule")
    if rule != "prior":
        raise StepFailed("preflight", f"{s.sweep_config} rule is {rule!r}, not 'prior': P6 v2 is not merged")
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


def run_calibration(s: Settings, host) -> tuple[float, float]:
    """PR-5's 2,000-step calibration through `blink train calibrate`; R_true from its output."""
    code, text = host.run("probe-calibrate", [*CALIBRATE, "--help"])
    if code != 0 or not calibrate_supported(text):
        raise StepFailed(
            "calibrate",
            f"`blink train calibrate ... --write` is not in {s.repo} (p7prep builds it): merge it, or "
            "calibrate by hand and pass --rate R_true",
        )
    args = [*CALIBRATE, "--config", s.config, "--steps", str(s.calib_steps), "--write"]
    args += ["--data", str(s.data)] if "--data" in text else []
    host.stop_endgame_screen()  # PR-4: the screen never runs during a calibration
    code, out = host.run("calibrate", args)
    if code != 0:
        raise StepFailed("calibrate", f"exit {code}: see p7v2-calibrate.out and .err")
    found = read_rate(out)
    if found is None:
        raise StepFailed("calibrate", "no R_true in p7v2-calibrate.out: pass --rate R_true from it")
    return found


def calibrated(s: Settings, host, state: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """R_true: --rate, else the recorded calibration, else a new one (never once leg 1 has started)."""
    status(s, "running", "calibrate")
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
    status(s, "running", "leg1", f"to step {plan.rung_start:,} ({plan.hours(plan.rung_start):.2f} h)")
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
    status(
        s,
        "running",
        "branch",
        f"{plan.rung_cooldown:,} cooldown steps ({plan.hours(plan.rung_cooldown):.2f} h)",
    )
    code, _ = host.run("branch", branch_args(s, plan, resume=bool(checkpoint_steps(s.runs / s.branch))))
    if code != 0:
        raise StepFailed(
            "branch", f"supervise exit {code}: see p7v2-branch.out and runs/{s.branch}/supervisor.json"
        )
    if not branch_done(s, plan):
        raise StepFailed(
            "branch", f"runs/{s.branch} has no full-valprobe EMA VAA at step {plan.rung_steps:,}"
        )


def run_guard(s: Settings, plan: Plan) -> dict[str, Any]:
    status(s, "running", "guard")
    try:
        verdict = guard(s)
    except (OSError, ValueError, KeyError) as exc:
        raise StepFailed("guard", str(exc)) from exc
    rung = {"steps": plan.rung_steps, "start": plan.rung_start, "cooldown": plan.rung_cooldown,
            "rule": plan.rung_rule, "rate_steps": plan.rate_rung_steps}  # fmt: skip
    write_atomic(s.home / "eval" / "size_guard.json", json.dumps({**verdict, "rung": rung}, indent=1) + "\n")
    return verdict


def relaunch(s: Settings, host, state: dict[str, Any], plan: Plan) -> None:
    status(s, "running", "launch")
    if train_table(s.config_path).get("vaa_reference", "") != s.branch:
        set_train_string(s.config_path, "vaa_reference", s.branch)
    code, out = host.run("launch", launch_args(s, plan))
    if code != 0:
        raise StepFailed("launch", f"ops launch exit {code}: see p7v2-launch.out and .err")
    launched = next((line for line in out.splitlines() if line.startswith("launched")), "launched")
    save_state(s, {**state, "launched": launched})
    ensure_keeper(s, host)


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


def finished(s: Settings, state: dict[str, Any], plan: Plan, verdict: dict[str, Any]) -> str:
    screen = SCREEN_NOTE if state.get("screen_stopped") else ""
    resumed = f"flagship resumed with vaa_reference {s.branch!r}; {after_launch(s, plan)}"
    return f"{resumed}. {describe(plan, verdict)}.{screen}"


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
        verdict = run_guard(s, plan)
        if not verdict["passed"]:
            detail = (
                f"P6 v2 guard failed, flagship paused at step {plan.rung_start:,}: {describe(plan, verdict)}"
            )
            mark_paused(s.runs / s.run, detail)
            status(s, "paused", "guard", detail)
            return EXIT_PAUSED
        status(s, "running", "choose")
        choose_n_star(s, host, "choose", [])
        relaunch(s, host, state, plan)
    except StepFailed as exc:
        status(s, "failed", exc.step, exc.detail)
        return EXIT_FAILED
    except Exception as exc:  # a detached driver must leave its reason behind, whatever it was
        step = read_json(s.path(".status.json")).get("step", "?") if s.path(".status.json").is_file() else "?"
        status(s, "failed", step, f"{type(exc).__name__}: {exc}")
        raise
    status(s, "done", "launch", finished(s, state, plan, verdict))
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
    return drive(s, Host(s))


if __name__ == "__main__":
    sys.exit(main())
