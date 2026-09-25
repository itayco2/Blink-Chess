"""PR-6's finish (EVAL.md section 5): the flagship re-planned so its final 20% 1-sqrt cooldown starts now.

    python tools\\p7_finish.py [--at-step C] [--dry-run]

Run it when Itay says the level is good enough (the strength checks, `blink eval strength --run long`,
show it beside DM-9M), from the worktree the flagship trains in, usually while he has Blink paused. At
step c, runs/long's latest checkpoint (after a user pause, the step it paused at) or --at-step c (which
must be a checkpoint), the final cooldown branches as run long-final with exactly ceil(c/4) cooldown
steps, so the 1-sqrt cooldown is the last 20% of c + ceil(c/4) steps and starts at c itself
(`blink train --preview-steps`, blink.train.preview.preview_config):

    blink ops launch --name p7-long-final -- supervise --bench-size m -- train --run long
        --data DATA --from-step c --preview-steps ceil(c/4) --preview-name long-final

It names no --config: the branch trains with the config runs/long's checkpoint c holds (blink train
refuses a --config that differs from it, so a later edit of long.toml would stop the branch).

runs/long's checkpoints and logs stay as they are. So that nothing but the branch trains once Resume
Blink removes the flag:
- nothing happens while any other Blink process could take the GPU, paused or not: a train, supervise,
  sweep or benchmark serving another run (a long-preview branch, a sweep arm, a calibration); it is
  looked for again just before the launch;
- nothing happens while runs/long trains (a trainer process for it is alive, or its heartbeat says
  running and is fresh): pause it with Pause Blink first;
- runs/long's supervisor is ended only while it holds no trainer and cannot start one: paused by the user
  (BLINK_HOME/PAUSE up and supervisor.json "paused: user"); a supervisor in any other state is refused,
  and one already gone (a stop rule, a reboot) needs nothing;
- the cooldown must end inside configs/long.toml's steps, PR-5's 120 h upper bound (PR-6);
- nothing happens while runs/long is held at gate P7-VAA (its supervisor or the guard left it there):
  only Itay clears that gate, and --vaa-gate-cleared says he has;
- just before the launch runs/long gets one file, finished_by.json (blink.train.finished): blink train
  and blink supervise then refuse to resume it (the P6 v2 driver's printed resume command included).
long-final trains with film off (blink.train.preview.preview_config): the flagship's film ends at c, and
its frames planned past c are never made; the record says so for the published write-up.
The branch is supervised like any run: it waits while the flag is up and trains once Resume Blink removes
it. The finish is done once the branch's trainer has written its config.json (blink train accepted the
command) or, while the flag is up, once its supervisor waits paused by the user; a crash of the trainer,
an ended supervisor, or neither within 10 minutes fails it. The plan goes to
<BLINK_HOME>/eval/p7_finish.json: c, the cooldown steps, the total, the training hours so far (from
runs/long's metrics rows), the time, the command, the supervisor it stopped and how the launch verified.
`--dry-run` prints all of it and changes nothing. It takes logs/p7v2.lock, so it never runs beside the
P6 v2 driver. Stdlib only (with psutil), like the driver.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from p7_guard import PAUSED_VAA
from p7_machine import (
    PAUSED_USER,
    Host,
    LockHeld,
    StepFailed,
    acquire_lock,
    blink_args,
    checkpoint_steps,
    flag_path,
    gpu_work,
    read_json_or_empty,
    read_rows,
    release_lock,
    run_state,
    served_run,
    train_table,
    write_atomic,
)

EXIT_DONE, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2
LIVE_S = 120.0  # a heartbeat this fresh that says "running" means the run trains
# the branch's trainer must have written config.json by then (its supervisor, with the flag up, its first
# record): building an M run from its parent checkpoint takes a minute or two
VERIFY_S, VERIFY_POLL_S = 600.0, 5.0
ENDED = ("finished", "stopped", "paused")  # supervisor.json states after which nothing trains
FINISHED_MARKER = "finished_by.json"  # blink.train.finished.MARKER: blink train and supervise honour it
GATE = "P7-VAA"  # blink.train.supervise.PENDING_GATE_VAA


@dataclass(frozen=True)
class Settings:
    repo: Path  # the worktree the flagship trains in
    python: Path
    home: Path  # BLINK_HOME
    data: Path
    config: str = "configs/long.toml"
    run: str = "long"
    branch: str = "long-final"
    bench_size: str = "m"
    launch_name: str = "p7-long-final"
    at_step: int | None = None
    dry_run: bool = False
    vaa_gate_cleared: bool = False  # Itay cleared runs/long's P7-VAA gate and says the finish may pass it

    @property
    def runs(self) -> Path:
        return self.home / "runs"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def config_path(self) -> Path:
        return self.repo / self.config


# ---------------------------------------------------------------- the plan


def cooldown_steps(c: int) -> int:
    """ceil(c / 4): the final 20% of c + ceil(c / 4) steps."""
    return -(-c // 4)


def training_hours(rows: list[dict[str, Any]], upto: int) -> float:
    """blink.train.calibrate.training_seconds, in hours, over the rows up to step `upto`: train-phase
    intervals, none whose rows two trainer sessions wrote (a pause, crash or relaunch between them)."""
    kept = [row for row in rows if int(row["step"]) <= upto]
    seconds = sum(
        float(later["time"]) - float(earlier["time"])
        for earlier, later in zip(kept, kept[1:], strict=False)
        if later.get("phase") == "train" and earlier.get("session") == later.get("session")
    )
    return seconds / 3600


def launch_args(s: Settings, c: int, k: int) -> list[str]:
    supervise = ["supervise", *(["--bench-size", s.bench_size] if s.bench_size else []), "--"]
    train = ["train", "--run", s.run, "--data", str(s.data), "--from-step", str(c)]
    train += ["--preview-steps", str(k), "--preview-name", s.branch]
    return ["ops", "launch", "--name", s.launch_name, "--", *supervise, *train]


def plan_finish(s: Settings) -> dict[str, Any]:
    """c, the cooldown and the command; StepFailed when this finish must not run."""
    branch_dir = s.runs / s.branch
    if (branch_dir / "config.json").is_file() or checkpoint_steps(branch_dir):
        why = f"runs/{s.branch} already exists: the finish ran before (eval/p7_finish.json has its command)"
        raise StepFailed("finish", f"{why}; resume that branch with the same command plus --resume")
    steps = checkpoint_steps(s.runs / s.run)
    if not steps:
        raise StepFailed("finish", f"runs/{s.run} has no checkpoint to branch the final cooldown from")
    c = steps[-1] if s.at_step is None else s.at_step
    if c not in steps:
        have = ", ".join(f"{step:,}" for step in steps[-5:])
        raise StepFailed("finish", f"--at-step {c:,} is not a checkpoint of runs/{s.run} (its last: {have})")
    train, k = train_table(s.config_path), cooldown_steps(c)
    if c <= int(train["warmup_steps"]):
        raise StepFailed("finish", f"step {c:,} is inside the {train['warmup_steps']}-step warmup")
    if c + k > int(train["steps"]):
        why = f"step {c:,} + {k:,} cooldown steps passes {s.config}'s {int(train['steps']):,} steps"
        raise StepFailed("finish", f"{why}, PR-5's 120 h upper bound (PR-6): let the flagship finish its "
                                   "own cooldown")  # fmt: skip
    args = launch_args(s, c, k)
    return {"at_step": c, "cooldown_steps": k, "total_steps": c + k, "cooldown_frac": k / (c + k),
            "upper_bound_steps": int(train["steps"]),
            "training_hours": training_hours(read_rows(s.runs / s.run / "metrics.jsonl"), c),
            "vaa_gate": gate_passed(s), "film": film_note(s, c),
            "args": args, "command": "python -m blink.cli " + " ".join(args)}  # fmt: skip


def gate_passed(s: Settings) -> str | None:
    """None when runs/long is not held at gate P7-VAA; StepFailed when it is, unless Itay cleared it
    (--vaa-gate-cleared), which the record then says. Its supervisor or the guard leaves the gate."""
    beat, record = run_state(s.runs / s.run)
    held = record.get("pending_gate") == GATE or PAUSED_VAA in (record.get("status"), beat.get("stopped"))
    if not held:
        return None
    said = f"supervisor.json {record.get('status') or record.get('state')}, heartbeat {beat.get('stopped')}"
    if s.vaa_gate_cleared:
        return f"cleared by Itay (--vaa-gate-cleared): runs/{s.run} was held at {GATE} ({said})"
    why = f"runs/{s.run} is held at gate {GATE} ({said}): only Itay clears it (EVAL.md)"
    raise StepFailed("finish", f"{why}; once he has, run p7_finish --vaa-gate-cleared")


def film_note(s: Settings, c: int) -> str:
    """The branch trains with film off (blink.train.preview.preview_config), so the flagship's film stops
    at c: said in the record, for the published write-up."""
    return (
        f"runs/{s.run}'s film ends at step {c:,}: runs/{s.branch} trains with film off "
        f"(blink.train.preview.preview_config), so no frame shows the final cooldown and the frames "
        f"planned past step {c:,} are never made"
    )


# ---------------------------------------------------------------- runs/long's own processes, and the others


def run_processes(host, run: str) -> tuple[list[dict], list[dict]]:
    """(trainers, supervisors) of `run` among the live processes; a dry run is neither."""
    trainers, supervisors = [], []
    for proc in host.processes():
        args = blink_args(list(proc.get("cmdline") or []))
        if "--dry-run" in args or served_run(args) != run:
            continue
        (trainers if args[0] == "train" else supervisors).append(proc)
    return trainers, supervisors


def own_tree(procs: list[dict], run: str) -> set[int]:
    """The pids that serve `run` (its trainers and supervisors, whatever their parents) and their
    descendants."""
    own = {proc["pid"] for proc in procs if served_run(blink_args(list(proc.get("cmdline") or []))) == run}
    while grown := {proc["pid"] for proc in procs if proc.get("ppid") in own} - own:
        own |= grown
    return own


def other_gpu_work(s: Settings, host) -> list[str]:
    """Every other Blink process that trains, benches or starts training runs (p7_machine.GPU_WORK), paused
    or not: once Resume Blink removes the flag it would train beside long-final. runs/long's own tree and
    dry runs are not counted."""
    procs = host.processes()
    own, found = own_tree(procs, s.run), []
    for proc in procs:
        args = blink_args(list(proc.get("cmdline") or []))
        if proc["pid"] in own or not gpu_work(args):
            continue
        run, command = served_run(args), " ".join(arg for arg in args[:2] if not arg.startswith("-"))
        found.append(f"pid {proc['pid']} (blink {command}{f', runs/{run}' if run else ''})")
    return found


def refuse_other_gpu_work(s: Settings, host, step: str, after: str = "") -> None:
    """StepFailed while anything but runs/long could take the GPU: the finish puts one run on it."""
    others = other_gpu_work(s, host)
    if others:
        what = f"{', '.join(others)}: another Blink run could take the GPU beside runs/{s.branch}"
        raise StepFailed(step, f"{what}{after}; end it or let it finish, then run p7_finish again")


def stoppable(s: Settings, host) -> list[int]:
    """The pids to end so runs/long's supervisor cannot resume it: [] when none is left; StepFailed while
    runs/long trains or its supervisor might start a trainer."""
    trainers, supervisors = run_processes(host, s.run)
    if trainers:
        raise StepFailed("finish", f"runs/{s.run} is training (pid {trainers[0]['pid']}): pause it with "
                                   "Pause Blink, then finish")  # fmt: skip
    beat, record = run_state(s.runs / s.run)
    age = host.clock() - float(beat.get("time") or 0)
    if beat.get("state") == "running" and age <= LIVE_S:
        raise StepFailed("finish", f"runs/{s.run}'s heartbeat says it is training (step {beat.get('step')}, "
                                   f"{age:.0f} s ago): pause it with Pause Blink, then finish")  # fmt: skip
    if not supervisors:
        return []
    flag = flag_path(s.home).exists()
    if not (flag and record.get("state") == PAUSED_USER):
        said = record.get("status") or record.get("state") or "unknown"
        down = "" if flag else f", and BLINK_HOME/{flag_path(s.home).name} is down"
        who = f"runs/{s.run}'s supervisor (pid {supervisors[0]['pid']}) is {said}{down}"
        raise StepFailed("finish", f"{who}: finish stops it only while it is paused by the user and holds no "
                                   "trainer (press Pause Blink, wait for it, then finish)")  # fmt: skip
    pids = {proc["pid"] for proc in supervisors}
    return sorted(proc["pid"] for proc in supervisors if proc.get("ppid") not in pids)


# ---------------------------------------------------------------- the finish


def describe(s: Settings, plan: dict[str, Any], pids: list[int]) -> list[str]:
    c, k = plan["at_step"], plan["cooldown_steps"]
    stop = f"  runs/{s.run} has no supervisor left to stop"
    if pids:
        verb = "would stop" if s.dry_run else "stops"
        stop = (
            f"  {verb} runs/{s.run}'s supervisor (pid {', '.join(map(str, pids))}): paused by the user, it "
        )
        stop += "holds no trainer"
    lines = [
        f"PR-6 finish of runs/{s.run}: the final cooldown branches at step {c:,} as runs/{s.branch}",
        f"  {k:,} cooldown steps (ceil(c/4)): the 1-sqrt cooldown is the last 20% of {c + k:,} steps",
        f"  runs/{s.run} has {plan['training_hours']:.1f} training hours up to step {c:,}; {s.config}'s "
        f"upper bound is {plan['upper_bound_steps']:,} steps",
        stop,
        f"  then closes runs/{s.run} ({FINISHED_MARKER}): blink train and supervise refuse to resume it",
        f"  {plan['command']}",
        f"  {plan['film']}",
    ]
    if flag_path(s.home).exists():
        lines.append("The branch waits while BLINK_HOME/PAUSE is up and trains once Resume Blink removes it.")
    return lines


def since(record: dict[str, Any], key: str, launched_at: float) -> bool:
    return float(record.get(key) or 0) >= launched_at - 1


def branch_state(s: Settings, launched_at: float) -> tuple[str | None, str | None]:
    """(how the launched branch stands, why it failed), each None while it is not known yet.

    The supervisor's first record says running before its child has parsed its arguments, so only the
    trainer's own config.json (or its running heartbeat) since the launch shows `blink train` accepted
    the branch. While the flag is up the supervisor starts no trainer: its "paused: user" record is all
    there is. A crash of the trainer (blink train refusing the command exits 2) or an ended supervisor
    fails the finish at once."""
    run_dir = s.runs / s.branch
    beat, record = run_state(run_dir)
    config = read_json_or_empty(run_dir / "config.json")
    if since(config, "created", launched_at) or (
        beat.get("state") == "running" and since(beat, "time", launched_at)
    ):
        return f"runs/{s.branch}'s trainer has started", None
    if not since(record, "started", launched_at):
        return None, None
    crashes = [event for event in record.get("events") or [] if event.get("event") == "crash"]
    if crashes:
        return None, f"its trainer exited {crashes[0].get('code')} (blink train refused it or crashed)"
    if record.get("state") in ENDED:
        return None, f"its supervisor is {record.get('status') or record.get('state')}"
    if record.get("state") == PAUSED_USER and flag_path(s.home).exists():
        return f"runs/{s.branch} waits for Resume Blink; its trainer starts then", None
    return None, None


def branch_started(s: Settings, host, launched_at: float) -> str:
    """How the branch stands once it is known to train (or to wait for Resume); StepFailed when it
    failed, or showed neither within VERIFY_S seconds."""
    deadline = host.clock() + VERIFY_S
    logs = f"logs/{s.launch_name}.err and runs/{s.branch}/supervisor.json"
    while True:
        state, failed = branch_state(s, launched_at)
        if state is not None:
            return state
        if failed is not None:
            raise StepFailed(
                "verify", f"runs/{s.branch} was launched but does not train: {failed}; see {logs}"
            )
        if host.clock() >= deadline:
            why = f"no config.json from its trainer in {VERIFY_S / 60:.0f} minutes"
            raise StepFailed("verify", f"runs/{s.branch} was launched but does not train: {why}; see {logs}")
        host.sleep(VERIFY_POLL_S)


def stop_supervisor(s: Settings, host) -> list[int]:
    """End runs/long's supervisor, looked at again just before (Itay may have resumed since the first
    look); StepFailed when anything of runs/long is still there afterwards, so nothing launches."""
    pids = stoppable(s, host)
    for pid in pids:
        host.kill_tree(pid)
    left = [proc["pid"] for group in run_processes(host, s.run) for proc in group]
    if left:
        raise StepFailed("stop", f"runs/{s.run}'s processes {left} are still there after the stop: nothing "
                                 "was launched")  # fmt: skip
    return pids


def finish(s: Settings, host) -> int:
    try:
        plan = plan_finish(s)
        refuse_other_gpu_work(s, host, "finish")
        pids = stoppable(s, host)
    except StepFailed as exc:
        print(f"p7_finish: refused: {exc.detail}", file=sys.stderr)
        return EXIT_REFUSED
    for line in describe(s, plan, pids):
        print(line)
    if s.dry_run:
        return EXIT_DONE
    try:
        pids = stop_supervisor(s, host)
        stopped = f" (runs/{s.run}'s supervisor {pids} was stopped)" if pids else ""
        refuse_other_gpu_work(s, host, "launch", f"{stopped}: nothing was launched")
    except StepFailed as exc:
        print(f"p7_finish: {'refused' if exc.step == 'finish' else 'failed'}: {exc.detail}", file=sys.stderr)
        return EXIT_REFUSED if exc.step == "finish" else EXIT_FAILED
    return launch(s, host, plan, pids)


def close_run(s: Settings, plan: dict[str, Any]) -> Path:
    """runs/long's finished_by.json: its checkpoints and logs stay, but nothing resumes it beside the branch
    (blink.train.finished)."""
    marker = s.runs / s.run / FINISHED_MARKER
    record = {"by": "tools/p7_finish.py (PR-6)", "at_step": plan["at_step"], "branch": s.branch,
              "time": time.strftime("%Y-%m-%dT%H:%M:%S")}  # fmt: skip
    write_atomic(marker, json.dumps(record, indent=1) + "\n")
    return marker


def launch(s: Settings, host, plan: dict[str, Any], pids: list[int]) -> int:
    """Record the plan, close runs/long, launch the branch and wait until it trains (or waits for Resume)."""
    record = {"rule": "PR-6 (EVAL.md section 5): the final 20% 1-sqrt cooldown from the current step",
              "run": s.run, "branch": s.branch, **{k: v for k, v in plan.items() if k != "args"},
              "stopped_supervisor": pids, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "launched": None}  # fmt: skip
    out_path = s.home / "eval" / "p7_finish.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_atomic(out_path, json.dumps(record, indent=1) + "\n")
    record["closed"] = str(close_run(s, plan))
    launched_at = host.clock()
    code, out = host.run("finish-launch", plan["args"])
    closed = (
        f"runs/{s.run} is closed ({FINISHED_MARKER}): run p7_finish again, or remove that file to resume it"
    )
    if code != 0:
        print(f"p7_finish: ops launch exit {code}: see logs/p7v2-finish-launch.out and .err; {closed}",
              file=sys.stderr)  # fmt: skip
        return EXIT_FAILED
    record["launched"] = next((line for line in out.splitlines() if line.startswith("launched")), "launched")
    write_atomic(out_path, json.dumps(record, indent=1) + "\n")
    try:
        state = branch_started(s, host, launched_at)
    except StepFailed as exc:
        record["verified"] = f"failed: {exc.detail}"
        write_atomic(out_path, json.dumps(record, indent=1) + "\n")
        print(f"p7_finish: failed: {exc.detail}; {closed}", file=sys.stderr)
        return EXIT_FAILED
    record["verified"] = state
    write_atomic(out_path, json.dumps(record, indent=1) + "\n")
    print(f"{record['launched']}: {state}; recorded in {out_path}")
    return EXIT_DONE


def settings_from(argv: list[str] | None = None) -> Settings:
    p = argparse.ArgumentParser(prog="p7_finish", description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--repo", default=r"C:\dev\blink-run", help="the worktree the flagship trains in")
    p.add_argument("--python", default=r"C:\dev\blink-chess\.venv\Scripts\python.exe")
    p.add_argument("--home", default=r"D:\blink", help="BLINK_HOME")
    p.add_argument("--data", help="the pack (default HOME/data/v1)")
    p.add_argument("--at-step", type=int, help="branch from this checkpoint (default: runs/long's latest)")
    p.add_argument("--bench-size", default="m", help="supervise's throughput benchmark ('' turns it off)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and the command, change nothing")
    p.add_argument("--vaa-gate-cleared", action="store_true",
                   help="Itay cleared runs/long's P7-VAA gate: the finish may pass it")  # fmt: skip
    args = p.parse_args(argv)
    home = Path(args.home)
    data = Path(args.data) if args.data else home / "data" / "v1"
    return Settings(repo=Path(args.repo), python=Path(args.python), home=home, data=data,
                    bench_size=args.bench_size, at_step=args.at_step, dry_run=args.dry_run,
                    vaa_gate_cleared=args.vaa_gate_cleared)  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    s = settings_from(argv)
    host = Host(s.repo, s.python, s.home, s.logs)
    if s.dry_run:
        return finish(s, host)
    s.logs.mkdir(parents=True, exist_ok=True)
    try:
        lock = acquire_lock(s.logs / "p7v2.lock")
    except LockHeld as exc:
        print(f"p7_finish: {exc}; nothing changed", file=sys.stderr)
        return EXIT_REFUSED
    try:
        return finish(s, host)
    finally:
        release_lock(lock)


if __name__ == "__main__":
    sys.exit(main())
