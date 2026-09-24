"""`blink supervise`: run the trainer as a child process and enforce every P7 stop rule without an agent.

Every `interval_s` (60 s) the supervisor reads the run's own files and applies, in order:
- nan: a non-finite loss. The first one rolls back once: the child is restarted from the last
  checkpoint with --resume --lr-scale 0.5; a second one stops the run.
- vaa: a 'vaa_check_failed' marker in evals.jsonl pauses the run (the child is terminated, checkpoints
  are kept) with the status "paused: P7-VAA", a pending gate only Itay clears.
- heartbeat: heartbeat.json older than 60 s (after a startup grace for imports and compile).
- throughput: train-phase samples/s more than 15% below the benchmark for 10 consecutive minutes;
  rows whose phase is eval or ckpt are skipped. Off when no benchmark rate is given.
- clip: the clip fraction over the last 1,000 steps is at least 20%.
- loss: the total loss has stayed above 3x its EMA for more than 1,000 steps (rows above the bar
  never move the EMA, so a sustained divergence cannot drag its own baseline up).
- crash: a child that exits non-zero for any other reason is resumed up to 3 times, 60 s apart.

A stop terminates the child's whole process tree and writes 'stopped: <rule>, <number>' into
heartbeat.json (state "stopped" or "paused", so `blink status` exits non-zero) and a STATUS-style
record into supervisor.json. Before each child starts, metrics.jsonl and evals.jsonl are cut back to
the step it resumes from (0 for a fresh start), exactly as the trainer's own resume does, so a row
the supervisor sees afterwards was written by that child. Torch-free.
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import psutil

from blink import heartbeat
from blink.train.atomic import write_text_atomic

EXIT_FINISHED, EXIT_STOPPED, EXIT_REFUSED, EXIT_PAUSED = 0, 1, 2, 3
RULES = ("nan", "vaa", "heartbeat", "throughput", "clip", "loss", "crash")
VAA_MARKER = "vaa_check_failed"
PAUSED_VAA = "paused: P7-VAA"
PENDING_GATE_VAA = "P7-VAA"
QUIET_PHASES = frozenset({"eval", "ckpt"})
LOSS_KEYS = ("loss_policy", "loss_value", "loss")
CHECKPOINT = re.compile(r"^ckpt_(\d+)\.pt$")
RECORD_FILE = "supervisor.json"
MAX_EVENTS = 200

Log = Callable[[str], None]


@dataclass(frozen=True)
class SuperviseConfig:
    interval_s: float = 60.0  # every stop rule is evaluated this often
    poll_s: float = 1.0  # the child's exit is noticed within this
    heartbeat_stale_s: float = 60.0
    startup_grace_s: float = 600.0  # before a child's first step: imports, data, compile
    bench_rate: float | None = None  # samples/s from bench.json; None turns the throughput rule off
    slow_frac: float = 0.15
    slow_window_s: float = 600.0
    clip_max: float = 0.20
    clip_window_steps: int = 1000
    loss_factor: float = 3.0
    loss_window_steps: int = 1000
    loss_ema_steps: float = 1000.0  # EMA time constant, in steps
    max_restarts: int = 3
    backoff_s: float = 60.0
    nan_lr_scale: float = 0.5
    max_rollbacks: int = 1
    terminate_timeout_s: float = 30.0
    disabled: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        unknown = sorted(set(self.disabled) - set(RULES))
        if unknown:
            raise ValueError(f"unknown stop rules {unknown}; the rules are {', '.join(RULES)}")
        if self.interval_s <= 0 or self.poll_s <= 0:
            raise ValueError("interval_s and poll_s must be positive")

    def on(self, rule: str) -> bool:
        return rule not in self.disabled


class Verdict(NamedTuple):
    rule: str
    number: str

    @property
    def status(self) -> str:
        return f"stopped: {self.rule}, {self.number}"


@dataclass(frozen=True)
class Outcome:
    state: str  # finished | stopped | paused | refused
    status: str
    exit_code: int


# ---------------------------------------------------------------- the run's files


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Complete JSON lines only: a torn last line from a live writer is skipped, as is garbage."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    rows = []
    for line in text.split("\n")[:-1]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def truncate_log(path: Path, step: int) -> None:
    """Keep complete rows whose step is <= step: what the trainer's resume does to its logs."""
    if not Path(path).exists():
        return
    kept = [row for row in read_jsonl(path) if isinstance(row.get("step"), int) and row["step"] <= step]
    write_text_atomic(Path(path), "".join(json.dumps(row) + "\n" for row in kept))


def checkpoint_steps(run_dir: Path) -> list[int]:
    if not Path(run_dir).is_dir():
        return []
    matches = (CHECKPOINT.match(p.name) for p in Path(run_dir).iterdir())
    return sorted(int(m.group(1)) for m in matches if m)


# ---------------------------------------------------------------- the rules (pure functions of rows)


def _finite(x: Any) -> bool:
    return not isinstance(x, float) or math.isfinite(x)


def nan_steps(rows: Sequence[dict]) -> frozenset[int]:
    return frozenset(r.get("step", -1) for r in rows if not all(_finite(r.get(k)) for k in LOSS_KEYS))


def vaa_steps(rows: Sequence[dict]) -> frozenset[int]:
    return frozenset(r.get("step", -1) for r in rows if r.get(VAA_MARKER) is True or VAA_MARKER in r.values())


def heartbeat_verdict(beat_time: float | None, started: float, now: float, progressed: bool, cfg):
    """Stale after the startup grace, or as soon as the child has made its first step."""
    if not progressed and now - started < cfg.startup_grace_s:
        return None
    reference = beat_time if beat_time is not None and beat_time >= started else started
    age = now - reference
    return Verdict("heartbeat", f"{age:.0f} s") if age > cfg.heartbeat_stale_s else None


def throughput_verdict(rows: Sequence[dict], cfg: SuperviseConfig) -> Verdict | None:
    """Rows carry their time in '_t'. Only train-phase rows count; a fast one resets the window."""
    if cfg.bench_rate is None:
        return None
    floor = (1.0 - cfg.slow_frac) * cfg.bench_rate
    since, last = None, None
    for row in rows:
        rate = row.get("samples_per_s")
        if row.get("phase", "train") in QUIET_PHASES or not isinstance(rate, int | float):
            continue
        if rate >= floor:
            since = None
            continue
        since = row["_t"] if since is None else since
        last = (row["_t"], rate)
    if since is None or last[0] - since < cfg.slow_window_s:
        return None
    return Verdict("throughput", f"{last[1]:,.0f} samples/s for {last[0] - since:.0f} s (floor {floor:,.0f})")


def clip_verdict(rows: Sequence[dict], cfg: SuperviseConfig) -> Verdict | None:
    """The clip fraction over the last clip_window_steps steps, each row weighted by the steps it covers."""
    ordered = [
        r for r in rows if isinstance(r.get("step"), int) and isinstance(r.get("clip_frac"), int | float)
    ]
    covered, weighted = 0, 0.0
    for i in range(len(ordered) - 1, -1, -1):
        span = ordered[i]["step"] - (ordered[i - 1]["step"] if i else 0)
        take = min(max(span, 0), cfg.clip_window_steps - covered)
        weighted += ordered[i]["clip_frac"] * take
        covered += take
        if covered >= cfg.clip_window_steps:
            break
    if covered < cfg.clip_window_steps:
        return None
    fraction = weighted / covered
    return Verdict("clip", f"{100 * fraction:.1f}%") if fraction >= cfg.clip_max else None


def _total_loss(row: dict) -> float | None:
    if isinstance(row.get("loss"), int | float):
        return float(row["loss"])
    parts = [row.get("loss_policy"), row.get("loss_value")]
    return float(sum(parts)) if all(isinstance(p, int | float) for p in parts) else None


def loss_verdict(rows: Sequence[dict], cfg: SuperviseConfig) -> Verdict | None:
    ema, above_from, previous = None, None, 0
    for row in rows:
        loss, step = _total_loss(row), row.get("step")
        if loss is None or not math.isfinite(loss) or not isinstance(step, int):
            continue
        if ema is None:
            ema = loss
        elif loss > cfg.loss_factor * ema:
            above_from = previous if above_from is None else above_from
            if step - above_from > cfg.loss_window_steps:
                return Verdict("loss", f"{loss / ema:.1f}x its EMA for {step - above_from:,} steps")
        else:
            above_from = None
            ema += (1.0 - math.exp(-(step - previous) / cfg.loss_ema_steps)) * (loss - ema)
        previous = step
    return None


# ---------------------------------------------------------------- command lines


def lr_scale_of(argv: Sequence[str]) -> float:
    for i, arg in enumerate(argv):
        if arg == "--lr-scale" and i + 1 < len(argv):
            return float(argv[i + 1])
        if arg.startswith("--lr-scale="):
            return float(arg.split("=", 1)[1])
    return 1.0


def restart_argv(argv: Sequence[str], resume: bool, lr_scale: float) -> list[str]:
    """argv without its --resume/--lr-scale flags, then the ones this restart needs."""
    out, skip = [], False
    for arg in argv:
        if skip:
            skip = False
        elif arg == "--lr-scale":
            skip = True
        elif arg != "--resume" and not arg.startswith("--lr-scale="):
            out.append(arg)
    out += ["--resume"] if resume else []
    return out + (["--lr-scale", f"{lr_scale:g}"] if lr_scale != 1.0 else [])


def _run_names(args: Sequence[str]) -> list[str]:
    names = [args[i + 1] for i, a in enumerate(args[:-1]) if a == "--run"]
    return names + [a.split("=", 1)[1] for a in args if a.startswith("--run=")]


def run_of(args: Sequence[str], run: str | None) -> str:
    """The run a supervise command serves: its own --run, else the one the train command names."""
    if run is not None:
        return run
    names = _run_names(args)
    if not names:
        raise ValueError("name the run: blink supervise --run NAME, or --run NAME in the train command")
    return names[0]


def train_argv(args: Sequence[str], run: str) -> list[str]:
    """The blink arguments for the child: a `train` command whose --run is this run."""
    args = list(args)
    if not args or args[0] != "train":
        raise ValueError(f"blink supervise wraps `train ...`, got {' '.join(args) or 'nothing'}")
    names = _run_names(args)
    if any(name != run for name in names):
        raise ValueError(f"the train command says --run {names[0]} but supervise says --run {run}")
    return args if names else args + ["--run", run]


def child_argv(blink_args: Sequence[str]) -> list[str]:
    return [sys.executable, "-m", "blink.cli", *blink_args]


def kill_tree(pid: int, timeout_s: float) -> None:
    """Terminate a process and all its descendants (Windows has no process groups to signal)."""
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for proc in procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(procs, timeout=timeout_s)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
    psutil.wait_procs(alive, timeout=timeout_s)


_JOB_HANDLE: list[int] = []  # the kill-on-close job this process sits in; its handle is never closed
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


def _job_limit_structure():
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_uint64 * 6)] + [
            (name, ctypes.c_size_t)
            for name in ("ProcessMemoryLimit", "JobMemoryLimit", "PeakProcessMemoryUsed", "PeakJobMemoryUsed")
        ]

    return Extended


def bind_children_to_this_process() -> str:
    """Windows: put this process in a kill-on-close job, so every child it starts dies with it.

    A supervisor that is killed must never leave its trainer running with no stop rule watching.
    Children inherit the job; the only handle to it is held here, so when this process exits for any
    reason Windows closes the handle and terminates the trainer too. Returns what happened.
    """
    if _JOB_HANDLE:
        return "bound: already in a kill-on-close job"
    if sys.platform != "win32":
        return "not bound: kill-on-close jobs are a Windows feature"
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    job = kernel32.CreateJobObjectW(None, None)
    limits = _job_limit_structure()()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    size = ctypes.sizeof(limits)
    configured = job and kernel32.SetInformationJobObject(
        ctypes.c_void_p(job), JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), size
    )
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    if not configured or not kernel32.AssignProcessToJobObject(
        ctypes.c_void_p(job), ctypes.c_void_p(kernel32.GetCurrentProcess())
    ):
        return f"not bound: Windows error {ctypes.get_last_error()}"
    _JOB_HANDLE.append(job)
    return "bound: children die with this supervisor"


# ---------------------------------------------------------------- the supervisor


@dataclass
class _Child:
    proc: subprocess.Popen
    argv: list[str]
    started: float
    first_step: int | None = None
    progressed: bool = False
    last_beat: float | None = None  # the newest heartbeat time read since this child started


class Supervisor:
    """Live supervision state. Mutable by nature: it owns a child process and counts its restarts."""

    def __init__(self, cfg: SuperviseConfig, run_dir: Path, argv: Sequence[str], log: Log, launch=None):
        self.cfg, self.run_dir, self.argv, self.log = cfg, Path(run_dir), list(argv), log
        self.launch_command = launch
        self.lr_scale = lr_scale_of(argv)
        self.restarts = self.rollbacks = 0
        self.started = time.time()
        self.child: _Child | None = None
        self.observed: dict[Any, float] = {}
        self.baseline_nan: frozenset[int] = frozenset()
        self.baseline_vaa: frozenset[int] = frozenset()
        self.events: list[dict[str, Any]] = []

    def _path(self, name: str) -> Path:
        return self.run_dir / name

    def run(self, deadline_s: float | None = None) -> Outcome:
        if "--resume" not in self.argv and checkpoint_steps(self.run_dir):
            status = f"refused: {self.run_dir.name} already has checkpoints; add --resume to continue it"
            self.log(f"supervise: {status}")
            return Outcome("refused", status, EXIT_REFUSED)
        self._event("job", result=bind_children_to_this_process())
        self._start(resume="--resume" in self.argv)
        next_check = time.time() + self.cfg.interval_s
        try:
            while True:
                code = self.child.proc.poll()
                outcome = self._on_exit(code) if code is not None else None
                if code is None:
                    now = time.time()
                    self._track_progress()
                    if deadline_s is not None and now - self.started >= deadline_s:
                        outcome = self._finish(
                            "stopped", Verdict("wall_clock", f"{now - self.started:.0f} s")
                        )
                    elif now >= next_check:
                        outcome, next_check = self._check(now), now + self.cfg.interval_s
                if outcome is not None:
                    return outcome
                time.sleep(self.cfg.poll_s)
        except KeyboardInterrupt:
            return self._finish("stopped", Verdict("supervisor", "interrupted"))

    def _start(self, resume: bool) -> None:
        steps = checkpoint_steps(self.run_dir)
        cut = steps[-1] if resume and steps else 0
        for name in ("metrics.jsonl", "evals.jsonl"):
            truncate_log(self._path(name), cut)
        self.observed = {}
        self.baseline_nan = nan_steps(read_jsonl(self._path("metrics.jsonl")))
        self.baseline_vaa = vaa_steps(read_jsonl(self._path("evals.jsonl")))
        argv = restart_argv(self.argv, resume, self.lr_scale)
        proc = subprocess.Popen(argv, env={**os.environ, "PYTHONUTF8": "1"})
        self.child = _Child(proc, argv, time.time())
        self._event("start", pid=proc.pid, resume=resume, from_step=cut, lr_scale=self.lr_scale)
        self._write_record("running", "running")

    def _track_progress(self) -> None:
        """Note the child's newest heartbeat, its first step, and whether it has moved on since.

        A heartbeat that cannot be read at this instant (the trainer is replacing it) changes nothing:
        only beats that were read count, so a momentary lock never looks like a stale run.
        """
        beat = heartbeat.read(self._path("heartbeat.json")) or {}
        beat_time = float(beat.get("time", 0.0))
        if beat_time < self.child.started:
            return
        self.child.last_beat = max(self.child.last_beat or beat_time, beat_time)
        if self.child.first_step is None:
            self.child.first_step = beat.get("step")
        elif beat.get("step") != self.child.first_step:
            self.child.progressed = True

    def _timed(self, rows: list[dict], now: float) -> list[dict]:
        """Each row with '_t': its own 'time' if it has one, else when the supervisor first saw it."""
        out = []
        for row in rows:
            t = row.get("time")
            if not isinstance(t, int | float):
                t = self.observed.setdefault(row.get("step"), now)
            out.append({**row, "_t": float(t)})
        return out

    def _check(self, now: float) -> Outcome | None:
        rows, evals = read_jsonl(self._path("metrics.jsonl")), read_jsonl(self._path("evals.jsonl"))
        fresh_nan = nan_steps(rows) - self.baseline_nan
        if self.cfg.on("nan") and fresh_nan:
            return self._nan(min(fresh_nan))
        if self.cfg.on("vaa") and vaa_steps(evals) - self.baseline_vaa:
            return self._finish("paused", None)
        self._track_progress()
        child, cfg = self.child, self.cfg
        checks = (
            (
                "heartbeat",
                lambda: heartbeat_verdict(child.last_beat, child.started, now, child.progressed, cfg),
            ),
            ("throughput", lambda: throughput_verdict(self._timed(rows, now), cfg)),
            ("clip", lambda: clip_verdict(rows, cfg)),
            ("loss", lambda: loss_verdict(rows, cfg)),
        )
        for rule, check in checks:
            verdict = check() if cfg.on(rule) else None
            if verdict is not None:
                return self._finish("stopped", verdict)
        self._write_record("running", "running", rows, evals)
        return None

    def _on_exit(self, code: int) -> Outcome | None:
        fresh_nan = nan_steps(read_jsonl(self._path("metrics.jsonl"))) - self.baseline_nan
        if self.cfg.on("nan") and (fresh_nan or self._beat_says_nan()):
            return self._nan(min(fresh_nan) if fresh_nan else -1)
        if self.cfg.on("vaa") and vaa_steps(read_jsonl(self._path("evals.jsonl"))) - self.baseline_vaa:
            return self._finish("paused", None)
        if code == 0:
            self._event("exit", code=0)
            return self._finish("finished", None)
        return self._crash(code)

    def _beat_says_nan(self) -> bool:
        beat = heartbeat.read(self._path("heartbeat.json")) or {}
        error = str(beat.get("error", ""))
        fresh = float(beat.get("time", 0.0)) >= self.child.started
        return fresh and ("non-finite" in error or "FloatingPointError" in error)

    def _nan(self, step: int) -> Outcome | None:
        if self.rollbacks >= self.cfg.max_rollbacks:
            return self._finish("stopped", Verdict("nan", f"step {step:,}"))
        self.rollbacks += 1
        self.lr_scale *= self.cfg.nan_lr_scale
        self._event("rollback", step=step, lr_scale=self.lr_scale)
        self._end_child()
        self._start(resume=bool(checkpoint_steps(self.run_dir)))
        return None

    def _crash(self, code: int) -> Outcome | None:
        if not self.cfg.on("crash") or self.restarts >= self.cfg.max_restarts:
            return self._finish("stopped", Verdict("crash", f"exit {code} after {self.restarts} restarts"))
        self.restarts += 1
        self._event("crash", code=code, restart=self.restarts, backoff_s=self.cfg.backoff_s)
        time.sleep(self.cfg.backoff_s)
        self._start(resume=bool(checkpoint_steps(self.run_dir)))
        return None

    def _end_child(self) -> None:
        if self.child is not None and self.child.proc.poll() is None:
            kill_tree(self.child.proc.pid, self.cfg.terminate_timeout_s)
            self.child.proc.wait(timeout=self.cfg.terminate_timeout_s)

    def _finish(self, state: str, verdict: Verdict | None) -> Outcome:
        """Stop, pause or finish: the child is gone afterwards and every file says why."""
        self._end_child()
        status = {"finished": "finished", "paused": PAUSED_VAA}.get(state) or verdict.status
        if state in ("stopped", "paused"):
            beat = heartbeat.read(self._path("heartbeat.json")) or {}
            payload = {**beat, "state": state, "stopped": status, "supervisor_pid": os.getpid()}
            try:
                heartbeat.write(self._path("heartbeat.json"), payload)
            except OSError as exc:  # a reader held the file past every retry; supervisor.json still says why
                self.log(f"supervise {self.run_dir.name}: could not write heartbeat.json: {exc}")
        self._event(state, status=status)
        self._write_record(state, status)
        code = {"finished": EXIT_FINISHED, "paused": EXIT_PAUSED}.get(state, EXIT_STOPPED)
        return Outcome(state, status, code)

    def _event(self, name: str, **fields: Any) -> None:
        self.events.append({"time": time.time(), "event": name, **fields})
        details = ", ".join(f"{k} {v}" for k, v in fields.items())
        self.log(f"supervise {self.run_dir.name}: {name}{': ' + details if details else ''}")

    def _write_record(self, state: str, status: str, rows=None, evals=None) -> None:
        rows = read_jsonl(self._path("metrics.jsonl")) if rows is None else rows
        evals = read_jsonl(self._path("evals.jsonl")) if evals is None else evals
        record = {
            "run": self.run_dir.name,
            "state": state,
            "status": status,
            "pending_gate": PENDING_GATE_VAA if status == PAUSED_VAA else None,
            "pid": os.getpid(),
            "child_pid": self.child.proc.pid if self.child else None,
            "launch_command": self.launch_command,
            "command": self.child.argv if self.child else self.argv,
            "resume_command": restart_argv(self.argv, True, self.lr_scale),
            "restarts": self.restarts,
            "rollbacks": self.rollbacks,
            "lr_scale": self.lr_scale,
            "disabled": list(self.cfg.disabled),
            "config": asdict(self.cfg),
            "started": self.started,
            "updated": time.time(),
            "last_metrics": rows[-1] if rows else None,
            "last_eval": evals[-1] if evals else None,
            "events": self.events[-MAX_EVENTS:],
        }
        try:
            write_text_atomic(self._path(RECORD_FILE), json.dumps(record, indent=2) + "\n")
        except OSError as exc:  # never let a locked status file take down the supervision itself
            self.log(f"supervise {self.run_dir.name}: could not write {RECORD_FILE}: {exc}")


def supervise(
    cfg: SuperviseConfig,
    run_dir: Path,
    argv: Sequence[str],
    log: Log = print,
    deadline_s: float | None = None,
    launch_command: str | None = None,
) -> Outcome:
    """Run argv under supervision until it finishes or a stop rule fires."""
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    return Supervisor(cfg, Path(run_dir), argv, log, launch_command).run(deadline_s)


def read_record(run_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(run_dir) / RECORD_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def status_exit_code(record: dict[str, Any] | None) -> int:
    """0 while the supervised run is running or has finished; 1 once any stop rule has fired."""
    return 0 if record is not None and record.get("state") in ("running", "finished") else 1
