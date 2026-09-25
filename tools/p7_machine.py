"""The machine side of the P7 tools (tools/p7_v2_driver.py, tools/p7_finish.py): files, blink commands,
processes, the user pause flag and the single-instance lock.

Like the tools, it imports nothing from the repo (only the stdlib and psutil), so code changes cannot
reach a detached driver mid-run; where it mirrors a blink rule, the docstring names the twin and the
tests hold the two to each other. Every wait here counts only the time Blink was not paused by the user:
a heartbeat or supervisor.json state "paused: user" (blink.train.userpause.PAUSED_USER) is alive.
"""

import json
import os
import re
import subprocess
import time
import tomllib
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import psutil

PAUSE_FLAG = "PAUSE"  # blink.train.userpause.FLAG_NAME: BLINK_HOME/PAUSE, the Pause Blink button's flag
PAUSED_USER = "paused: user"  # blink.train.userpause.PAUSED_USER
EXIT_USER_PAUSE = 75  # blink.train.supervise.EXIT_USER_PAUSE: `blink train (calibrate)` let go for the flag
PAUSE_POLL_S = 30.0  # how often a paused tool looks for the flag again
CHECKPOINT = re.compile(r"^ckpt_(\d+)\.pt$")
KEEPER = "keep_training_priority.ps1"
DETACHED = 0x00000008 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
LOCK_GRACE_S = 5.0  # an unreadable lock younger than this is being written by its owner


class StepFailed(RuntimeError):
    """A step of a P7 tool failed; `step` and `detail` go to its status file."""

    def __init__(self, step: str, detail: str) -> None:
        super().__init__(f"{step}: {detail}")
        self.step, self.detail = step, detail


# ---------------------------------------------------------------- files


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_json_or_empty(path: Path) -> dict[str, Any]:
    """A status file another process replaces atomically: missing, torn or locked reads as empty."""
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Complete JSON lines (a torn last line from a live writer is skipped)."""
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").split("\n")[:-1]:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return [row for row in rows if isinstance(row, dict)]


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


def checkpoint_steps(run_dir: Path) -> list[int]:
    if not Path(run_dir).is_dir():
        return []
    return sorted(int(m.group(1)) for p in Path(run_dir).iterdir() if (m := CHECKPOINT.match(p.name)))


def checkpoint_name(step: int) -> str:
    return f"ckpt_{step:09d}.pt"


def train_table(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """A config's [train] table with its `base` chain resolved (blink.model.config.read_tables)."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"config base cycle at {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    base = train_table(path.parent / str(data["base"]), (*seen, path)) if "base" in data else {}
    return {**base, **data.get("train", {})}


def train_literal(path: Path, key: str) -> str | None:
    """The literal text of a number the file's own [train] table sets (its base is not read), or None."""
    table, pattern = None, re.compile(rf"^\s*{re.escape(key)}\s*=\s*([-+0-9._eE]+)")
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            table = stripped.split("#", 1)[0].strip()
        elif table == "[train]" and (match := pattern.match(stripped)):
            return match.group(1)
    return None


def parse_number(text: str) -> tuple[float, float]:
    """'2,621.44' or '1.6e-3' -> the value and half its last printed digit ((2621.44, 0.005))."""
    try:
        exponent = Decimal(text.replace(",", "").replace("_", "")).as_tuple().exponent
    except InvalidOperation:
        raise ValueError(f"{text!r} is not a number") from None
    return float(text.replace(",", "").replace("_", "")), 0.5 * 10.0 ** int(exponent)


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


# ---------------------------------------------------------------- the user pause


def flag_path(home: Path) -> Path:
    return Path(home) / PAUSE_FLAG


def run_state(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """(heartbeat.json, supervisor.json) of a run, each {} when missing or mid-write."""
    return read_json_or_empty(Path(run_dir) / "heartbeat.json"), read_json_or_empty(
        Path(run_dir) / "supervisor.json"
    )


def user_paused(home: Path, *records: dict[str, Any]) -> bool:
    """The flag is up, or a heartbeat or supervisor.json says "paused: user": Blink is alive, only paused."""
    return flag_path(home).exists() or any(record.get("state") == PAUSED_USER for record in records)


def wait_while_flagged(home: Path, host, on_wait) -> float:
    """Block while BLINK_HOME/PAUSE is up, calling on_wait() before each look; the seconds waited."""
    started = host.clock()
    while flag_path(home).exists():
        on_wait()
        host.sleep(PAUSE_POLL_S)
    return host.clock() - started


# ---------------------------------------------------------------- the single-instance lock


class LockHeld(RuntimeError):
    pass


def process_identity(pid: int) -> float | None:
    """A process's create time: with its pid, what tells it from a later process that reused the pid."""
    try:
        return psutil.Process(pid).create_time()
    except psutil.Error:
        return None


def _holder_alive(record: dict[str, Any], identity) -> bool:
    pid, created = record.get("pid"), record.get("create_time")
    return isinstance(pid, int) and created is not None and identity(pid) == created


def acquire_lock(path: Path, pid: int | None = None, identity=process_identity) -> Path:
    """Create the lock with O_EXCL, holding this pid and its create time. A lock whose holder is alive
    refuses (LockHeld); a stale one (its pid dead, or reused by another process) is replaced."""
    pid = os.getpid() if pid is None else pid
    record = json.dumps({"pid": pid, "create_time": identity(pid), "time": time.time()})
    for _ in range(3):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            held = read_json_or_empty(path)
            young = time.time() - _mtime(path) < LOCK_GRACE_S
            if _holder_alive(held, identity) or (not held and young):
                raise LockHeld(f"another instance (pid {held.get('pid', '?')}) holds {path}") from None
            Path(path).unlink(missing_ok=True)  # stale: its holder is gone
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(record)
        return Path(path)
    raise LockHeld(f"could not take {path}: it keeps coming back")


def _mtime(path: Path) -> float:
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def release_lock(path: Path, pid: int | None = None) -> None:
    """Remove the lock if this process holds it."""
    if read_json_or_empty(path).get("pid") == (os.getpid() if pid is None else pid):
        Path(path).unlink(missing_ok=True)


# ---------------------------------------------------------------- the machine


def decode_lines(stdout: str | None) -> list[str]:
    """PowerShell's non-empty output lines; no output at all is no lines."""
    return [line for line in (stdout or "").splitlines() if line.strip()]


POWERSHELL_UTF8 = "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "


class Host:
    """The real machine: blink commands as children, process listings, detached starts, the clock."""

    def __init__(self, repo: Path, python: Path, home: Path, logs: Path) -> None:
        self.repo, self.python, self.home, self.logs = Path(repo), Path(python), Path(home), Path(logs)
        self.env = {**os.environ, "BLINK_HOME": str(home), "PYTHONUTF8": "1"}

    def run(self, step: str, blink_args: list[str]) -> tuple[int, str]:
        """`python -m blink.cli ARGS` from the repo; stdout and stderr go to logs/p7v2-<step>.out|err."""
        out_path, err_path = self.logs / f"p7v2-{step}.out", self.logs / f"p7v2-{step}.err"
        argv = [str(self.python), "-m", "blink.cli", *blink_args]
        with open(out_path, "w", encoding="utf-8") as out, open(err_path, "w", encoding="utf-8") as err:
            code = subprocess.run(argv, cwd=self.repo, env=self.env, stdout=out, stderr=err).returncode
        return code, out_path.read_text(encoding="utf-8", errors="replace")

    def powershell(self, query: str) -> list[str]:
        """A PowerShell query's output lines, sent as UTF-8 and decoded so no byte can fail the read."""
        argv = ["powershell", "-NoProfile", "-Command", POWERSHELL_UTF8 + query]
        done = subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="backslashreplace"
        )
        return decode_lines(done.stdout)

    def command_lines(self) -> list[str]:
        """Every python.exe and powershell.exe command line (Windows has no pgrep)."""
        return self.powershell(
            "Get-CimInstance Win32_Process | Where-Object { $_.Name -in 'python.exe','powershell.exe' } | "
            "ForEach-Object { $_.CommandLine }"
        )

    def stop_endgame_screen(self) -> None:
        """Kill any `blink eval endgames` tree (its Stockfish children too); it resumes from its cache."""
        self.powershell(
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            r"Where-Object { $_.CommandLine -match 'blink\.cli\s+eval\s+endgames' } | "
            "ForEach-Object { taskkill /PID $_.ProcessId /T /F }"
        )

    def start_keeper(self, script: Path) -> None:
        args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-WindowStyle", "Hidden", "-File"]
        subprocess.Popen([*args, str(script)], cwd=script.parent, creationflags=DETACHED)

    def processes(self) -> list[dict[str, Any]]:
        """pid, ppid and cmdline of every process that has a command line."""
        found = []
        for proc in psutil.process_iter(["pid", "ppid", "cmdline"]):
            if proc.info.get("cmdline"):
                found.append(dict(proc.info))
        return found

    def kill_tree(self, pid: int) -> None:
        """End a process and its descendants (supervise.kill_tree): terminate, then kill what is left."""
        try:
            parent = psutil.Process(pid)
            procs = [*parent.children(recursive=True), parent]
        except psutil.NoSuchProcess:
            return
        for proc in procs:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                continue
        _, alive = psutil.wait_procs(procs, timeout=30)
        for proc in alive:
            try:
                proc.kill()
            except psutil.NoSuchProcess:
                continue

    def clock(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)
