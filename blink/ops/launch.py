"""`blink ops launch --name NAME -- <blink args>` and `blink ops ps`.

A long job must outlive the agent session that starts it (PF38), so it is created by WMI
(`Invoke-CimMethod -ClassName Win32_Process -MethodName Create`): its parent is the WMI provider host,
not this shell, so it sits outside any session job object and survives the session closing. The
created process is `cmd.exe /d /s /c "set ...&& python -m blink.cli <args> 1>>out 2>>err"`: cmd sets
PYTHONUTF8=1, UV_CACHE_DIR and BLINK_HOME for that process tree only (never setx, never this
process) and redirects stdout and stderr to BLINK_HOME/logs/NAME.out and NAME.err. A WMI-created
process starts from the user's default environment, so BLINK_HOME is passed explicitly.

Arguments go through cmd.exe, so any character cmd would reinterpret (" % ^ & | < > or a line break)
is refused with an explicit error rather than escaped.
"""

import base64
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from blink import heartbeat, paths
from blink.train.atomic import write_text_atomic
from blink.train.status import valid_run_name

UV_CACHE_DIR = r"D:\uv-cache"
CMD_UNSAFE = frozenset('"%^&|<>\r\n')
POWERSHELL = "powershell.exe"
POWERSHELL_TIMEOUT_S = 60
CHILD_WAIT_S = 5.0
WMI_ERRORS = {
    2: "access denied",
    3: "insufficient privilege",
    8: "unknown failure",
    9: "path not found",
    21: "invalid parameter",
}
REPO_ROOT = Path(__file__).resolve().parents[2]


class LaunchError(RuntimeError):
    pass


@dataclass(frozen=True)
class LaunchPlan:
    name: str
    argv: tuple[str, ...]  # the python command the detached cmd.exe runs
    env: tuple[tuple[str, str], ...]
    cwd: str
    out: Path
    err: Path
    heartbeat: Path | None

    @property
    def command_line(self) -> str:
        sets = "".join(f'set "{key}={value}"&& ' for key, value in self.env)
        command = subprocess.list2cmdline(self.argv)
        return f'cmd.exe /d /s /c "{sets}{command} 1>>"{self.out}" 2>>"{self.err}""'

    @property
    def script(self) -> str:
        return powershell_script(self.command_line, self.cwd)


@dataclass(frozen=True)
class LaunchResult:
    pid: int  # the detached cmd.exe
    python_pids: tuple[int, ...]  # its python descendants, found shortly after the launch
    record: Path


def _ps_quote(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def powershell_script(command_line: str, cwd: str) -> str:
    return "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            "$class = Get-CimClass -ClassName Win32_ProcessStartup",
            "$startup = New-CimInstance -CimClass $class -Property @{ShowWindow = [uint16]0} -ClientOnly",
            "$arguments = @{",
            f"    CommandLine = {_ps_quote(command_line)}",
            f"    CurrentDirectory = {_ps_quote(cwd)}",
            "    ProcessStartupInformation = $startup",
            "}",
            "$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments $arguments",
            "Write-Output ('{0} {1}' -f $result.ReturnValue, $result.ProcessId)",
        ]
    )


def encode_command(script: str) -> str:
    """PowerShell's -EncodedCommand form: base64 of UTF-16LE, so no quoting layer can mangle it."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def powershell_argv(script: str) -> list[str]:
    flags = ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
    return [POWERSHELL, *flags, "-EncodedCommand", encode_command(script)]


def _check_safe(args: Iterable[str]) -> None:
    for arg in args:
        bad = sorted(set(arg) & CMD_UNSAFE)
        if bad:
            raise ValueError(
                f"argument {arg!r} holds {bad}, which cmd.exe would reinterpret; refusing to launch"
            )


def heartbeat_of(blink_args: Sequence[str], home: Path) -> Path | None:
    """The heartbeat a launched command writes: runs/<run>/heartbeat.json, or a probe's --out file."""
    args = list(blink_args)
    if "--run" in args[:-1]:
        return home / "runs" / args[args.index("--run") + 1] / "heartbeat.json"
    if "heartbeat-probe" in args and "--out" in args[:-1]:
        return Path(args[args.index("--out") + 1])
    return None


def plan_launch(
    name: str,
    blink_args: Sequence[str],
    home: Path | None = None,
    python: str | None = None,
    cwd: str | None = None,
) -> LaunchPlan:
    """What `blink ops launch` will run. Pure: nothing is created and os.environ is untouched."""
    if not valid_run_name(name):
        raise ValueError(f"bad launch name {name!r} (letters, digits, _ - . only)")
    home = Path(home) if home is not None else paths.home()
    argv = (python or sys.executable, "-m", "blink.cli", *blink_args)
    _check_safe([*argv, str(home), cwd or str(REPO_ROOT)])
    env = (("PYTHONUTF8", "1"), ("UV_CACHE_DIR", UV_CACHE_DIR), ("BLINK_HOME", str(home)))
    logs = home / "logs"
    return LaunchPlan(
        name,
        argv,
        env,
        cwd or str(REPO_ROOT),
        logs / f"{name}.out",
        logs / f"{name}.err",
        heartbeat_of(blink_args, home),
    )


def _python_children(pid: int) -> list[int]:
    """The python processes under the detached cmd.exe (the venv launcher starts one more)."""
    deadline = time.monotonic() + CHILD_WAIT_S
    while time.monotonic() < deadline:
        try:
            found = [p.pid for p in psutil.Process(pid).children(recursive=True)]
        except psutil.NoSuchProcess:
            return []
        if found:
            return found
        time.sleep(0.1)
    return []


def _record_path(plan: LaunchPlan) -> Path:
    return plan.out.parent / f"{plan.name}.launch.json"


def _refuse_if_alive(plan: LaunchPlan) -> None:
    try:
        record = json.loads(_record_path(plan).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if _alive(record.get("pid"), record.get("create_time")):
        raise LaunchError(f"{plan.name} is already running as pid {record['pid']}; pick another --name")


def _alive(pid: Any, create_time: Any) -> bool:
    try:
        return isinstance(pid, int) and psutil.Process(pid).create_time() == create_time
    except psutil.Error:
        return False


def _failure_text(done: subprocess.CompletedProcess) -> str:
    """What PowerShell said on failure: stderr, else stdout, else the exit code alone."""
    said = (done.stderr or "").strip() or (done.stdout or "").strip()
    return said[:300] if said else f"exit code {done.returncode}, no output"


def launch(
    plan: LaunchPlan,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    find_children: Callable[[int], list[int]] = _python_children,
) -> LaunchResult:
    """Create the detached process and record it in BLINK_HOME/logs/NAME.launch.json."""
    plan.out.parent.mkdir(parents=True, exist_ok=True)
    _refuse_if_alive(plan)
    try:
        # PowerShell writes a redirected stream in the console code page, not UTF-8. A strict decode
        # fails in subprocess's reader thread and silently turns that stream into None (PF39).
        done = runner(
            powershell_argv(plan.script),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="backslashreplace",
            timeout=POWERSHELL_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LaunchError(f"could not run {POWERSHELL}: {exc}") from exc
    fields = (done.stdout or "").split()
    if done.returncode != 0 or len(fields) < 2 or not all(f.isdigit() for f in fields[-2:]):
        raise LaunchError(f"PowerShell could not create the process: {_failure_text(done)}")
    code, pid = int(fields[-2]), int(fields[-1])
    if code != 0:
        raise LaunchError(f"Win32_Process.Create returned {code} ({WMI_ERRORS.get(code, 'unknown code')})")
    children = tuple(find_children(pid))
    try:
        create_time = psutil.Process(pid).create_time()
    except psutil.Error:
        create_time = None
    record = {
        "name": plan.name,
        "pid": pid,
        "python_pids": list(children),
        "create_time": create_time,
        "launched": time.time(),
        "argv": list(plan.argv),
        "command_line": plan.command_line,
        "cwd": plan.cwd,
        "out": str(plan.out),
        "err": str(plan.err),
        "heartbeat": None if plan.heartbeat is None else str(plan.heartbeat),
    }
    write_text_atomic(_record_path(plan), json.dumps(record, indent=2) + "\n")
    return LaunchResult(pid, children, _record_path(plan))


# ---------------------------------------------------------------- blink ops ps


def is_blink(cmdline: Sequence[str]) -> bool:
    names = [Path(arg).name.lower() for arg in cmdline]
    if any(name in ("blink.exe", "blink-uci.exe", "blink", "blink-uci") for name in names):
        return True
    return any(
        arg == "-m" and nxt in ("blink.cli", "blink.uci")
        for arg, nxt in zip(cmdline, cmdline[1:], strict=False)
    )


def _describe_beat(path: Path | None, now: float) -> str:
    if path is None:
        return "no heartbeat"
    beat = heartbeat.read(path)
    if beat is None or "time" not in beat:
        return "no heartbeat file"
    parts = [f"{now - float(beat['time']):.0f} s ago"]
    parts += [str(beat[k]) for k in ("state",) if k in beat]
    parts += [f"step {beat['step']}" for k in ("step",) if beat.get(k) is not None]
    parts += [str(beat["stopped"])] if "stopped" in beat else []
    return ", ".join(parts)


def ps_rows(
    processes: Iterable[dict[str, Any]], home: Path, now: float | None = None
) -> list[dict[str, Any]]:
    """One row per Blink process: pid, age, the run it serves and that run's heartbeat."""
    now = time.time() if now is None else now
    rows = []
    for proc in processes:
        cmdline = list(proc.get("cmdline") or [])
        if not is_blink(cmdline):
            continue
        beat = heartbeat_of(cmdline, Path(home))
        run = cmdline[cmdline.index("--run") + 1] if "--run" in cmdline[:-1] else ""
        rows.append(
            {
                "pid": proc["pid"],
                "age_s": now - float(proc.get("create_time") or now),
                "run": run,
                "heartbeat": _describe_beat(beat, now),
                "command": " ".join(cmdline),
            }
        )
    return sorted(rows, key=lambda row: row["pid"])


def _own_lineage() -> set[int]:
    """This process and its ancestors: `blink ops ps` never lists itself or its venv launcher."""
    mine = {os.getpid()}
    with contextlib.suppress(psutil.Error):
        mine |= {p.pid for p in psutil.Process().parents()}
    return mine


def blink_processes(home: Path | None = None) -> list[dict[str, Any]]:
    skip = _own_lineage()
    found = []
    for proc in psutil.process_iter(["pid", "cmdline", "create_time"]):
        info = proc.info
        if info["pid"] not in skip and info.get("cmdline"):
            found.append(info)
    return ps_rows(found, home or paths.home())


def _age(seconds: float) -> str:
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m{rest % 60:02d}s"


def format_ps(rows: Sequence[dict[str, Any]], width: int = 90) -> str:
    if not rows:
        return "no Blink processes"
    lines = [f"{'PID':>7}  {'AGE':>7}  {'RUN':<14}  HEARTBEAT / COMMAND"]
    for row in rows:
        lines.append(f"{row['pid']:>7}  {_age(row['age_s']):>7}  {row['run'][:14]:<14}  {row['heartbeat']}")
        lines.append(f"{'':>34}{row['command'][:width]}")
    return "\n".join(lines)


def launch_records(home: Path | None = None) -> list[dict[str, Any]]:
    """Every BLINK_HOME/logs/*.launch.json, newest first, each marked alive or not."""
    logs = (home or paths.home()) / "logs"
    records = []
    for path in sorted(logs.glob("*.launch.json")) if logs.is_dir() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        beat = Path(record["heartbeat"]) if record.get("heartbeat") else None
        alive = _alive(record.get("pid"), record.get("create_time"))
        records.append({**record, "alive": alive, "beat": _describe_beat(beat, time.time())})
    return sorted(records, key=lambda r: r.get("launched", 0.0), reverse=True)
