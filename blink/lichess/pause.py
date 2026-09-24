"""Pausing the Lichess bot: the agent's only action on a running bot is stopping its process by PID.

    blink lichess pause --bot NAME [--timeout S] [--reason TEXT]
    blink lichess resume-note

`pause` first writes BLINK_HOME/lichess/PAUSED, so `start-bot.ps1` (and a Task Scheduler start) will
not start the bot again. It then polls the public status API, without a token, until the bot is in
no game, so a pause never forfeits a live game unless the game outlasts --timeout. Then it finds the
lichess-bot process by its command line (`lichess-bot.py`, run from under D:\\blink-bot), stops that
process and every process below it (the game workers and each game's blink-uci engine), and records
what it did in BLINK_HOME/lichess/pause.json.

lichess-bot keeps accepting challenges while the pause waits, because nothing tells it to stop; the
wait ends at the first poll that shows no game. `resume-note` only deletes the flag: Itay restarts the
bot himself with start-bot.ps1. The agent never starts the bot and never reads its token or its
environment (a process's command line and working folder are all this module looks at).
"""

import contextlib
import datetime
import json
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePath

import psutil

from blink.lichess import snapshot
from blink.train.atomic import write_text_atomic

BOT_ROOT = Path("D:/blink-bot")
BOT_SCRIPT = "lichess-bot.py"
FLAG_NAME = "PAUSED"
RECORD_NAME = "pause.json"
DEFAULT_TIMEOUT_S = 1800.0
DEFAULT_POLL_S = 15.0
STOP_GRACE_S = 10.0
RESTART_NOTE = "Itay restarts the bot himself with D:\\blink-bot\\start-bot.ps1; the agent never starts it."


@dataclass(frozen=True)
class ProcessRow:
    pid: int
    ppid: int
    create_time: float
    cmdline: tuple[str, ...]
    exe: str | None
    cwd: str | None


@dataclass(frozen=True)
class BotProcess:
    pid: int
    ppid: int
    create_time: float
    cmdline: tuple[str, ...]


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class PauseDeps:
    is_playing: Callable[[], bool]
    find_bot: Callable[[], Sequence[BotProcess]]
    stop_tree: Callable[[BotProcess], Sequence[int]]
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], str] = _utc_now
    log: Callable[[str], None] = print


@dataclass(frozen=True)
class WaitResult:
    polls: int
    waited_s: float
    live: bool


@dataclass(frozen=True)
class PauseRecord:
    bot: str
    reason: str
    flag: str
    paused_at: str
    stopped_at: str
    polls: int
    waited_s: float
    timeout_s: float
    timed_out: bool
    live_game_at_stop: bool
    processes: tuple[dict, ...]


# ---------------------------------------------------------------- finding and stopping the bot


def _norm(path: str | PurePath) -> str:
    return str(path).replace("\\", "/").rstrip("/").lower()


def is_under(path: str | None, root: str | PurePath) -> bool:
    return bool(path) and _norm(path).startswith(_norm(root) + "/")


def _name(path: str | None) -> str:
    return PurePath((path or "").replace("\\", "/")).name.lower()


def _is_bot(row: ProcessRow, root: str | PurePath) -> bool:
    """A Python interpreter running lichess-bot.py from under `root` (an editor with the file open is not)."""
    program = _name(row.exe) or _name(row.cmdline[0] if row.cmdline else None)
    is_python = program.startswith("python")
    runs_script = any(_name(arg) == BOT_SCRIPT for arg in row.cmdline[1:])
    places = (row.exe, row.cwd, *row.cmdline)
    return is_python and runs_script and any(is_under(place, root) for place in places)


def _process_rows() -> list[ProcessRow]:
    rows = []
    for proc in psutil.process_iter(["pid", "ppid", "create_time", "cmdline", "exe"]):
        info = proc.info
        cwd = None
        with contextlib.suppress(psutil.Error):
            cwd = proc.cwd()
        cmdline = tuple(info.get("cmdline") or ())
        rows.append(
            ProcessRow(
                info["pid"],
                info.get("ppid") or 0,
                info.get("create_time") or 0.0,
                cmdline,
                info.get("exe"),
                cwd,
            )
        )
    return rows


def find_bot_processes(
    root: str | PurePath = BOT_ROOT, rows: Iterable[ProcessRow] | None = None
) -> tuple[BotProcess, ...]:
    """The top lichess-bot processes run from under `root` (a match whose parent also matches is skipped)."""
    matches = {row.pid: row for row in (rows if rows is not None else _process_rows()) if _is_bot(row, root)}
    return tuple(
        BotProcess(row.pid, row.ppid, row.create_time, row.cmdline)
        for row in matches.values()
        if row.ppid not in matches
    )


def stop_tree(bot: BotProcess, grace_s: float = STOP_GRACE_S) -> tuple[int, ...]:
    """Stop the bot process, then everything below it; a PID now owned by another process is left alone.

    The parent goes first so that lichess-bot's worker pool cannot start replacements for the
    workers being stopped.
    """
    try:
        root = psutil.Process(bot.pid)
        if root.create_time() != bot.create_time:
            return ()
        family = [root, *root.children(recursive=True)]
    except psutil.Error:
        return ()
    for proc in family:
        with contextlib.suppress(psutil.Error):
            proc.terminate()
    _, alive = psutil.wait_procs(family, timeout=grace_s)
    for proc in alive:
        with contextlib.suppress(psutil.Error):
            proc.kill()
    return tuple(proc.pid for proc in family)


# ---------------------------------------------------------------- the pause


def write_flag(path: Path, reason: str, when: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    note = (
        f"Blink bot paused at {when}: {reason}\nDelete this file when the bot is restarted.\n{RESTART_NOTE}\n"
    )
    write_text_atomic(path, note)
    return path


def _still_live(deps: PauseDeps) -> bool:
    try:
        return deps.is_playing()
    except (snapshot.ApiError, OSError, ValueError) as exc:
        deps.log(f"  could not read the bot's status ({exc}); counting it as still in a game")
        return True


def wait_for_no_live_game(deps: PauseDeps, timeout_s: float, poll_s: float) -> WaitResult:
    """Poll until the bot is in no game, or until `timeout_s` has passed."""
    start = deps.clock()
    polls = 0
    while True:
        polls += 1
        live = _still_live(deps)
        waited = deps.clock() - start
        if not live or waited >= timeout_s:
            return WaitResult(polls=polls, waited_s=waited, live=live)
        deps.log(f"  the bot is in a game; checking again in {poll_s:g} s ({waited:.0f} of {timeout_s:g} s)")
        deps.sleep(poll_s)


def pause(
    bot: str, deps: PauseDeps, lichess_dir: Path, timeout_s: float, poll_s: float, reason: str
) -> PauseRecord:
    snapshot.check_name(bot)
    lichess_dir = Path(lichess_dir)
    paused_at = deps.now()
    flag = write_flag(lichess_dir / FLAG_NAME, reason, paused_at)
    deps.log(f"flag written: {flag}")
    wait = wait_for_no_live_game(deps, timeout_s, poll_s)
    if wait.live:
        deps.log(f"  still in a game after {wait.waited_s:.0f} s: stopping anyway (--timeout {timeout_s:g})")
    processes = tuple(
        {"pid": proc.pid, "cmdline": list(proc.cmdline), "stopped": list(deps.stop_tree(proc))}
        for proc in deps.find_bot()
    )
    record = PauseRecord(
        bot=bot,
        reason=reason,
        flag=str(flag),
        paused_at=paused_at,
        stopped_at=deps.now(),
        polls=wait.polls,
        waited_s=round(wait.waited_s, 1),
        timeout_s=timeout_s,
        timed_out=wait.live,
        live_game_at_stop=wait.live,
        processes=processes,
    )
    write_text_atomic(lichess_dir / RECORD_NAME, json.dumps(asdict(record), indent=2) + "\n")
    return record


def default_deps(bot: str, api: snapshot.PublicApi, root: str | PurePath = BOT_ROOT) -> PauseDeps:
    return PauseDeps(
        is_playing=lambda: api.is_playing(bot),
        find_bot=lambda: find_bot_processes(root),
        stop_tree=stop_tree,
    )


def format_record(record: PauseRecord) -> list[str]:
    if not record.processes:
        lines = [f"no lichess-bot process found under {BOT_ROOT}; nothing to stop (the flag stays)"]
    else:
        lines = [
            f"stopped the bot: pid {p['pid']} and {len(p['stopped']) - 1} below it"
            if p["stopped"]
            else f"pid {p['pid']} had already gone (or its PID was reused)"
            for p in record.processes
        ]
    game = "a live game was cut off" if record.live_game_at_stop else "no live game"
    lines.append(f"{game}; waited {record.waited_s:g} s over {record.polls} polls; recorded in {RECORD_NAME}")
    lines.append(f"to restart: `blink lichess resume-note` deletes the flag, then {RESTART_NOTE}")
    return lines


def resume_note(flag: Path) -> str:
    flag = Path(flag)
    if flag.exists():
        flag.unlink()
        return f"deleted {flag}. {RESTART_NOTE}"
    return f"no PAUSED flag at {flag}. {RESTART_NOTE}"
