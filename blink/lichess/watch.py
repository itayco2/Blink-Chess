"""The bot's stop rule, enforced with no agent session alive (plan P9 and section 4, long jobs).

    blink lichess watch --bot NAME [--every S] [--window N] [--pgn-dir DIR] [--rounds N]

The bot runs for days and comes back at every logon, so its stop rule cannot wait for an agent to
poll. `watch` runs beside it, started by a Task Scheduler task of its own (RUNBOOK section 8). Every
--every seconds (default 120) it reads the PGNs lichess-bot saves and the public export (no token),
applies the stop rule (monitor.py), and writes a heartbeat to BLINK_HOME/lichess/watch.json.

When the rule fires on a game that no earlier stop acted on, the watcher pauses the bot at once. A
stop-rule pause does not wait for the live game: while it waits, a broken engine loses that game
anyway and lichess-bot keeps accepting new ones. The games a stop acted on are kept in
BLINK_HOME/lichess/stop-rule.json, so a bot Itay restarts is not stopped again for the same games
while they are still among the last 50; any new time loss or abort by the bot counts at once.

While the PAUSED flag exists the bot is down, and the watcher only writes its heartbeat. When the
public API fails, the round judges the local PGNs alone and its heartbeat says so. A round that
cannot pause (a folder it cannot write, say) is logged, and the next round tries again.
"""

import datetime
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from blink.lichess import monitor, pause, snapshot
from blink.train.atomic import write_text_atomic

DEFAULT_EVERY_S = 120.0
HEARTBEAT_NAME = "watch.json"
ACTED_NAME = "stop-rule.json"


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class WatchDeps:
    exported: Callable[[], Sequence[snapshot.BotGame]]  # the public export; may raise ApiError
    local: Callable[[], Sequence[snapshot.BotGame]]  # lichess-bot's own PGNs
    pause_now: Callable[[str], object]  # pause the bot with this reason, without waiting
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], str] = utc_now
    log: Callable[[str], None] = print


@dataclass(frozen=True)
class RoundResult:
    at: str
    paused_flag: bool
    api: str
    verdict: str
    action: str


# ---------------------------------------------------------------- acting once per piece of evidence


def _acted_on(lichess_dir: Path, log: Callable[[str], None]) -> frozenset[str]:
    """Games an earlier stop acted on; an unreadable record counts as none (the rule errs toward stopping)."""
    path = lichess_dir / ACTED_NAME
    if not path.is_file():
        return frozenset()
    try:
        return frozenset(json.loads(path.read_text(encoding="utf-8")).get("games", []))
    except (OSError, ValueError, AttributeError) as exc:
        log(f"  cannot read {path} ({exc}); treating every counted game as new")
        return frozenset()


def _record_stop(lichess_dir: Path, verdict: monitor.StopVerdict, now: str, acted: frozenset[str]) -> None:
    record = {
        "stopped_at": now,
        "reasons": list(verdict.reasons),
        "games": sorted(acted | set(verdict.offenses)),
    }
    write_text_atomic(lichess_dir / ACTED_NAME, json.dumps(record, indent=2) + "\n")


def enforce(
    verdict: monitor.StopVerdict,
    lichess_dir: Path,
    pause_now: Callable[[str], object],
    now: str,
    log: Callable[[str], None] = print,
) -> str:
    """Pause the bot when the rule fires on a game no earlier stop acted on; returns what was done."""
    if not verdict.stop:
        return "none"
    lichess_dir = Path(lichess_dir)
    acted = _acted_on(lichess_dir, log)
    fresh = [game for game in verdict.offenses if game not in acted]
    if not fresh:
        return f"none: the rule fires only on games an earlier stop acted on ({', '.join(verdict.offenses)})"
    reason = "stop rule: " + "; ".join(verdict.reasons)
    pause_now(reason)
    _record_stop(lichess_dir, verdict, now, acted)
    return f"paused the bot ({reason}; new: {', '.join(fresh)})"


# ---------------------------------------------------------------- the loop


def _write_heartbeat(lichess_dir: Path, bot: str, round_no: int, every_s: float, result: RoundResult) -> None:
    beat = {"bot": bot, "pid": os.getpid(), "round": round_no, "every_s": every_s, **asdict(result)}
    lichess_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(lichess_dir / HEARTBEAT_NAME, json.dumps(beat, indent=2) + "\n")


def _judge(bot: str, deps: WatchDeps, lichess_dir: Path, window: int, at: str) -> RoundResult:
    local = tuple(deps.local())
    try:
        exported, api = tuple(deps.exported()), "ok"
    except (snapshot.ApiError, OSError, ValueError) as exc:
        exported, api = (), f"failed ({exc}); judged the local PGNs alone"
    verdict = monitor.stop_rule(monitor.merge(exported, local), window=window)
    action = enforce(verdict, lichess_dir, deps.pause_now, at, deps.log)
    return RoundResult(at, False, api, monitor.format_verdict(bot, verdict), action)


def watch_round(
    bot: str,
    deps: WatchDeps,
    lichess_dir: Path,
    window: int = monitor.WINDOW,
    round_no: int = 1,
    every_s: float = DEFAULT_EVERY_S,
) -> RoundResult:
    """One check: skipped while the bot is paused, otherwise the rule over local and public games."""
    lichess_dir = Path(lichess_dir)
    at = deps.now()
    if (lichess_dir / pause.FLAG_NAME).exists():
        result = RoundResult(at, True, "not read", "not judged", "none: the bot is paused")
    else:
        result = _judge(bot, deps, lichess_dir, window, at)
    _write_heartbeat(lichess_dir, bot, round_no, every_s, result)
    deps.log(f"{at} {result.verdict}; action: {result.action}")
    return result


def watch(
    bot: str,
    deps: WatchDeps,
    lichess_dir: Path,
    every_s: float = DEFAULT_EVERY_S,
    rounds: int | None = None,
    window: int = monitor.WINDOW,
) -> int:
    """Check every `every_s` seconds, forever or for `rounds` rounds; returns the rounds run."""
    done = 0
    while rounds is None or done < rounds:
        if done:
            deps.sleep(every_s)
        done += 1
        try:
            watch_round(bot, deps, lichess_dir, window, done, every_s)
        except OSError as exc:
            deps.log(f"round {done} failed ({exc}); the next round tries again in {every_s:g} s")
    return done
