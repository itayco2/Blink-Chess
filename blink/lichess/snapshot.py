"""The bot's public numbers from the public Lichess API: no token, one request at a time.

    blink lichess snapshot --bot NAME [--no-write]

Three GET requests, none authenticated: the profile (`/api/user/NAME`: blitz rating, RD and the rated
blitz game count N), the rating history, and the public game export (`/api/games/user/NAME`, rated
blitz, ndjson). From the exported games it computes, from the bot's side:

- human share: games against opponents without the BOT title, of the games actually played;
- performance vs humans and vs bots: the rating R at which the expected score
  sum 1/(1 + 10^((r_i - R)/400)) equals the real score, capped 800 beyond the opponents' range
  at 0% or 100% (aborted games excluded);
- time-loss rate: games the bot lost on the clock (`outoftime`) or by leaving (`timeout`), of all games;
- abort rate: `aborted` or `noStart` games, of all games;
- duplicate rate: games whose first 20 plies repeat an earlier game's against the same opponent, of
  the games at least 20 plies long (deterministic play repeats itself; plan F18).

Lichess asks API users to make one request at a time and to wait a full minute after a 429. The
snapshot is written to results/lichess.json through blink.report.results_schema.LichessSnapshot,
which marks it publishable only at N >= 200 and RD < 75.
"""

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from blink.report.results_schema import LichessSnapshot, lichess_to_json
from blink.train.atomic import write_text_atomic

BASE_URL = "https://lichess.org"
USER_AGENT = "blink-chess/0.1 (public read-only snapshot; github.com/itayco2/Blink-Chess)"
NDJSON = "application/x-ndjson"
RETRY_AFTER_429_S = 60.0
RETRIES_429 = 3
MIN_GAP_S = 1.0
TIMEOUT_S = 60.0
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,29}$")

ABORT_STATUSES = frozenset({"aborted", "noStart"})
TIME_LOSS_STATUSES = frozenset({"outoftime", "timeout"})
DUPLICATE_PLIES = 20
PERF_CAP = 800.0
BLITZ = "blitz"


class ApiError(RuntimeError):
    """The public API answered with an error status (after the 429 retries)."""


def check_name(name: str) -> str:
    """A Lichess username (2-30 letters, digits, _ or -); anything else never reaches a URL."""
    if not NAME.match(name or ""):
        raise ValueError(f"bad Lichess bot name {name!r}: 2-30 letters, digits, _ or -")
    return name


# ---------------------------------------------------------------- the public API client


class PublicApi:
    """GET requests to the public Lichess API, serialised, with no credentials of any kind."""

    def __init__(
        self,
        opener: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        base: str = BASE_URL,
        min_gap_s: float = MIN_GAP_S,
        retries: int = RETRIES_429,
    ) -> None:
        self._open = opener or urllib.request.urlopen
        self._sleep = sleep
        self._clock = clock
        self._base = base.rstrip("/")
        self._min_gap_s = min_gap_s
        self._retries = retries
        self._last: float | None = None

    def _url(self, path: str, params: Mapping[str, Any] | None) -> str:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        return f"{self._base}{path}" + (f"?{query}" if query else "")

    def _wait_turn(self) -> None:
        if self._last is None:
            return
        wait = self._min_gap_s - (self._clock() - self._last)
        if wait > 0:
            self._sleep(wait)

    def _read(self, path: str, params: Mapping[str, Any] | None, accept: str) -> bytes:
        url = self._url(path, params)
        headers = {"Accept": accept, "User-Agent": USER_AGENT}
        for attempt in range(self._retries + 1):
            self._wait_turn()
            try:
                with self._open(
                    urllib.request.Request(url, headers=headers, method="GET"), timeout=TIMEOUT_S
                ) as r:
                    return r.read()
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == self._retries:
                    raise ApiError(f"GET {path}: HTTP {exc.code}") from None
                self._sleep(RETRY_AFTER_429_S)
            finally:
                self._last = self._clock()
        raise ApiError(f"GET {path}: gave up")  # pragma: no cover (the loop always returns or raises)

    def get_json(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return json.loads(self._read(path, params, "application/json"))

    def get_ndjson(self, path: str, params: Mapping[str, Any] | None = None) -> list[dict]:
        text = self._read(path, params, NDJSON).decode("utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def user(self, name: str) -> dict:
        return self.get_json(f"/api/user/{check_name(name)}")

    def rating_history(self, name: str) -> list:
        return self.get_json(f"/api/user/{check_name(name)}/rating-history")

    def games(
        self, name: str, max_games: int, perf_type: str | None = BLITZ, rated: bool | None = True
    ) -> list[dict]:
        """The newest `max_games` finished games, newest first (the export's own order)."""
        params = {
            "max": int(max_games),
            "perfType": perf_type,
            "rated": None if rated is None else str(rated).lower(),
            "moves": "true",
            "clocks": "false",
            "evals": "false",
            "opening": "false",
            "pgnInJson": "false",
        }
        return self.get_ndjson(f"/api/games/user/{check_name(name)}", params)

    def is_playing(self, name: str) -> bool:
        """Whether the public status shows the bot in a game right now."""
        rows = self.get_json("/api/users/status", {"ids": check_name(name)})
        return any(row.get("id") == name.lower() and row.get("playing") for row in rows)


def default_api() -> PublicApi:
    return PublicApi()


# ---------------------------------------------------------------- games, from the bot's side


@dataclass(frozen=True)
class BotGame:
    id: str
    created_at: int
    status: str
    rated: bool
    bot_color: str
    result: str  # "win" | "loss" | "draw" (an aborted game has no winner, so it reads as a draw)
    opponent: str
    opponent_is_bot: bool
    opponent_rating: int | None
    opening: tuple[str, ...]  # the first DUPLICATE_PLIES plies, SAN
    plies: int

    @property
    def aborted(self) -> bool:
        return self.status in ABORT_STATUSES

    @property
    def time_loss(self) -> bool:
        return self.status in TIME_LOSS_STATUSES and self.result == "loss"

    @property
    def score(self) -> float:
        return {"win": 1.0, "draw": 0.5, "loss": 0.0}[self.result]


def _player_id(side: Mapping) -> str:
    return str((side.get("user") or {}).get("id", "")).lower()


def parse_game(record: Mapping, bot: str) -> BotGame:
    players = record.get("players") or {}
    colors = [c for c in ("white", "black") if _player_id(players.get(c) or {}) == bot.lower()]
    if not colors:
        raise ValueError(f"{bot} is not a player in game {record.get('id')!r}")
    color = colors[0]
    other = players.get("black" if color == "white" else "white") or {}
    winner = record.get("winner")
    moves = str(record.get("moves") or "").split()
    return BotGame(
        id=str(record.get("id")),
        created_at=int(record.get("createdAt") or 0),
        status=str(record.get("status")),
        rated=bool(record.get("rated")),
        bot_color=color,
        result="draw" if winner is None else ("win" if winner == color else "loss"),
        opponent=_player_id(other),
        opponent_is_bot=(other.get("user") or {}).get("title") == "BOT",
        opponent_rating=other.get("rating"),
        opening=tuple(moves[:DUPLICATE_PLIES]),
        plies=len(moves),
    )


def parse_records(records: Iterable[Mapping], bot: str) -> tuple[BotGame, ...]:
    """Games in the order they were played (the export streams newest first)."""
    return tuple(sorted((parse_game(r, bot) for r in records), key=lambda g: (g.created_at, g.id)))


def parse_ndjson(lines: Iterable[str | bytes], bot: str) -> tuple[BotGame, ...]:
    return parse_records((json.loads(line) for line in lines if line.strip()), bot)


# ---------------------------------------------------------------- the rates


def _share(count: int, total: int) -> float | None:
    return count / total if total else None


def abort_rate(games: Sequence[BotGame]) -> float | None:
    return _share(sum(g.aborted for g in games), len(games))


def time_loss_rate(games: Sequence[BotGame]) -> float | None:
    return _share(sum(g.time_loss for g in games), len(games))


def played(games: Sequence[BotGame]) -> list[BotGame]:
    return [g for g in games if not g.aborted]


def human_share(games: Sequence[BotGame]) -> float | None:
    real = played(games)
    return _share(sum(not g.opponent_is_bot for g in real), len(real))


def duplicate_rate(games: Sequence[BotGame], plies: int = DUPLICATE_PLIES) -> float | None:
    long_enough = [g for g in games if g.plies >= plies]
    seen: set[tuple[str, tuple[str, ...]]] = set()
    repeats = 0
    for game in long_enough:
        key = (game.opponent, game.opening[:plies])
        repeats += key in seen
        seen.add(key)
    return _share(repeats, len(long_enough))


def _expected(rating: float, opponents: Sequence[float]) -> float:
    return sum(1.0 / (1.0 + 10.0 ** ((r - rating) / 400.0)) for r in opponents)


def performance(results: Sequence[tuple[float, float]]) -> float | None:
    """The rating whose expected score against these (opponent rating, score) pairs is the real score."""
    if not results:
        return None
    opponents = [float(r) for r, _ in results]
    score = sum(s for _, s in results)
    low, high = min(opponents) - PERF_CAP, max(opponents) + PERF_CAP
    if score <= _expected(low, opponents):
        return low
    if score >= _expected(high, opponents):
        return high
    for _ in range(100):  # bisection: the expected score rises with the rating
        mid = (low + high) / 2.0
        low, high = (mid, high) if _expected(mid, opponents) < score else (low, mid)
    return (low + high) / 2.0


def _performance_of(games: Iterable[BotGame]) -> float | None:
    return performance([(g.opponent_rating, g.score) for g in games if g.opponent_rating is not None])


def performance_split(games: Sequence[BotGame]) -> tuple[float | None, float | None]:
    """(performance vs humans, performance vs bots), over the games actually played."""
    real = played(games)
    humans = _performance_of(g for g in real if not g.opponent_is_bot)
    bots = _performance_of(g for g in real if g.opponent_is_bot)
    return humans, bots


# ---------------------------------------------------------------- the snapshot


@dataclass(frozen=True)
class SnapshotReport:
    snapshot: LichessSnapshot
    exported: int
    title: str | None
    history: dict


def _rounded(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def history_summary(history: Sequence[Mapping], perf: str = "Blitz") -> dict:
    points = next((h.get("points") or [] for h in history if h.get("name") == perf), [])
    ratings = [p[3] for p in points if len(p) >= 4]
    if not ratings:
        return {"points": 0}
    return {"points": len(ratings), "first": ratings[0], "min": min(ratings), "last": ratings[-1]}


def build_snapshot(bot: str, user: Mapping, games: Sequence[BotGame], snapshot_date: str) -> LichessSnapshot:
    blitz = (user.get("perfs") or {}).get(BLITZ) or {}
    humans, bots = performance_split(games)
    return LichessSnapshot(
        bot=str(user.get("username") or bot),
        rating=int(blitz.get("rating", 0)),
        rd=int(blitz.get("rd", 0)),
        n=int(blitz.get("games", 0)),
        snapshot_date=snapshot_date,
        human_share=_rounded(human_share(games), 4),
        perf_vs_humans=_rounded(humans, 1),
        perf_vs_bots=_rounded(bots, 1),
        time_loss_rate=_rounded(time_loss_rate(games), 4),
        abort_rate=_rounded(abort_rate(games), 4),
        duplicate_rate=_rounded(duplicate_rate(games), 4),
    )


def take_snapshot(api: PublicApi, bot: str, max_games: int, snapshot_date: str) -> SnapshotReport:
    user = api.user(bot)
    history = api.rating_history(bot)
    games = parse_records(api.games(bot, max_games), bot)
    return SnapshotReport(
        snapshot=build_snapshot(bot, user, games, snapshot_date),
        exported=len(games),
        title=user.get("title"),
        history=history_summary(history),
    )


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _perf(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}"


def format_report(report: SnapshotReport, max_games: int) -> list[str]:
    s = report.snapshot
    gate = "publishable" if s.publishable else "NOT publishable yet (needs N >= 200 and RD < 75)"
    history = report.history
    lines = [
        f"{s.bot} ({report.title or 'no title'}) on {s.snapshot_date}: blitz {s.rating}, RD {s.rd}, "
        f"N {s.n} rated blitz games; {gate}",
        f"  over the last {report.exported} exported rated blitz games: human share {_pct(s.human_share)}, "
        f"performance vs humans {_perf(s.perf_vs_humans)}, vs bots {_perf(s.perf_vs_bots)}",
        f"  time losses {_pct(s.time_loss_rate)}, aborts {_pct(s.abort_rate)}, "
        f"duplicate games {_pct(s.duplicate_rate)}",
    ]
    if history.get("points"):
        lines.append(
            f"  blitz history: {history['points']} points, first {history['first']}, "
            f"lowest {history['min']}, last {history['last']}"
        )
    if report.exported >= max_games:
        lines.append(f"  (the export stopped at --max-games {max_games}; the rates cover only those games)")
    if report.title != "BOT":
        lines.append("  warning: this account does not carry the BOT title")
    return lines


def write_snapshot(snapshot: LichessSnapshot, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, lichess_to_json(snapshot) + "\n")
    return out
