"""The bot's stop rule, read from the public API: time losses > 2% or aborts > 1% over the last 50 games.

    blink lichess check --bot NAME [--stop]

The rule looks at the bot's newest 50 games of every kind (rated or casual, any speed), in the order
they were played. With fewer than 50 games it looks at all of them, so the rates are over the games
actually played and an early time loss stops the bot sooner, not later. A time loss is a game the bot
lost on the clock or by leaving it; an abort is an `aborted` or `noStart` game, whoever caused it.

The public export never contains `aborted` games (lila exports only status >= mate), so the check
also reads the `Termination "Abandoned"` PGNs lichess-bot saves in its pgn_directory
(BLINK_HOME/lichess/pgn for the rated config) and merges them in by game id. When the rule fires,
`--stop` pauses the bot exactly as `blink lichess pause` does, and Itay decides when it restarts.
"""

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import chess.pgn

from blink.lichess import snapshot

WINDOW = 50
MAX_TIME_LOSS_RATE = 0.02
MAX_ABORT_RATE = 0.01
ABANDONED = "Abandoned"
NEWEST_PGN_FILES = 500


@dataclass(frozen=True)
class StopVerdict:
    games: int
    time_losses: int
    aborts: int
    reasons: tuple[str, ...]

    @property
    def stop(self) -> bool:
        return bool(self.reasons)

    @property
    def time_loss_rate(self) -> float:
        return self.time_losses / self.games if self.games else 0.0

    @property
    def abort_rate(self) -> float:
        return self.aborts / self.games if self.games else 0.0


def _breach(label: str, count: int, games: int, limit: float) -> tuple[str, ...]:
    if not games or count / games <= limit:
        return ()
    return (f"{label} {count}/{games} = {100 * count / games:.1f}% > {100 * limit:g}%",)


def stop_rule(
    games: Sequence[snapshot.BotGame],
    window: int = WINDOW,
    max_time_loss: float = MAX_TIME_LOSS_RATE,
    max_abort: float = MAX_ABORT_RATE,
) -> StopVerdict:
    """A pure function of the games (oldest first): fires when either rate is over its limit."""
    recent = tuple(games)[-window:] if window > 0 else ()
    time_losses = sum(g.time_loss for g in recent)
    aborts = sum(g.aborted for g in recent)
    reasons = _breach("time losses", time_losses, len(recent), max_time_loss)
    reasons += _breach("aborts", aborts, len(recent), max_abort)
    return StopVerdict(games=len(recent), time_losses=time_losses, aborts=aborts, reasons=reasons)


def _utc_ms(date: str, clock: str) -> int:
    try:
        moment = datetime.datetime.strptime(f"{date} {clock}", "%Y.%m.%d %H:%M:%S")
    except ValueError:
        return 0
    return int(moment.replace(tzinfo=datetime.UTC).timestamp() * 1000)


def _abandoned(headers: Mapping[str, str], bot: str) -> snapshot.BotGame | None:
    """An aborted game from its PGN headers, seen from the bot's side; None for anything else."""
    white, black = headers.get("White", ""), headers.get("Black", "")
    if headers.get("Termination") != ABANDONED or bot.lower() not in (white.lower(), black.lower()):
        return None
    color, opponent, side = (
        ("white", black, "Black") if white.lower() == bot.lower() else ("black", white, "White")
    )
    return snapshot.BotGame(
        id=headers.get("Site", "").rstrip("/").rsplit("/", 1)[-1],
        created_at=_utc_ms(headers.get("UTCDate", ""), headers.get("UTCTime", "")),
        status="aborted",
        rated=headers.get("Event", "").startswith("Rated"),
        bot_color=color,
        result="draw",
        opponent=opponent.lower(),
        opponent_is_bot=headers.get(f"{side}Title") == "BOT",
        opponent_rating=None,
        opening=(),
        plies=0,
    )


def aborted_from_pgns(
    pgn_dir: Path, bot: str, newest: int = NEWEST_PGN_FILES
) -> tuple[snapshot.BotGame, ...]:
    """Aborted games from the PGNs lichess-bot writes to its pgn_directory (Termination "Abandoned").

    lila's game export keeps only games with status >= mate (Query.finished), so an `aborted` game
    (status 25) never reaches the public API. lichess-bot still fetches and saves every game's PGN,
    and lila writes `Termination "Abandoned"` for both aborted and noStart games. Only the newest
    `newest` files are read.
    """
    pgn_dir = Path(pgn_dir)
    if not pgn_dir.is_dir():
        return ()
    files = sorted(pgn_dir.glob("*.pgn"), key=lambda p: p.stat().st_mtime)[-newest:]
    games = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while (headers := chess.pgn.read_headers(handle)) is not None:
                game = _abandoned(headers, bot)
                games += [game] if game else []
    return tuple(games)


def merge(
    exported: Sequence[snapshot.BotGame], local: Sequence[snapshot.BotGame]
) -> tuple[snapshot.BotGame, ...]:
    """One game per id, in the order played; the public record wins when both sources have a game."""
    by_id = {game.id: game for game in local}
    by_id.update({game.id: game for game in exported})
    return tuple(sorted(by_id.values(), key=lambda g: (g.created_at, g.id)))


def check(api, bot: str, window: int = WINDOW, pgn_dir: Path | None = None) -> StopVerdict:
    """The stop rule over the bot's newest `window` games: the public export plus local aborted games."""
    records = api.games(bot, max_games=window, perf_type=None, rated=None)
    local = aborted_from_pgns(pgn_dir, bot) if pgn_dir is not None else ()
    return stop_rule(merge(snapshot.parse_records(records, bot), local), window=window)


def format_verdict(bot: str, verdict: StopVerdict) -> str:
    counts = (
        f"{verdict.games} games: {verdict.time_losses} time losses ({100 * verdict.time_loss_rate:.1f}%), "
        f"{verdict.aborts} aborts ({100 * verdict.abort_rate:.1f}%)"
    )
    if verdict.stop:
        return f"STOP {bot}: {'; '.join(verdict.reasons)} (last {counts})"
    return f"ok {bot}: last {counts}; limits {100 * MAX_TIME_LOSS_RATE:g}% and {100 * MAX_ABORT_RATE:g}%"
