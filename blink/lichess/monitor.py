"""The bot's stop rule, read from the public API: time losses > 2% or aborts > 1% over the last 50 games.

    blink lichess check --bot NAME [--stop]

The rule looks at the bot's newest 50 games of every kind (rated or casual, any speed), in the order
they were played. With fewer than 50 games it looks at all of them, so the rates are over the games
actually played and an early time loss stops the bot sooner, not later. A time loss is a game the bot
lost on the clock or by leaving it. An abort counts only when the bot caused it: an `aborted` game
that ended with the bot to move, or a `noStart` game the bot did not win (lila names the side that did
move as the winner). lichess-bot itself aborts a game whose opponent makes no move within abort_time,
and lila records that the same way, so those opponent no-shows are reported beside the rule and never
count against the bot.

The public export never contains `aborted` games (lila exports only status >= mate), so the check
also reads the PGNs lichess-bot saves in its pgn_directory (BLINK_HOME/lichess/pgn for the rated
config) and merges them in by game id: every game the bot finished there, with time forfeits read from
the Result tag and aborts from `Termination "Abandoned"`. When the rule fires, `--stop` pauses the bot
exactly as `blink lichess pause` does, and Itay decides when it restarts.
"""

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chess.pgn

from blink.lichess import snapshot

WINDOW = 50
MAX_TIME_LOSS_RATE = 0.02
MAX_ABORT_RATE = 0.01
ABANDONED = "Abandoned"
TIME_FORFEIT = "Time forfeit"
UNTERMINATED = "Unterminated"
NEWEST_PGN_FILES = 500
WINNER_BY_RESULT = {"1-0": "white", "0-1": "black"}


@dataclass(frozen=True)
class StopVerdict:
    games: int
    time_losses: int
    aborts: int  # aborts the bot caused
    reasons: tuple[str, ...]
    opponent_aborts: int = 0  # opponent no-shows: reported, never counted
    offenses: tuple[str, ...] = ()  # ids of the counted time losses and aborts in the window

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


def bot_caused_abort(game: snapshot.BotGame) -> bool:
    """An unstarted game the bot is to blame for: it was the side to move when the game ended."""
    if game.status == "noStart":
        return game.result != "win"  # lila names the side that did move as the winner
    if game.status == "aborted":
        white_to_move = game.plies % 2 == 0
        return white_to_move == (game.bot_color == "white")
    return False


def stop_rule(
    games: Sequence[snapshot.BotGame],
    window: int = WINDOW,
    max_time_loss: float = MAX_TIME_LOSS_RATE,
    max_abort: float = MAX_ABORT_RATE,
) -> StopVerdict:
    """A pure function of the games (oldest first): fires when either rate is over its limit."""
    recent = tuple(games)[-window:] if window > 0 else ()
    time_losses = tuple(g for g in recent if g.time_loss)
    aborts = tuple(g for g in recent if bot_caused_abort(g))
    reasons = _breach("time losses", len(time_losses), len(recent), max_time_loss)
    reasons += _breach("aborts", len(aborts), len(recent), max_abort)
    return StopVerdict(
        games=len(recent),
        time_losses=len(time_losses),
        aborts=len(aborts),
        reasons=reasons,
        opponent_aborts=sum(g.aborted for g in recent) - len(aborts),
        offenses=tuple(g.id for g in (*time_losses, *aborts)),
    )


def _utc_ms(date: str, clock: str) -> int:
    try:
        moment = datetime.datetime.strptime(f"{date} {clock}", "%Y.%m.%d %H:%M:%S")
    except ValueError:
        return 0
    return int(moment.replace(tzinfo=datetime.UTC).timestamp() * 1000)


def _status(termination: str, winner: str | None) -> str:
    if termination == ABANDONED:
        return "noStart" if winner else "aborted"  # lila gives only a noStart game a winner
    if termination == TIME_FORFEIT:
        return "outoftime"
    return termination.lower() or "unknown"


def _from_pgn(game: chess.pgn.Game, bot: str) -> snapshot.BotGame | None:
    """A finished game from lichess-bot's PGN, seen from the bot's side; None when the bot is not in it."""
    headers = game.headers
    white, black = headers.get("White", ""), headers.get("Black", "")
    termination = headers.get("Termination", "")
    if bot.lower() not in (white.lower(), black.lower()) or termination == UNTERMINATED:
        return None
    color, opponent, side = (
        ("white", black, "Black") if white.lower() == bot.lower() else ("black", white, "White")
    )
    winner = WINNER_BY_RESULT.get(headers.get("Result", "*"))
    return snapshot.BotGame(
        id=headers.get("Site", "").rstrip("/").rsplit("/", 1)[-1],
        created_at=_utc_ms(headers.get("UTCDate", ""), headers.get("UTCTime", "")),
        status=_status(termination, winner),
        rated=headers.get("Event", "").startswith("Rated"),
        bot_color=color,
        result="draw" if winner is None else ("win" if winner == color else "loss"),
        opponent=opponent.lower(),
        opponent_is_bot=headers.get(f"{side}Title") == "BOT",
        opponent_rating=None,
        opening=(),
        plies=sum(1 for _ in game.mainline_moves()),  # read only for an abort: who was to move
    )


class _MovesOnlyWhenAbandoned(chess.pgn.GameBuilder):
    """Every game's headers; the moves only of an abandoned game, where they decide whose abort it was."""

    def end_headers(self):
        return None if self.game.headers.get("Termination") == ABANDONED else chess.pgn.SKIP


def games_from_pgns(pgn_dir: Path, bot: str, newest: int = NEWEST_PGN_FILES) -> tuple[snapshot.BotGame, ...]:
    """Every game the bot finished, from the PGNs lichess-bot writes to its pgn_directory.

    lila's game export keeps only games with status >= mate (Query.finished), so an `aborted` game
    (status 25) never reaches the public API. lichess-bot still fetches and saves every game's PGN,
    and lila writes `Termination "Abandoned"` for both aborted and noStart games (only noStart has a
    winner) and "Time forfeit" for a loss on the clock or by leaving. Only the newest `newest` files
    are read; a game still in progress ("Unterminated") is skipped.
    """
    pgn_dir = Path(pgn_dir)
    if not pgn_dir.is_dir():
        return ()
    files = sorted(pgn_dir.glob("*.pgn"), key=lambda p: p.stat().st_mtime)[-newest:]
    games = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as handle:
            while (game := chess.pgn.read_game(handle, Visitor=_MovesOnlyWhenAbandoned)) is not None:
                found = _from_pgn(game, bot)
                games += [found] if found else []
    return tuple(games)


def aborted_from_pgns(
    pgn_dir: Path, bot: str, newest: int = NEWEST_PGN_FILES
) -> tuple[snapshot.BotGame, ...]:
    """The aborted and noStart games among games_from_pgns (the ones the public export never has)."""
    return tuple(game for game in games_from_pgns(pgn_dir, bot, newest) if game.aborted)


def merge(
    exported: Sequence[snapshot.BotGame], local: Sequence[snapshot.BotGame]
) -> tuple[snapshot.BotGame, ...]:
    """One game per id, in the order played; the public record wins when both sources have a game."""
    by_id = {game.id: game for game in local}
    by_id.update({game.id: game for game in exported})
    return tuple(sorted(by_id.values(), key=lambda g: (g.created_at, g.id)))


def exported_games(api, bot: str, window: int = WINDOW) -> tuple[snapshot.BotGame, ...]:
    """The bot's newest `window` games of every kind from the public export (no token)."""
    return snapshot.parse_records(api.games(bot, max_games=window, perf_type=None, rated=None), bot)


def check(api, bot: str, window: int = WINDOW, pgn_dir: Path | None = None) -> StopVerdict:
    """The stop rule over the bot's newest `window` games: the public export plus lichess-bot's PGNs."""
    local = games_from_pgns(pgn_dir, bot) if pgn_dir is not None else ()
    return stop_rule(merge(exported_games(api, bot, window), local), window=window)


def format_verdict(bot: str, verdict: StopVerdict) -> str:
    counts = (
        f"{verdict.games} games: {verdict.time_losses} time losses ({100 * verdict.time_loss_rate:.1f}%), "
        f"{verdict.aborts} aborts by the bot ({100 * verdict.abort_rate:.1f}%), "
        f"{verdict.opponent_aborts} opponent no-shows (not counted)"
    )
    if verdict.stop:
        return f"STOP {bot}: {'; '.join(verdict.reasons)} (last {counts})"
    return f"ok {bot}: last {counts}; limits {100 * MAX_TIME_LOSS_RATE:g}% and {100 * MAX_ABORT_RATE:g}%"
