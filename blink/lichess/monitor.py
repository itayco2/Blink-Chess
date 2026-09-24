"""The bot's stop rule, read from the public API: time losses > 2% or aborts > 1% over the last 50 games.

    blink lichess check --bot NAME [--stop]

The rule looks at the bot's newest 50 games of every kind (rated or casual, any speed), in the order
they were played. With fewer than 50 games it looks at all of them, so the rates are over the games
actually played and an early time loss stops the bot sooner, not later. A time loss is a game the bot
lost on the clock or by leaving it; an abort is an `aborted` or `noStart` game, whoever caused it (the
public export does not say). When the rule fires, `--stop` pauses the bot exactly as
`blink lichess pause` does, and Itay decides when it restarts.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from blink.lichess import snapshot

WINDOW = 50
MAX_TIME_LOSS_RATE = 0.02
MAX_ABORT_RATE = 0.01


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


def check(api, bot: str, window: int = WINDOW) -> StopVerdict:
    """The stop rule over the bot's newest `window` games of every kind."""
    records = api.games(bot, max_games=window, perf_type=None, rated=None)
    return stop_rule(snapshot.parse_records(records, bot), window=window)


def format_verdict(bot: str, verdict: StopVerdict) -> str:
    counts = (
        f"{verdict.games} games: {verdict.time_losses} time losses ({100 * verdict.time_loss_rate:.1f}%), "
        f"{verdict.aborts} aborts ({100 * verdict.abort_rate:.1f}%)"
    )
    if verdict.stop:
        return f"STOP {bot}: {'; '.join(verdict.reasons)} (last {counts})"
    return f"ok {bot}: last {counts}; limits {100 * MAX_TIME_LOSS_RATE:g}% and {100 * MAX_ABORT_RATE:g}%"
