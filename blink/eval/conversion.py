"""E8 conversion and E2b, the epsilon choice (plan P8).

Conversion: Blink plays the winning side of each screened endgame (blink.eval.endgames) against
Stockfish 19 at st=0.1, and the metric is the share it wins by checkmate within 100 of its own moves
(200 plies from the start position; a longer game is stopped and counts as not converted, as does any
draw). Games are played in process: SF19 gets `go movetime 100` with the whole game, and Blink needs no
clock because its compute never depends on time (N4). "Rules off" (F16) is Blink's network with only R1:
no mate-now shortcut (R2), no rule-draw handling (R3) and no tie-break (R4); it still goes through the
EvalBudget, so the no-search audit covers it too.

The epsilon rule (the R4 tie window, section 1): each epsilon in {0, 1/256, 1/128} converts the 200 dev
endgames; the highest conversion rate wins (a tie goes to the smallest epsilon). A winner other than 0
must then score at least 50% - 1 SE over 1,000 dev-slice games against epsilon = 0, or epsilon = 0 ships.
The result goes to results/epsilon.json. EVAL.md holds only the frozen rule, and this module never opens
it for writing.
"""

import json
import math
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np

from blink.eval import match, puzzles
from blink.eval.books import Opening
from blink.eval.endgames import Endgame
from blink.play.agents import Agent, Decision, expand
from blink.play.budget import EvalBudget
from blink.play.evaluator import Evaluator

MAX_BLINK_MOVES = 100
MAX_PLIES = 2 * MAX_BLINK_MOVES
EPSILONS = (0.0, 1 / 256, 1 / 128)
NO_REGRESSION_GAMES = 1000
EPSILON_FILE = "epsilon.json"
EPSILON_RULE = (
    "highest conversion rate on the 200 dev endgames (a tie goes to the smallest epsilon); a winner other "
    "than 0 must score at least 50% - 1 SE over 1,000 dev-slice games against epsilon 0, else 0 ships"
)


@dataclass(frozen=True)
class ConversionGame:
    line: int
    winner: str
    result: str
    reason: str
    plies: int
    converted: bool


@dataclass(frozen=True)
class ConversionResult:
    agent: str
    games: tuple[ConversionGame, ...]

    @property
    def n(self) -> int:
        return len(self.games)

    @property
    def converted(self) -> int:
        return sum(g.converted for g in self.games)

    @property
    def rate(self) -> float:
        return self.converted / self.n if self.n else 0.0

    def as_dict(self) -> dict:
        low, high = puzzles.wilson(self.converted, self.n)
        return {
            "agent": self.agent,
            "n": self.n,
            "converted": self.converted,
            "pct": 100 * self.rate,
            "wilson95": [100 * low, 100 * high],
            "reasons": dict(Counter(g.reason for g in self.games)),
            "games": [asdict(g) for g in self.games],
        }


def _converted(record: match.GameRecord, blink_white: bool) -> bool:
    won = record.result == ("1-0" if blink_white else "0-1")
    return won and record.reason == "checkmate"


def play_conversion(
    blink: Agent, opponent: Agent, endgames: Sequence[Endgame], pgn: Path, max_plies: int = MAX_PLIES
) -> ConversionResult:
    """Blink on the winning side of every endgame, in order; the PGN gets one game per endgame."""
    pgn.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for endgame in endgames:
        blink_white = endgame.blink_color == chess.WHITE
        white, black = (blink, opponent) if blink_white else (opponent, blink)
        game, record = match.play_game(
            white, black, Opening(endgame.line, endgame.fen, ()), f"e{endgame.line}", max_plies
        )
        game.headers["Event"] = "Blink conversion"
        game.headers["Round"] = str(endgame.line)
        with open(pgn, "a", encoding="utf-8") as handle:
            print(game, file=handle, end="\n\n")
        converted = _converted(record, blink_white)
        out.append(
            ConversionGame(
                endgame.line, endgame.winner, record.result, record.reason, record.engine_plies, converted
            )
        )
    return ConversionResult(blink.name, tuple(out))


# ------------------------------------------------------------------------------ rules off (F16)


@dataclass(frozen=True)
class RulesOffAgent:
    """Blink's network with R1 only: policy = masked argmax; value = the child worst for the opponent.

    Ties go to the lowest vocabulary index. No R2 (a mate is played only if the network finds it), no
    R3 (a rule-draw child is valued by the network like any other) and no R4 tie-break."""

    evaluator: Evaluator
    mode: str
    name: str = "Blink-rules-off"

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        children = expand(board)
        budget = EvalBudget(self.evaluator, board, mode=self.mode, game=game)
        if self.mode == "policy":
            logits = budget.evaluate(budget.root_codes[None]).policy_logits[0]
            scores = np.array([logits[c.index] for c in children])
        else:
            codes = budget.child_codes()
            evaluation = budget.evaluate(np.stack([budget.root_codes] + [codes[c.move] for c in children]))
            scores = 1.0 - evaluation.win_probability()[1:]
        pick = int(np.argmax(scores))
        record = budget.finish(("R1",))
        return Decision(children[pick].move, budget.n_rows, budget.n_calls, ("R1",), None, record)


# ------------------------------------------------------------------------------ E2b, the epsilon rule


def select_epsilon(rates: Mapping[float, float]) -> float:
    """The epsilon with the highest conversion rate; a tie goes to the smallest epsilon."""
    return min(rates, key=lambda eps: (-rates[eps], eps))


def score_se(scores: Sequence[float]) -> tuple[float, float]:
    """(mean score, standard error of the mean) over per-game scores 0 / 0.5 / 1."""
    n = len(scores)
    if n == 0:
        raise ValueError("no games to score")
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    return mean, math.sqrt(var / n)


def no_regression(scores: Sequence[float]) -> dict:
    """The candidate's per-game scores against epsilon 0: passes when score >= 50% - 1 SE."""
    mean, se = score_se(scores)
    threshold = 0.5 - se
    return {
        "games": len(scores),
        "score": mean,
        "se": se,
        "threshold": threshold,
        "passed": mean >= threshold,
    }


def epsilon_decision(conversions: Mapping[float, ConversionResult], check: dict | None) -> dict:
    """The rule applied to measured numbers: which epsilon ships, and why."""
    rates = {eps: result.rate for eps, result in conversions.items()}
    winner = select_epsilon(rates)
    if winner == 0.0:
        chosen, reason = 0.0, "epsilon 0 converted the most (or tied for the most)"
    elif check is None:
        raise ValueError(f"epsilon {winner} won the conversion: its no-regression check must run first")
    elif check["passed"]:
        chosen, reason = winner, f"epsilon {winner} converted the most and passed the no-regression check"
    else:
        chosen, reason = 0.0, f"epsilon {winner} converted the most but failed the no-regression check"
    return {
        "epsilon": chosen,
        "reason": reason,
        "winner_by_conversion": winner,
        "candidates": list(conversions),
        "conversion": {str(eps): conversions[eps].as_dict() for eps in conversions},
        "no_regression": check,
        "rule": EPSILON_RULE,
        "protocol": "EVAL.md holds the frozen rule; this file holds the result",
    }


def write_epsilon(decision: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / EPSILON_FILE
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(decision, indent=2) + "\n")
    os.replace(tmp, path)
    return path


def run_epsilon_selection(
    convert: Callable[[float], ConversionResult],
    check: Callable[[float], Sequence[float]],
    results_dir: Path,
    epsilons: Sequence[float] = EPSILONS,
) -> dict:
    """convert(eps) plays the dev endgames; check(eps) plays the no-regression match against epsilon 0 and
    returns the candidate's per-game scores. Writes results/epsilon.json and returns the decision."""
    conversions = {eps: convert(eps) for eps in epsilons}
    winner = select_epsilon({eps: r.rate for eps, r in conversions.items()})
    verdict = None if winner == 0.0 else no_regression(check(winner))
    decision = epsilon_decision(conversions, verdict)
    write_epsilon(decision, results_dir)
    return decision
