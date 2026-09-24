"""Who picks the move: Blink's two modes and the two baselines.

Every agent takes the game so far as a python-chess Board whose move stack is the real history
(the halfmove clock and repetitions come from it) and returns a Decision. The input board is never
mutated. Blink's agents route their single network call through an EvalBudget.

- PolicyAgent, "one look": exactly one row, P; the masked argmax of its policy logits.
- ValueAgent, "one look per move": one batch of P plus every child; the child worst for the
  opponent wins (value for the mover = 1 - the child's win probability).
- RandomAgent and MaterialAgent (1-ply material 1/3/3/5/9) are baselines, not Blink. Their only
  randomness is a tie-break seeded by (seed, game, position), so they hold no state between moves.
  The ladder's rungs play value mode through ValueAgent with a tie_seed that does the same for R4
  ties: their policy is flat, so without it every tie went to the lowest vocab index. Blink's own
  agents never carry a tie seed (N4, no randomisation).
"""

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import chess
import numpy as np

from blink.board import encode, moves
from blink.play import rules
from blink.play.budget import DecisionRecord, EvalBudget
from blink.play.evaluator import Evaluator
from blink.play.oracles import material_balance

Sink = Callable[[DecisionRecord], None]
MATE_SCORE = 1000  # the material agent's score for a checkmating move


@dataclass(frozen=True)
class Decision:
    move: chess.Move
    n_rows: int = 0
    n_calls: int = 0
    rules: tuple[str, ...] = ()
    win: float | None = None  # the mover's expected win probability for this move, when a network scored it
    record: DecisionRecord | None = None

    @property
    def mate_now(self) -> bool:
        return "R2" in self.rules


class Agent(Protocol):
    name: str

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision: ...


@dataclass(frozen=True)
class Child:
    index: int  # vocabulary index of the move, in the mover's frame
    move: chess.Move
    board: chess.Board


def expand(board: chess.Board) -> tuple[Child, ...]:
    """R1: every legal move and its child position, in vocabulary order."""
    parent = board.copy(stack=False)
    out = []
    for move in parent.legal_moves:
        child = parent.copy(stack=False)
        child.push(move)
        out.append(Child(moves.encode_move(parent, move), move, child))
    if not out:
        raise ValueError(f"no legal moves: the game is over at {board.fen()}")
    return tuple(sorted(out, key=lambda c: c.index))


def _applied(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted({rule for group in groups for rule in group}))


def _mate_now(budget: EvalBudget, children: tuple[Child, ...], extra: tuple[str, ...]) -> Decision | None:
    """R2: a checkmating child is played before any network call (lowest vocab index)."""
    mate = next((child for child in children if child.board.is_checkmate()), None)
    if mate is None:
        return None
    applied = _applied(("R1", "R2"), extra)
    return Decision(mate.move, 0, 0, applied, 1.0, budget.finish(applied))


def policy_decision(
    evaluator: Evaluator, board: chess.Board, game: str, sink: Sink | None, extra: tuple[str, ...] = ()
) -> Decision:
    """One look: one row, P. R3 uses the root value from the same call (delta 0.10)."""
    children = expand(board)
    budget = EvalBudget(evaluator, board, mode="policy", game=game, sink=sink)
    mate = _mate_now(budget, children, extra)
    if mate is not None:
        return mate
    evaluation = budget.evaluate(budget.root_codes[None])
    logits = evaluation.policy_logits[0]
    order = [c.index for c in sorted(children, key=lambda c: -logits[c.index])]
    history = rules.History.from_board(board)
    draws = {c.index for c in children if rules.rule_draw(c.board, history) is not None}
    root_win = float(evaluation.win_probability()[0])
    choice, fired = rules.policy_draw_choice(order, draws, root_win)
    applied = _applied(("R1",), extra, ("R3",) if fired else ())
    move = next(c.move for c in children if c.index == choice)
    return Decision(move, budget.n_rows, budget.n_calls, applied, root_win, budget.finish(applied))


def _tie_rng(seed: int, game: str, board: chess.Board) -> random.Random:
    """The FEN holds the move counters, so a position that recurs later in a game draws afresh."""
    return random.Random(f"{seed}:{game}:{board.fen()}")


def value_decision(
    evaluator: Evaluator,
    board: chess.Board,
    game: str,
    sink: Sink | None,
    epsilon: float,
    tie_seed: int | None = None,
) -> Decision:
    """One look per move: one batch of P plus every child; rule-draw children are worth 0.5.

    With a tie_seed (baselines only), an R4 tie is drawn from rules.tie_set by (seed, game, position)
    rather than going to the lowest vocab index. It picks among moves already scored, so the row and
    call counts are the same.
    """
    children = expand(board)
    budget = EvalBudget(evaluator, board, mode="value", game=game, sink=sink)
    mate = _mate_now(budget, children, ())
    if mate is not None:
        return mate
    child_codes = budget.child_codes()
    evaluation = budget.evaluate(np.stack([budget.root_codes] + [child_codes[c.move] for c in children]))
    history = rules.History.from_board(board)
    draws = np.array([rules.rule_draw(c.board, history) is not None for c in children])
    mover = np.where(draws, 0.5, 1.0 - evaluation.win_probability()[1:])
    root_logits = evaluation.policy_logits[0][[c.index for c in children]]
    pick, tied = rules.tie_break(mover, root_logits, epsilon)
    if tied and tie_seed is not None:
        tie = rules.tie_set(mover, root_logits, epsilon).tolist()
        pick = _tie_rng(tie_seed, game, board).choice(tie)
    applied = _applied(("R1",), ("R3",) if draws.any() else (), ("R4",) if tied else ())
    record = budget.finish(applied)
    return Decision(children[pick].move, budget.n_rows, budget.n_calls, applied, float(mover[pick]), record)


@dataclass(frozen=True)
class PolicyAgent:
    evaluator: Evaluator
    sink: Sink | None = None
    name: str = "Blink-policy"

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        return policy_decision(self.evaluator, board, game, self.sink)


@dataclass(frozen=True)
class ValueAgent:
    evaluator: Evaluator
    epsilon: float = rules.DEFAULT_EPSILON
    p99_s: float = 0.0  # measured value-mode move time, for the R5 clock guard
    sink: Sink | None = None
    name: str = "Blink-value"
    tie_seed: int | None = None  # flat-policy baselines only: R4 ties drawn by (seed, game, position)

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        if rules.clock_guard(remaining_s, self.p99_s):
            return policy_decision(self.evaluator, board, game, self.sink, extra=("R5",))
        return value_decision(self.evaluator, board, game, self.sink, self.epsilon, self.tie_seed)


@dataclass(frozen=True)
class RandomAgent:
    """Baseline: a uniformly random legal move."""

    seed: int = 0
    name: str = "Random"

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        legal = sorted(board.legal_moves, key=chess.Move.uci)
        if not legal:
            raise ValueError(f"no legal moves: the game is over at {board.fen()}")
        return Decision(_tie_rng(self.seed, game, board).choice(legal))


@dataclass(frozen=True)
class MaterialAgent:
    """Baseline: 1-ply material (1/3/3/5/9) after each move, mates first, ties broken at random."""

    seed: int = 0
    name: str = "Material"

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        children = expand(board)
        scores = [
            MATE_SCORE if c.board.is_checkmate() else -int(material_balance(encode.encode_board(c.board)))
            for c in children
        ]
        best = max(scores)
        tied = [c.move for c, score in zip(children, scores, strict=True) if score == best]
        return Decision(_tie_rng(self.seed, game, board).choice(tied))
