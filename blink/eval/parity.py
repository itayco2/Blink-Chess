"""Fast-mode parity: does a fast evaluator (bf16, compiled) play the moves fp32 plays?

For each real position the reference (fp32) and the fast evaluator each make the two decisions Blink
can make there, through the play code itself, so the rows are exactly what play sends:
- one look: agents.policy_decision, one row (P). Its policy top-1 is the argmax of P's policy logits
  over the legal moves (lowest vocabulary index on a tie, as PolicyAgent orders them).
- one look per move: agents.value_decision, one batch of L+1 rows (P and every child). Its move is
  the value agent's real choice (rules R3 and R4 included), and every row's win% is compared.

Positions where a mate in one is played by rule R2 never reach the network and are counted apart.
The report gives policy top-1 agreement, value choice agreement and max/mean |d win%| in points.
Torch-free: any two Evaluators compare; `blink bench parity` builds them from one model selector.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

from blink.board import encode
from blink.data import children, valprobe
from blink.play import agents, rules
from blink.play.evaluator import Evaluation, Evaluator

GAME = "parity"
MAX_EXAMPLES = 20  # value-choice disagreements kept in the report, with their FENs


class Recorder:
    """An Evaluator that remembers every call: (the rows it was sent, what it answered)."""

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.calls: list[tuple[np.ndarray, Evaluation]] = []

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        result = self.evaluator.evaluate(codes)
        self.calls.append((np.array(codes, dtype=np.uint8, copy=True), result))
        return result


@dataclass(frozen=True)
class Look:
    """What one evaluator made of one position."""

    top1: int  # the one-look policy argmax over the legal moves (a vocabulary index)
    value_move: chess.Move  # the value agent's move
    rows: np.ndarray  # the value batch it scored, uint8 [L+1, 64]
    win: np.ndarray  # win probability of each value-batch row, for that row's side to move


@dataclass(frozen=True)
class PositionParity:
    fen: str
    top1_agree: bool
    reference_move: chess.Move
    fast_move: chess.Move
    dwin_pct: np.ndarray  # |d win%| of each value-batch row, in points

    @property
    def value_agree(self) -> bool:
        return self.reference_move == self.fast_move


def mate_now(board: chess.Board) -> bool:
    """Rule R2 fires: some legal move mates, and play takes it without a network call."""
    return any(child.board.is_checkmate() for child in agents.expand(board))


def look(evaluator: Evaluator, board: chess.Board, epsilon: float) -> Look:
    """The one-look and value decisions at `board` (no R2 mate), each through its own EvalBudget."""
    recorder = Recorder(evaluator)
    agents.policy_decision(recorder, board, GAME, None)
    value = agents.value_decision(recorder, board, GAME, None, epsilon)
    (_, one), (rows, batch) = recorder.calls
    legal = [child.index for child in agents.expand(board)]
    top1 = legal[int(np.argmax(one.policy_logits[0][legal]))]
    return Look(top1, value.move, rows, batch.win_probability())


def position_parity(
    reference: Evaluator, fast: Evaluator, board: chess.Board, epsilon: float = rules.DEFAULT_EPSILON
) -> PositionParity | None:
    """None when R2 plays a mate in one (no network call to compare)."""
    if mate_now(board):
        return None
    ref, other = look(reference, board, epsilon), look(fast, board, epsilon)
    if not np.array_equal(ref.rows, other.rows):
        raise RuntimeError(f"the two evaluators were sent different value rows at {board.fen()}")
    dwin = np.abs(ref.win - other.win) * 100.0
    return PositionParity(board.fen(), ref.top1 == other.top1, ref.value_move, other.value_move, dwin)


def _share(count: int, total: int) -> float | None:
    return count / total if total else None


def summarize(results: Sequence[PositionParity | None]) -> dict:
    scored = [r for r in results if r is not None]
    dwin = np.concatenate([r.dwin_pct for r in scored]) if scored else np.zeros(0)
    examples = [
        {
            "fen": r.fen,
            "fp32": r.reference_move.uci(),
            "fast": r.fast_move.uci(),
            "max_abs_dwin_pct": float(r.dwin_pct.max()),
        }
        for r in scored
        if not r.value_agree
    ]
    return {
        "positions": len(results),
        "mate_now": len(results) - len(scored),
        "scored": len(scored),
        "value_rows": len(dwin),
        "policy_top1_agreement": _share(sum(r.top1_agree for r in scored), len(scored)),
        "value_choice_agreement": _share(sum(r.value_agree for r in scored), len(scored)),
        "max_abs_dwin_pct": float(dwin.max()) if len(dwin) else None,
        "mean_abs_dwin_pct": float(dwin.mean()) if len(dwin) else None,
        "value_disagreements": examples[:MAX_EXAMPLES],
    }


def compare(
    reference: Evaluator,
    fast: Evaluator,
    boards: Sequence[chess.Board],
    epsilon: float = rules.DEFAULT_EPSILON,
    log: Callable[[str], None] | None = None,
    every: int = 500,
) -> dict:
    """Parity of `fast` against `reference` over `boards` (see summarize for the keys)."""
    results = []
    for count, board in enumerate(boards, start=1):
        results.append(position_parity(reference, fast, board, epsilon))
        if log is not None and count % every == 0:
            log(f"  {count} of {len(boards)} positions")
    return summarize(results)


def val_positions(pack_dir: Path, n: int) -> list[chess.Board]:
    """The first n val roots VAA would take (deep or mate labels, unique), as boards with White to move.

    A root is stored from its side to move's view, so its board is the colour-normalised twin, which
    encodes to the very same network rows.
    """
    roots = valprobe.read_val_roots(pack_dir)
    boards = (children.codes_to_board(encode.unpack(roots["board"][i])) for i in valprobe.select(roots, n))
    return [board for board in boards if any(board.generate_legal_moves())]
