"""The closed list of rule checks Blink may apply at play time (NSC-1, plan section 2).

Nothing outside R1-R5 may influence a move. None of these checks evaluates a position with the
network: they look at the legal moves of P, at each child P.m, and at the real game history H.
A rule query on a child may enumerate the child's legal replies (is it mate or stalemate?), but
those replies are never scored, compared or fed to the network.

Repetition is counted with Blink's own history counter. python-chess's
`can_claim_threefold_repetition` is never called: it pushes every legal reply, a grandchild.
"""

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import chess
import numpy as np

RULES: Mapping[str, str] = MappingProxyType(
    {
        "R1": "legal-move generation, and masking the policy softmax to legal moves",
        "R2": "mate now: a checkmating child is played before any network call (lowest vocab index)",
        "R3": "rule draws: stalemate, insufficient material, halfmove clock 100, or a third repetition",
        "R4": "value-mode tie-break: within epsilon of the best child, the highest root policy logit",
        "R5": "clock guard: below max(3 s, 10 x p99 move time) remaining, play one look",
    }
)
ALLOWED = frozenset(RULES)

DRAW_DELTA = 0.10  # pre-registered: policy mode acts on rule draws only when |v(P) - 0.5| > delta
HALFMOVE_DRAW = 100
CLOCK_FLOOR_S = 3.0
CLOCK_P99_FACTOR = 10.0
DEFAULT_EPSILON = 0.0  # replaced by results/epsilon.json once P8 has chosen it

RepetitionKey = tuple[int, ...]


def repetition_key(board: chess.Board) -> RepetitionKey:
    """Placement, side to move, castling rights and the en-passant square only when a capture is legal.

    Placement is read from python-chess's public bitboards, which is exact and far cheaper than a FEN.
    """
    ep = board.ep_square if board.has_legal_en_passant() else -1
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        int(board.turn),
        board.clean_castling_rights(),
        ep,
    )


@dataclass(frozen=True)
class History:
    """How often each position occurred in the real game so far, including the current one."""

    counts: Mapping[RepetitionKey, int]
    halfmove_clock: int
    has_repeats: bool = field(init=False)  # False: no position occurred twice, so no child can be a third

    def __post_init__(self) -> None:
        object.__setattr__(self, "has_repeats", any(count >= 2 for count in self.counts.values()))

    @classmethod
    def from_board(cls, board: chess.Board) -> "History":
        """Count positions back to the last irreversible move.

        A capture or pawn move resets the halfmove clock and can never be undone, so no position
        before it can occur again; lost castling rights are part of the key. Walking back
        `halfmove_clock` plies is therefore exact, and it only ever pops (never pushes).
        """
        walker = board.copy()
        counts = Counter([repetition_key(walker)])
        for _ in range(min(board.halfmove_clock, len(board.move_stack))):
            walker.pop()
            counts[repetition_key(walker)] += 1
        return cls(MappingProxyType(dict(counts)), board.halfmove_clock)

    def occurrences_with(self, child: chess.Board) -> int:
        """How often the child's position would have occurred once the child is played."""
        return self.counts.get(repetition_key(child), 0) + 1


def rule_draw(child: chess.Board, history: History) -> str | None:
    """R3: the reason the child is a draw by rule, or None. A mate is never a draw."""
    if child.is_checkmate():
        return None
    if child.is_stalemate():
        return "stalemate"
    if child.is_insufficient_material():
        return "insufficient material"
    if child.halfmove_clock >= HALFMOVE_DRAW:
        return "fifty-move rule"
    if history.has_repeats and history.occurrences_with(child) >= 3:
        return "threefold repetition"
    return None


def policy_draw_choice(order: Sequence[int], draws: Collection[int], root_win: float) -> tuple[int, bool]:
    """R3 in policy mode. `order` is the legal vocab indices by descending policy logit.

    Returns (chosen index, whether R3 fired). Winning clearly: rule-draw moves go below all others.
    Losing clearly: the highest-policy rule-draw move is played. Otherwise the plain argmax.
    """
    if not draws or abs(root_win - 0.5) <= DRAW_DELTA:
        return order[0], False
    if root_win > 0.5:
        others = [index for index in order if index not in draws]
        return (others[0] if others else order[0]), True
    return next(index for index in order if index in draws), True


def tie_break(values: np.ndarray, root_logits: np.ndarray, epsilon: float) -> tuple[int, bool]:
    """R4: among values within epsilon of the best, the highest root policy logit wins.

    Both arrays are aligned with the children. Returns (position in that order, whether R4 fired).
    """
    near = np.flatnonzero(values >= values.max() - epsilon)
    if len(near) == 1:
        return int(near[0]), False
    return int(near[np.argmax(root_logits[near])]), True


def tie_set(values: np.ndarray, root_logits: np.ndarray, epsilon: float) -> np.ndarray:
    """R4's whole tie, in child order: within epsilon of the best value and at the highest root logit.

    tie_break plays the first of it. A ladder baseline's policy is flat, so for it this is every move
    within epsilon; it draws one with its tie seed (ValueAgent.tie_seed) instead. Blink never does.
    """
    near = np.flatnonzero(values >= values.max() - epsilon)
    logits = root_logits[near]
    return near[logits == logits.max()]


def clock_guard(remaining_s: float | None, p99_s: float) -> bool:
    """R5: True when the clock is too low for one look per move. It only ever reduces evaluations."""
    if remaining_s is None:
        return False
    return remaining_s < max(CLOCK_FLOOR_S, CLOCK_P99_FACTOR * p99_s)
