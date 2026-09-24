"""EvalBudget: the no-search counter around every network call of one decision (NSC-1, N1 and N2).

One budget wraps an Evaluator for exactly one decision at position P. It hashes P and every child
P.m from its own copy of P (never from rows the agent built), so it can refuse, before the network
sees anything:
  - a second call in the same decision;
  - more than L(P)+1 rows;
  - a row that is not P or a child of P (a grandchild, a random position);
  - a row repeated beyond its multiplicity in {P} U children.
At the end of the decision `finish` checks the rules that fired and logs
{game, ply, mode, n_legal, n_rows, n_calls, rule}.
"""

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType

import chess
import numpy as np

from blink.board import encode
from blink.play.evaluator import Evaluation, Evaluator
from blink.play.rules import ALLOWED


class NoSearchViolation(RuntimeError):
    """A decision tried to look further than NSC-1 allows. Never caught inside Blink."""


@dataclass(frozen=True)
class DecisionRecord:
    game: str
    ply: int
    mode: str
    n_legal: int
    n_rows: int
    n_calls: int
    rule: str

    def as_dict(self) -> dict:
        return asdict(self)


def _row_key(codes: np.ndarray) -> bytes:
    return encode.pack(codes).tobytes()


class EvalBudget:
    """Wraps an Evaluator for ONE decision at `board`; see the module docstring for what it refuses."""

    def __init__(
        self,
        evaluator: Evaluator,
        board: chess.Board,
        mode: str,
        game: str = "",
        sink: Callable[[DecisionRecord], None] | None = None,
    ) -> None:
        self._evaluator = evaluator
        self._mode = mode
        self._game = game
        self._sink = sink
        self._board = board.copy(stack=False)
        self._ply = board.ply()
        self._n_legal = board.legal_moves.count()
        self.root_codes = encode.encode_board(board)
        self._root_key = _row_key(self.root_codes)
        self._child_codes: Mapping[chess.Move, np.ndarray] | None = None
        self.n_calls = 0
        self.n_rows = 0

    def child_codes(self) -> Mapping[chess.Move, np.ndarray]:
        """The square codes of every child P.m, computed by the budget itself from its own copy of P.

        They are hashed the first time anything but P is asked for; a one-look decision never needs them.
        """
        if self._child_codes is None:
            walker, codes = self._board.copy(stack=False), {}
            for move in walker.legal_moves:
                walker.push(move)
                codes[move] = encode.encode_board(walker)
                walker.pop()
            self._child_codes = MappingProxyType(codes)
        return self._child_codes

    def _allowed(self, keys: Counter) -> Counter:
        if keys == Counter({self._root_key: 1}):
            return keys
        return Counter([self._root_key, *(_row_key(c) for c in self.child_codes().values())])

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        codes = np.asarray(codes, dtype=np.uint8)
        if codes.ndim != 2 or codes.shape[1] != 64 or len(codes) == 0:
            raise ValueError(f"expected uint8 [N>=1, 64] square codes, got {codes.shape}")
        if self.n_calls >= 1:
            raise NoSearchViolation(f"a second network call in one decision (ply {self._ply})")
        if len(codes) > self._n_legal + 1:
            raise NoSearchViolation(f"{len(codes)} rows exceed L+1 = {self._n_legal + 1} (ply {self._ply})")
        keys = Counter(_row_key(row) for row in codes)
        allowed_keys = self._allowed(keys)
        for key, count in keys.items():
            allowed = allowed_keys.get(key, 0)
            if allowed == 0:
                raise NoSearchViolation(f"a row outside {{P}} U children (ply {self._ply})")
            if count > allowed:
                raise NoSearchViolation(f"a repeated row: {count} copies of one position (ply {self._ply})")
        self.n_calls = 1
        self.n_rows = len(codes)
        return self._evaluator.evaluate(codes)

    def finish(self, rules: Sequence[str]) -> DecisionRecord:
        unknown = sorted(set(rules) - ALLOWED)
        if unknown:
            raise NoSearchViolation(f"rules outside the closed list R1-R5: {unknown}")
        mate_now = "R2" in rules
        if self.n_calls == 0 and not mate_now:
            raise NoSearchViolation("a decision without a network call is allowed only when R2 fired")
        if self.n_calls > 0 and mate_now:
            raise NoSearchViolation("R2 fires before the network call, so it cannot follow one")
        record = DecisionRecord(
            game=self._game,
            ply=self._ply,
            mode=self._mode,
            n_legal=self._n_legal,
            n_rows=self.n_rows,
            n_calls=self.n_calls,
            rule=",".join(rules),
        )
        if self._sink is not None:
            self._sink(record)
        return record
