"""Evaluators that need no trained weights: the harness's stand-ins for a network.

RandomLogitEvaluator gives pseudo-random logits that depend only on the input row, so it is
deterministic (N4: no randomisation) and the same history always gives the same move. The UCI
engine's --random flag uses it to test the harness end to end without a model.

MaterialEvaluator is a test oracle: its value head reads the material balance 1/3/3/5/9 from the
side to move's view, so value mode with it behaves like a 1-ply material player up to about +10,
where value.win_probability's cp clamp and its one-bin value stop telling bigger leads apart. The
ladder's rung 1 uses it at LADDER_CP_PER_POINT with an exact (two-hot) value, which ranks every
balance a game can reach by material alone.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from blink.board import encode, moves, value
from blink.play.evaluator import Evaluation

PIECE_VALUES = (1, 3, 3, 5, 9, 0)  # pawn, knight, bishop, rook, queen, king
MATERIAL_CP = 100  # centipawns per material point
MAX_BALANCE = 103  # nine queens, two rooks, two bishops and two knights against a bare king
LADDER_CP_PER_POINT = value.CP_CLAMP // MAX_BALANCE  # 9: no balance reaches the clamp, so none share a value


def _code_values() -> np.ndarray:
    table = np.zeros(encode.NUM_CODES, dtype=np.int32)
    for offset, points in enumerate(PIECE_VALUES):
        table[encode.OWN + offset] = points
        table[encode.OPP + offset] = -points
    table[encode.OWN_CASTLING_ROOK] = PIECE_VALUES[3]
    table[encode.OPP_CASTLING_ROOK] = -PIECE_VALUES[3]
    return table


CODE_VALUES = _code_values()


def material_balance(codes: np.ndarray) -> np.ndarray:
    """Own material minus the opponent's, per row of square codes."""
    return CODE_VALUES[np.asarray(codes, dtype=np.uint8)].sum(axis=-1)


def one_hot_value(win: np.ndarray) -> np.ndarray:
    """A value distribution with all its mass in the bin holding each win probability."""
    probs = np.zeros((len(win), value.NUM_BINS), dtype=np.float32)
    bins = np.minimum(value.NUM_BINS - 1, (np.asarray(win) * value.NUM_BINS).astype(np.int64))
    probs[np.arange(len(win)), bins] = 1.0
    return probs


def two_hot(win: np.ndarray) -> np.ndarray:
    """float32 [N, 128]: mass split between the two bin centres around each win probability.

    Its mean is exactly the win probability (clipped to the outer bin centres), so unlike one_hot_value
    it keeps two close win probabilities apart.
    """
    position = np.clip(np.asarray(win, dtype=np.float64) * value.NUM_BINS - 0.5, 0.0, value.NUM_BINS - 1)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, value.NUM_BINS - 1)
    upper = position - low
    probs = np.zeros((len(position), value.NUM_BINS), dtype=np.float64)
    rows = np.arange(len(position))
    np.add.at(probs, (rows, low), 1.0 - upper)
    np.add.at(probs, (rows, high), upper)
    return probs.astype(np.float32)


TABLE_ROWS = 1024  # each output row sums one row of each of two tables: about a million distinct outputs


@lru_cache(maxsize=4)
def _random_tables(seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    policy = rng.standard_normal((2, TABLE_ROWS, moves.NUM_MOVES), dtype=np.float32) / np.sqrt(2)
    value_logits = rng.standard_normal((2, TABLE_ROWS, value.NUM_BINS), dtype=np.float32) / np.sqrt(2)
    return policy[0], policy[1], value_logits[0], value_logits[1]


@dataclass(frozen=True)
class RandomLogitEvaluator:
    """Logits picked by blake2b(seed, row) from two fixed random tables: random-looking, fully deterministic.

    The output for a row depends only on that row and the seed, never on the rest of the batch.
    """

    seed: int = 0

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        codes = np.asarray(codes, dtype=np.uint8)
        salt = self.seed.to_bytes(8, "little")
        hashes = np.array([encode.key_hash(salt + row.tobytes()) for row in codes], dtype=np.uint64)
        low = (hashes % TABLE_ROWS).astype(np.int64)
        high = ((hashes >> np.uint64(32)) % TABLE_ROWS).astype(np.int64)
        p1, p2, v1, v2 = _random_tables(self.seed)
        logits = v1[low] + v2[high]
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (weights / weights.sum(axis=1, keepdims=True)).astype(np.float32)
        return Evaluation((p1[low] + p2[high]).astype(np.float32), probs)


@dataclass(frozen=True)
class MaterialEvaluator:
    """Value = the Lichess win probability of the material balance at cp_per_point cp a point; flat policy.

    The defaults are the test oracle. For the ladder's rung 1 (cp_per_point=LADDER_CP_PER_POINT,
    exact=True) the win probability rises with every point of balance and the value holds it exactly,
    so value mode keeps preferring more material however far ahead it is.
    """

    cp_per_point: int = MATERIAL_CP
    exact: bool = False  # a two-hot value whose mean is the win probability, not one bin's centre

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        balance = material_balance(codes)
        win = np.array([value.win_probability(cp=int(self.cp_per_point * b)) for b in balance])
        policy = np.zeros((len(balance), moves.NUM_MOVES), dtype=np.float32)
        return Evaluation(policy, two_hot(win) if self.exact else one_hot_value(win))
