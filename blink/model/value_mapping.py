"""Which win probability each (cp, mate) label becomes as the value head's target (train.value_mapping).

"lichess" (Recipe D) is blink.board.value's frozen mapping: the Lichess logistic with cp clamped to
+-1000 and the mate ladder cp = (21 - min(10, |m|)) * 100, so mates outrank every clamped score and
shorter mates rank higher.

"deepmind" (P5 arm a08) is searchless_chess's mapping: the same logistic, 1 / (1 + exp(-0.00368208 cp)),
with no clamp (utils.centipawns_to_win_probability, github.com/google-deepmind/searchless_chess
src/utils.py), and every mate a certain result: 1.0 for the side that mates, 0.0 for the side being
mated ("mates ... to a win percentage of 100%", arXiv 2402.04494). Inside +-1000 cp the two agree bit
for bit; they differ beyond the clamp and on mates, where DeepMind puts mate-in-1 and mate-in-15 in
the same top bin (the "indecisiveness" the a08 mate-preserving guard watches for).

Only the value target moves. The soft policy target keeps Lichess W (blink.train.batch w_best and
w_alt), so a08 changes one thing. Scores are from the side to move's point of view, and a record
whose cp is CP_NONE carries a mate: mate > 0 mates, mate < 0 is mated, mate == 0 is checkmated now.
This module is torch-free.
"""

import numpy as np

from blink.board.value import CP_NONE, LICHESS_K
from blink.board.value import win_probability_array as lichess_win_probability_array
from blink.model.config import VALUE_MAPPINGS

LICHESS, DEEPMIND = VALUE_MAPPINGS


def deepmind_win_probability_array(cp: np.ndarray, mate: np.ndarray) -> np.ndarray:
    """Float64 win probabilities: the unclamped logistic in cp, and 1.0 / 0.0 for every mate."""
    cp = np.asarray(cp, dtype=np.int32)
    mate = np.asarray(mate, dtype=np.int32)
    is_mate = cp == CP_NONE
    prob = 1.0 / (1.0 + np.exp(-LICHESS_K * cp.astype(np.float64)))  # |K cp| <= 121: exp cannot overflow
    return np.where(is_mate, (mate > 0).astype(np.float64), prob)


def win_probability_array(cp: np.ndarray, mate: np.ndarray, mapping: str = LICHESS) -> np.ndarray:
    """The value targets of record fields under a train.value_mapping name."""
    if mapping == LICHESS:
        return lichess_win_probability_array(cp, mate)
    if mapping == DEEPMIND:
        return deepmind_win_probability_array(cp, mate)
    raise ValueError(f"train.value_mapping must be one of {VALUE_MAPPINGS}, got {mapping!r}")
