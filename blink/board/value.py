"""Scores to win probability to 128 bins.

The Lichess UI mapping (winningChances.ts): logistic in centipawns with the score clamped to +-1000,
and a mate ladder cp = (21 - min(10, |m|)) * 100 that keeps mates above every clamped score and
shorter mates higher. Scores are from the side to move's point of view; mate > 0 means the side to
move mates, mate < 0 means it is mated, and mate == 0 means it is checkmated now.
The HL-Gauss target spreads each label over nearby bins with sigma = 0.75 bin widths
(Farebrother et al. 2024, arXiv 2403.03950).
"""

import math

import numpy as np

NUM_BINS = 128
SIGMA = 0.75 / NUM_BINS
LICHESS_K = 0.00368208
CP_CLAMP = 1000
CP_NONE = -32768  # int16 sentinel: the score is a mate, read the mate field
EDGES = np.linspace(0.0, 1.0, NUM_BINS + 1)
BIN_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2


def mate_to_cp(mate: int) -> float:
    return math.copysign((21 - min(10, abs(mate))) * 100, mate)


def win_probability(cp: int | None = None, mate: int | None = None) -> float:
    if mate is not None:
        if mate == 0:
            return 0.0
        score = mate_to_cp(mate)
    elif cp is not None:
        score = max(-CP_CLAMP, min(CP_CLAMP, cp))
    else:
        raise ValueError("give cp or mate")
    return 1.0 / (1.0 + math.exp(-LICHESS_K * score))


def win_probability_array(cp: np.ndarray, mate: np.ndarray) -> np.ndarray:
    """Vectorised win_probability over record fields (cp == CP_NONE marks a mate)."""
    cp = np.asarray(cp, dtype=np.int32)
    mate = np.asarray(mate, dtype=np.int32)
    is_mate = cp == CP_NONE
    mate_cp = np.sign(mate) * (21 - np.minimum(10, np.abs(mate))) * 100
    score = np.where(is_mate, mate_cp, np.clip(cp, -CP_CLAMP, CP_CLAMP)).astype(np.float64)
    prob = 1.0 / (1.0 + np.exp(-LICHESS_K * score))
    return np.where(is_mate & (mate == 0), 0.0, prob)


def to_bin(p: float) -> int:
    return min(NUM_BINS - 1, int(p * NUM_BINS))


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def hl_gauss(p: float, sigma: float = SIGMA) -> np.ndarray:
    """The HL-Gauss target: a Gaussian around p integrated over each bin, renormalised to [0, 1]."""
    cdf = _normal_cdf((EDGES - p) / sigma)
    mass = np.diff(cdf)
    return mass / mass.sum()
