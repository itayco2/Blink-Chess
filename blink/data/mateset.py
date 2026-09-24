"""mateset: val roots where the side to move mates in 2 to 5, with every legal child (valprobe arrays).

It measures whether value mode keeps a forced mate and picks the shortest one (a08 and the E2 static
metrics). The arrays are valprobe's plus mate_in [N] int8.
"""

from pathlib import Path

import numpy as np

from blink.board.value import CP_NONE
from blink.data import valprobe

MATE_MIN, MATE_MAX = 2, 5
OUTPUT = "mateset.npz"


def eligible(roots: np.ndarray) -> np.ndarray:
    mate = roots["mate"].astype(np.int64)
    return (roots["cp"] == CP_NONE) & (mate >= MATE_MIN) & (mate <= MATE_MAX)


def select(roots: np.ndarray, n: int) -> np.ndarray:
    return valprobe.first_unique(roots, eligible(roots), n)


def build(roots: np.ndarray, n: int) -> dict[str, np.ndarray]:
    chosen = roots[select(roots, n)]
    return {**valprobe.probe_arrays(chosen), "mate_in": chosen["mate"].astype(np.int8)}


def run(pack_dir: Path, n: int) -> dict:
    return valprobe.run(pack_dir, n, builder=build, output=OUTPUT)
