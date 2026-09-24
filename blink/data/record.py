"""Fixed-width, little-endian record formats. Part of the frozen contract.

Scores are stored raw and from the side to move's point of view, so the value mapping can change
without a data rebuild. cp == CP_NONE (-32768) means "this score is a mate: read the mate field";
mate > 0: the side to move mates in that many moves; mate < 0: it is mated; mate == 0 with
cp == CP_NONE: it is checkmated now.
"""

import numpy as np

NO_MOVE = 0xFFFF
NUM_ALTERNATIVES = 4

ROOT_DTYPE = np.dtype(
    [
        ("board", "u1", 32),  # 64 square codes, 4 bits each (blink.board.encode.pack)
        ("move", "<u2"),  # best move index in the 1880 vocabulary
        ("cp", "<i2"),
        ("mate", "i1"),
        ("depth", "u1"),
        ("npv", "u1"),  # PV count of evals[0]
        ("flags", "u1"),
        ("alt_move", "<u2", NUM_ALTERNATIVES),  # PV 2..5 first moves, NO_MOVE when absent
        ("alt_cp", "<i2", NUM_ALTERNATIVES),
        ("alt_mate", "i1", NUM_ALTERNATIVES),
        ("fen_hash", "<u8"),  # blake2b-8 of the colour-normalised key
    ]
)

CHILD_DTYPE = np.dtype(
    [
        ("board", "u1", 32),
        ("cp", "<i2"),
        ("mate", "i1"),
        ("depth", "u1"),
        ("fen_hash", "<u8"),
    ]
)
