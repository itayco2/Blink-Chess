"""768 input bits for the baselines: 12 piece planes x 64 squares, from the side-to-move codes.

Plane p covers squares [64 p, 64 p + 64). Planes 0-5 are the mover's pawn..king, planes 6-11 the
opponent's. A castling-rook code counts as a rook of its colour; the en-passant code is ignored.
Because the codes are already in the side-to-move frame, a position and its colour mirror give the
same bits, exactly as they give the same network input. Numpy only: the evaluator runs torch-free.
"""

import numpy as np

from blink.board import encode

NUM_PLANES = 12
NUM_FEATURES = NUM_PLANES * 64
OWN_PAWN, OWN_KNIGHT, OWN_BISHOP, OWN_ROOK, OWN_QUEEN, OWN_KING = range(6)
OPP_PAWN, OPP_KNIGHT, OPP_BISHOP, OPP_ROOK, OPP_QUEEN, OPP_KING = range(6, 12)
NO_PLANE = -1


def _plane_of_code() -> np.ndarray:
    table = np.full(encode.NUM_CODES, NO_PLANE, dtype=np.int64)
    for offset in range(6):
        table[encode.OWN + offset] = OWN_PAWN + offset
        table[encode.OPP + offset] = OPP_PAWN + offset
    table[encode.OWN_CASTLING_ROOK] = OWN_ROOK
    table[encode.OPP_CASTLING_ROOK] = OPP_ROOK
    return table


PLANE_OF_CODE = _plane_of_code()  # code -> plane, NO_PLANE for empty squares and the ep code


def _checked_codes(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != 64:
        raise ValueError(f"codes must be [N, 64] square codes, got shape {codes.shape}")
    if codes.size and (codes.min() < 0 or codes.max() >= encode.NUM_CODES):
        raise ValueError(f"square codes must be in 0..{encode.NUM_CODES - 1}")
    return codes.astype(np.int64)


def features(codes: np.ndarray) -> np.ndarray:
    """uint8 [N, 768] bits from uint8 [N, 64] square codes."""
    codes = _checked_codes(codes)
    planes = PLANE_OF_CODE[codes]
    rows, squares = np.nonzero(planes != NO_PLANE)
    out = np.zeros((len(codes), NUM_FEATURES), dtype=np.uint8)
    out[rows, planes[rows, squares] * 64 + squares] = 1
    return out


def features_from_packed(boards: np.ndarray) -> np.ndarray:
    """The same bits from packed record boards (uint8 [N, 32])."""
    return features(encode.unpack(boards))
