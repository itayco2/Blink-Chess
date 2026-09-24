"""The board as 64 square codes (16-letter alphabet), always seen from the side to move.

Codes: 0 empty; 1-6 own pawn..king; 7-12 opponent pawn..king; 13 own rook that can still castle;
14 opponent rook that can still castle; 15 the en-passant target square, set only when a legal
en-passant capture exists. Castling rights and en passant live in the squares, so the board is
exactly 64 tokens and a position and its colour mirror encode to the same bytes.
"""

import hashlib

import chess
import numpy as np

from blink.board.moves import frame

EMPTY = 0
OWN = 1  # own piece code = OWN + piece_type - 1
OPP = 7  # opponent piece code = OPP + piece_type - 1
OWN_CASTLING_ROOK = 13
OPP_CASTLING_ROOK = 14
EP_SQUARE = 15
NUM_CODES = 16


def encode_board(board: chess.Board) -> np.ndarray:
    codes = np.zeros(64, dtype=np.uint8)
    turn = board.turn
    for square, piece in board.piece_map().items():
        base = OWN if piece.color == turn else OPP
        codes[frame(square, turn)] = base + piece.piece_type - 1
    for square in chess.scan_forward(board.clean_castling_rights()):
        rook = board.piece_at(square)
        own = rook is not None and rook.color == turn
        codes[frame(square, turn)] = OWN_CASTLING_ROOK if own else OPP_CASTLING_ROOK
    if board.has_legal_en_passant() and board.ep_square is not None:
        codes[frame(board.ep_square, turn)] = EP_SQUARE
    return codes


def pack(codes: np.ndarray) -> np.ndarray:
    """64 codes -> 32 bytes (two squares per byte, even square in the low nibble). Works on [..., 64]."""
    codes = np.asarray(codes, dtype=np.uint8)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)


def unpack(packed: np.ndarray) -> np.ndarray:
    """32 bytes -> 64 codes. Works on [..., 32]."""
    packed = np.asarray(packed, dtype=np.uint8)
    out = np.empty(packed.shape[:-1] + (64,), dtype=np.uint8)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = packed >> 4
    return out


def position_key(board: chess.Board) -> bytes:
    """The colour-normalised key: what the network sees, so mirror twins share it."""
    return pack(encode_board(board)).tobytes()


def key_hash(key: bytes) -> int:
    """blake2b-8 of a key as an unsigned 64-bit int. Stable across processes (never Python's hash())."""
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")


def position_hash(board: chess.Board) -> int:
    return key_hash(position_key(board))
