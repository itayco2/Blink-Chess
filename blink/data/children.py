"""Child records: the position after each PV's first move, scored from the child's side to move.

A root whose evals[0] has at least 2 PVs gives one child per PV i. The child board is built from the
root's 64 square codes with numpy (no python-chess), in the mover's frame, then turned to the child's
frame: ranks flipped and own/opponent codes swapped. python-chess is asked only when a double push lands
next to an enemy pawn, to decide whether the en-passant square is legal (PF10).

Scores follow the plan: the child's value is 1 - W_i, so cp flips sign; a root line mate +m (the mover
mates) becomes mate -(m-1) for the child, and m = 1 means the child is checkmated now (mate 0, value 0);
a root line mate -m becomes mate +m. The child's depth is the root's search depth, so de-duplication can
keep the deepest root's label.
"""

import hashlib
from collections.abc import Sequence
from typing import NamedTuple

import chess
import numpy as np

from blink.board import encode, moves
from blink.board.value import CP_NONE
from blink.data.record import CHILD_DTYPE, NO_MOVE, NUM_ALTERNATIVES

OWN_PAWN = encode.OWN + chess.PAWN - 1
OWN_ROOK = encode.OWN + chess.ROOK - 1
OWN_KING = encode.OWN + chess.KING - 1
OPP_PAWN = encode.OPP + chess.PAWN - 1
PV_SLOTS = 1 + NUM_ALTERNATIVES  # PV 1 plus the 4 alternatives a root record keeps

_PROMO_FROM_TO = [pair for pair in moves.PROMO_PAIRS for _ in moves.PROMO_PIECES]
MOVE_FROM = np.array([f for f, _ in moves.FROM_TO] + [f for f, _ in _PROMO_FROM_TO], dtype=np.int64)
MOVE_TO = np.array([t for _, t in moves.FROM_TO] + [t for _, t in _PROMO_FROM_TO], dtype=np.int64)
MOVE_PROMO = np.array(
    [0] * moves.NUM_FROM_TO + [piece for _ in moves.PROMO_PAIRS for piece in moves.PROMO_PIECES],
    dtype=np.uint8,
)
# own <-> opponent; the old en-passant marker (15) never survives a move
SWAP = np.array([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 14, 13, 0], dtype=np.uint8)
FLIP = np.arange(64) ^ 56
_MOVABLE = np.zeros(16, dtype=bool)
_MOVABLE[[1, 2, 3, 4, 5, 6, encode.OWN_CASTLING_ROOK]] = True


class Children(NamedTuple):
    records: np.ndarray  # CHILD_DTYPE, grouped by root in PV order
    parent: np.ndarray  # int64 index of each child's root in the roots given


class ExtraPv(NamedTuple):
    """A PV beyond the fifth (the root record keeps only 4 alternatives)."""

    root: int
    move: int
    cp: int
    mate: int


def codes_to_board(codes: np.ndarray) -> chess.Board:
    """64 side-to-move codes -> a python-chess board with White to move (the colour-normalised twin)."""
    board = chess.Board.empty()
    pieces = {}
    rights = 0
    ep = None
    for square, code in enumerate(np.asarray(codes).tolist()):
        if code == encode.EMPTY:
            continue
        if code == encode.EP_SQUARE:
            ep = square
        elif code in (encode.OWN_CASTLING_ROOK, encode.OPP_CASTLING_ROOK):
            pieces[square] = chess.Piece(chess.ROOK, code == encode.OWN_CASTLING_ROOK)
            rights |= chess.BB_SQUARES[square]
        elif code < encode.OPP:
            pieces[square] = chess.Piece(code - encode.OWN + 1, chess.WHITE)
        else:
            pieces[square] = chess.Piece(code - encode.OPP + 1, chess.BLACK)
    board.set_piece_map(pieces)
    board.castling_rights = rights
    board.ep_square = ep
    board.turn = chess.WHITE
    return board


def _check_movers(piece: np.ndarray, moves_: np.ndarray) -> None:
    bad = ~_MOVABLE[piece]
    if bad.any():
        index = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"{int(bad.sum())} moves do not start on an own piece, first: move {int(moves_[index])} "
            f"from square {int(MOVE_FROM[moves_[index]])} holding code {int(piece[index])}"
        )


def _castle(c: np.ndarray, rows: np.ndarray, is_king: np.ndarray, frm: np.ndarray, to: np.ndarray) -> None:
    """Move the rook of a castle, and strip the rights of every king move (in place on the copy)."""
    for king_to, rook_from, rook_to in ((chess.G1, chess.H1, chess.F1), (chess.C1, chess.A1, chess.D1)):
        hit = rows[is_king & (frm == chess.E1) & (to == king_to)]
        c[hit, rook_from] = encode.EMPTY
        c[hit, rook_to] = OWN_ROOK
    kings = rows[is_king]
    sub = c[kings]
    sub[sub == encode.OWN_CASTLING_ROOK] = OWN_ROOK
    c[kings] = sub


def _en_passant(child: np.ndarray, mover: np.ndarray, rows: np.ndarray, frm, to, is_pawn) -> None:
    """Mark the child's ep square where a double push lands next to an enemy pawn that may legally take."""
    double = is_pawn & (to - frm == 16)
    file = to % 8
    left = double & (file > 0) & (mover[rows, np.maximum(to - 1, 0)] == OPP_PAWN)
    right = double & (file < 7) & (mover[rows, np.minimum(to + 1, 63)] == OPP_PAWN)
    for row in np.flatnonzero(left | right):
        ep = int(frm[row] + 8) ^ 56  # the skipped square, in the child's frame
        board = codes_to_board(child[row])
        board.ep_square = ep
        if board.has_legal_en_passant():
            child[row, ep] = encode.EP_SQUARE


def apply_moves(codes: np.ndarray, move_ids: np.ndarray) -> np.ndarray:
    """[M, 64] side-to-move codes and [M] vocabulary moves -> [M, 64] codes of each child, in its frame.

    The moves must be legal (the parser checked them); a move that does not start on an own piece
    raises. Castling rights and en passant follow python-chess exactly (the parity tests prove it).
    """
    c = np.array(codes, dtype=np.uint8, copy=True).reshape(-1, 64)
    move_ids = np.asarray(move_ids, dtype=np.int64)
    rows = np.arange(len(c))
    frm, to, promo = MOVE_FROM[move_ids], MOVE_TO[move_ids], MOVE_PROMO[move_ids]
    piece = c[rows, frm]
    _check_movers(piece, move_ids)
    is_pawn, is_king = piece == OWN_PAWN, piece == OWN_KING
    takes_ep = is_pawn & (c[rows, to] == encode.EP_SQUARE)
    c[rows[takes_ep], to[takes_ep] - 8] = encode.EMPTY
    _castle(c, rows, is_king, frm, to)
    landed = np.where(piece == encode.OWN_CASTLING_ROOK, OWN_ROOK, piece)
    landed = np.where(promo > 0, encode.OWN + promo - 1, landed).astype(np.uint8)
    c[rows, frm] = encode.EMPTY
    c[rows, to] = landed
    c[c == encode.EP_SQUARE] = encode.EMPTY
    child = SWAP[c][:, FLIP]
    _en_passant(child, c, rows, frm, to, is_pawn)
    return child


def child_scores(cp: np.ndarray, mate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Root PV scores (mover's view) -> the child's scores (the opponent's view)."""
    cp = np.asarray(cp, dtype=np.int32)
    mate = np.asarray(mate, dtype=np.int32)
    is_mate = cp == CP_NONE
    child_cp = np.where(is_mate, CP_NONE, -cp).astype(np.int16)
    child_mate = np.where(is_mate, np.where(mate > 0, -(mate - 1), -mate), 0).astype(np.int8)
    return child_cp, child_mate


def hash_boards(packed: np.ndarray) -> np.ndarray:
    """blake2b-8 of each 32-byte packed board as uint64 (equal to encode.key_hash of its bytes)."""
    raw = np.ascontiguousarray(packed, dtype=np.uint8).tobytes()
    digests = b"".join(
        hashlib.blake2b(raw[i : i + 32], digest_size=8).digest() for i in range(0, len(raw), 32)
    )
    return np.frombuffer(digests, dtype="<u8").astype(np.uint64)


def _pv_table(roots: np.ndarray, extras: Sequence[ExtraPv]) -> tuple[np.ndarray, ...]:
    """(parent, move, cp, mate) for every usable PV of every root with at least 2 PVs, root by root."""
    move = np.concatenate([roots["move"][:, None], roots["alt_move"]], axis=1).astype(np.int64)
    cp = np.concatenate([roots["cp"][:, None], roots["alt_cp"]], axis=1).astype(np.int32)
    mate = np.concatenate([roots["mate"][:, None], roots["alt_mate"]], axis=1).astype(np.int32)
    usable = (move != NO_MOVE) & (roots["npv"] >= 2)[:, None] & ~((cp == CP_NONE) & (mate == 0))
    parent, slot = np.nonzero(usable)
    columns = [parent, move[parent, slot], cp[parent, slot], mate[parent, slot]]
    if extras:
        extra = np.array([tuple(e) for e in extras], dtype=np.int64).reshape(-1, 4)
        columns = [np.concatenate([col, extra[:, i]]) for i, col in enumerate(columns)]
    order = np.argsort(columns[0], kind="stable")
    return tuple(col[order] for col in columns)


def children_of(roots: np.ndarray, extras: Sequence[ExtraPv] = ()) -> Children:
    """Every child of `roots` (ROOT_DTYPE), plus the children of PVs beyond the fifth given in extras."""
    parent, move, cp, mate = _pv_table(roots, extras)
    out = np.zeros(len(parent), dtype=CHILD_DTYPE)
    if not len(parent):
        return Children(out, parent.astype(np.int64))
    packed = encode.pack(apply_moves(encode.unpack(roots["board"][parent]), move))
    out["board"] = packed
    out["cp"], out["mate"] = child_scores(cp, mate)
    out["depth"] = roots["depth"][parent]
    out["fen_hash"] = hash_boards(packed)
    return Children(out, parent.astype(np.int64))


def dedupe_deepest(records: np.ndarray) -> tuple[np.ndarray, int]:
    """One record per fen_hash: the deepest label, the first one on a tie. Kept rows stay in order."""
    if not len(records):
        return records, 0
    arrival = np.arange(len(records))
    order = np.lexsort((arrival, -records["depth"].astype(np.int64), records["fen_hash"]))
    hashes = records["fen_hash"][order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = hashes[1:] != hashes[:-1]
    keep = np.sort(order[first])
    return records[keep], len(records) - len(keep)
