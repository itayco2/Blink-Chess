"""The 1880-move vocabulary, always in the side-to-move frame.

1792 from-to pairs (1456 queen lines + 336 knight jumps) plus 88 promotions (22 from-to pairs from the
7th rank to the 8th, times queen, rook, bishop, knight). A black move is flipped to White's view first,
so e7e5 for Black and e2e4 for White share one index. Castling is the king's two-square move (e1g1).
"""

import chess
import numpy as np

PROMO_PIECES = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


def _is_queen_line(frm: int, to: int) -> bool:
    dr = chess.square_rank(to) - chess.square_rank(frm)
    df = chess.square_file(to) - chess.square_file(frm)
    return frm != to and (dr == 0 or df == 0 or abs(dr) == abs(df))


def _is_knight_jump(frm: int, to: int) -> bool:
    dr = abs(chess.square_rank(to) - chess.square_rank(frm))
    df = abs(chess.square_file(to) - chess.square_file(frm))
    return {dr, df} == {1, 2}


FROM_TO: tuple[tuple[int, int], ...] = tuple(
    (frm, to) for frm in range(64) for to in range(64) if _is_queen_line(frm, to) or _is_knight_jump(frm, to)
)
PROMO_PAIRS: tuple[tuple[int, int], ...] = tuple(
    (frm, to)
    for frm in range(chess.A7, chess.H7 + 1)
    for to in range(chess.A8, chess.H8 + 1)
    if abs(chess.square_file(to) - chess.square_file(frm)) <= 1
)
NUM_FROM_TO = len(FROM_TO)
NUM_MOVES = NUM_FROM_TO + len(PROMO_PAIRS) * len(PROMO_PIECES)

_FT_INDEX = np.full((64, 64), -1, dtype=np.int16)
for _i, (_f, _t) in enumerate(FROM_TO):
    _FT_INDEX[_f, _t] = _i
_PROMO_INDEX = {pair: i for i, pair in enumerate(PROMO_PAIRS)}


def frame(square: int, turn: chess.Color) -> int:
    """A square as the side to move sees it: unchanged for White, rank-flipped for Black."""
    return square if turn == chess.WHITE else chess.square_mirror(square)


def encode_move(board: chess.Board, move: chess.Move) -> int:
    frm, to = frame(move.from_square, board.turn), frame(move.to_square, board.turn)
    if move.promotion:
        pair = _PROMO_INDEX.get((frm, to))
        if pair is None:
            raise ValueError(f"{move.uci()} is not a promotion from the 7th to the 8th rank")
        return NUM_FROM_TO + pair * len(PROMO_PIECES) + PROMO_PIECES.index(move.promotion)
    index = int(_FT_INDEX[frm, to])
    if index < 0:
        raise ValueError(f"{move.uci()} is not a queen line or a knight jump")
    return index


def decode_move(board: chess.Board, index: int) -> chess.Move:
    if not 0 <= index < NUM_MOVES:
        raise ValueError(f"move index {index} outside 0..{NUM_MOVES - 1}")
    if index >= NUM_FROM_TO:
        pair, piece = divmod(index - NUM_FROM_TO, len(PROMO_PIECES))
        (frm, to), promotion = PROMO_PAIRS[pair], PROMO_PIECES[piece]
    else:
        (frm, to), promotion = FROM_TO[index], None
    return chess.Move(frame(frm, board.turn), frame(to, board.turn), promotion=promotion)


def legal_mask(board: chess.Board) -> np.ndarray:
    """True at the vocabulary index of every legal move."""
    mask = np.zeros(NUM_MOVES, dtype=bool)
    for move in board.legal_moves:
        mask[encode_move(board, move)] = True
    return mask


_CASTLE_TARGETS = {
    (chess.E1, chess.H1): chess.G1,
    (chess.E1, chess.A1): chess.C1,
    (chess.E8, chess.H8): chess.G8,
    (chess.E8, chess.A8): chess.C8,
}


def uci960_to_standard(board: chess.Board, uci: str) -> str:
    """The eval DB writes castling as king-takes-rook (e1h1). Convert it only when the king is on e1/e8."""
    move = chess.Move.from_uci(uci)
    target = _CASTLE_TARGETS.get((move.from_square, move.to_square))
    piece = board.piece_at(move.from_square)
    if target is None or piece is None or piece.piece_type != chess.KING or piece.color != board.turn:
        return uci
    return chess.Move(move.from_square, target).uci()
