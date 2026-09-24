"""canonical_epd: the EPD text a packed board stands for, written without python-chess.

The 64 side-to-move codes hold every EPD field except the side to move: the pieces, the castling rights
(a castling rook) and the en-passant square (set only when a legal capture exists). This module decodes
them straight to text. python-chess's Board.epd() writes a castling right only when its king and rook
stand ready and an en-passant square only when a legal capture exists, so a pack whose boards give the
same text is canonical in python-chess's sense (the plan's canonical_epd parity, P2). The parity tests
compare it with epd() of the source row; verify compares it with epd() of each sampled board.

Standard chess only: Chess960 rows never reach a pack (rows.CHESS960), so a castling rook off a1, h1,
a8 or h8 raises ValueError, as do two en-passant squares.
"""

import numpy as np

from blink.board import encode

PIECE_LETTERS = "pnbrqk"
FILES = "abcdefgh"
# a castling rook's real square -> (the right it gives, the rook is White's)
CASTLING_SQUARES = {7: ("K", True), 0: ("Q", True), 63: ("k", False), 56: ("q", False)}
RIGHTS_ORDER = "KQkq"


def square_name(square: int) -> str:
    return f"{FILES[square % 8]}{square // 8 + 1}"


def _placement(letters: list[str]) -> str:
    ranks = []
    for rank in range(7, -1, -1):
        text, empty = "", 0
        for letter in letters[rank * 8 : rank * 8 + 8]:
            if letter:
                text += (str(empty) if empty else "") + letter
                empty = 0
            else:
                empty += 1
        ranks.append(text + (str(empty) if empty else ""))
    return "/".join(ranks)


def _castling_right(square: int, white: bool) -> str:
    right, rook_is_white = CASTLING_SQUARES.get(square, (None, None))
    if right is None or rook_is_white != white:
        colour = "white" if white else "black"
        raise ValueError(f"a {colour} castling rook on {square_name(square)} is off its back-rank corners")
    return right


def canonical_epd(codes: np.ndarray, white_to_move: bool = True) -> str:
    """64 side-to-move codes -> 'placement side castling ep'. The default gives the colour-normalised twin."""
    letters = [""] * 64
    rights: list[str] = []
    ep = "-"
    for square, code in enumerate(np.asarray(codes).tolist()):
        if code == encode.EMPTY:
            continue
        real = square if white_to_move else square ^ 56
        if code == encode.EP_SQUARE:
            if ep != "-":
                raise ValueError(f"two en-passant squares: {ep} and {square_name(real)}")
            ep = square_name(real)
            continue
        own = code < encode.OPP or code == encode.OWN_CASTLING_ROOK
        white = own == white_to_move
        if code in (encode.OWN_CASTLING_ROOK, encode.OPP_CASTLING_ROOK):
            rights.append(_castling_right(real, white))
            letter = "r"
        else:
            letter = PIECE_LETTERS[code - (encode.OWN if own else encode.OPP)]
        letters[real] = letter.upper() if white else letter
    castling = "".join(sorted(rights, key=RIGHTS_ORDER.index)) or "-"
    return f"{_placement(letters)} {'w' if white_to_move else 'b'} {castling} {ep}"
