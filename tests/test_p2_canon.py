"""canonical_epd: a packed board written back as EPD text with no python-chess, checked against epd()."""

import random

import chess
import numpy as np
import orjson
import pytest
from data_fakes import fixture_lines

from blink.board import encode
from blink.data import canon, parse, rows

EDGE_FENS = (
    # a d6 square with no white pawn beside d5: python-chess drops it
    "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6",
    # a usable d3 square, Black to move
    "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3",
    # the only pawn that could take on d6 is pinned to its king along rank 5: no legal ep, square dropped
    "8/8/8/K1Pp3r/8/8/8/7k w - d6",
    # the pinned pawn case with the rook gone: the capture is legal again, square kept
    "8/8/8/K1Pp4/8/8/8/7k w - d6",
    # a K right with no rook on h1 is dropped; with the black king off e8 both black rights are
    "r3k3/8/8/8/8/8/8/R3K3 w KQq -",
    "1r3k1r/8/8/8/8/8/8/R3K2R b KQkq -",
    # every right, both sides
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq -",
)


def random_boards(n: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randrange(0, 90)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        out.append(board.copy(stack=False))
    return out


def packed_codes(board: chess.Board) -> np.ndarray:
    return encode.unpack(encode.pack(encode.encode_board(board)))


def test_canonical_epd_equals_python_chess_epd():
    checked = 0
    for line in fixture_lines():
        try:
            record = rows.parse_root(line)
        except parse.Rejected:
            continue
        board = chess.Board(orjson.loads(line)["fen"] + " 0 1")
        assert canon.canonical_epd(encode.unpack(record["board"]), board.turn) == board.epd(), line[:80]
        checked += 1
    for board in random_boards(500, seed=3) + [chess.Board(fen + " 0 1") for fen in EDGE_FENS]:
        assert canon.canonical_epd(packed_codes(board), board.turn) == board.epd(), board.fen()
        checked += 1
    assert checked > 500


def test_the_colour_normalised_canonical_epd_is_the_mirror_twins_epd():
    for board in random_boards(150, seed=5) + [chess.Board(fen + " 0 1") for fen in EDGE_FENS]:
        twin = board if board.turn == chess.WHITE else board.mirror()
        assert canon.canonical_epd(packed_codes(board)) == twin.epd(), board.fen()


@pytest.mark.parametrize(
    ("fen", "expected"),
    [
        ("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6", "KQkq -"),
        ("8/8/8/K1Pp3r/8/8/8/7k w - d6", "- -"),
        ("8/8/8/K1Pp4/8/8/8/7k w - d6", "- d6"),
        ("rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3", "KQkq d3"),
        ("r3k3/8/8/8/8/8/8/R3K3 w KQq -", "Qq -"),
    ],
)
def test_a_dead_en_passant_square_and_a_stale_castling_right_are_dropped_as_python_chess_drops_them(
    fen, expected
):
    board = chess.Board(fen + " 0 1")
    epd = canon.canonical_epd(packed_codes(board), board.turn)
    assert epd.split(" ", 2)[2] == expected


def test_a_castling_rook_off_the_corner_squares_is_refused():
    codes = np.zeros(64, dtype=np.uint8)
    codes[chess.E1] = encode.OWN + chess.KING - 1
    codes[chess.E8] = encode.OPP + chess.KING - 1
    codes[chess.B1] = encode.OWN_CASTLING_ROOK
    with pytest.raises(ValueError, match="b1"):
        canon.canonical_epd(codes)


def test_two_en_passant_squares_are_refused():
    codes = np.zeros(64, dtype=np.uint8)
    codes[chess.E1] = encode.OWN + chess.KING - 1
    codes[chess.E8] = encode.OPP + chess.KING - 1
    codes[chess.D6] = codes[chess.F6] = encode.EP_SQUARE
    with pytest.raises(ValueError, match="en-passant"):
        canon.canonical_epd(codes)
