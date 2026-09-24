"""The 768-bit baseline input: 12 piece planes x 64 squares, read from the side-to-move codes."""

import chess
import numpy as np

from blink.baselines import features
from blink.board import encode


def _bits(board: chess.Board) -> np.ndarray:
    return features.features(encode.encode_board(board)[None])[0]


def _plane(bits: np.ndarray, plane: int) -> set[int]:
    return set(np.flatnonzero(bits[plane * 64 : (plane + 1) * 64]).tolist())


def test_features_are_768_bits_in_the_side_to_move_frame():
    start = chess.Board()
    bits = _bits(start)
    assert bits.shape == (768,) and bits.dtype == np.uint8
    assert set(np.unique(bits).tolist()) == {0, 1}
    assert int(bits.sum()) == 32
    assert _plane(bits, features.OWN_PAWN) == set(range(8, 16))
    assert _plane(bits, features.OPP_PAWN) == set(range(48, 56))
    assert _plane(bits, features.OWN_KING) == {chess.E1}
    # Black to move after 1.e4: Black's pieces are now "own" and seen from Black's side of the board.
    after = chess.Board()
    after.push_uci("e2e4")
    black_view = _bits(after)
    assert _plane(black_view, features.OWN_PAWN) == set(range(8, 16))
    assert chess.square_mirror(chess.E4) in _plane(black_view, features.OPP_PAWN)
    # A position and its colour mirror are the same network input, so the same 768 bits.
    assert np.array_equal(_bits(after), _bits(after.mirror()))


def test_castling_rooks_count_as_rooks_and_the_en_passant_code_is_ignored():
    board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 2")
    codes = encode.encode_board(board)
    assert encode.OWN_CASTLING_ROOK in codes and encode.EP_SQUARE in codes
    bits = features.features(codes[None])[0]
    assert _plane(bits, features.OWN_ROOK) == {chess.A1, chess.H1}
    assert _plane(bits, features.OPP_ROOK) == {chess.A8, chess.H8}
    assert int(bits.sum()) == 8  # 2 kings, 4 rooks, 2 pawns: the ep square adds nothing
    no_rights = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w - - 0 2")
    assert np.array_equal(bits, features.features(encode.encode_board(no_rights)[None])[0])


def test_features_of_packed_records_equal_features_of_their_codes():
    boards = [chess.Board(), chess.Board("8/8/8/8/8/5k2/6q1/7K w - - 0 1")]
    codes = np.stack([encode.encode_board(b) for b in boards])
    records_board = encode.pack(codes)
    assert np.array_equal(features.features_from_packed(records_board), features.features(codes))


def test_features_refuse_rows_that_are_not_64_codes():
    try:
        features.features(np.zeros((2, 63), dtype=np.uint8))
    except ValueError as exc:
        assert "64" in str(exc)
    else:
        raise AssertionError("expected a ValueError")
