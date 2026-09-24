"""The board contract: every checkpoint, the JS tokenizer and the WORLD hash depend on it."""

import random

import chess
import numpy as np
import pytest

from blink.board import encode, moves, value

# -- helpers ------------------------------------------------------------------------------------


def random_positions(n: int, seed: int = 7, max_plies: int = 120) -> list[chess.Board]:
    """Positions from random playouts, including castling, promotions and en passant chances."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randrange(0, max_plies)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if not board.is_game_over():
            out.append(board.copy(stack=False))
    return out


# -- move vocabulary ----------------------------------------------------------------------------


def test_the_vocabulary_has_1792_from_to_moves_plus_88_promotions():
    assert moves.NUM_FROM_TO == 1792
    assert moves.NUM_MOVES == 1880


def test_every_legal_move_round_trips_through_the_1880_vocab():
    seen = 0
    for board in random_positions(2000):
        for move in board.legal_moves:
            index = moves.encode_move(board, move)
            assert 0 <= index < moves.NUM_MOVES
            assert moves.decode_move(board, index) == move
            seen += 1
    assert seen > 40_000


def test_queen_promotion_and_a_plain_rank_7_move_get_different_indices():
    board = chess.Board("4k3/P7/8/8/8/8/8/R3K3 w Q - 0 1")
    promo = moves.encode_move(board, chess.Move.from_uci("a7a8q"))
    board2 = chess.Board("4k3/R7/8/8/8/8/8/4K3 w - - 0 1")
    rook = moves.encode_move(board2, chess.Move.from_uci("a7a8"))
    assert promo != rook
    assert promo >= moves.NUM_FROM_TO > rook


def test_a_black_move_is_encoded_in_the_side_to_move_frame():
    white = chess.Board()
    black = chess.Board()
    black.push_uci("g1f3")
    black_mirror = black.mirror()  # colours swapped, board flipped: white to move
    assert moves.encode_move(black, chess.Move.from_uci("e7e5")) == moves.encode_move(
        white, chess.Move.from_uci("e2e4")
    )
    assert moves.encode_move(black, chess.Move.from_uci("g8f6")) == moves.encode_move(
        black_mirror, chess.Move.from_uci("g1f3")
    )


# -- castling notation --------------------------------------------------------------------------


def test_e1h1_becomes_e1g1_only_when_the_king_stands_on_e1():
    castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    assert moves.uci960_to_standard(castle, "e1h1") == "e1g1"
    assert moves.uci960_to_standard(castle, "e1a1") == "e1c1"
    rook_on_e1 = chess.Board("4k3/8/8/8/8/8/8/K3R2R w - - 0 1")
    assert moves.uci960_to_standard(rook_on_e1, "e1h1") == "e1h1"


def test_e8a8_becomes_e8c8_for_black():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    assert moves.uci960_to_standard(board, "e8a8") == "e8c8"
    assert moves.uci960_to_standard(board, "e8h8") == "e8g8"


def test_castling_conversion_equals_python_chess_canonical_uci():
    for board in random_positions(1500, seed=11):
        for move in board.legal_moves:
            if board.is_castling(move):
                as960 = board.uci(move, chess960=True)
                assert moves.uci960_to_standard(board, as960) == move.uci()


# -- board encoding -----------------------------------------------------------------------------


def test_the_board_is_64_codes_below_16_and_packs_to_32_bytes():
    board = chess.Board()
    codes = encode.encode_board(board)
    assert codes.shape == (64,) and codes.dtype == np.uint8 and codes.max() < 16
    packed = encode.pack(codes)
    assert packed.shape == (32,)
    assert np.array_equal(encode.unpack(packed), codes)


def test_a_position_and_its_colour_mirror_encode_identically():
    for board in random_positions(500, seed=3):
        assert np.array_equal(encode.encode_board(board), encode.encode_board(board.mirror()))


def test_castling_rights_are_folded_into_the_rook_squares():
    full = encode.encode_board(chess.Board())
    none = encode.encode_board(chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"))
    assert full[chess.A1] == encode.OWN_CASTLING_ROOK and full[chess.H8] == encode.OPP_CASTLING_ROOK
    assert none[chess.A1] == encode.OWN + chess.ROOK - 1


def test_the_en_passant_code_matches_python_chess_legal_ep():
    legal = chess.Board("rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 3")
    assert legal.has_legal_en_passant()
    codes = encode.encode_board(legal)
    assert (codes == encode.EP_SQUARE).sum() == 1
    no_capturer = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    assert (encode.encode_board(no_capturer) == encode.EP_SQUARE).sum() == 0


def test_the_colour_normalised_key_ignores_which_side_is_white():
    board = random_positions(1, seed=5)[0]
    assert encode.position_key(board) == encode.position_key(board.mirror())
    assert len(encode.position_key(board)) == 32


def test_the_position_hash_is_stable_across_processes():
    assert encode.position_hash(chess.Board()) == encode.position_hash(chess.Board())
    assert encode.position_hash(chess.Board()) != encode.position_hash(
        chess.Board("8/8/8/8/8/8/8/K1k5 w - - 0 1")
    )


# -- value mapping ------------------------------------------------------------------------------


@pytest.mark.parametrize("cp,expected", [(100, 0.5910), (200, 0.6762), (400, 0.8135), (1000, 0.9755)])
def test_win_probability_matches_lichess_at_100_200_400_1000_cp(cp, expected):
    assert abs(value.win_probability(cp=cp) - expected) < 5e-4
    assert abs(value.win_probability(cp=-cp) - (1 - expected)) < 5e-4


def test_scores_beyond_1000_cp_are_clamped():
    assert value.win_probability(cp=5000) == value.win_probability(cp=1000)


def test_mates_land_in_bins_125_to_127_and_clamped_scores_top_out_at_124():
    assert value.to_bin(value.win_probability(cp=1000)) == 124
    for m in range(1, 8):
        assert value.to_bin(value.win_probability(mate=m)) == 127
    assert value.to_bin(value.win_probability(mate=8)) == 126
    assert value.to_bin(value.win_probability(mate=9)) == 126
    assert value.to_bin(value.win_probability(mate=10)) == 125
    assert value.to_bin(value.win_probability(mate=25)) == 125
    assert value.to_bin(value.win_probability(mate=-3)) == 0


def test_being_checkmated_now_is_a_certain_loss():
    assert value.win_probability(mate=0) == 0.0


def test_hl_gauss_rows_sum_to_one_and_mean_the_label_including_0_and_1():
    for p in (0.0, 0.001, 0.25, 0.5, 0.9755, 0.999, 1.0):
        row = value.hl_gauss(p)
        assert row.shape == (value.NUM_BINS,)
        assert abs(row.sum() - 1.0) < 1e-6
        assert abs(float(row @ value.BIN_CENTERS) - p) <= 1.0 / value.NUM_BINS


def test_sigma_is_three_quarters_of_a_bin():
    assert abs(value.SIGMA - 0.75 / 128) < 1e-12


def test_score_arrays_convert_in_one_vectorised_call():
    cp = np.array([100, value.CP_NONE, -200], dtype=np.int16)
    mate = np.array([0, 3, 0], dtype=np.int8)
    probs = value.win_probability_array(cp, mate)
    assert np.allclose(
        probs, [value.win_probability(cp=100), value.win_probability(mate=3), value.win_probability(cp=-200)]
    )
