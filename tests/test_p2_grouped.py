"""test_grouped: whole groups (pawn structure + material) held out, chosen by a salted hash."""

import itertools

import chess
import numpy as np
import pytest

from blink.board import encode
from blink.data import grouped


def board_with_pawns(own: tuple[int, ...], opp: tuple[int, ...] = (), knights: int = 0) -> np.ndarray:
    codes = np.zeros(64, dtype=np.uint8)
    codes[chess.E1] = encode.OWN + chess.KING - 1
    codes[chess.E8] = encode.OPP + chess.KING - 1
    for square in own:
        codes[square] = encode.OWN + chess.PAWN - 1
    for square in opp:
        codes[square] = encode.OPP + chess.PAWN - 1
    for square in (chess.B1, chess.G1, chess.C1)[:knights]:
        codes[square] = encode.OWN + chess.KNIGHT - 1
    return encode.pack(codes)


def packed(fen: str) -> np.ndarray:
    return encode.pack(encode.encode_board(chess.Board(fen)))


def test_a_group_is_pawns_plus_material_whatever_the_pieces_do():
    a = packed("4k3/pp6/8/8/8/8/PP6/4K1N1 w - - 0 1")
    knight_moved = packed("4k3/pp6/8/8/8/5N2/PP6/4K3 w - - 0 1")
    pawn_moved = packed("4k3/pp6/8/8/8/P7/1P6/4K1N1 w - - 0 1")
    knight_became_bishop = packed("4k3/pp6/8/8/8/8/PP6/4K1B1 w - - 0 1")
    keys = grouped.group_hashes(np.stack([a, knight_moved, pawn_moved, knight_became_bishop]), salt=3)
    assert keys[0] == keys[1]
    assert len(set(keys.tolist())) == 3


def test_a_position_and_its_colour_mirror_share_a_group():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    twin = packed(board.mirror().fen())
    assert (
        grouped.group_hashes(np.stack([packed(board.fen()), twin]), salt=11).tolist()[0]
        == (grouped.group_hashes(twin[None], salt=11).tolist()[0])
    )


def test_membership_is_the_salted_group_hash_mod_1000_equal_to_7():
    boards = np.stack([board_with_pawns((s,)) for s in range(8, 48)])
    for salt in (0, 5):
        hashes = grouped.group_hashes(boards, salt)
        assert (grouped.selected(boards, salt) == (hashes % np.uint64(1000) == np.uint64(7))).all()
    assert (grouped.group_hashes(boards, 0) != grouped.group_hashes(boards, 5)).any()


def _giant_for_salt_zero() -> np.ndarray:
    for squares in itertools.combinations(range(8, 48), 3):
        board = board_with_pawns(squares)
        if grouped.selected(board[None], 0)[0]:
            return board
    raise AssertionError("no three-pawn group is selected by salt 0")


def test_grouped_split_never_selects_a_giant_group():
    giant = _giant_for_salt_zero()
    small = [
        board_with_pawns(sq, knights=k) for sq in itertools.combinations(range(8, 48), 2) for k in (0, 1)
    ]
    boards = np.stack([giant] * 500 + small)
    choice = grouped.choose_salt(boards, giant_share=0.01)
    assert choice.salt != 0  # salt 0 would hold out the giant group
    assert not grouped.selected(giant[None], choice.salt)[0]
    assert choice.largest_selected_roots <= 0.01 * len(boards)
    assert choice.tried == choice.salt + 1
    assert choice.selected_roots == int(grouped.selected(boards, choice.salt).sum()) > 0


def test_the_default_giant_threshold_is_one_hundredth_of_a_percent():
    assert grouped.GIANT_SHARE == 1e-4


def test_choose_salt_gives_up_with_a_clear_error():
    giant = _giant_for_salt_zero()
    with pytest.raises(ValueError, match="no salt in 0..0"):
        grouped.choose_salt(np.stack([giant] * 10 + [board_with_pawns((9,))]), giant_share=0.5, max_tries=1)
    with pytest.raises(ValueError, match="every one of 1 groups"):
        grouped.choose_salt(np.stack([giant] * 10), giant_share=0.01)


def test_the_choice_reports_the_numbers_the_manifest_records():
    boards = np.stack([board_with_pawns(sq) for sq in itertools.combinations(range(8, 48), 2)])
    report = grouped.choose_salt(boards, giant_share=0.01).as_dict()
    assert set(report) >= {"salt", "probe_roots", "groups", "selected_groups", "selected_roots", "rule"}
    assert report["probe_roots"] == len(boards) == report["groups"]
