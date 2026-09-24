"""The sign check: value mode with the true child sign against a flipped one, on ROOT records."""

import random

import chess
import numpy as np
import pytest

from blink import paths
from blink.board import encode, moves
from blink.data.record import ROOT_DTYPE
from blink.eval import signcheck
from blink.play.oracles import MaterialEvaluator, RandomLogitEvaluator

FREE_QUEENS = (
    ("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1", "d1d5"),
    ("3qk3/8/8/8/3Q4/8/8/4K3 b - - 0 1", "d8d4"),
    ("4k3/8/2q5/8/8/5B2/8/4K3 w - - 0 1", "f3c6"),
    ("4k3/8/5b2/8/8/2Q5/8/4K3 b - - 0 1", "f6c3"),
)


def records_for(positions) -> np.ndarray:
    records = np.zeros(len(positions), dtype=ROOT_DTYPE)
    for i, (fen, best) in enumerate(positions):
        board = chess.Board(fen)
        records[i]["board"] = encode.pack(encode.encode_board(board))
        records[i]["move"] = moves.encode_move(board, chess.Move.from_uci(best))
    return records


def random_boards(count: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out, board = [], chess.Board()
    while len(out) < count:
        if board.is_game_over() or board.ply() > 120:
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        out.append(board.copy(stack=False))
    return out


def test_decoded_codes_re_encode_identically():
    specials = [
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1"),
        chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"),
        chess.Board("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1"),
    ]
    for board in specials + random_boards(1000, seed=4):
        codes = encode.encode_board(board)
        assert np.array_equal(encode.encode_board(signcheck.decode_codes(codes)), codes)


def test_the_decoded_board_has_the_same_moves_in_the_same_vocab():
    for board in random_boards(200, seed=9):
        decoded = signcheck.decode_codes(encode.encode_board(board))
        assert decoded.turn == chess.WHITE
        original = sorted(moves.encode_move(board, m) for m in board.legal_moves)
        assert sorted(moves.encode_move(decoded, m) for m in decoded.legal_moves) == original


def test_the_true_sign_beats_the_flipped_sign_with_a_material_oracle():
    result = signcheck.signcheck(MaterialEvaluator(), records_for(FREE_QUEENS))
    assert result["n"] == 4
    assert result["value_top1_true_sign"] == 1.0
    assert result["value_top1_flipped_sign"] == 0.0
    assert result["passes_3x"] is True


def test_batches_never_exceed_the_row_budget():
    sizes = []

    class Counting(RandomLogitEvaluator):
        def evaluate(self, codes):
            sizes.append(len(codes))
            return super().evaluate(codes)

    boards = random_boards(40, seed=1)
    positions = [(b.fen(), next(iter(b.legal_moves)).uci()) for b in boards if not b.is_game_over()]
    result = signcheck.signcheck(Counting(), records_for(positions), max_rows=80)
    assert result["n"] == len(positions)
    assert max(sizes) <= 80
    assert sum(sizes) == len(positions) + sum(chess.Board(fen).legal_moves.count() for fen, _ in positions)


def test_records_are_read_front_to_back_with_a_limit(tmp_path):
    records = records_for(FREE_QUEENS)
    path = tmp_path / "val.bin"
    records.tofile(path)
    assert np.array_equal(signcheck.read_records(path, limit=3), records[:3])
    assert len(signcheck.read_records(path)) == 4


SKELETON_VAL = paths.home() / "data" / "skeleton" / "val.bin"


@pytest.mark.local
@pytest.mark.skipif(not SKELETON_VAL.is_file(), reason="the skeleton pack is not on this machine")
def test_every_real_val_record_decodes_to_a_board_where_its_label_is_legal():
    records = signcheck.read_records(SKELETON_VAL, limit=2000)
    for record in records:
        board = signcheck.decode_codes(encode.unpack(record["board"]))
        assert moves.decode_move(board, int(record["move"])) in board.legal_moves
