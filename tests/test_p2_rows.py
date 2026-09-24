"""rows: the P2 wrapper around the frozen reference parser, with a Chess960 reason code and children."""

import chess
import numpy as np
import pytest
from data_fakes import fixture_lines
from test_p2_fakes import board_line, db_line

from blink.data import children, parse, rows

CHESS960_START = "bqnbrkrn/pppppppp/8/8/8/8/PPPPPPPP/BQNBRKRN w KQkq -"


def test_a_position_with_chess960_castling_rights_is_rejected_as_chess960():
    line = db_line(CHESS960_START, [("e2e4", {"cp": 20}), ("d2d4", {"cp": 15})])
    with pytest.raises(parse.Rejected) as caught:
        rows.parse_root(line)
    assert caught.value.reason == "chess960"


def test_a_chess960_castle_as_best_move_is_rejected_as_chess960_not_illegal_best_move():
    # king f1, rook h1, right "H" (Shredder-FEN): the DB writes the castle as f1h1
    fen = "4k3/8/8/8/8/8/8/5K1R w H -"
    line = db_line(fen, [("f1h1", {"cp": 50})])
    with pytest.raises(parse.Rejected) as caught:
        parse.parse_line(line)
    assert caught.value.reason in ("illegal_best_move", "chess960")
    with pytest.raises(parse.Rejected) as caught:
        rows.parse_root(line)
    assert caught.value.reason == "chess960"


def test_a_castle_whose_rights_look_standard_but_the_move_is_chess960_is_still_chess960(monkeypatch):
    line = db_line("4k3/8/8/8/8/8/8/5K1R w - -", [("f1h1", {"cp": 50})])
    monkeypatch.setattr(rows, "_chess960_castle", lambda raw: True)
    with pytest.raises(parse.Rejected) as caught:
        rows.parse_root(line)
    assert caught.value.reason == "chess960"


def test_an_ordinary_illegal_best_move_keeps_its_reason():
    line = db_line("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -", [("e2e5", {"cp": 10})])
    with pytest.raises(parse.Rejected) as caught:
        rows.parse_root(line)
    assert caught.value.reason == "illegal_best_move"


@pytest.mark.parametrize(
    ("placement", "rights", "chess960"),
    [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "KQkq", False),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR", "HAha", False),  # Shredder letters, standard squares
        ("r3k2r/8/8/8/8/8/8/R3K2R", "-", False),
        ("r3k1r1/8/8/8/8/8/8/R3K2R", "Kq", False),
        ("r3k1r1/8/8/8/8/8/8/R3K2R", "Kk", True),  # k needs a rook on h8
        ("4k3/8/8/8/8/8/8/5K1R", "K", True),  # king off e1
        ("4k3/8/8/8/8/8/8/R3K1R1", "G", True),  # a g-file rook right
    ],
)
def test_castling_rights_are_chess960_unless_king_and_rook_stand_on_standard_squares(
    placement, rights, chess960
):
    assert rows.chess960_rights(placement, rights) is chess960


def test_every_fixture_row_parses_exactly_like_the_reference_parser():
    for line in fixture_lines():
        mine = rows.parse_root(line)
        assert mine.tobytes() == parse.parse_line(line).tobytes()


def test_parse_rows_counts_rejects_by_reason_and_collects_extra_pvs():
    board = chess.Board()
    firsts = ["e2e4", "d2d4", "g1f3", "c2c4", "b1c3", "f2f4"]
    many = board_line(board, [(uci, {"cp": 30 - i}) for i, uci in enumerate(firsts)])
    lines = [
        many,
        db_line(CHESS960_START, [("e2e4", {"cp": 20})]),
        b"not json",
        board_line(board, [("e2e4", {"cp": 30})]),
    ]
    got = rows.parse_rows(lines)
    assert len(got.roots) == 2
    assert got.rejects == {"chess960": 1, "bad_row": 1}
    assert got.errors == {}
    assert [(e.root, e.cp) for e in got.extras] == [(0, 25)]
    kids = children.children_of(got.roots, got.extras)
    assert len(kids.records) == 6 and set(kids.parent.tolist()) == {0}


def test_parse_children_of_a_single_pv_line_is_empty():
    line = board_line(chess.Board(), [("e2e4", {"cp": 30})])
    assert len(rows.parse_children(line)) == 0


def test_an_unexpected_exception_is_counted_as_an_error_with_a_sample(monkeypatch):
    def boom(line):
        raise KeyError("surprise")

    monkeypatch.setattr(rows, "parse_root", boom)
    got = rows.parse_rows([b"{}"])
    assert got.errors == {"KeyError": 1}
    assert "surprise" in got.error_samples[0]
    assert len(got.roots) == 0 and got.roots.dtype == np.dtype(parse.ROOT_DTYPE)
