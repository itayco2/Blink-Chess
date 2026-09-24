"""Record formats and raw-line parsing, checked on real (CC0) rows from the Lichess eval DB."""

from pathlib import Path

import chess
import numpy as np
import orjson

from blink.board import encode, moves, value
from blink.data import parse, record

FIXTURE = Path(__file__).parent / "fixtures" / "eval_lines_first100.jsonl"


def fixture_lines() -> list[bytes]:
    return FIXTURE.read_bytes().splitlines()


def test_record_dtypes_are_68_and_44_bytes():
    assert record.ROOT_DTYPE.itemsize == 68
    assert record.CHILD_DTYPE.itemsize == 44


def test_raw_eval_db_scores_are_white_pov_on_real_rows():
    """Real row 0: Black to move, PV scores 69, 163, 229. Rising scores only fit White's view."""
    row = orjson.loads(fixture_lines()[0])
    assert row["fen"].split()[1] == "b"
    cps = [pv["cp"] for pv in row["evals"][0]["pvs"]]
    assert cps[:3] == [69, 163, 229]
    assert cps == sorted(cps)  # White's view: Black's best line is the one best for Black, lowest for White


def test_black_to_move_scores_flip_so_the_best_pv_is_highest_for_the_mover():
    rec = parse.parse_line(fixture_lines()[0])
    assert rec["cp"] == -69
    alt = rec["alt_cp"][:2].tolist()
    assert alt == [-163, -229]
    assert rec["cp"] > alt[0] > alt[1]


def test_evals_zero_pv_zero_is_the_label_and_alternatives_come_from_evals_zero():
    line = fixture_lines()[0]
    row = orjson.loads(line)
    rec = parse.parse_line(line)
    board = chess.Board(row["fen"] + " 0 1")
    best = chess.Move.from_uci(row["evals"][0]["pvs"][0]["line"].split()[0])
    assert moves.decode_move(board, int(rec["move"])) == best
    assert rec["npv"] == len(row["evals"][0]["pvs"])
    assert rec["depth"] == row["evals"][0]["depth"]


def test_every_fixture_row_parses_to_a_legal_best_move():
    for line in fixture_lines():
        rec = parse.parse_line(line)
        board = chess.Board(orjson.loads(line)["fen"] + " 0 1")
        assert moves.decode_move(board, int(rec["move"])) in board.legal_moves
        assert np.array_equal(encode.unpack(rec["board"]), encode.encode_board(board))
        assert rec["fen_hash"] == encode.position_hash(board)


def test_a_mate_score_uses_the_cp_sentinel_and_side_to_move_sign():
    line = orjson.dumps(
        {
            "fen": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - -",
            "evals": [{"pvs": [{"mate": 1, "line": "d1d8"}], "depth": 30}],
        }
    )
    rec = parse.parse_line(line)
    assert rec["cp"] == value.CP_NONE and rec["mate"] == 1
    black = orjson.dumps(
        {
            "fen": "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - -",
            "evals": [{"pvs": [{"mate": -1, "line": "d8d1"}], "depth": 30}],
        }
    )
    assert parse.parse_line(black)["mate"] == 1  # White-POV -1 is "Black mates in 1" for the mover


def test_castling_in_the_best_line_is_converted_to_standard_uci():
    line = orjson.dumps(
        {
            "fen": "r3k2r/pppq1ppp/8/8/8/8/PPPQ1PPP/R3K2R w KQkq -",
            "evals": [{"pvs": [{"cp": 10, "line": "e1h1"}], "depth": 20}],
        }
    )
    rec = parse.parse_line(line)
    board = chess.Board("r3k2r/pppq1ppp/8/8/8/8/PPPQ1PPP/R3K2R w KQkq - 0 1")
    assert moves.decode_move(board, int(rec["move"])) == chess.Move.from_uci("e1g1")


def test_a_row_without_evals_is_rejected_with_a_reason():
    line = orjson.dumps({"fen": "8/8/8/8/8/8/8/K1k5 w - -", "evals": []})
    try:
        parse.parse_line(line)
    except parse.Rejected as exc:
        assert exc.reason == "no_evals"
        return
    raise AssertionError("expected Rejected")


def test_an_illegal_best_move_is_rejected_with_a_reason():
    line = orjson.dumps(
        {"fen": "8/8/8/8/8/8/8/K1k5 w - -", "evals": [{"pvs": [{"cp": 0, "line": "a1a8"}], "depth": 20}]}
    )
    try:
        parse.parse_line(line)
    except parse.Rejected as exc:
        assert exc.reason == "illegal_best_move"
        return
    raise AssertionError("expected Rejected")


def test_unused_alternative_slots_hold_the_no_move_sentinel():
    rec = parse.parse_line(fixture_lines()[0])
    assert rec["npv"] >= 2
    single = orjson.dumps(
        {"fen": "8/8/8/8/8/8/8/K1k5 w - -", "evals": [{"pvs": [{"cp": 0, "line": "a1a2"}], "depth": 20}]}
    )
    assert parse.parse_line(single)["alt_move"].tolist() == [record.NO_MOVE] * 4
