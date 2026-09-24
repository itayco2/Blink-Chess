"""Helpers for the P2 tests: hand-written eval DB lines with exact scores, and a tiny pzstd source.

Scores are given as the eval DB stores them, from White's point of view. The two tests at the bottom
check the helpers themselves against the reference parser.
"""

from pathlib import Path

import chess
import orjson
from data_fakes import lines_text, write_pzstd

from blink.board import value
from blink.data import parse


def db_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])  # the eval DB drops the move counters


def db_uci(board: chess.Board, move: chess.Move) -> str:
    """The eval DB writes castling as king-takes-rook (e1h1)."""
    if board.is_castling(move):
        rook_file = 7 if chess.square_file(move.to_square) > chess.square_file(move.from_square) else 0
        rook = chess.square(rook_file, chess.square_rank(move.from_square))
        return chess.square_name(move.from_square) + chess.square_name(rook)
    return move.uci()


def db_line(fen: str, pvs: list[tuple[str, dict]], depth: int = 30) -> bytes:
    """One eval DB line: pvs are (raw uci first move, {"cp": x} or {"mate": m}) in White's view."""
    rows = [{**score, "line": uci} for uci, score in pvs]
    return orjson.dumps({"fen": fen, "evals": [{"pvs": rows, "knodes": 1000, "depth": depth}]})


def board_line(board: chess.Board, pvs: list[tuple[str, dict]], depth: int = 30) -> bytes:
    """db_line for a python-chess board, with moves given in standard uci (castling as e1g1)."""
    raw = [(db_uci(board, chess.Move.from_uci(uci)), score) for uci, score in pvs]
    return db_line(db_fen(board), raw, depth)


def write_source(path: Path, lines: list[bytes], frame_bytes: int = 4_000) -> Path:
    write_pzstd(path, lines_text(lines), frame_bytes)
    return path


def test_db_line_round_trips_through_the_reference_parser():
    board = chess.Board()
    rec = parse.parse_line(board_line(board, [("e2e4", {"cp": 30}), ("d2d4", {"cp": 25})], depth=33))
    assert int(rec["cp"]) == 30 and int(rec["depth"]) == 33 and int(rec["npv"]) == 2


def test_board_line_writes_castling_as_king_takes_rook():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    line = board_line(board, [("e1g1", {"mate": 3})])
    assert b'"line":"e1h1"' in line
    assert int(parse.parse_line(line)["cp"]) == value.CP_NONE
