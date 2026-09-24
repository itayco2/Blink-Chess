"""DeepMind's puzzle scorer (an exact port, Apache-2.0) and Blink's puzzle runner around it."""

import csv
import json
from pathlib import Path

import chess
import pytest

from blink.eval import puzzles
from blink.play import agents
from blink.play.oracles import RandomLogitEvaluator

DM_PUZZLES = puzzles.resolve_set("dm10k")
TWO_MATES = "6k1/5ppp/8/n7/8/8/5PPP/R3R1K1 b - - 0 1"  # after ...Nc4 both Ra8# and Re8# mate
DOUBLE_ROOKS = "3qr1k1/p4ppp/8/8/8/8/4RPPP/4R1K1 b - - 0 1"  # ...a6 Rxe8+ Qxe8 Rxe8#
DOUBLE_ROOKS_LINE = ["a7a6", "e2e8", "d8e8", "e1e8"]
SCHOLAR = {"PuzzleId": "s1", "Rating": "650", "PGN": "1. e4 e5 2. Bc4 Nc6 3. Qh5", "Moves": "g8f6 h5f7"}


class Scripted:
    """DeepMind's Engine protocol: play(board) returns the next scripted move."""

    def __init__(self, *ucis: str) -> None:
        self.moves = list(ucis)
        self.seen: list[tuple[str, int]] = []

    def play(self, board: chess.Board) -> chess.Move:
        self.seen.append((board.fen(), len(board.move_stack)))
        return chess.Move.from_uci(self.moves.pop(0))


def solve(fen: str, line: list[str], engine) -> bool:
    return puzzles.evaluate_puzzle_from_board(board=chess.Board(fen), moves=line, engine=engine)


def test_puzzle_scorer_accepts_an_alternative_mate():
    assert solve(TWO_MATES, ["a5c4", "e1e8"], Scripted("e1e8"))
    assert solve(TWO_MATES, ["a5c4", "e1e8"], Scripted("a1a8"))


def test_puzzle_scorer_fails_any_other_deviation():
    assert not solve(TWO_MATES, ["a5c4", "e1e8"], Scripted("a1a7"))
    assert solve(DOUBLE_ROOKS, DOUBLE_ROOKS_LINE, Scripted("e2e8", "e1e8"))
    assert not solve(DOUBLE_ROOKS, DOUBLE_ROOKS_LINE, Scripted("e2e7"))
    assert not solve(DOUBLE_ROOKS, DOUBLE_ROOKS_LINE, Scripted("e2e8", "e1e2"))
    assert not solve(DOUBLE_ROOKS, DOUBLE_ROOKS_LINE, Scripted("e1e8"))


def test_the_scorer_starts_from_the_end_of_the_pgn_with_its_real_history():
    engine = Scripted("h5f7")
    assert puzzles.evaluate_puzzle_from_pandas_row(puzzle=SCHOLAR, engine=engine)
    fen, plies = engine.seen[0]
    assert plies == 6
    assert fen == "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def test_rating_bands_follow_the_eval_protocol():
    assert [puzzles.band_of(r) for r in (399, 999, 1000, 1499, 1500, 2499, 2500, 2867)] == [
        "<1000",
        "<1000",
        "1000-1500",
        "1000-1500",
        "1500-2000",
        "2000-2500",
        "2500+",
        "2500+",
    ]


def test_wilson_interval_matches_the_textbook_formula():
    low, high = puzzles.wilson(8, 10)
    assert low == pytest.approx(0.4902, abs=1e-4)
    assert high == pytest.approx(0.9433, abs=1e-4)
    assert puzzles.wilson(0, 0) == (0.0, 1.0)


def write_puzzle_csv(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(puzzles.REQUIRED_COLUMNS) + ["Solution", "FEN"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"Solution": "", "FEN": "", **row})
    return path


def test_the_runner_writes_a_per_puzzle_csv_and_accuracy_by_band(tmp_path):
    rows = [
        SCHOLAR,
        {"PuzzleId": "m2", "Rating": "1600", "PGN": "1. e4 e5", "Moves": "g1f3 b8c6"},
        {**SCHOLAR, "PuzzleId": "s3", "Rating": "2600"},
    ]
    source = write_puzzle_csv(tmp_path / "set.csv", rows)
    agent = agents.PolicyAgent(RandomLogitEvaluator())
    summary = puzzles.run_puzzle_set(source, agent, mode="policy", out_dir=tmp_path, limit=2, label="t")
    assert (summary["n"], summary["illegal_moves"]) == (2, 0)
    assert summary["bands"]["<1000"] == {"n": 1, "correct": 1, "accuracy": 1.0}
    with open(tmp_path / "puzzles_t_policy.csv", encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert [(r["puzzle_id"], r["band"], r["correct"]) for r in written][0] == ("s1", "<1000", "1")
    assert json.loads((tmp_path / "puzzles_t_policy.json").read_text(encoding="utf-8"))["n"] == 2


def test_a_file_without_the_scorer_columns_is_refused(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("id,fen\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Moves"):
        list(puzzles.read_puzzles(bad))


@pytest.mark.local
@pytest.mark.skipif(not DM_PUZZLES.is_file(), reason="DeepMind puzzles.csv is not downloaded")
def test_the_dm10k_file_has_the_scorer_columns_and_its_pgn_ends_at_its_fen():
    first = list(puzzles.read_puzzles(DM_PUZZLES, limit=50))
    assert len(first) == 50
    for row in first:
        board = puzzles.board_from_pgn(row["PGN"])
        assert board.fen() == row["FEN"]
