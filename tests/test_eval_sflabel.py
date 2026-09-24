"""Stockfish labels at a node budget, cached by (fen, move) so an interrupted run never searches twice."""

import chess
import pytest

from blink.eval import fastchess, sflabel

START = chess.STARTING_FEN


class FakeStockfish:
    def __init__(self) -> None:
        self.calls = []

    def __call__(self, board, nodes, move):
        self.calls.append((board.fen(), nodes, None if move is None else move.uci()))
        return sflabel.SfLabel(cp=30 if move is None else -20, mate=None, depth=12, best="e2e4")


def test_a_label_is_searched_once_then_read_from_the_cache(tmp_path):
    fake = FakeStockfish()
    labeler = sflabel.SfLabeler(1000, cache_path=tmp_path / "c.jsonl", analyse=fake)
    first = labeler.label(START)
    assert labeler.label(START) == first
    assert fake.calls == [(START, 1000, None)] and labeler.searched == 1


def test_the_cache_survives_a_new_process(tmp_path):
    sflabel.SfLabeler(1000, cache_path=tmp_path / "c.jsonl", analyse=FakeStockfish()).label(START, "e2e4")
    fake = FakeStockfish()
    again = sflabel.SfLabeler(1000, cache_path=tmp_path / "c.jsonl", analyse=fake)
    assert again.label(START, chess.Move.from_uci("e2e4")).cp == -20
    assert fake.calls == [] and len(again.cache) == 1


def test_a_move_label_restricts_the_search_to_that_move(tmp_path):
    fake = FakeStockfish()
    sflabel.SfLabeler(500, cache_path=tmp_path / "c.jsonl", analyse=fake).label(START, "g1f3")
    assert fake.calls == [(START, 500, "g1f3")]


def test_an_illegal_move_is_refused_before_any_search(tmp_path):
    fake = FakeStockfish()
    with pytest.raises(ValueError, match="not legal"):
        sflabel.SfLabeler(500, cache_path=tmp_path / "c.jsonl", analyse=fake).label(START, "e2e5")
    assert fake.calls == []


def test_label_win_uses_the_lichess_mapping_and_mates():
    assert sflabel.SfLabel(0, None, 1, None).win == pytest.approx(0.5)
    assert sflabel.SfLabel(None, 3, 1, None).win > 0.97
    assert sflabel.SfLabel(None, -3, 1, None).win < 0.03
    assert (
        sflabel.SfLabel(None, -2, 1, None).pawns == -100.0
        and sflabel.SfLabel(250, None, 1, None).pawns == 2.5
    )


def test_without_an_executable_a_cache_miss_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Stockfish"):
        sflabel.SfLabeler(500, cache_path=tmp_path / "c.jsonl").label(START)


SF = fastchess.stockfish_exe()


@pytest.mark.local
@pytest.mark.skipif(not SF.is_file(), reason="Stockfish 19 is not installed here")
def test_real_stockfish_sees_a_mate_in_one_for_the_side_to_move(tmp_path):
    fen = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
    with sflabel.SfLabeler(20_000, exe=SF, cache_path=tmp_path / "c.jsonl") as labeler:
        label = labeler.label(fen)
        restricted = labeler.label(fen, "g1f1")
    assert (label.mate, label.best) == (1, "d1d8")
    assert restricted.mate is None or restricted.mate != 1


@pytest.mark.local
@pytest.mark.skipif(not SF.is_file(), reason="Stockfish 19 is not installed here")
def test_label_many_spreads_the_misses_over_processes_and_caches_them(tmp_path):
    board, fens = chess.Board(), []
    for move in list(board.legal_moves)[:20]:
        board.push(move)
        fens.append(board.fen())
        board.pop()
    labeler = sflabel.SfLabeler(2_000, exe=SF, cache_path=tmp_path / "c.jsonl", procs=2)
    labels = labeler.label_many([(fen, None) for fen in fens])
    assert len(labels) == 20 and labeler.searched == 20 and len(labeler.cache) == 20
    again = sflabel.SfLabeler(2_000, cache_path=tmp_path / "c.jsonl", procs=2)
    assert again.label_many([(fens[0], None)]) == [labels[0]] and again.searched == 0
