"""In-process pieces the P8 blocks share: SPRT game pairs, pentanomials, per-player audits, SF as agent."""

import chess
import chess.pgn
import pytest

from blink.eval import books, fastchess, match, sprt
from blink.play import agents, factory
from blink.play.oracles import RandomLogitEvaluator

START = chess.STARTING_FEN
OPENINGS = [books.Opening(i + 1, START, moves) for i, moves in enumerate([("e2e4",), ("d2d4",), ("c2c4",)])]


def read_games(path):
    with open(path, encoding="utf-8") as handle:
        return [g for g in iter(lambda: chess.pgn.read_game(handle), None)]


def test_a_pair_player_plays_one_opening_with_both_colours(tmp_path):
    play = match.pair_player(
        agents.MaterialAgent(), agents.RandomAgent(), OPENINGS, tmp_path / "s.pgn", max_plies=40
    )
    first, second = play(0)
    assert first in (0.0, 0.5, 1.0) and second in (0.0, 0.5, 1.0)
    games = read_games(tmp_path / "s.pgn")
    assert [(g.headers["White"], g.headers["Black"]) for g in games] == [
        ("Material", "Random"),
        ("Random", "Material"),
    ]
    assert [g.headers["Round"] for g in games] == ["1", "2"]
    assert all(next(iter(g.mainline_moves())).uci() == "e2e4" for g in games)


def test_a_pair_player_refuses_to_run_out_of_openings(tmp_path):
    play = match.pair_player(agents.RandomAgent(), agents.RandomAgent(), OPENINGS[:1], tmp_path / "s.pgn")
    with pytest.raises(ValueError, match="openings"):
        play(1)


def test_a_match_summary_carries_the_pentanomial(tmp_path):
    summary = match.run_match(
        agents.MaterialAgent(), agents.RandomAgent(), OPENINGS, 4, tmp_path / "m.pgn", max_plies=30
    )
    assert summary["penta"] is not None and sum(summary["penta"]) == 2
    scores = [r["result"] for r in summary["records"]]
    assert len(scores) == 4


def test_an_sprt_between_two_tiny_agents_runs_from_the_pair_player(tmp_path):
    play = match.pair_player(
        agents.MaterialAgent(), agents.RandomAgent(), OPENINGS, tmp_path / "s.pgn", max_plies=40
    )
    result = sprt.run_sprt(play, sprt.SprtConfig(cap_games=6))
    assert result.games == 6 and result.capped
    assert len(read_games(tmp_path / "s.pgn")) == 6


def test_every_searchless_player_is_audited_by_its_own_name(tmp_path):
    evaluator = RandomLogitEvaluator(0)
    blink = factory.make_agent("value", evaluator)
    dm_like = factory.make_agent("policy", evaluator)
    blink = type(blink)(blink.evaluator, name="Blink-value-x")
    dm_like = type(dm_like)(dm_like.evaluator, name="DM-9M")
    match.run_match(blink, dm_like, OPENINGS, 2, tmp_path / "b.pgn", max_plies=10)
    audits = match.audit_players([tmp_path / "b.pgn"], ["Blink-value-x", "DM-9M"])
    assert set(audits["Blink-value-x"]["players"]) == {"Blink-value-x"}
    assert set(audits["DM-9M"]["players"]) == {"DM-9M"}
    assert min(a["decisions"] for a in audits.values()) > 0
    assert all(a["compliant"] for a in audits.values())


def test_an_engine_agent_needs_exactly_one_limit():
    with pytest.raises(ValueError, match="exactly one"):
        match.EngineAgent("SF", "sf.exe")
    with pytest.raises(ValueError, match="exactly one"):
        match.EngineAgent("SF", "sf.exe", movetime=0.1, nodes=100)


def test_stockfish_agents_are_named_like_the_fastchess_engines():
    exe = fastchess.stockfish_exe()
    anchor = match.stockfish_agent(exe, elo=1500)
    assert anchor.name == "SF1500" and anchor.options["UCI_Elo"] == 1500 and anchor.limit.time == 0.1
    nodes = match.stockfish_agent(exe, nodes=256)
    assert nodes.name == "SF19-n256" and nodes.limit.nodes == 256 and "UCI_Elo" not in nodes.options


SF = fastchess.stockfish_exe()


@pytest.mark.local
@pytest.mark.skipif(not SF.is_file(), reason="Stockfish 19 is not installed here")
def test_stockfish_in_process_mates_in_one_and_quits(tmp_path):
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    with match.stockfish_agent(SF, nodes=2000) as engine:
        assert engine.choose(board, game="g1").move == chess.Move.from_uci("d1d8")
    assert engine._engine is None


def test_two_matches_never_append_to_one_pgn(tmp_path):
    first = match.unique_path(tmp_path / "a.pgn")
    first.write_text("x", encoding="utf-8")
    second = match.unique_path(tmp_path / "a.pgn")
    second.write_text("y", encoding="utf-8")
    assert (first.name, second.name, match.unique_path(tmp_path / "a.pgn").name) == (
        "a.pgn",
        "a-2.pgn",
        "a-3.pgn",
    )
