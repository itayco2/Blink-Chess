"""E8 conversion, the endgame screen and E2b, the epsilon rule (plan P8)."""

import json
import os

import chess
import pytest

from blink.eval import conversion, endgames, sflabel
from blink.play import agents
from blink.play.oracles import MaterialEvaluator, RandomLogitEvaluator

MATE_IN_ONE_WHITE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"
MATE_IN_ONE_BLACK_TO_BE_MATED = "3r2k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1"


def endgame(fen, winner="white", line=1):
    return endgames.Endgame(line, fen, winner, 6.0, 6.0)


def fake_labeler(tmp_path, name, scores):
    """A labeler whose Stockfish returns the given pawns (side to move) for each FEN."""

    def analyse(board, nodes, move):
        pawns = scores[board.fen()]
        return sflabel.SfLabel(int(pawns * 100), None, 20, None)

    return sflabel.SfLabeler(1, cache_path=tmp_path / f"{name}.jsonl", analyse=analyse)


# ------------------------------------------------------------------------------ the screen


def test_the_screen_keeps_positions_at_plus_5_on_both_searches_for_the_same_side(tmp_path):
    fens = [
        chess.Board(f).fen() for f in (MATE_IN_ONE_WHITE, MATE_IN_ONE_BLACK_TO_BE_MATED, chess.STARTING_FEN)
    ]
    first = fake_labeler(tmp_path, "s", {fens[0]: 6.0, fens[1]: -7.0, fens[2]: 0.3})
    second = fake_labeler(tmp_path, "c", {fens[0]: 5.5, fens[1]: -4.0, fens[2]: 0.0})
    result = endgames.screen(iter(enumerate(fens, start=1)), first, second)
    assert [(e.line, e.winner) for e in result.kept] == [(1, "white")]
    assert result.screened == 3 and result.passed_screen == 2


def test_a_black_to_move_position_lost_for_black_is_won_for_white(tmp_path):
    fen = chess.Board(MATE_IN_ONE_BLACK_TO_BE_MATED).fen()
    first = fake_labeler(tmp_path, "s", {fen: -9.0})
    second = fake_labeler(tmp_path, "c", {fen: -8.0})
    found = endgames.screen_one(7, fen, first, second)
    assert found == endgames.Endgame(7, fen, "white", 9.0, 8.0) and found.blink_color == chess.WHITE


def test_the_screen_stops_at_want_and_splits_200_dev_then_500_final(tmp_path):
    fens = [chess.Board(MATE_IN_ONE_WHITE).fen()] * 5
    labeler = fake_labeler(tmp_path, "s", {fens[0]: 6.0})
    result = endgames.screen(iter(enumerate(fens, start=1)), labeler, labeler, want=3)
    assert len(result.kept) == 3 and result.screened == 3
    kept = tuple(endgame(fens[0], line=i) for i in range(750))
    split = endgames.ScreenResult(kept, 750, 750)
    assert (len(split.dev), len(split.final)) == (200, 500)
    assert split.final[0].line == 200


def test_the_sets_round_trip_through_their_files(tmp_path):
    kept = tuple(endgame(MATE_IN_ONE_WHITE, line=i) for i in range(1, 4))
    summary = endgames.write_sets(endgames.ScreenResult(kept, 10, 4), tmp_path)
    assert summary["kept"] == 3 and summary["dev"] == 3 and not summary["complete"]
    assert endgames.read_set(tmp_path, "dev") == list(kept)


def test_epd_lines_become_full_fens(tmp_path):
    epd = tmp_path / "e.epd"
    epd.write_text("8/8/8/4k3/8/8/4P3/4K3 w - - 0 31\n8/8/8/4k3/8/8/4P3/4K3 b - -\n", encoding="utf-8")
    assert [f for _, f in endgames.read_positions(epd)] == [
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 31",
        "8/8/8/4k3/8/8/4P3/4K3 b - - 0 1",
    ]


# ------------------------------------------------------------------------------ conversion


def test_a_conversion_counts_blinks_checkmate(tmp_path):
    result = conversion.play_conversion(
        agents.MaterialAgent(), agents.RandomAgent(), [endgame(MATE_IN_ONE_WHITE)], tmp_path / "c.pgn"
    )
    assert (result.n, result.converted) == (1, 1)
    assert result.as_dict()["pct"] == 100.0 and result.games[0].reason == "checkmate"


def test_blink_plays_the_winning_side_even_when_it_is_black(tmp_path):
    fen = "6k1/5ppp/8/8/8/8/5PPP/3r2K1 b - - 0 1"
    result = conversion.play_conversion(
        agents.MaterialAgent(name="Blink-x"),
        agents.RandomAgent(),
        [endgame(fen, "black")],
        tmp_path / "c.pgn",
    )
    assert result.games[0].result == "0-1" and result.converted == 1


def test_a_game_not_won_within_the_move_limit_is_not_converted(tmp_path):
    fen = "7k/8/8/8/8/8/8/R6K w - - 0 1"
    result = conversion.play_conversion(
        agents.RandomAgent(), agents.RandomAgent(seed=1), [endgame(fen)], tmp_path / "c.pgn", max_plies=4
    )
    assert result.converted == 0 and result.games[0].reason == "4 engine plies"


def test_rules_off_never_fires_r2_and_value_mode_scores_every_child(tmp_path):
    board = chess.Board(MATE_IN_ONE_WHITE)
    off = conversion.RulesOffAgent(RandomLogitEvaluator(3), "value")
    decision = off.choose(board)
    assert decision.rules == ("R1",) and decision.n_calls == 1
    assert decision.n_rows == board.legal_moves.count() + 1
    policy = conversion.RulesOffAgent(RandomLogitEvaluator(3), "policy").choose(board)
    assert policy.rules == ("R1",) and policy.n_rows == 1


def test_rules_off_value_mode_still_wins_material_with_a_material_oracle():
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    decision = conversion.RulesOffAgent(MaterialEvaluator(), "value").choose(board)
    assert decision.move == chess.Move.from_uci("e4d5")


# ------------------------------------------------------------------------------ the epsilon rule


def converted(rate, n=10):
    games = tuple(conversion.ConversionGame(i, "white", "*", "x", 1, i < rate * n) for i in range(n))
    return conversion.ConversionResult("Blink-value", games)


def test_the_best_conversion_wins_and_a_tie_goes_to_the_smallest_epsilon():
    assert conversion.select_epsilon({0.0: 0.5, 1 / 256: 0.7, 1 / 128: 0.6}) == 1 / 256
    assert conversion.select_epsilon({0.0: 0.7, 1 / 256: 0.7, 1 / 128: 0.6}) == 0.0
    assert conversion.select_epsilon({0.0: 0.5, 1 / 256: 0.7, 1 / 128: 0.7}) == 1 / 256


def test_the_no_regression_check_is_score_at_least_half_minus_one_se():
    passed = conversion.no_regression([0.5] * 10 + [1.0, 0.0])
    assert passed["score"] == 0.5 and passed["passed"]
    near = conversion.no_regression([0.0] * 6 + [1.0] * 4)
    assert near["score"] == pytest.approx(0.4)
    assert near["se"] == pytest.approx((0.24 / 10) ** 0.5)
    assert near["passed"]  # 0.40 >= 0.50 - 0.155
    failed = conversion.no_regression([0.0] * 8 + [1.0] * 2)
    assert failed["threshold"] == pytest.approx(0.5 - (0.16 / 10) ** 0.5)
    assert not failed["passed"]


def test_the_epsilon_rule_writes_results_epsilon_json_and_never_edits_eval_md(tmp_path):
    eval_md = tmp_path / "EVAL.md"
    eval_md.write_text("# EVAL\nthe frozen epsilon rule\n", encoding="utf-8")
    os.utime(eval_md, (1_000_000_000, 1_000_000_000))
    before = (eval_md.read_bytes(), eval_md.stat().st_mtime)
    rates = {0.0: 0.5, 1 / 256: 0.8, 1 / 128: 0.6}
    checked = []

    def check(eps):
        checked.append(eps)
        return [0.5] * 1000

    decision = conversion.run_epsilon_selection(
        lambda eps: converted(rates[eps]), check, tmp_path / "results"
    )
    written = json.loads((tmp_path / "results" / "epsilon.json").read_text(encoding="utf-8"))
    assert written == decision
    assert written["epsilon"] == 1 / 256 and written["no_regression"]["passed"]
    assert checked == [1 / 256]
    assert (eval_md.read_bytes(), eval_md.stat().st_mtime) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["EVAL.md", "results"]


def test_a_winner_that_regresses_leaves_epsilon_at_zero(tmp_path):
    rates = {0.0: 0.5, 1 / 256: 0.9, 1 / 128: 0.6}
    decision = conversion.run_epsilon_selection(
        lambda eps: converted(rates[eps]), lambda eps: [0.0] * 600 + [1.0] * 400, tmp_path
    )
    assert decision["epsilon"] == 0.0 and decision["winner_by_conversion"] == 1 / 256
    assert "failed" in decision["reason"]


def test_when_zero_converts_best_no_check_is_played(tmp_path):
    decision = conversion.run_epsilon_selection(
        lambda eps: converted(0.9 if eps == 0 else 0.1), lambda eps: pytest.fail("no check"), tmp_path
    )
    assert decision["epsilon"] == 0.0 and decision["no_regression"] is None


def test_the_batched_screen_keeps_a_position_won_for_black(tmp_path):
    black_wins = chess.Board("3r2k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1").fen()
    first = fake_labeler(tmp_path, "s", {black_wins: -8.0})
    second = fake_labeler(tmp_path, "c", {black_wins: -7.0})
    result = endgames.screen(iter([(3, black_wins)]), first, second)
    assert [(e.line, e.winner) for e in result.kept] == [(3, "black")]


def test_an_empty_conversion_has_no_percentage():
    empty = conversion.ConversionResult("Blink-value", ())
    assert empty.as_dict()["pct"] is None and empty.as_dict()["wilson95"] is None
