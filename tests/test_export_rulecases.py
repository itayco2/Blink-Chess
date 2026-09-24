"""site/tests/rules.json: what NSC-1's R2 and R3 say about every child, for the page's rules.js (P1, P10).

python-chess is the reference; site/tests/rules.test.mjs checks the browser port against this file.
"""

import chess
import pytest


def _board(fen: str, moves: str = "") -> chess.Board:
    board = chess.Board(fen)
    for uci in moves.split():
        board.push_uci(uci)
    return board


def _child(board: chess.Board, uci: str) -> chess.Board:
    child = board.copy()
    child.push_uci(uci)
    return child


def _draw(board: chess.Board, uci: str) -> str | None:
    from blink.export import rulecases

    return rulecases.rule_draw(_child(board, uci), rulecases.history_counts(board))


def test_the_tracked_rules_json_is_current(repo_root):
    from blink.export import rulecases

    tracked = repo_root / "site" / "tests" / "rules.json"
    assert tracked.read_text(encoding="utf-8") == rulecases.render(rulecases.build())


def test_rule_cases_cover_mate_now_every_draw_reason_and_both_colours():
    from blink.export import rulecases

    cases = rulecases.build()["cases"]
    reasons = {reason for case in cases for _, _, reason in case["draws"]}
    assert reasons == {"stalemate", "insufficient material", "fifty-move rule", "threefold repetition"}
    assert any(len(case["mates"]) >= 2 for case in cases), "no case has two mates (lowest index wins)"
    assert any(case["turn"] == "b" and case["mates"] for case in cases)
    assert any(case["turn"] == "b" and case["draws"] for case in cases)
    assert any(case["mates"] and case["draws"] for case in cases), "no case has a mate beside a draw"
    assert any(not case["mates"] and not case["draws"] for case in cases), "no case where no rule applies"


def test_a_mate_is_never_a_rule_draw_even_on_the_hundredth_halfmove():
    board = _board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80")
    assert _child(board, "a1a8").is_checkmate()
    assert _draw(board, "a1a8") is None
    assert _draw(board, "a1a7") == "fifty-move rule"


def test_the_third_occurrence_of_a_position_is_a_threefold_draw():
    board = _board(chess.STARTING_FEN, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1")
    assert _draw(board, "f6g8") == "threefold repetition"
    assert _draw(board, "b8c6") is None


def test_repetition_ignores_an_en_passant_square_nobody_can_capture_on():
    board = _board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", "e2e4 e8e7 e1d1 e7e8 d1e1 e8e7 e1d1 e7e8")
    assert _draw(board, "d1e1") == "threefold repetition"


def test_repetition_counts_a_capturable_en_passant_square_as_a_different_position():
    board = _board("4k3/8/8/8/5p2/8/4P3/4K3 w - - 0 1", "e2e4 e8e7 e1d1 e7e8 d1e1 e8e7 e1d1 e7e8")
    assert _draw(board, "d1e1") is None


def test_castling_rights_make_otherwise_equal_positions_different():
    board = _board("r3k3/8/8/8/8/8/8/4K2R w Kq - 0 1", "h1h2 a8a7 h2h1 a7a8 h1h2 a8a7 h2h1")
    assert _draw(board, "a7a8") is None


def test_rule_case_entries_list_moves_by_vocabulary_index_with_their_uci():
    from blink.board import moves
    from blink.export import rulecases

    for case in rulecases.build()["cases"]:
        board = _board(case["fen"], " ".join(case["moves"]))
        for index, uci, *_ in case["mates"] + case["draws"]:
            assert moves.encode_move(board, chess.Move.from_uci(uci)) == index, (case["name"], uci)
        assert [index for index, *_ in case["mates"]] == sorted(index for index, *_ in case["mates"])


def test_rule_cases_agree_with_the_play_areas_rules_module():
    rules = pytest.importorskip("blink.play.rules", reason="blink/play/rules.py lands with the play area")
    from blink.export import rulecases

    for case in rulecases.build()["cases"]:
        board = _board(case["fen"], " ".join(case["moves"]))
        history = rules.History.from_board(board)
        want = {uci: reason for _, uci, reason in case["draws"]}
        for move in board.legal_moves:
            assert rules.rule_draw(_child(board, move.uci()), history) == want.get(move.uci()), case["name"]
