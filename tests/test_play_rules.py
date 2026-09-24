"""The closed list of rule checks (NSC-1, R1-R5) and Blink's own repetition counter."""

import chess
import numpy as np

from blink.play import rules

KNIGHT_SHUFFLE = ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1")


def board_after(ucis, fen: str = chess.STARTING_FEN) -> chess.Board:
    board = chess.Board(fen)
    for uci in ucis:
        board.push_uci(uci)
    return board


def child(board: chess.Board, uci: str) -> chess.Board:
    out = board.copy(stack=False)
    out.push_uci(uci)
    return out


def test_rules_allowed_is_exactly_r1_to_r5():
    assert sorted(rules.ALLOWED) == ["R1", "R2", "R3", "R4", "R5"]
    assert isinstance(rules.ALLOWED, frozenset)
    assert set(rules.RULES) == rules.ALLOWED
    assert rules.DRAW_DELTA == 0.10


def test_the_history_counts_a_third_occurrence_of_the_start_position():
    board = board_after(KNIGHT_SHUFFLE)
    history = rules.History.from_board(board)
    back_home = child(board, "f6g8")
    assert history.occurrences_with(back_home) == 3
    assert rules.rule_draw(back_home, history) == "threefold repetition"
    assert rules.rule_draw(child(board, "e7e5"), history) is None


def test_the_history_only_looks_back_to_the_last_irreversible_move():
    board = board_after(("e2e4", "e7e5", *KNIGHT_SHUFFLE))
    history = rules.History.from_board(board)
    assert history.occurrences_with(child(board, "f6g8")) == 3
    assert sum(history.counts.values()) == board.halfmove_clock + 1


def test_the_repetition_key_ignores_an_en_passant_square_with_no_legal_capture():
    with_ep = chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1")
    without = chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - - 0 1")
    assert rules.repetition_key(with_ep) == rules.repetition_key(without)
    legal_ep = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1")
    no_ep = chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - - 0 1")
    assert rules.repetition_key(legal_ep) != rules.repetition_key(no_ep)


def test_a_stalemate_child_is_a_rule_draw():
    board = chess.Board("7k/8/5K2/6Q1/8/8/8/8 w - - 0 1")
    history = rules.History.from_board(board)
    assert rules.rule_draw(child(board, "g5g6"), history) == "stalemate"


def test_an_insufficient_material_child_is_a_rule_draw():
    board = chess.Board("8/8/8/4k3/8/8/3r4/4K3 w - - 0 1")
    history = rules.History.from_board(board)
    assert rules.rule_draw(child(board, "e1d2"), history) == "insufficient material"


def test_the_hundredth_halfmove_child_is_a_rule_draw_unless_it_mates():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 99 80")
    history = rules.History.from_board(board)
    assert rules.rule_draw(child(board, "g1g2"), history) == "fifty-move rule"
    assert rules.rule_draw(child(board, "a1a8"), history) is None


def test_policy_mode_demotes_rule_draws_when_clearly_winning():
    order = [7, 3, 9]
    choice, fired = rules.policy_draw_choice(order, draws={7}, root_win=0.61)
    assert (choice, fired) == (3, True)
    choice, fired = rules.policy_draw_choice(order, draws={7, 3, 9}, root_win=0.9)
    assert (choice, fired) == (7, True)


def test_policy_mode_plays_the_best_rule_draw_when_clearly_losing():
    choice, fired = rules.policy_draw_choice([7, 3, 9], draws={9, 3}, root_win=0.39)
    assert (choice, fired) == (3, True)


def test_policy_mode_ignores_rule_draws_inside_the_delta_band():
    for root_win in (0.40, 0.5, 0.60):
        assert rules.policy_draw_choice([7, 3, 9], draws={7}, root_win=root_win) == (7, False)
    assert rules.policy_draw_choice([7, 3, 9], draws=set(), root_win=0.9) == (7, False)


def test_the_value_tie_break_uses_the_root_policy_logit_within_epsilon():
    values = np.array([0.70, 0.699, 0.60])
    logits = np.array([0.0, 5.0, 9.0])
    assert rules.tie_break(values, logits, epsilon=0.0) == (0, False)
    assert rules.tie_break(values, logits, epsilon=1 / 256) == (1, True)
    assert rules.tie_break(np.array([0.5, 0.5]), np.array([1.0, 2.0]), epsilon=0.0) == (1, True)


def test_the_clock_guard_fires_below_the_larger_of_3_seconds_and_10_p99():
    assert not rules.clock_guard(None, p99_s=0.05)
    assert rules.clock_guard(2.9, p99_s=0.05)
    assert not rules.clock_guard(3.1, p99_s=0.05)
    assert rules.clock_guard(4.9, p99_s=0.5)
    assert not rules.clock_guard(5.1, p99_s=0.5)
