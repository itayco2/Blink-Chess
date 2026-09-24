"""EvalBudget: the no-search counter that wraps every network call of one decision (NSC-1, N1-N2)."""

import chess
import numpy as np
import pytest

from blink.board import encode
from blink.play.budget import EvalBudget, NoSearchViolation
from blink.play.oracles import RandomLogitEvaluator

START = chess.Board()


def codes_of(*boards: chess.Board) -> np.ndarray:
    return np.stack([encode.encode_board(board) for board in boards])


def children(board: chess.Board) -> list[chess.Board]:
    out = []
    for move in board.legal_moves:
        child = board.copy(stack=False)
        child.push(move)
        out.append(child)
    return out


def budget_for(board: chess.Board, sink=None) -> EvalBudget:
    return EvalBudget(RandomLogitEvaluator(), board, mode="value", game="g1", sink=sink)


def test_the_root_plus_every_child_once_is_one_allowed_call():
    budget = budget_for(START)
    evaluation = budget.evaluate(codes_of(START, *children(START)))
    assert evaluation.policy_logits.shape == (21, 1880)
    assert (budget.n_calls, budget.n_rows) == (1, 21)


def test_a_grandchild_evaluation_raises():
    grandchild = START.copy()
    grandchild.push_uci("e2e4")
    grandchild.push_uci("e7e5")
    with pytest.raises(NoSearchViolation, match="outside"):
        budget_for(START).evaluate(codes_of(START, grandchild))


def test_a_second_call_in_one_decision_raises():
    budget = budget_for(START)
    budget.evaluate(codes_of(START))
    with pytest.raises(NoSearchViolation, match="second"):
        budget.evaluate(codes_of(START))


def test_a_repeated_row_raises():
    with pytest.raises(NoSearchViolation, match="repeated"):
        budget_for(START).evaluate(codes_of(START, START))


def test_more_than_l_plus_one_rows_raises():
    rows = codes_of(START, *children(START), START)
    with pytest.raises(NoSearchViolation, match="L\\+1"):
        budget_for(START).evaluate(rows)


def test_a_rejected_call_never_reaches_the_network():
    calls = []

    class Spy(RandomLogitEvaluator):
        def evaluate(self, codes):
            calls.append(len(codes))
            return super().evaluate(codes)

    budget = EvalBudget(Spy(), START, mode="policy")
    with pytest.raises(NoSearchViolation):
        budget.evaluate(codes_of(START, START))
    assert calls == []


def test_a_decision_without_a_call_must_be_a_mate_now():
    with pytest.raises(NoSearchViolation, match="R2"):
        budget_for(START).finish(("R1",))
    budget = budget_for(START)
    budget.evaluate(codes_of(START))
    with pytest.raises(NoSearchViolation, match="R2"):
        budget.finish(("R1", "R2"))


def test_a_rule_outside_the_closed_list_raises():
    budget = budget_for(START)
    budget.evaluate(codes_of(START))
    with pytest.raises(NoSearchViolation, match="R6"):
        budget.finish(("R1", "R6"))


def test_each_decision_logs_game_ply_mode_legal_rows_calls_and_rule():
    board = chess.Board()
    board.push_uci("d2d4")
    logged = []
    budget = budget_for(board, sink=logged.append)
    budget.evaluate(codes_of(board, *children(board)))
    record = budget.finish(("R1", "R4"))
    assert logged == [record]
    assert record.as_dict() == {
        "game": "g1",
        "ply": 1,
        "mode": "value",
        "n_legal": 20,
        "n_rows": 21,
        "n_calls": 1,
        "rule": "R1,R4",
    }
