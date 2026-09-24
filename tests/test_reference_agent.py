"""DeepMindAgent: DeepMind's released ActionValueEngine play logic, L rows per decision, no fallback."""

from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from blink import uci
from blink.eval import match, nosearch
from blink.eval.books import Opening
from blink.play.agents import RandomAgent
from blink.reference import agent as dm_agent
from blink.reference import deepmind

JAX_LOGITS = Path(r"D:\blink\dm\9M-jax-logits.npz")
WEIGHTS = Path(r"D:\blink\dm\9M-params.npz")
# 1.Nf3 Nf6 2.Ng1 Ng8 3.Nf3 Nf6 4.Ng1: Black's ...Ng8 would repeat the start position a third time.
KNIGHT_DANCE = ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1")

pytestmark = pytest.mark.torch


def log_probs_for(win: float) -> np.ndarray:
    """Mass on the two bucket centres around `win`, weighted so the expected win is exactly `win`."""
    position = win * deepmind.NUM_RETURN_BUCKETS - 0.5
    low = int(np.clip(np.floor(position), 0, deepmind.NUM_RETURN_BUCKETS - 2))
    probs = np.zeros(deepmind.NUM_RETURN_BUCKETS)
    probs[low], probs[low + 1] = 1 - (position - low), position - low
    return np.log(np.maximum(probs, 1e-30)).astype(np.float32)


class WinTable:
    """A scorer whose rows have an exact expected win per move (default 0.3)."""

    def __init__(self, wins: dict[str, float] | None = None, default: float = 0.3) -> None:
        self.wins, self.default = wins or {}, default
        self.calls: list[np.ndarray] = []

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        self.calls.append(rows.copy())
        moves = (deepmind.ACTION_TO_MOVE[int(row[deepmind.SEQUENCE_LENGTH])] for row in rows)
        return np.stack([log_probs_for(self.wins.get(move, self.default)) for move in moves])


def played(*ucis: str) -> chess.Board:
    board = chess.Board()
    for move in ucis:
        board.push_uci(move)
    return board


def test_the_agent_plays_the_argmax_of_expected_win_and_reports_l_rows():
    scorer = WinTable({"g8f6": 0.9, "e7e5": 0.6})
    board = played("e2e4")
    decision = dm_agent.DeepMindAgent(scorer).choose(board)
    assert decision.move == chess.Move.from_uci("g8f6")
    assert decision.n_rows == board.legal_moves.count() == 20
    assert decision.n_calls == 1 and len(scorer.calls) == 1
    rows = scorer.calls[0]
    assert rows.shape == (20, 79)
    assert rows[:, 77].tolist() == sorted(rows[:, 77].tolist())
    assert (rows[:, :77] == deepmind.tokenize_board(board)).all()
    assert decision.win == pytest.approx(0.9)
    assert decision.rules == () and not decision.mate_now


def test_ties_go_to_the_lowest_action_index_like_np_argmax():
    board = played("d2d4")
    decision = dm_agent.DeepMindAgent(WinTable()).choose(board)
    assert decision.move == deepmind.ordered_legal_moves(board)[0]


def test_a_move_that_allows_a_threefold_claim_is_worth_one_half():
    board = played(*KNIGHT_DANCE)
    records = []
    avoid = dm_agent.DeepMindAgent(WinTable({"f6g8": 0.9, "e7e5": 0.6}), sink=records.append).choose(board)
    assert avoid.move == chess.Move.from_uci("e7e5")  # 0.9 became 0.5 < 0.6
    assert records[-1].rule == dm_agent.REPETITION_RULE
    seek = dm_agent.DeepMindAgent(WinTable({"f6g8": 0.1}, default=0.2)).choose(board)
    assert seek.move == chess.Move.from_uci("f6g8")  # 0.1 became 0.5 > 0.2: the released behaviour
    assert seek.win == 0.5


def test_the_agent_never_mutates_the_board_it_is_given():
    board = played(*KNIGHT_DANCE)
    fen, stack = board.fen(), list(board.move_stack)
    dm_agent.DeepMindAgent(WinTable({"f6g8": 0.9})).choose(board)
    assert board.fen() == fen and board.move_stack == stack


def test_the_sink_gets_one_action_value_record_per_decision():
    records = []
    board = played("e2e4", "e7e5")
    dm_agent.DeepMindAgent(WinTable(), sink=records.append).choose(board, game="g7")
    (record,) = records
    assert record.as_dict() == {
        "game": "g7",
        "ply": 2,
        "mode": "action-value",
        "n_legal": 29,
        "n_rows": 29,
        "n_calls": 1,
        "rule": "",
    }


def test_there_is_no_stockfish_fallback_when_every_move_is_winning():
    board = chess.Board("6k1/8/6K1/8/8/8/8/Q7 w - - 0 1")
    wins = {m.uci(): 0.995 for m in board.legal_moves} | {"a1a2": 0.997}  # all top-5 above 99%
    decision = dm_agent.DeepMindAgent(WinTable(wins)).choose(board)
    assert decision.move == chess.Move.from_uci("a1a2")  # plain argmax, even with the mate a1a8 on the board


def test_the_uci_info_line_counts_l_rows():
    board = played("e2e4")
    decision = dm_agent.DeepMindAgent(WinTable({"c7c5": 0.7})).choose(board)
    line = uci.info_line(decision)
    assert line.startswith("info depth 1 nodes 20 score cp ")
    assert line.endswith(" pv c7c5")


def test_a_scorer_with_the_wrong_shape_is_refused():
    with pytest.raises(ValueError, match="log-probs"):
        dm_agent.DeepMindAgent(lambda rows: np.zeros((len(rows), 5), np.float32)).choose(chess.Board())


def test_an_in_process_match_passes_the_no_search_audit(tmp_path):
    pgn = tmp_path / "dm.pgn"
    opening = Opening(1, chess.STARTING_FEN, ("e2e4", "e7e5"))
    dm = dm_agent.DeepMindAgent(WinTable(), name="DM-9M")
    summary = match.run_match(dm, RandomAgent(seed=3), [opening], 2, pgn, max_plies=24)
    assert summary["illegal_moves"] == 0 and summary["crashes"] == 0
    report = nosearch.audit([pgn], engine="dm")
    assert report["compliant"] and report["decisions"] > 0
    assert report["value_mode_full_batches"] == 0  # L rows, never L + 1


def test_the_torch_scorer_returns_float32_log_probs_on_cpu():
    config = deepmind.DeepMindConfig(embedding_dim=16, num_layers=1, num_heads=2)
    scorer = dm_agent.TorchScorer(deepmind.ActionValueTransformer(config).eval(), device="cpu")
    board = chess.Board()
    out = scorer(deepmind.sequences(board, deepmind.ordered_legal_moves(board)))
    assert isinstance(scorer.model, torch.nn.Module)
    assert out.dtype == np.float32 and out.shape == (20, 128)
    assert np.allclose(np.exp(out).sum(axis=1), 1.0, atol=1e-5)


@pytest.mark.local
@pytest.mark.skipif(not (JAX_LOGITS.is_file() and WEIGHTS.is_file()), reason="run tools/dm_convert.py first")
def test_with_the_real_weights_the_agent_plays_the_jax_argmax():
    saved = np.load(JAX_LOGITS, allow_pickle=False)
    model = deepmind.load_model(WEIGHTS, deepmind.CONFIGS["9M"], device="cpu")
    agent = dm_agent.DeepMindAgent(dm_agent.TorchScorer(model, device="cpu"))
    offsets, checked = saved["offsets"], 0
    for i in range(0, 100, 5):
        start, end = offsets[i], offsets[i + 1]
        win = deepmind.win_probabilities(saved["log_probs_params"][start:end])
        top = np.sort(win)[-2:]
        if len(win) > 1 and top[1] - top[0] < 1e-3:
            continue  # a near tie could flip on float noise
        decision = agent.choose(chess.Board(str(saved["fens"][i])))
        assert decision.move.uci() == saved["moves"][start + int(np.argmax(win))]
        checked += 1
    assert checked >= 10
