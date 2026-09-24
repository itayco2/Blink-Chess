"""E2, the static metrics: every formula on hand-checkable inputs, then one tiny end-to-end run."""

import math

import chess
import numpy as np
import pytest

from blink.board import encode, moves, value
from blink.data import games10k, valprobe
from blink.eval import sflabel, static
from blink.play import factory
from blink.play.evaluator import Evaluation
from blink.play.oracles import MaterialEvaluator, RandomLogitEvaluator
from blink.report.results_schema import DiagnosticsRow

MATE_IN_TWO = "6k1/5ppp/8/8/8/8/5PPP/1R1R2K1 w - - 0 1"


def test_kendall_tau_is_computed_on_scores_not_argsorts():
    """PF37: DeepMind's formula ranks argsorts; on these scores it says -1.0 where the truth is +0.33."""
    predicted, stockfish = [0.1, 0.3, 0.2], [0.2, 0.3, 0.1]
    assert static.kendall_tau_b(predicted, stockfish) == pytest.approx(1 / 3)
    assert static.kendall_tau_b(np.argsort(predicted), np.argsort(stockfish)) == pytest.approx(-1.0)


def test_kendall_tau_b_handles_ties_and_constant_vectors():
    assert static.kendall_tau_b([1, 2, 2], [1, 2, 3]) == pytest.approx(2 / math.sqrt(2 * 3))
    assert math.isnan(static.kendall_tau_b([1, 1], [1, 2]))


def test_brier_and_ece_on_known_numbers():
    pred, label = np.array([0.9, 0.1, 0.5, 0.5]), np.array([1.0, 0.0, 0.3, 0.7])
    assert static.brier(pred, label) == pytest.approx((0.01 + 0.01 + 0.04 + 0.04) / 4)
    assert static.ece(np.array([0.05, 0.05, 0.95]), np.array([0.25, 0.05, 0.95])) == pytest.approx(0.2 / 3)


def test_a_temperature_above_one_is_fitted_for_an_overconfident_head():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0.2, 0.8, 4000)
    labels = np.clip(truth + rng.normal(0, 0.08, 4000), 0, 0.999)
    centers = value.BIN_CENTERS
    sharp = np.exp(-((centers[None] - truth[:, None]) ** 2) / (2 * 0.01**2))
    sharp /= sharp.sum(axis=1, keepdims=True)
    assert static.fit_temperature(sharp, labels) > 1.5
    flat = np.exp(-((centers[None] - truth[:, None]) ** 2) / (2 * 0.3**2))
    flat /= flat.sum(axis=1, keepdims=True)
    assert static.fit_temperature(flat, labels) < 1.0


def test_divider_phases_on_three_positions():
    assert static.phase(chess.Board()) == "opening"
    assert static.phase(chess.Board("8/8/4k3/8/8/8/3RK3/8 w - - 0 1")) == "endgame"
    middlegame = chess.Board("r4rk1/pp3ppp/2n1b3/3p4/3P4/2N1BN2/PP3PPP/R4RK1 w - - 0 15")
    assert static.phase(middlegame) == "middlegame"
    developed = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2QK2R w KQ - 0 9")
    assert static.phase(developed) == "opening"  # 13 majors and minors, full back ranks, mixedness 95
    assert static.mixedness(chess.Board()) == 0


def record(fen, best, cp=None, mate=None, alts=()):
    board = chess.Board(fen)
    rec = games10k.to_record(board, chess.Move.from_uci(best), cp, mate, 20)
    for slot, (uci, alt_cp) in enumerate(alts):
        rec["alt_move"][slot] = moves.encode_move(board, chess.Move.from_uci(uci))
        rec["alt_cp"][slot] = alt_cp
    return rec


class Favours:
    """Policy logits that rank the given vocabulary indices first; a flat, half-way value head."""

    def __init__(self, ranking):
        self.ranking = ranking

    def evaluate(self, codes):
        logits = np.zeros((len(codes), moves.NUM_MOVES), dtype=np.float32)
        for rank, index in enumerate(self.ranking):
            logits[:, index] = 10.0 - rank
        probs = np.full((len(codes), value.NUM_BINS), 1 / value.NUM_BINS, dtype=np.float32)
        return Evaluation(logits, probs)


def test_top_k_counts_the_rank_of_stockfishs_move_among_legal_moves():
    board = chess.Board()
    rec = record(chess.STARTING_FEN, "e2e4", cp=30)
    root = static.root_from_record(rec, board)
    ranking = [moves.encode_move(board, chess.Move.from_uci(u)) for u in ("d2d4", "g1f3", "e2e4")]
    (out,) = static.evaluate_roots(Favours(ranking), [root])
    assert out.label_rank == 2 and out.policy_pick == ranking[0]
    summary = static.summarize([out])
    assert (summary["top1"]["value"], summary["top3"]["value"]) == (0.0, 1.0)
    assert out.legal_mass == pytest.approx(1.0, abs=1e-3) or 0 < out.legal_mass <= 1


def test_vaa_counts_an_alternative_with_the_identical_score_as_correct():
    board = chess.Board("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1")
    rec = record(board.fen(), "e1f1", cp=900, alts=[("e4d5", 900), ("e1f2", -300)])
    root = static.root_from_record(rec, board)
    assert moves.encode_move(board, chess.Move.from_uci("e4d5")) in root.ties
    (out,) = static.evaluate_roots(MaterialEvaluator(), [root], value_limit=None)
    assert out.value_pick == moves.encode_move(board, chess.Move.from_uci("e4d5"))
    assert out.vaa is True and out.near_best_value is True


def test_value_mode_plays_a_checkmating_child_first():
    board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    root = static.root_from_record(record(board.fen(), "d1d8", mate=1), board)
    (out,) = static.evaluate_roots(RandomLogitEvaluator(0), [root], value_limit=None)
    assert out.value_pick == moves.encode_move(board, chess.Move.from_uci("d1d8")) and out.vaa


def test_regret_searches_only_moves_that_differ_from_stockfishs(tmp_path):
    board = chess.Board()
    root = static.root_from_record(record(board.fen(), "e2e4", cp=0), board)
    searched = []

    def analyse(b, nodes, move):
        searched.append(move.uci())
        return sflabel.SfLabel(-100, None, 20, None)

    labeler = sflabel.SfLabeler(1, cache_path=tmp_path / "c.jsonl", analyse=analyse)
    e2e4 = moves.encode_move(board, chess.Move.from_uci("e2e4"))
    a2a3 = moves.encode_move(board, chess.Move.from_uci("a2a3"))
    result = static.regret([root, root], [e2e4, a2a3], labeler)
    assert searched == ["a2a3"]
    assert result["value"] == pytest.approx((0.5 - value.win_probability(cp=-100)) / 2)


def test_the_mateset_rates_shortest_and_preserving_mates(tmp_path):
    rec = record(MATE_IN_TWO, "d1d8", mate=2)
    arrays = {**valprobe.probe_arrays(np.array([rec])), "mate_in": np.array([2], dtype=np.int8)}
    labeler = sflabel.SfLabeler(
        1, cache_path=tmp_path / "c.jsonl", analyse=lambda b, n, m: sflabel.SfLabel(None, 3, 20, None)
    )
    rates = static.mate_rates(MaterialEvaluator(), arrays, labeler)
    assert set(rates) == {"policy", "value"}
    for mode in rates.values():
        assert mode["shortest"]["n"] == 1
        assert mode["preserving"]["value"] == 1.0  # the fake SF says every pick still mates


def test_the_puzzle_rating_equivalent_recovers_a_known_rating():
    rng = np.random.default_rng(0)
    ratings = rng.uniform(600, 2600, 3000).round()
    solved = rng.uniform(size=3000) < 1 / (1 + 10 ** ((ratings - 1500) / 400))
    rows = [{"rating": int(r), "correct": int(s)} for r, s in zip(ratings, solved, strict=True)]
    result = static.puzzle_rating_equivalent(rows, samples=50)
    assert abs(result["value"] - 1500) < 60
    low, high = result["ci95"]
    assert low < result["value"] < high
    assert static.puzzle_rating_mle(np.array([800.0, 900.0]), np.array([True, True])) is None


def test_band_accuracy_uses_wilson_and_lichess_bands_are_200_points_wide():
    rows = [{"rating": 950, "correct": 1}, {"rating": 990, "correct": 0}, {"rating": 1210, "correct": 1}]
    bands = static.band_accuracy(rows, static.lichess_band)
    assert bands["800-1000"]["value"] == 0.5 and bands["800-1000"]["n"] == 2
    assert bands["1200-1400"]["wilson95"][1] == 1.0
    assert static.lichess_band(400) == "400-600" and static.lichess_band(2799) == "2600-2800"


def test_lichess_band_puzzles_are_scored_from_their_fen_and_moves():
    rows = [
        {
            "PuzzleId": "p1",
            "FEN": "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1",
            "Moves": "g8h8 d1d8",
            "Rating": "600",
        },
    ]
    agent = factory.make_agent("value", MaterialEvaluator())
    (row,) = static.score_lichess_puzzles(rows, agent)
    assert row == {"puzzle_id": "p1", "rating": 600, "correct": 1, "illegal": 0}


def write_roots(path, fens_and_best):
    records = np.array([record(fen, best, cp=20) for fen, best in fens_and_best])
    records.tofile(path)
    return records


def test_e2_runs_end_to_end_and_fills_two_diagnostics_rows(tmp_path):
    positions = [(chess.STARTING_FEN, "e2e4"), ("4k3/8/8/3q4/4P3/8/8/4K3 w - - 0 1", "e4d5")]
    write_roots(tmp_path / "test_iid.bin", positions)
    write_roots(tmp_path / "val.bin", positions)
    write_roots(tmp_path / "test_grouped.bin", positions[:1])
    evaluator = RandomLogitEvaluator(1)
    agents = {mode: factory.make_agent(mode, evaluator) for mode in factory.MODES}
    inputs = static.StaticInputs(
        test_iid=tmp_path / "test_iid.bin",
        val=tmp_path / "val.bin",
        test_grouped=tmp_path / "test_grouped.bin",
    )
    e2 = static.run_e2(evaluator, agents, inputs, static.StaticLimits(value_roots=2))
    assert e2["test_iid"]["roots"] == 2 and e2["test_iid"]["value_roots"] == 2
    assert e2["temperature"] > 0 and set(e2["grouped_gap"]) == {"top1", "vaa"}
    rows = static.diagnostics_rows(e2, "Blink-test")
    assert [r.mode for r in rows] == ["policy", "value"] and all(isinstance(r, DiagnosticsRow) for r in rows)
    assert rows[0].top1 is not None and rows[0].vaa is None
    assert rows[1].vaa is not None and rows[1].top1 is None


def test_decoded_record_boards_encode_back_to_the_record():
    rec = record("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8", cp=0)
    root = static.root_from_record(rec)
    assert (encode.encode_board(root.board) == root.codes).all()
