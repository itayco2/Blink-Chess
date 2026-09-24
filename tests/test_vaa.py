import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_model_config  # noqa: E402

from blink.board import encode, moves  # noqa: E402
from blink.board import value as board_value  # noqa: E402
from blink.data.record import NO_MOVE, ROOT_DTYPE  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train import vaa  # noqa: E402
from blink.train.telemetry import board_from_codes  # noqa: E402

pytestmark = pytest.mark.torch


class CodeValueNet(torch.nn.Module):
    """A stand-in network: a position's win probability for its side to move is (square 0's code) / 16."""

    def forward(self, tokens):
        w = tokens[:, 0].float() / 16
        bins = (w * 128).long().clamp(0, 127)
        value = torch.nn.functional.one_hot(bins, 128).float() * 60.0
        return torch.zeros(len(tokens), moves.NUM_MOVES), value


def _board_with_code(code: int) -> np.ndarray:
    codes = np.zeros(64, dtype=np.uint8)
    codes[0] = code
    return encode.pack(codes)


def _probe(children_per_root: list[list[tuple[int, bool, int]]]) -> vaa.Probe:
    """Each child is (square-0 code, is_best, terminal)."""
    offsets = np.cumsum([0] + [len(c) for c in children_per_root]).astype(np.int64)
    flat = [child for children in children_per_root for child in children]
    n = len(children_per_root)
    return vaa.Probe(
        root_board=np.zeros((n, 32), dtype=np.uint8),
        root_best=np.zeros(n, dtype=np.uint16),
        child_offset=offsets,
        child_board=np.stack([_board_with_code(code) for code, _, _ in flat]),
        child_move=np.arange(len(flat), dtype=np.uint16),
        child_is_best=np.array([best for _, best, _ in flat]),
        child_terminal=np.array([terminal for _, _, terminal in flat], dtype=np.int8),
    )


def _score(probe: vaa.Probe) -> dict:
    return vaa.evaluate_vaa(CodeValueNet(), probe, torch.device("cpu"))


def test_value_mode_picks_the_child_worst_for_the_opponent():
    right = _probe([[(12, False, 0), (2, True, 0), (8, False, 0)]])
    wrong = _probe([[(12, True, 0), (2, False, 0), (8, False, 0)]])
    assert _score(right) == {"vaa": 1.0, "n": 1}
    assert _score(wrong)["vaa"] == 0.0


def test_a_tie_child_counts_as_correct():
    probe = _probe([[(2, True, 0), (9, False, 0)], [(3, True, 0), (1, True, 0), (10, False, 0)]])
    assert _score(probe) == {"vaa": 1.0, "n": 2}


def test_a_checkmating_child_is_taken_and_a_rule_draw_is_worth_half():
    mate = _probe([[(1, False, 0), (15, True, 1)]])
    assert _score(mate)["vaa"] == 1.0
    draw = _probe([[(12, False, 0), (14, True, 2), (11, False, 0)]])  # every real move loses
    assert _score(draw)["vaa"] == 1.0
    no_draw = _probe([[(4, True, 0), (14, False, 2)]])  # a winning move beats the draw
    assert _score(no_draw)["vaa"] == 1.0


def test_a_subset_keeps_the_first_roots_and_renumbers_their_children():
    probe = _probe([[(2, True, 0)], [(3, False, 0), (1, True, 0)], [(5, True, 0)]])
    sub = probe.subset(2)
    assert sub.n_roots == 2 and sub.child_offset.tolist() == [0, 1, 3]
    assert len(sub.child_board) == 3 and _score(sub) == {"vaa": 1.0, "n": 2}
    assert probe.subset(10) is probe


def test_a_probe_round_trips_through_npz_and_a_missing_array_is_named(tmp_path):
    probe = _probe([[(2, True, 0), (9, False, 2)]])
    path = tmp_path / "valprobe.npz"
    vaa.save_probe(path, probe)
    loaded = vaa.load_probe(path)
    assert np.array_equal(loaded.child_terminal, probe.child_terminal)
    arrays = dict(np.load(path))
    del arrays["child_is_best"]
    np.savez(tmp_path / "broken.npz", **arrays)
    with pytest.raises(ValueError, match="child_is_best"):
        vaa.load_probe(tmp_path / "broken.npz")


def _root(board: chess.Board, best: chess.Move, cp: int, mate: int, alts=()) -> np.ndarray:
    record = np.zeros(1, dtype=ROOT_DTYPE)
    record["board"] = encode.pack(encode.encode_board(board))
    record["move"] = moves.encode_move(board, best)
    record["cp"], record["mate"] = cp, mate
    record["alt_move"] = NO_MOVE
    for i, (move, alt_cp, alt_mate) in enumerate(alts):
        record["alt_move"][0, i] = moves.encode_move(board, move)
        record["alt_cp"][0, i], record["alt_mate"][0, i] = alt_cp, alt_mate
    return record


def test_probe_from_roots_lists_every_legal_child_with_the_best_and_ties_marked():
    records = fixture_records()[:20]
    probe = vaa.probe_from_roots(records)
    assert probe.n_roots == 20
    for i, codes in enumerate(encode.unpack(records["board"])):
        board = board_from_codes(codes)
        lo, hi = probe.child_offset[i], probe.child_offset[i + 1]
        assert hi - lo == board.legal_moves.count()
        assert sorted(probe.child_move[lo:hi].tolist()) == sorted(np.flatnonzero(moves.legal_mask(board)))
        best = probe.child_move[lo:hi][probe.child_is_best[lo:hi]]
        assert records["move"][i] in best

    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/5NP1/PPPPPP1P/RNBQKB1R b KQkq - 1 2")
    tie = chess.Move.from_uci("b8c6")
    record = _root(
        board, chess.Move.from_uci("g8f6"), 20, 0, alts=[(tie, 20, 0), (chess.Move.from_uci("a7a6"), 5, 0)]
    )
    probe = vaa.probe_from_roots(record)
    marked = {moves.decode_move(board, int(m)).uci() for m in probe.child_move[probe.child_is_best]}
    assert marked == {"g8f6", "b8c6"}


def test_a_mate_in_one_child_is_terminal_and_stalemate_is_a_rule_draw():
    fools = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    probe = vaa.probe_from_roots(_root(fools, chess.Move.from_uci("d8h4"), board_value.CP_NONE, 1))
    mates = [moves.decode_move(fools, int(m)).uci() for m in probe.child_move[probe.child_terminal == 1]]
    assert mates == ["d8h4"]
    stale = chess.Board("7k/8/6Q1/8/8/8/8/K7 w - - 0 1")
    probe = vaa.probe_from_roots(_root(stale, chess.Move.from_uci("g6f7"), 900, 0))
    drawn = {moves.decode_move(stale, int(m)).uci() for m in probe.child_move[probe.child_terminal == 2]}
    assert drawn == {"g6f7", "a1a2", "a1b1", "a1b2"}  # Qf7 or any king move stalemates the h8 king


def test_a_real_model_scores_the_probe_in_bounded_chunks():
    probe = vaa.probe_from_roots(fixture_records()[:10])
    result = vaa.evaluate_vaa(BlinkNet(tiny_model_config()), probe, torch.device("cpu"), chunk=7)
    assert result["n"] == 10 and 0.0 <= result["vaa"] <= 1.0


def test_check_steps_sit_at_5_25_30_50_and_100_percent():
    assert vaa.check_steps(10_000) == {500: "5%", 2500: "25%", 3000: "30%", 5000: "50%", 10_000: "100%"}
    assert vaa.check_steps(10_000, preview=True) == {10_000: "preview"}


def _reference() -> vaa.Reference:
    rows = (
        {"step": 1000, "samples": 1_024_000, "ema_vaa": 0.30},
        {"step": 2000, "samples": 2_048_000, "ema_vaa": 0.40},
        {"step": 3000, "samples": 3_072_000, "ema_vaa": 0.45},
        {"step": 4000, "samples": 4_096_000, "ema_vaa": 0.55},  # cooldown: never the 5% reference
    )
    return vaa.Reference(name="s6h", rows=rows, cooldown_start=3500)


def test_the_reference_is_its_stable_phase_vaa_at_equal_samples():
    ref = _reference()
    assert ref.stable_at(2_500_000) == 0.40
    assert ref.stable_at(9_000_000) == 0.45
    assert ref.stable_at(10) == 0.30
    assert ref.final() == 0.55


def test_the_5_percent_check_fails_below_the_reference_minus_2_sigma():
    ref = _reference()
    ok = vaa.apply_check("5%", 0.385, [], ref, samples=2_100_000, sigma=0.01)
    assert ok["check"] == "5%" and "vaa_check_failed" not in ok
    assert ok["check_threshold"] == pytest.approx(0.38)
    bad = vaa.apply_check("5%", 0.37, [], ref, samples=2_100_000, sigma=0.01)
    assert bad["vaa_check_failed"] is True and bad["check_failure"]["check"] == "5%"
    skipped = vaa.apply_check("5%", 0.37, [], None, samples=2_100_000, sigma=0.01)
    assert "vaa_check_failed" not in skipped and "no reference" in skipped["check_skipped"]


def test_the_25_and_50_percent_checks_compare_with_the_previous_check():
    history = [{"step": 500, "check": "5%", "ema_vaa": 0.40}, {"step": 3000, "check": "30%", "ema_vaa": 0.50}]
    assert "vaa_check_failed" not in vaa.apply_check("25%", 0.381, history[:1], None, 0, sigma=0.01)
    failed = vaa.apply_check("25%", 0.379, history[:1], None, 0, sigma=0.01)
    assert failed["check_failure"]["previous_check"] == "5%"
    assert "vaa_check_failed" in vaa.apply_check("50%", 0.47, history, None, 0, sigma=0.01)
    assert "vaa_check_failed" not in vaa.apply_check("50%", 0.49, history, None, 0, sigma=0.01)
    assert "vaa_check_failed" not in vaa.apply_check("30%", 0.01, history, None, 0, sigma=0.01)


def test_the_preview_must_beat_the_reference_final_vaa():
    ref = _reference()
    assert "vaa_check_failed" in vaa.apply_check("preview", 0.55, [], ref, 0, sigma=0.01)
    assert "vaa_check_failed" not in vaa.apply_check("preview", 0.56, [], ref, 0, sigma=0.01)


def test_a_failed_check_marks_its_row_with_vaa_check_failed_true():
    """The supervisor pauses on a row whose vaa_check_failed is True (P3 reads exactly that)."""
    history = [{"step": 500, "check": "5%", "ema_vaa": 0.40}]
    row = vaa.apply_check("25%", 0.30, history, None, 0, sigma=0.01)
    assert row["vaa_check_failed"] is True
    assert row["check_failure"] == {
        "check": "25%",
        "ema_vaa": 0.30,
        "threshold": pytest.approx(0.38),
        "rule": row["check_rule"],
        "previous_check": "5%",
    }


def test_a_full_pass_also_scores_its_first_roots_as_the_subset():
    probe = _probe([[(12, False, 0), (2, True, 0)], [(3, False, 0), (9, True, 0)], [(5, True, 0)]])
    full = vaa.evaluate_vaa(CodeValueNet(), probe, torch.device("cpu"), subset=2)
    alone = _score(probe.subset(2))
    assert (full["vaa_subset"], full["n_subset"]) == (alone["vaa"], alone["n"]) == (0.5, 2)
    assert (full["vaa"], full["n"]) == (pytest.approx(2 / 3), 3)


def test_the_5_percent_check_compares_subset_with_subset_when_the_reference_row_is_a_subset_row():
    rows = (
        {"step": 1000, "samples": 1_024_000, "ema_vaa": 0.40, "vaa_set": "subset", "vaa_n": 2000},
        {"step": 4000, "samples": 4_096_000, "ema_vaa": 0.50, "vaa_set": "full", "vaa_n": 20000},
    )
    ref = vaa.Reference(name="m6h", rows=rows, cooldown_start=3500)
    same_roots = vaa.apply_check("5%", 0.45, [], ref, 2_000_000, sigma=0.01, subset=(2000, 0.37))
    assert same_roots["vaa_check_failed"] is True and same_roots["check_failure"]["ema_vaa"] == 0.37
    assert "subset" in same_roots["check_rule"]
    other_size = vaa.apply_check("5%", 0.45, [], ref, 2_000_000, sigma=0.01, subset=(1000, 0.37))
    assert "vaa_check_failed" not in other_size and "full" in other_size["check_rule"]


def test_scoring_ticks_after_every_chunk_so_a_long_check_keeps_the_heartbeat_fresh():
    """A full-valprobe check is minutes of forward passes; the supervisor kills a 60 s stale heartbeat."""
    probe = vaa.probe_from_roots(fixture_records()[:10])
    ticks = []
    model = BlinkNet(tiny_model_config())
    vaa.evaluate_vaa(model, probe, torch.device("cpu"), chunk=7, tick=lambda: ticks.append(1))
    assert len(ticks) == -(-len(probe.child_board) // 7)
