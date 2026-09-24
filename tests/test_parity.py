"""Fast-mode parity: the helper's arithmetic, its rows (the value agent's own) and `blink bench parity`."""

import json
import random
from dataclasses import dataclass, fields

import chess
import numpy as np
import pytest

from blink import cli
from blink.board import encode, value
from blink.eval import parity
from blink.play import agents
from blink.play.evaluator import Evaluation
from blink.play.oracles import MaterialEvaluator, RandomLogitEvaluator

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"


def positions(n: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randrange(2, 50)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if any(board.generate_legal_moves()) and not parity.mate_now(board):
            out.append(board.copy(stack=False))
    return out


@dataclass(frozen=True)
class OneBinUp:
    """Every row's value distribution moved up one bin: every win% rises by exactly 100/128 points."""

    evaluator: object

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        base = self.evaluator.evaluate(codes)
        return Evaluation(base.policy_logits, np.roll(base.value_probs, 1, axis=1))


def test_an_evaluator_agrees_with_itself_everywhere():
    report = parity.compare(RandomLogitEvaluator(0), RandomLogitEvaluator(0), positions(20, 1))
    assert report["scored"] == 20 and report["mate_now"] == 0
    assert report["policy_top1_agreement"] == 1.0 and report["value_choice_agreement"] == 1.0
    assert report["max_abs_dwin_pct"] == 0.0 and report["value_disagreements"] == []


def test_the_value_rows_are_the_value_agents_own_l_plus_1_rows():
    evaluator = RandomLogitEvaluator(3)
    for board in positions(10, 2):
        look = parity.look(evaluator, board, epsilon=0.0)
        kids = agents.expand(board)
        assert len(look.rows) == len(kids) + 1 == board.legal_moves.count() + 1
        assert np.array_equal(look.rows[0], encode.encode_board(board))
        assert {r.tobytes() for r in look.rows[1:]} == {encode.encode_board(c.board).tobytes() for c in kids}
        assert look.value_move == agents.ValueAgent(evaluator).choose(board).move


def _independent(reference, fast, boards) -> tuple[float, float, float]:
    """Top-1, value choice and max |d win%| counted without the helper."""
    top1 = choice = 0
    dwin = 0.0
    for board in boards:
        kids = agents.expand(board)
        legal = [c.index for c in kids]
        root = encode.encode_board(board)[None]
        best = [legal[int(np.argmax(e.evaluate(root).policy_logits[0][legal]))] for e in (reference, fast)]
        top1 += best[0] == best[1]
        moves = [agents.ValueAgent(e).choose(board).move for e in (reference, fast)]
        choice += moves[0] == moves[1]
        rows = np.stack([root[0]] + [encode.encode_board(c.board) for c in kids])
        wins = [e.evaluate(rows).win_probability() for e in (reference, fast)]
        dwin = max(dwin, float(np.abs(wins[0] - wins[1]).max()) * 100)
    return top1 / len(boards), choice / len(boards), dwin


def test_the_helpers_numbers_match_an_independent_count():
    boards = positions(40, 3)
    reference, fast = RandomLogitEvaluator(0), RandomLogitEvaluator(1)
    report = parity.compare(reference, fast, boards)
    top1, choice, dwin = _independent(reference, fast, boards)
    assert report["policy_top1_agreement"] == pytest.approx(top1)
    assert report["value_choice_agreement"] == pytest.approx(choice)
    assert report["max_abs_dwin_pct"] == pytest.approx(dwin)
    assert 0.0 < choice < 1.0  # two different networks disagree somewhere, not everywhere
    disagreements = round((1 - choice) * len(boards))
    assert len(report["value_disagreements"]) == min(disagreements, parity.MAX_EXAMPLES)


def test_a_one_bin_shift_moves_every_win_by_exactly_one_bin():
    reference = MaterialEvaluator(exact=True)
    report = parity.compare(reference, OneBinUp(reference), positions(20, 4))
    one_bin = 100.0 / value.NUM_BINS
    assert report["max_abs_dwin_pct"] == pytest.approx(one_bin, abs=1e-4)
    assert report["mean_abs_dwin_pct"] == pytest.approx(one_bin, abs=1e-4)
    assert report["policy_top1_agreement"] == 1.0  # the policy did not change


def test_a_mate_in_one_is_played_by_rule_and_counted_apart():
    board = chess.Board(MATE_IN_ONE)
    assert parity.position_parity(RandomLogitEvaluator(0), RandomLogitEvaluator(1), board) is None
    report = parity.compare(RandomLogitEvaluator(0), RandomLogitEvaluator(1), [board])
    assert (report["positions"], report["mate_now"], report["scored"]) == (1, 1, 0)
    assert report["policy_top1_agreement"] is None and report["max_abs_dwin_pct"] is None


def test_evaluators_sent_different_rows_are_an_error(monkeypatch):
    board = positions(1, 5)[0]
    looks = iter(
        [
            parity.look(RandomLogitEvaluator(0), board, 0.0),
            parity.look(RandomLogitEvaluator(0), chess.Board(), 0.0),
        ]
    )
    monkeypatch.setattr(parity, "look", lambda evaluator, board, epsilon: next(looks))
    with pytest.raises(RuntimeError, match="different value rows"):
        parity.position_parity(RandomLogitEvaluator(0), RandomLogitEvaluator(1), board)


# ---------------------------------------------------------------- positions and the command


def _write_val_roots(pack) -> np.ndarray:
    from train_helpers import fixture_records

    records = fixture_records()
    records["depth"] = 30  # every fixture root counts as deep, as VAA takes them
    pack.mkdir(parents=True, exist_ok=True)
    records.tofile(pack / "val_roots.bin")
    return records


def test_val_positions_are_the_roots_vaa_takes_as_boards(tmp_path):
    records = _write_val_roots(tmp_path / "pack")
    boards = parity.val_positions(tmp_path / "pack", 7)
    assert len(boards) == 7
    for board, record in zip(boards, records[:7], strict=True):
        assert np.array_equal(encode.encode_board(board), encode.unpack(record["board"]))


@pytest.mark.torch
def test_bench_parity_on_a_tiny_model_writes_the_report(tmp_path):
    torch = pytest.importorskip("torch")
    from train_helpers import tiny_model_config

    from blink.model.transformer import BlinkNet

    _write_val_roots(tmp_path / "pack")
    config = tiny_model_config()
    model = BlinkNet(config)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=torch.Generator().manual_seed(1)) * 0.05)
    weights = tmp_path / "tiny.pt"
    torch.save(
        {"config": {f.name: getattr(config, f.name) for f in fields(config)}, "model": model.state_dict()},
        weights,
    )
    out = tmp_path / "parity.json"
    argv = ["bench", "parity", "--model", str(weights), "--device", "cpu", "--positions", "12"]
    assert cli.main([*argv, "--data", str(tmp_path / "pack"), "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert (report["precision"], report["compile"], report["device"]) == ("fp32", False, "cpu")
    assert report["positions"] == 12 and report["scored"] + report["mate_now"] == 12
    assert report["policy_top1_agreement"] == 1.0 and report["value_choice_agreement"] == 1.0
    assert report["max_abs_dwin_pct"] == 0.0 and report["value_rows"] > report["scored"]


def test_bench_parity_refuses_bf16_on_the_cpu(tmp_path, capsys):
    argv = ["bench", "parity", "--model", "ship", "--device", "cpu", "--precision", "bf16"]
    assert cli.main([*argv, "--data", str(tmp_path), "--out", str(tmp_path / "p.json")]) == 2
    assert "CUDA only" in capsys.readouterr().err
