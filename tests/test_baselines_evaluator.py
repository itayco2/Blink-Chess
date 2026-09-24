"""BaselineEvaluator (interface 6) and the one agent wrapper every baseline plays through."""

import chess
import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from blink.baselines import evaluator, models  # noqa: E402
from blink.board import encode, value  # noqa: E402
from blink.play.agents import ValueAgent  # noqa: E402
from blink.play.evaluator import Evaluation  # noqa: E402

pytestmark = pytest.mark.torch


def test_two_hot_puts_the_mean_exactly_on_the_win_probability():
    win = np.array([0.0, 0.001, 0.25, 0.5, 0.73, 0.999, 1.0])
    probs = evaluator.two_hot(win)
    assert probs.shape == (7, value.NUM_BINS) and probs.dtype == np.float32
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert ((probs > 0).sum(axis=1) <= 2).all()
    inner = slice(2, 5)
    assert np.allclose(probs[inner] @ value.BIN_CENTERS, win[inner], atol=1e-6)
    assert probs[0, 0] == 1.0 and probs[-1, -1] == 1.0  # beyond the outer centres: all in the end bin


def test_baseline_evaluator_returns_zero_policy_and_a_two_hot_value():
    model = models.build("linear")
    with torch.no_grad():
        model.linear.weight.zero_()
        model.linear.bias.fill_(1.0)  # sigmoid(1) = 0.7311 for every row
    codes = np.stack([encode.encode_board(chess.Board())] * 3)
    result = evaluator.BaselineEvaluator(model).evaluate(codes)
    assert isinstance(result, Evaluation)
    assert np.array_equal(result.policy_logits, np.zeros((3, 1880), dtype=np.float32))
    assert np.allclose(result.win_probability(), 1 / (1 + np.exp(-1.0)), atol=1e-6)


def _mate_in_one() -> chess.Board:
    return chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")


def _stalemate_trap() -> chess.Board:
    """White to move with no mate in one: b5-b6 stalemates, the other moves keep the game going."""
    return chess.Board("k7/2K5/8/1P6/8/8/8/8 w - - 0 1")


@pytest.mark.parametrize("kind", ["material", "linear", "mlp"])
def test_every_baseline_uses_the_same_agent_wrapper_and_rules(kind, tmp_path):
    if kind == "material":
        agent = evaluator.baseline_agent("material")
    else:
        path = tmp_path / f"{kind}.pt"
        models.save(path, models.build(kind), kind, {"note": "untrained"})
        agent = evaluator.baseline_agent(str(path), device="cpu")
    assert isinstance(agent, ValueAgent)
    mate = agent.choose(_mate_in_one())
    assert mate.move == chess.Move.from_uci("a1a8") and mate.n_calls == 0 and "R2" in mate.rules
    trap = _stalemate_trap()
    decision = agent.choose(trap)
    assert decision.n_calls == 1 and decision.n_rows == trap.legal_moves.count() + 1
    assert "R1" in decision.rules and "R3" in decision.rules


def test_a_baseline_selector_by_kind_reads_blink_home_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    path = tmp_path / "runs" / "baseline-mlp" / "model.pt"
    path.parent.mkdir(parents=True)
    models.save(path, models.build("mlp"), "mlp", {})
    agent = evaluator.baseline_agent("mlp", device="cpu")
    assert agent.name == "MLP"
    with pytest.raises(FileNotFoundError, match="baseline-linear"):
        evaluator.baseline_agent("linear", device="cpu")


def test_the_torch_features_equal_the_numpy_features():
    from blink.baselines import features

    rng = np.random.default_rng(0)
    codes = rng.integers(0, encode.NUM_CODES, size=(50, 64), dtype=np.uint8)
    expected = features.features(codes).astype(np.float32)
    got = models.features_torch(torch.from_numpy(codes.astype(np.int64))).numpy()
    assert np.array_equal(got, expected)
    packed = torch.from_numpy(encode.pack(codes))
    assert np.array_equal(models.unpack_torch(packed).numpy(), codes.astype(np.int64))


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((2, 63), dtype=np.uint8),
        np.full((1, 64), -1, dtype=np.int64),
        np.full((1, 64), encode.NUM_CODES, dtype=np.int64),
    ],
)
def test_the_baseline_evaluator_refuses_exactly_what_the_reference_features_refuse(bad):
    from blink.baselines import features

    with pytest.raises(ValueError) as reference:
        features.features(bad)
    with pytest.raises(ValueError) as played:
        evaluator.BaselineEvaluator(models.build("linear")).evaluate(bad)
    assert str(played.value) == str(reference.value)
