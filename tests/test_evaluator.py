import copy

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import tiny_model_config  # noqa: E402

from blink.board import encode  # noqa: E402
from blink.model.evaluator import TorchEvaluator  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.play.evaluator import Evaluation, Evaluator  # noqa: E402

pytestmark = pytest.mark.torch


def _codes(n: int = 3) -> np.ndarray:
    board = chess.Board()
    rows = []
    for move in ("e2e4", "e7e5", "g1f3")[:n]:
        rows.append(encode.encode_board(board))
        board.push_uci(move)
    return np.stack(rows)


def _trained_looking_model() -> BlinkNet:
    model = BlinkNet(tiny_model_config())
    generator = torch.Generator().manual_seed(4)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.05)
    return model


def test_the_torch_evaluator_satisfies_the_play_protocol():
    evaluator: Evaluator = TorchEvaluator(BlinkNet(tiny_model_config()), "cpu")
    result = evaluator.evaluate(_codes())
    assert isinstance(result, Evaluation)
    assert result.policy_logits.shape == (3, 1880) and result.policy_logits.dtype == np.float32
    assert result.value_probs.shape == (3, 128) and result.value_probs.dtype == np.float32
    np.testing.assert_allclose(result.value_probs.sum(axis=1), 1.0, atol=1e-5)


def test_a_fresh_network_is_uniform_with_win_probability_one_half():
    result = TorchEvaluator(BlinkNet(tiny_model_config()), "cpu").evaluate(_codes(1))
    np.testing.assert_allclose(result.value_probs, 1 / 128, atol=1e-7)
    assert result.win_probability()[0] == pytest.approx(0.5, abs=1e-6)


def test_one_evaluate_call_is_exactly_one_forward_pass():
    model = _trained_looking_model()
    calls = []
    model.register_forward_hook(lambda module, args, output: calls.append(args[0].shape[0]))
    TorchEvaluator(model, "cpu").evaluate(_codes(3))
    assert calls == [3]


def test_the_evaluator_returns_the_models_fp32_outputs_with_a_softmaxed_value():
    model = _trained_looking_model()
    codes = _codes(2)
    result = TorchEvaluator(model, "cpu").evaluate(codes)
    with torch.no_grad():
        policy, value = model(torch.from_numpy(codes.astype(np.int64)))
    np.testing.assert_allclose(result.policy_logits, policy.numpy(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.value_probs, torch.softmax(value, -1).numpy(), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((2, 63), dtype=np.uint8), np.zeros(64, dtype=np.uint8), np.full((1, 64), 16, dtype=np.int64)],
)
def test_malformed_square_codes_are_refused(bad):
    with pytest.raises(ValueError):
        TorchEvaluator(BlinkNet(tiny_model_config()), "cpu").evaluate(bad)


@pytest.mark.cuda
def test_the_evaluator_runs_on_the_gpu_in_fp32():
    model = _trained_looking_model()
    gpu = TorchEvaluator(copy.deepcopy(model), "cuda").evaluate(_codes(2))
    cpu = TorchEvaluator(model, "cpu").evaluate(_codes(2))
    np.testing.assert_allclose(gpu.policy_logits, cpu.policy_logits, atol=1e-3)
