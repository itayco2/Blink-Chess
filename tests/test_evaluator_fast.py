"""The opt-in fast play modes of TorchEvaluator: bf16 trunk with fp32 heads, and a compiled trunk.

CPU tests pin the default path (bitwise the model's own output), the CPU refusal of bf16, and the
structure of the fast path with torch.compile replaced by a spy. The cuda tests measure the real
thing: parity of bf16 against fp32 on a model with non-trivial heads, and the compiled trunk at every
row count play sends (1 row for one look, L+1 for value mode).
"""

import math
import random
from dataclasses import fields

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import tiny_model_config  # noqa: E402

from blink.board import encode, value  # noqa: E402
from blink.model import loading  # noqa: E402
from blink.model.config import ModelConfig  # noqa: E402
from blink.model.evaluator import TorchEvaluator  # noqa: E402
from blink.model.transformer import BlinkNet, Trunk  # noqa: E402
from blink.play.agents import expand  # noqa: E402

pytestmark = pytest.mark.torch


def _codes(n: int = 3) -> np.ndarray:
    board = chess.Board()
    rows = []
    for move in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5")[:n]:
        rows.append(encode.encode_board(board))
        board.push_uci(move)
    return np.stack(rows)


def _perturbed_model(config: ModelConfig | None = None, seed: int = 4) -> BlinkNet:
    model = BlinkNet(config or tiny_model_config())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn(p.shape, generator=generator) * 0.05)
    return model


def _direct(model: BlinkNet, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        policy, value_logits = model(torch.from_numpy(codes.astype(np.int64)))
    return policy.numpy(), torch.softmax(value_logits, dim=-1).numpy()


class CompileSpy:
    """Stands in for torch.compile: records what it was given and returns a plain callable."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.nn.Module, dict]] = []

    def __call__(self, module, **kwargs):
        self.calls.append((module, kwargs))
        return lambda tokens: module(tokens)


def _row_counts(model: BlinkNet) -> dict[str, list[int]]:
    calls: dict[str, list[int]] = {"model": [], "trunk": [], "policy": [], "value": []}
    for name, module in (
        ("model", model),
        ("trunk", model.trunk),
        ("policy", model.policy),
        ("value", model.value),
    ):
        module.register_forward_hook(lambda m, args, out, name=name: calls[name].append(args[0].shape[0]))
    return calls


def test_the_default_path_is_bitwise_the_models_own_output():
    model = _perturbed_model()
    codes = _codes(3)
    policy, probs = _direct(model, codes)
    for evaluator in (
        TorchEvaluator(model, "cpu"),
        TorchEvaluator(model, "cpu", precision="fp32", compile=False),
    ):
        result = evaluator.evaluate(codes)
        assert np.array_equal(result.policy_logits, policy)
        assert np.array_equal(result.value_probs, probs)


def test_the_default_path_is_one_call_of_the_whole_model():
    model = _perturbed_model()
    calls = _row_counts(model)
    TorchEvaluator(model, "cpu").evaluate(_codes(3))
    assert calls == {"model": [3], "trunk": [3], "policy": [3], "value": [3]}


def test_bf16_is_refused_on_the_cpu_rather_than_played_as_fp32():
    with pytest.raises(ValueError, match="CUDA only"):
        TorchEvaluator(_perturbed_model(), "cpu", precision="bf16")


def test_an_unknown_precision_is_refused():
    with pytest.raises(ValueError, match="precision must be one of"):
        TorchEvaluator(_perturbed_model(), "cpu", precision="fp16")


def test_compile_wraps_only_the_trunk_with_dynamic_shapes_and_leaves_the_model_alone(monkeypatch):
    spy = CompileSpy()
    monkeypatch.setattr(torch, "compile", spy)
    model = _perturbed_model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    evaluator = TorchEvaluator(model, "cpu", compile=True)
    assert len(spy.calls) == 1
    module, kwargs = spy.calls[0]
    assert module is model.trunk and kwargs == {"dynamic": True}
    assert isinstance(model.trunk, Trunk)  # the attribute is not replaced by the compiled wrapper
    evaluator.evaluate(_codes(3))
    after = model.state_dict()
    assert before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)


def test_the_fast_path_is_still_one_forward_pass_over_every_row(monkeypatch):
    monkeypatch.setattr(torch, "compile", CompileSpy())
    model = _perturbed_model()
    calls = _row_counts(model)
    TorchEvaluator(model, "cpu", compile=True).evaluate(_codes(5))
    assert calls == {"model": [], "trunk": [5], "policy": [5], "value": [5]}


def test_the_fast_path_in_fp32_gives_the_models_own_numbers(monkeypatch):
    """Trunk then heads is BlinkNet.forward spelled out: in fp32 it is the same arithmetic."""
    monkeypatch.setattr(torch, "compile", CompileSpy())
    model = _perturbed_model()
    codes = _codes(4)
    policy, probs = _direct(model, codes)
    result = TorchEvaluator(model, "cpu", compile=True).evaluate(codes)
    assert np.array_equal(result.policy_logits, policy)
    assert np.array_equal(result.value_probs, probs)


def test_warm_up_runs_one_row_then_two(monkeypatch):
    monkeypatch.setattr(torch, "compile", CompileSpy())
    model = _perturbed_model()
    calls = _row_counts(model)
    TorchEvaluator(model, "cpu", compile=True).warm_up()
    assert calls["trunk"] == [1, 2]


# ---------------------------------------------------------------- cuda

PARITY_POSITIONS = 100
PARITY_CONFIG = ModelConfig(d_model=128, n_layers=2, n_heads=4, head_dim=32, gab=True)


def non_trivial_model(config: ModelConfig = PARITY_CONFIG, seed: int = 3) -> BlinkNet:
    """A model whose heads say something: distinct square embeddings (a trained trunk's squares are
    far apart), a perturbed GAB-lite trunk, a policy head with a logit gap near 1 at the median, and
    a value head whose win% moves with the position. Freshly initialised heads are zero (uniform
    outputs), which no precision can disagree about."""
    torch.manual_seed(seed)
    model = BlinkNet(config)
    generator = torch.Generator().manual_seed(seed)
    scale = 1.0 / math.sqrt(config.d_model)

    def normal(shape, std):
        return torch.randn(shape, generator=generator) * std

    with torch.no_grad():
        trunk = model.trunk
        trunk.token_embedding.weight.copy_(normal(trunk.token_embedding.weight.shape, 1.0))
        trunk.square_embedding.copy_(normal(trunk.square_embedding.shape, 1.0))
        for name, p in trunk.named_parameters():
            if "embedding" not in name and p.dim() >= 2:
                p.add_(normal(p.shape, 0.05))
        for layer in (model.policy.query, model.policy.key, model.policy.promotion):
            layer.weight.copy_(normal(layer.weight.shape, 2.0 * scale))
        model.value.hidden.weight.copy_(normal(model.value.hidden.weight.shape, 16.0 * scale))
        model.value.out.weight.copy_(normal(model.value.out.weight.shape, 2.0 * scale))
    return model.eval()


def random_game_positions(n: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randrange(4, 60)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if any(board.generate_legal_moves()):
            out.append(board.copy(stack=False))
    return out


def value_batch(board: chess.Board) -> tuple[np.ndarray, list[list[int]]]:
    """The value-mode rows (P and every child) and each row's legal moves, as vocabulary indices."""
    kids = expand(board)
    rows = [encode.encode_board(board)] + [encode.encode_board(c.board) for c in kids]
    legal = [[c.index for c in kids]]
    legal += [[g.index for g in expand(c.board)] if any(c.board.generate_legal_moves()) else [] for c in kids]
    return np.stack(rows), legal


@pytest.mark.cuda
def test_bf16_runs_the_trunk_in_bf16_and_both_heads_in_fp32():
    model = non_trivial_model()
    seen: dict[str, object] = {}
    model.trunk.blocks[0].ffn_in.register_forward_hook(lambda m, a, out: seen.update(trunk=out.dtype))
    for name in ("policy", "value"):
        getattr(model, name).register_forward_pre_hook(
            lambda m, args, name=name: seen.update({name: (args[0].dtype, torch.is_autocast_enabled("cuda"))})
        )
    result = TorchEvaluator(model, "cuda", precision="bf16").evaluate(_codes(3))
    assert seen["trunk"] == torch.bfloat16
    assert seen["policy"] == (torch.float32, False) and seen["value"] == (torch.float32, False)
    assert result.policy_logits.dtype == np.float32 and result.value_probs.dtype == np.float32


@pytest.mark.cuda
def test_bf16_with_fp32_heads_keeps_parity_on_a_model_with_non_trivial_heads():
    """Every value-mode row of 100 positions (about 3,300): policy top-1 over each row's legal moves
    agrees with fp32 on >= 99% of rows, and no row's win% moves by more than 1 point."""
    model = non_trivial_model()
    reference = TorchEvaluator(model, "cuda")
    fast = TorchEvaluator(model, "cuda", precision="bf16")
    agree = rows = 0
    dwin_max, wins = 0.0, []
    for board in random_game_positions(PARITY_POSITIONS, seed=11):
        codes, legal = value_batch(board)
        ref, got = reference.evaluate(codes), fast.evaluate(codes)
        for row, moves_ in enumerate(legal):
            if moves_:
                rows += 1
                agree += (
                    moves_[np.argmax(ref.policy_logits[row][moves_])]
                    == moves_[np.argmax(got.policy_logits[row][moves_])]
                )
        dwin_max = max(dwin_max, float(np.abs(ref.win_probability() - got.win_probability()).max()) * 100)
        wins.append(ref.win_probability())
    assert np.concatenate(wins).std() * 100 > 1.0  # the value head is not flat
    assert agree / rows >= 0.99, f"top-1 agreement {agree / rows:.4f} over {rows} rows"
    assert dwin_max <= 1.0, f"max |d win%| {dwin_max:.3f} pt"


@pytest.mark.cuda
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_the_compiled_trunk_serves_every_row_count_play_sends(precision):
    model = non_trivial_model()
    eager = TorchEvaluator(model, "cuda", precision=precision)
    compiled = TorchEvaluator(model, "cuda", precision=precision, compile=True)
    compiled.warm_up()
    atol = 1e-3 if precision == "fp32" else 5e-2
    for n in (1, 2, 7, 37, 219, 1, 64):
        codes = np.random.default_rng(n).integers(0, encode.NUM_CODES, size=(n, 64), dtype=np.uint8)
        got, want = compiled.evaluate(codes), eager.evaluate(codes)
        assert got.policy_logits.shape == (n, 1880) and got.value_probs.shape == (n, value.NUM_BINS)
        np.testing.assert_allclose(got.policy_logits, want.policy_logits, atol=atol, rtol=1e-2)
        np.testing.assert_allclose(got.win_probability(), want.win_probability(), atol=atol)
    assert isinstance(model.trunk, Trunk)
    assert not any("_orig_mod" in key for key in model.state_dict())


# ---------------------------------------------------------------- loading


@pytest.fixture
def slim_weights(tmp_path):
    model = _perturbed_model()
    path = tmp_path / "slim.pt"
    config = tiny_model_config()
    torch.save(
        {"config": {f.name: getattr(config, f.name) for f in fields(config)}, "model": model.state_dict()},
        path,
    )
    return path, model


def test_load_evaluator_plays_fp32_uncompiled_by_default(slim_weights):
    path, model = slim_weights
    evaluator = loading.load_evaluator(str(path), device="cpu")
    assert (evaluator.precision, evaluator.compile) == ("fp32", False)
    policy, probs = _direct(model, _codes(3))
    result = evaluator.evaluate(_codes(3))
    assert np.array_equal(result.policy_logits, policy) and np.array_equal(result.value_probs, probs)


def test_load_evaluator_checks_the_mode_before_it_loads_any_weights():
    with pytest.raises(ValueError, match="CUDA only"):
        loading.load_evaluator("run:no-such-run", device="cpu", precision="bf16")


def test_load_evaluator_warms_a_compiled_evaluator_up_before_returning_it(slim_weights, monkeypatch):
    seen = []

    def spy(module, **kwargs):
        return lambda tokens: (seen.append(tokens.shape[0]), module(tokens))[1]

    monkeypatch.setattr(torch, "compile", spy)
    evaluator = loading.load_evaluator(str(slim_weights[0]), device="cpu", compile=True)
    assert evaluator.compile and seen == [1, 2]
