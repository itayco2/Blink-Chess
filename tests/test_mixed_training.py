import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_model_config  # noqa: E402

from blink.board import value as board_value  # noqa: E402
from blink.data.record import CHILD_DTYPE  # noqa: E402
from blink.model import losses  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train import step  # noqa: E402
from blink.train.batch import make_batch, make_child_batch  # noqa: E402
from blink.train.source import StepData  # noqa: E402

pytestmark = pytest.mark.torch


def children_from(roots: np.ndarray) -> np.ndarray:
    """Child records reusing root boards, with scores shifted so their targets differ from the roots'."""
    children = np.zeros(len(roots), dtype=CHILD_DTYPE)
    for name in ("board", "cp", "mate", "depth", "fen_hash"):
        children[name] = roots[name]
    is_cp = children["cp"] != board_value.CP_NONE
    children["cp"] = np.where(is_cp, -np.clip(children["cp"], -900, 900), children["cp"])
    return children


def _logits(rows: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    policy = torch.randn(rows, 1880, generator=generator, requires_grad=True)
    value = torch.randn(rows, 128, generator=generator, requires_grad=True)
    return policy, value


def test_children_add_value_loss_but_no_policy_loss():
    roots = make_batch(fixture_records()[:6], "cpu")
    children = make_child_batch(children_from(fixture_records()[6:10]), "cpu")
    policy, value = _logits(10, seed=0)
    policy_ce, value_ce = losses.mixed_losses(policy, value, roots, children.w, alpha=0.5, tau=0.05)
    assert policy_ce.shape == (6,) and value_ce.shape == (10,)

    policy_ce.sum().backward(retain_graph=True)
    assert torch.count_nonzero(policy.grad[6:]) == 0 and torch.count_nonzero(policy.grad[:6]) > 0
    value_ce.sum().backward()
    assert torch.count_nonzero(value.grad[6:]) > 0

    flipped = make_child_batch(children_from(fixture_records()[10:14]), "cpu")
    _, other = losses.mixed_losses(policy, value, roots, flipped.w, alpha=0.5, tau=0.05)
    assert torch.equal(other[:6], value_ce[:6]) and not torch.equal(other[6:], value_ce[6:])


def test_the_child_value_target_is_its_own_side_to_move_win_probability():
    records = children_from(fixture_records()[:5])
    batch = make_child_batch(records, "cpu")
    expected = board_value.win_probability_array(records["cp"], records["mate"]).astype(np.float32)
    np.testing.assert_allclose(batch.w.numpy(), expected)
    assert batch.tokens.shape == (5, 64) and batch.tokens.dtype == torch.int64


def _step_data(n_roots: int, n_children: int, weights: bool = False) -> StepData:
    roots = fixture_records()[:n_roots]
    children = children_from(fixture_records()[n_roots : n_roots + n_children])
    rng = np.random.default_rng(3)
    return StepData(
        roots=roots,
        children=children,
        root_weight=rng.uniform(0.2, 5, n_roots).astype(np.float32) if weights else None,
        child_weight=rng.uniform(0.2, 5, n_children).astype(np.float32) if weights else None,
    )


def _grads(model: BlinkNet, data: StepData, micro: int) -> tuple[tuple[float, float], dict]:
    model.zero_grad(set_to_none=True)
    out = step.accumulate(model, data, torch.device("cpu"), micro, alpha=0.5, tau=0.05, lambda_v=1.0)
    return (out.policy.item(), out.value.item()), {n: p.grad.clone() for n, p in model.named_parameters()}


def test_rebalancing_weights_multiply_both_losses():
    torch.manual_seed(0)
    model = BlinkNet(tiny_model_config())
    data = _step_data(8, 4)
    (policy, value), _ = _grads(model, data, micro=12)
    doubled = StepData(data.roots, data.children, np.full(8, 2.0, np.float32), np.full(4, 2.0, np.float32))
    (policy2, value2), _ = _grads(model, doubled, micro=12)
    assert policy2 == pytest.approx(2 * policy, rel=1e-6) and value2 == pytest.approx(2 * value, rel=1e-6)

    no_children = StepData(data.roots, data.children, np.ones(8, np.float32), np.zeros(4, np.float32))
    (policy3, value3), _ = _grads(model, no_children, micro=12)
    roots_only = StepData(data.roots, data.children[:0])
    (policy4, value4), _ = _grads(model, roots_only, micro=12)
    assert policy3 == pytest.approx(policy4, rel=1e-6)
    assert value3 == pytest.approx(value4 * 8 / 12, rel=1e-6)  # the value mean still counts all 12 rows


def test_gradient_accumulation_over_micro_batches_equals_one_full_batch():
    torch.manual_seed(0)
    model = BlinkNet(tiny_model_config(gab=True))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    data = _step_data(21, 11, weights=True)
    whole, whole_grads = _grads(model, data, micro=32)
    parts, part_grads = _grads(model, data, micro=8)
    assert parts == pytest.approx(whole, rel=1e-5)
    for name, grad in whole_grads.items():
        torch.testing.assert_close(part_grads[name], grad, rtol=1e-4, atol=1e-6, msg=name)


def test_micro_slices_cover_every_row_once_and_keep_the_mix():
    slices = step.micro_slices(717, 307, parts=4)
    roots = [s.stop - s.start for s, _ in slices]
    children = [c.stop - c.start for _, c in slices]
    assert sum(roots) == 717 and sum(children) == 307 and len(slices) == 4
    assert max(roots) - min(roots) <= 1 and max(children) - min(children) <= 1
    assert slices[0][0].start == 0 and slices[-1][0].stop == 717


@pytest.mark.cuda
def test_a_mixed_step_accumulates_in_bf16_on_cuda():
    model = BlinkNet(tiny_model_config(gab=True)).cuda()
    data = _step_data(20, 8, weights=True)
    out = step.accumulate(model, data, torch.device("cuda"), 7, alpha=0.5, tau=0.05, lambda_v=1.0)
    assert out.rows == 28 and out.passes == 4
    assert np.isfinite(out.policy.item()) and np.isfinite(out.value.item())
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
