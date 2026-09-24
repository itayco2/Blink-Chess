import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_model_config  # noqa: E402

from blink.board import value as board_value  # noqa: E402
from blink.data.record import NO_MOVE  # noqa: E402
from blink.model import losses  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train.batch import make_batch  # noqa: E402

pytestmark = pytest.mark.torch


def test_initial_losses_are_ln_1880_and_ln_128():
    batch = make_batch(fixture_records(), "cpu")
    policy, value = BlinkNet(tiny_model_config())(batch.tokens)
    loss_policy, loss_value = losses.compute_losses(policy, value, batch, alpha=0.5, tau=0.05)
    assert math.log(1880) == pytest.approx(7.54, abs=0.01)
    assert math.log(128) == pytest.approx(4.85, abs=0.01)
    assert loss_policy.item() == pytest.approx(7.54, abs=0.1)
    assert loss_value.item() == pytest.approx(4.85, abs=0.05)


def test_soft_policy_target_sums_to_one_and_favours_the_best_move():
    batch = make_batch(fixture_records(), "cpu")
    target = losses.soft_policy_target(
        batch.move, batch.alt_move, batch.alt_valid, batch.w_best, batch.w_alt, alpha=0.5, tau=0.05
    )
    torch.testing.assert_close(target.sum(-1), torch.ones(len(target)), rtol=0, atol=1e-6)
    assert torch.equal(target.argmax(-1), batch.move)
    best_mass = target.gather(1, batch.move[:, None]).squeeze(1)
    assert torch.all(best_mass >= 0.5)


def test_the_soft_target_follows_the_win_probability_gaps():
    move = torch.tensor([10])
    alt_move = torch.tensor([[11, 12, NO_MOVE, NO_MOVE]])
    alt_valid = torch.tensor([[True, True, False, False]])
    w_best = torch.tensor([0.60])
    w_alt = torch.tensor([[0.55, 0.60, 0.5, 0.5]])
    target = losses.soft_policy_target(move, alt_move, alt_valid, w_best, w_alt, alpha=0.5, tau=0.05)
    weights = np.exp(np.array([0.0, -0.05 / 0.05, 0.0]))
    soft = weights / weights.sum()
    assert target[0, 10].item() == pytest.approx(0.5 + 0.5 * soft[0], abs=1e-6)
    assert target[0, 11].item() == pytest.approx(0.5 * soft[1], abs=1e-6)
    assert target[0, 12].item() == pytest.approx(0.5 * soft[2], abs=1e-6)
    assert target.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_a_position_without_alternatives_gets_a_one_hot_target():
    move = torch.tensor([7])
    alt_valid = torch.zeros(1, 4, dtype=torch.bool)
    target = losses.soft_policy_target(
        move, torch.full((1, 4), NO_MOVE), alt_valid, torch.tensor([0.5]), torch.full((1, 4), 0.5), 0.5, 0.05
    )
    assert target[0, 7].item() == pytest.approx(1.0)
    assert torch.count_nonzero(target).item() == 1


@pytest.mark.parametrize("p", [0.0, 0.004, 0.3, 0.5, 0.97, 1.0])
def test_the_torch_hl_gauss_matches_the_numpy_contract(p):
    ours = losses.hl_gauss_target(torch.tensor([p], dtype=torch.float32))[0].double().numpy()
    np.testing.assert_allclose(ours, board_value.hl_gauss(p), atol=2e-6)


def test_value_ce_of_a_perfect_prediction_is_the_target_entropy():
    target = losses.hl_gauss_target(torch.tensor([0.3, 0.8]))
    logits = torch.log(target.clamp_min(1e-30))
    entropy = -(target * torch.log(target.clamp_min(1e-30))).sum(-1).mean()
    assert losses.soft_cross_entropy(logits, target).item() == pytest.approx(entropy.item(), abs=1e-5)
