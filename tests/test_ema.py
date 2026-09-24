import pytest

torch = pytest.importorskip("torch")

from blink.train.ema import Ema, ema_decay  # noqa: E402

pytestmark = pytest.mark.torch


def test_ema_decay_warms_up_as_1_plus_t_over_10_plus_t():
    assert ema_decay(0) == pytest.approx(0.1)
    assert ema_decay(10) == pytest.approx(11 / 20)
    for t in (1, 5, 90, 1_000, 50_000):
        assert ema_decay(t) == pytest.approx(min(0.9999, (1 + t) / (10 + t)))
    assert ema_decay(89_990) == pytest.approx(0.9999)
    assert ema_decay(10**7) == 0.9999
    assert all(ema_decay(t) <= ema_decay(t + 1) for t in range(0, 200))


def test_the_decay_cap_is_configurable():
    assert ema_decay(10**7, max_decay=0.999) == 0.999


def test_an_ema_update_moves_the_shadow_by_one_minus_the_decay():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = Ema(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
    ema.update(model, step=0)  # decay 0.1: 0.1 * 1 + 0.9 * 3
    torch.testing.assert_close(ema.module.weight, torch.full((2, 2), 2.8))
    ema.update(model, step=10)  # decay 0.55: 0.55 * 2.8 + 0.45 * 3
    torch.testing.assert_close(ema.module.weight, torch.full((2, 2), 0.55 * 2.8 + 0.45 * 3.0))


def test_the_ema_copy_is_frozen_and_separate_from_the_model():
    model = torch.nn.Linear(2, 2)
    ema = Ema(model)
    assert all(not p.requires_grad for p in ema.module.parameters())
    assert ema.module.weight.data_ptr() != model.weight.data_ptr()


def test_the_ema_state_round_trips():
    model = torch.nn.Linear(3, 3)
    ema = Ema(model)
    ema.update(model, step=4)
    other = Ema(torch.nn.Linear(3, 3))
    other.load_state_dict(ema.state_dict())
    for a, b in zip(ema.module.parameters(), other.module.parameters(), strict=True):
        assert torch.equal(a, b)
