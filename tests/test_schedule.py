import math

import pytest

from blink.train.schedule import cooldown_start, wsd_lr

PEAK, WARMUP, TOTAL = 1e-3, 100, 1000


def _lrs() -> list[float]:
    return [wsd_lr(step, PEAK, WARMUP, TOTAL, cooldown_frac=0.2) for step in range(TOTAL)]


def test_wsd_warms_up_holds_then_cools_to_zero():
    lrs = _lrs()
    warm, stable, cool = lrs[:WARMUP], lrs[WARMUP:800], lrs[800:]
    assert warm[0] == pytest.approx(PEAK / WARMUP)
    assert all(a < b for a, b in zip(warm, warm[1:], strict=False))
    assert warm[-1] == pytest.approx(PEAK)
    assert stable == [PEAK] * len(stable)
    assert all(a > b for a, b in zip(cool, cool[1:], strict=False))
    assert lrs[-1] == 0.0


def test_the_cooldown_is_the_last_20_percent_and_follows_one_minus_sqrt():
    assert cooldown_start(TOTAL, 0.2) == 800
    halfway = 800 + 100 - 1  # (step + 1 - start) / length == 0.5
    assert wsd_lr(halfway, PEAK, WARMUP, TOTAL, 0.2) == pytest.approx(PEAK * (1 - math.sqrt(0.5)))


def test_the_learning_rate_stays_at_zero_past_the_end():
    assert wsd_lr(TOTAL, PEAK, WARMUP, TOTAL, 0.2) == 0.0
    assert wsd_lr(TOTAL + 50, PEAK, WARMUP, TOTAL, 0.2) == 0.0


def test_no_warmup_starts_at_the_peak():
    assert wsd_lr(0, PEAK, 0, TOTAL, 0.2) == PEAK


def test_a_negative_step_is_refused():
    with pytest.raises(ValueError):
        wsd_lr(-1, PEAK, WARMUP, TOTAL, 0.2)
