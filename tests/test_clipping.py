import math

import numpy as np
import pytest

from blink.train.clipping import GradClip


def test_auto_clip_fixes_at_twice_the_95th_percentile_of_the_warmup_norms():
    clip = GradClip("auto", warmup_steps=100)
    messages = [clip.observe(step, float(step + 1)) for step in range(100)]
    assert clip.value == pytest.approx(2 * np.percentile(np.arange(1, 101), 95))
    assert clip.limit() == clip.value and not clip.measuring
    assert [m for m in messages if m] == [messages[-1]]
    assert "2 x p95" in messages[-1] and "100 warmup" in messages[-1]


def test_auto_clip_leaves_warmup_gradients_unclipped():
    clip = GradClip("auto", warmup_steps=5)
    assert clip.measuring and math.isinf(clip.limit())
    assert clip.value is None


def test_a_numeric_clip_is_fixed_from_the_start_and_never_measures():
    clip = GradClip(1.0, warmup_steps=5)
    assert not clip.measuring and clip.limit() == 1.0
    assert clip.observe(0, 50.0) is None and clip.limit() == 1.0


def test_steps_after_warmup_are_not_recorded():
    clip = GradClip("auto", warmup_steps=3)
    for step, norm in enumerate((2.0, 4.0, 6.0, 1000.0, 1000.0)):
        clip.observe(step, norm)
    assert clip.value == pytest.approx(2 * np.percentile([2.0, 4.0, 6.0], 95))


def test_the_clip_state_survives_a_resume_mid_warmup():
    clip = GradClip("auto", warmup_steps=4)
    clip.observe(0, 3.0)
    clip.observe(1, 5.0)
    resumed = GradClip("auto", warmup_steps=4)
    resumed.load_state_dict(clip.state_dict())
    for step, norm in ((2, 7.0), (3, 9.0)):
        clip.observe(step, norm)
        resumed.observe(step, norm)
    assert resumed.value == clip.value == pytest.approx(2 * np.percentile([3.0, 5.0, 7.0, 9.0], 95))


class _DeviceNorm:
    """A stand-in for a 0-dim CUDA tensor: reading it on the host (float) is a device sync."""

    reads = 0

    def __init__(self, value: float) -> None:
        self.value = value

    def __float__(self) -> float:
        _DeviceNorm.reads += 1
        return self.value


def test_warmup_norms_stay_on_the_device_until_the_warmup_ends():
    """PF60: reading every warmup norm back (.item()) synced each step and cost 37% of S's rate."""
    _DeviceNorm.reads = 0
    clip = GradClip("auto", warmup_steps=50)
    for step in range(49):
        assert clip.observe(step, _DeviceNorm(float(step + 1))) is None
    assert _DeviceNorm.reads == 0 and clip.measuring
    assert clip.observe(49, _DeviceNorm(50.0)).startswith("clip auto: ")
    assert _DeviceNorm.reads == 50
    assert clip.value == pytest.approx(2 * np.percentile(np.arange(1, 51), 95))
    assert clip.state_dict()["norms"] == [float(n) for n in range(1, 51)]


def test_a_mid_warmup_checkpoint_stores_plain_floats():
    clip = GradClip("auto", warmup_steps=10)
    clip.observe(0, _DeviceNorm(3.0))
    assert clip.state_dict() == {"value": None, "norms": [3.0]}


def test_a_resume_past_warmup_without_a_measured_clip_is_refused():
    clip = GradClip("auto", warmup_steps=4)
    with pytest.raises(ValueError, match="warmup"):
        clip.load_state_dict({"value": None, "norms": [1.0]}, step=10)
