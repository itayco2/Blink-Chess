"""Gradient clipping: a fixed norm, or "auto" measured over the warmup.

The P1 skeleton ran with clip 1.0 against a median gradient norm of 5.9, so 98% of its steps were
clipped: the clip was acting as a second learning rate and would trip the P7 stop rule (clip
fraction >= 20%). With clip_norm = "auto" the warmup steps are left unclipped while their gradient
norms are recorded, and at the end of the warmup the clip is fixed at 2 x their 95th percentile. The
recorded norms and the chosen clip travel in every checkpoint, so a resume inside the warmup picks
up the same measurement.
"""

import math
from typing import Any

import numpy as np

AUTO = "auto"
MULTIPLIER = 2.0
PERCENTILE = 95


class GradClip:
    def __init__(self, setting: float | str, warmup_steps: int) -> None:
        self.auto = setting == AUTO
        self.warmup_steps = warmup_steps
        self.value: float | None = None if self.auto else float(setting)
        self.norms: list[float] = []

    @property
    def measuring(self) -> bool:
        return self.value is None

    def limit(self) -> float:
        """The max norm to pass to clip_grad_norm_ (infinite while the warmup is measured)."""
        return math.inf if self.value is None else self.value

    def observe(self, step: int, norm: float) -> str | None:
        """Record a warmup step's norm; returns the log line when the clip gets fixed."""
        if not self.measuring or step >= self.warmup_steps:
            return None
        self.norms.append(float(norm))
        if step < self.warmup_steps - 1:
            return None
        norms = np.asarray(self.norms)
        p95 = float(np.percentile(norms, PERCENTILE))
        self.value = MULTIPLIER * p95
        return (
            f"clip auto: {self.value:.4g} = 2 x p95 of {len(norms)} warmup grad norms "
            f"(median {np.median(norms):.4g}, p95 {p95:.4g}, max {norms.max():.4g})"
        )

    def state_dict(self) -> dict[str, Any]:
        return {"value": self.value, "norms": list(self.norms)}

    def load_state_dict(self, state: dict[str, Any], step: int = 0) -> None:
        if not self.auto:
            return  # a numeric clip comes from the config, never from the checkpoint
        self.value = state.get("value")
        self.norms = [float(n) for n in state.get("norms", [])]
        if self.value is None and step >= self.warmup_steps:
            raise ValueError(
                f"clip_norm 'auto' resumed at step {step}, past the {self.warmup_steps}-step warmup, "
                "without a measured clip in the checkpoint"
            )
