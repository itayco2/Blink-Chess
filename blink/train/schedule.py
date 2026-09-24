"""Warmup-stable-decay learning rate: linear warmup, a flat peak, then a 1 - sqrt cooldown to zero.

`step` is the 0-based index of the optimizer update. The last update (step == total - 1) runs at
exactly zero, and any step past the end stays at zero.
"""

import math


def cooldown_start(total: int, cooldown_frac: float) -> int:
    return total - int(round(cooldown_frac * total))


def wsd_lr(step: int, peak: float, warmup: int, total: int, cooldown_frac: float = 0.2) -> float:
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if step >= total:
        return 0.0
    if step < warmup:
        return peak * (step + 1) / warmup
    start = cooldown_start(total, cooldown_frac)
    if step < start:
        return peak
    progress = (step + 1 - start) / (total - start)
    return peak * max(0.0, 1.0 - math.sqrt(progress))
