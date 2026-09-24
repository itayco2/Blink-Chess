"""Exponential moving average of the weights with a warmup: decay_t = min(0.9999, (1 + t) / (10 + t)).

Early on the average follows the weights closely (decay 0.1 at t = 0), so the EMA is useful from
the first evaluation instead of being dominated by the random initialisation.
"""

import copy

import torch
from torch import nn

EMA_MAX = 0.9999


def ema_decay(step: int, max_decay: float = EMA_MAX) -> float:
    return min(max_decay, (1.0 + step) / (10.0 + step))


class Ema:
    def __init__(self, model: nn.Module, max_decay: float = EMA_MAX) -> None:
        self.max_decay = max_decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Blend the model's parameters in with weight 1 - decay(step); buffers are copied."""
        weight = 1.0 - ema_decay(step, self.max_decay)
        shadow = list(self.module.parameters())
        live = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(shadow, live, weight)
        for mine, theirs in zip(self.module.buffers(), model.buffers(), strict=True):
            mine.copy_(theirs)

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state)
