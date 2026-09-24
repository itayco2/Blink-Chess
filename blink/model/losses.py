"""Training targets and losses (Recipe D, roots only in P1).

Policy: cross-entropy over all 1880 moves (unmasked) against
    (1 - alpha) * onehot(best) + alpha * softmax((W_i - W_1) / tau)
where the softmax runs over the best move and the available alternatives, and W is the win
probability of each PV from blink.board.value.

Value: cross-entropy against the HL-Gauss target of the best line's win probability (a torch port of
blink.board.value.hl_gauss, sigma = 0.75 / 128).
"""

import math

import torch
from torch.nn import functional as F

from blink.board import moves, value


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean over rows of -sum(target * log_softmax(logits)), computed in fp32."""
    return -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def soft_policy_target(
    move: torch.Tensor,
    alt_move: torch.Tensor,
    alt_valid: torch.Tensor,
    w_best: torch.Tensor,
    w_alt: torch.Tensor,
    alpha: float,
    tau: float,
) -> torch.Tensor:
    """[B, 1880] rows summing to one. Missing alternatives (alt_valid False) get no mass."""
    batch = move.shape[0]
    gaps = torch.cat([torch.zeros_like(w_best)[:, None], (w_alt - w_best[:, None]) / tau], dim=1)
    valid = torch.cat([torch.ones_like(alt_valid[:, :1]), alt_valid], dim=1)
    soft = torch.softmax(gaps.float().masked_fill(~valid, -math.inf), dim=1)
    index = torch.cat([move[:, None], torch.where(alt_valid, alt_move, move[:, None])], dim=1).long()
    target = torch.zeros(batch, moves.NUM_MOVES, dtype=torch.float32, device=move.device)
    target = target.scatter_add(1, index, alpha * soft)
    onehot = torch.full((batch, 1), 1.0 - alpha, dtype=torch.float32, device=move.device)
    return target.scatter_add(1, move[:, None].long(), onehot)


def hl_gauss_target(
    p: torch.Tensor, num_bins: int = value.NUM_BINS, sigma: float = value.SIGMA
) -> torch.Tensor:
    """[B, num_bins]: a Gaussian around each p integrated over each bin of [0, 1], renormalised."""
    edges = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64, device=p.device)
    z = (edges[None, :] - p.double()[:, None]) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(z))
    mass = cdf[:, 1:] - cdf[:, :-1]
    return (mass / mass.sum(dim=1, keepdim=True)).float()


def compute_losses(
    policy_logits: torch.Tensor, value_logits: torch.Tensor, batch, alpha: float, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """(policy CE, value CE) for a batch from blink.train.batch.make_batch."""
    policy_target = soft_policy_target(
        batch.move, batch.alt_move, batch.alt_valid, batch.w_best, batch.w_alt, alpha, tau
    )
    value_target = hl_gauss_target(batch.w_best)
    return soft_cross_entropy(policy_logits, policy_target), soft_cross_entropy(value_logits, value_target)
