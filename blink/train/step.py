"""One optimizer step's gradients: roots and children, in micro-batches, weighted, accumulated.

Each micro-batch holds a proportional slice of the step's roots and of its children in one forward
pass. Roots add policy and value loss, children value loss only, and every row's losses are
multiplied by its rebalancing weight. The sums are divided by the whole step's row counts before
backward, so the accumulated gradient equals the gradient of one pass over the full batch:

    loss = sum_roots(w * policy CE) / n_roots + lambda_v * sum_rows(w * value CE) / n_rows
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from blink.model.losses import mixed_losses
from blink.model.value_mapping import LICHESS
from blink.train.batch import make_batch, make_child_batch, take
from blink.train.source import StepData


@dataclass(frozen=True)
class Accumulated:
    policy: torch.Tensor  # weighted mean policy CE over the roots (detached, on the device)
    value: torch.Tensor  # weighted mean value CE over roots and children
    rows: int
    passes: int


def _bounds(n: int, parts: int) -> list[int]:
    return [n * i // parts for i in range(parts + 1)]


def micro_slices(n_roots: int, n_children: int, parts: int) -> list[tuple[slice, slice]]:
    """`parts` (root slice, child slice) pairs covering every row once, each with the step's mix."""
    parts = max(1, min(parts, n_roots))
    roots, children = _bounds(n_roots, parts), _bounds(n_children, parts)
    return [(slice(roots[i], roots[i + 1]), slice(children[i], children[i + 1])) for i in range(parts)]


def _weights(weight: np.ndarray | None, n: int, device: torch.device) -> torch.Tensor:
    if weight is None:
        return torch.ones(n, dtype=torch.float32, device=device)
    if len(weight) != n:
        raise ValueError(f"got {len(weight)} weights for {n} records")
    return torch.from_numpy(np.asarray(weight, dtype=np.float32)).to(device, non_blocking=True)


def accumulate(
    model: torch.nn.Module,
    data: StepData,
    device: torch.device,
    micro: int,
    alpha: float,
    tau: float,
    lambda_v: float,
    value_mapping: str = LICHESS,
) -> Accumulated:
    """Backward over the whole step in passes of at most `micro` rows (0: one pass). Grads accumulate.

    `value_mapping` (train.value_mapping) picks the value targets of roots and children alike.
    """
    roots = make_batch(data.roots, device, value_mapping)
    children = make_child_batch(data.children, device, value_mapping)
    root_w = _weights(data.root_weight, len(roots), device)
    child_w = _weights(data.child_weight, len(children), device)
    n_roots, rows = len(roots), len(roots) + len(children)
    slices = micro_slices(n_roots, len(children), math.ceil(rows / micro) if micro else 1)
    policy_sum = torch.zeros((), device=device)
    value_sum = torch.zeros((), device=device)
    for root_rows, child_rows in slices:
        part_roots, part_children = take(roots, root_rows), take(children, child_rows)
        tokens = torch.cat([part_roots.tokens, part_children.tokens])
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            policy, value = model(tokens)
        policy_ce, value_ce = mixed_losses(policy, value, part_roots, part_children.w, alpha, tau)
        policy_part = (policy_ce * root_w[root_rows]).sum()
        value_part = (value_ce * torch.cat([root_w[root_rows], child_w[child_rows]])).sum()
        (policy_part / n_roots + lambda_v * value_part / rows).backward()
        policy_sum = policy_sum + policy_part.detach()
        value_sum = value_sum + value_part.detach()
    return Accumulated(policy_sum / n_roots, value_sum / rows, rows, len(slices))
