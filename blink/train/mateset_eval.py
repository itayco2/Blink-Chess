"""Mate keeping on the mateset: does value mode keep a forced mate, and does it take a shortest one?

The mateset (blink.data.mateset) holds val roots where the side to move mates in 2 to 5 (mate_in [N]),
each with every legal child in the valprobe's arrays. A root is decided exactly as value mode plays
(blink.play.agents.value_decision): R1 lists every legal child in vocabulary order; R2 plays a
checkmating child first, lowest vocabulary index; R3 values a rule-draw child at 0.5; otherwise the
child worst for the opponent (1 - W) wins, and R4 breaks ties within epsilon by the root's policy
logit. R3's fifty-move and repetition draws and R5's clock guard need a game history and a clock,
which a stored position does not have: there they never fire, the same as for VAA. Forward passes
run as VAA's do (bf16 on CUDA, bounded chunks).

- shortest_mate: the chosen move is Stockfish's PV 1 or an alternative with the identical mate score
  (child_is_best), i.e. a move on a shortest mate line the label knows. Ties count as VAA counts them.
- mate_preserving: the chosen move keeps a forced mate for the mating side (a checkmate counts). That
  needs every legal move's mate status. The mateset the data step writes does not have it: it marks
  only the PV moves tied with PV 1 (child_is_best), drops the alternatives' own mate scores, and the
  eval DB labels at most 5 PVs per root. It is scored only from `child_mate_in` [M] int8 when a file
  has it: the mover's mate-in after that move, counting the move (1 = it mates, mate_in = a shortest
  line), 0 when no forced mate remains. Without it the metric is not scored rather than guessed.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from blink.board import encode
from blink.play import rules
from blink.train import vaa

CHILD_MATE_IN = "child_mate_in"


@dataclass(frozen=True)
class Mateset:
    probe: vaa.Probe
    mate_in: np.ndarray  # int8 [N], the root's mate distance for its side to move
    child_mate_in: np.ndarray | None = None  # int8 [M], see the module docstring; None when not labelled

    def __post_init__(self) -> None:
        if len(self.mate_in) != self.probe.n_roots:
            raise ValueError(f"mate_in has {len(self.mate_in)} rows for {self.probe.n_roots} roots")
        children = len(self.probe.child_board)
        if self.child_mate_in is not None and len(self.child_mate_in) != children:
            raise ValueError(f"{CHILD_MATE_IN} has {len(self.child_mate_in)} rows for {children} children")


def load(path: Path) -> Mateset:
    probe = vaa.load_probe(path)
    with np.load(path, allow_pickle=False) as data:
        if "mate_in" not in data.files:
            raise ValueError(f"{path} lacks mate_in: a valprobe, not a mateset")
        labels = data[CHILD_MATE_IN] if CHILD_MATE_IN in data.files else None
        return Mateset(probe, data["mate_in"], labels)


@torch.no_grad()
def root_policy_logits(
    model: torch.nn.Module,
    boards: np.ndarray,
    device: torch.device,
    chunk: int = vaa.VAA_CHUNK,
    tick: Callable[[], None] | None = None,
) -> np.ndarray:
    """float32 [N, 1880] policy logits of the packed roots (R4's tie-break), in VAA's precision."""
    codes = encode.unpack(boards)
    parts = []
    for start in range(0, len(codes), chunk):
        tokens = torch.from_numpy(codes[start : start + chunk].astype(np.int64)).to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            policy, _ = model(tokens)
        parts.append(policy.float().cpu())
        if tick is not None:
            tick()
    return torch.cat(parts).numpy() if parts else np.zeros((0, 0), dtype=np.float32)


def choose(
    values: np.ndarray, logits: np.ndarray, listed: np.ndarray, mates: np.ndarray, epsilon: float
) -> int:
    """One root's value-mode choice, as a position in its children's arrays.

    `values` are the mover's (1 - W, or 0.5 for a rule draw), `logits` the root policy logit of each
    child's move, `listed` each child's vocabulary index and `mates` which children are checkmate."""
    if mates.any():  # R2
        candidates = np.flatnonzero(mates)
        return int(candidates[np.argmin(listed[candidates])])
    order = np.argsort(listed, kind="stable")  # value mode lists the children in vocabulary order
    pick, _ = rules.tie_break(values[order], logits[order], epsilon)  # R4
    return int(order[pick])


def choices(
    w_child: np.ndarray, root_logits: np.ndarray, probe: vaa.Probe, epsilon: float = rules.DEFAULT_EPSILON
) -> np.ndarray:
    """The chosen child (an index into the probe's children) of every root; -1 for a root without one.

    A NaN value (weights that diverged) is never the best: it counts as -inf, since with a NaN best
    R4's tie set would be empty and the choice would raise instead of scoring what it can."""
    values = np.where(probe.child_terminal == vaa.RULE_DRAW, vaa.DRAW_VALUE, 1.0 - w_child)  # R3
    values = np.where(np.isnan(values), -np.inf, values)
    mates = probe.child_terminal == vaa.CHECKMATE
    chosen = np.full(probe.n_roots, -1, dtype=np.int64)
    for root in range(probe.n_roots):
        lo, hi = int(probe.child_offset[root]), int(probe.child_offset[root + 1])
        if hi == lo:
            continue
        listed = probe.child_move[lo:hi]
        logits = root_logits[root, listed.astype(np.int64)]
        chosen[root] = lo + choose(values[lo:hi], logits, listed, mates[lo:hi], epsilon)
    return chosen


def value_mode_choices(
    model: torch.nn.Module,
    probe: vaa.Probe,
    device: torch.device,
    chunk: int = vaa.VAA_CHUNK,
    epsilon: float = rules.DEFAULT_EPSILON,
    tick: Callable[[], None] | None = None,
) -> np.ndarray:
    """What value mode plays at every root: one forward pass per root and child, in eval mode."""
    was_training = model.training
    model.eval()
    try:
        w_child = vaa.child_win_probability(model, probe.child_board, device, chunk, tick)
        root_logits = root_policy_logits(model, probe.root_board, device, chunk, tick)
    finally:
        model.train(was_training)
    return choices(w_child, root_logits, probe, epsilon)


def rates(chosen: np.ndarray, mates: Mateset) -> dict[str, float | int]:
    """{"n", "shortest_mate"} of the chosen children, and "mate_preserving" when the file can say it."""
    picked = chosen[chosen >= 0]
    n = len(picked)
    if n == 0:
        return {"n": 0}
    probe = mates.probe
    out: dict[str, float | int] = {"n": n, "shortest_mate": float(probe.child_is_best[picked].mean())}
    if mates.child_mate_in is not None:
        keeps = (mates.child_mate_in[picked] > 0) | (probe.child_terminal[picked] == vaa.CHECKMATE)
        out["mate_preserving"] = float(keeps.mean())
    return out


def evaluate(
    model: torch.nn.Module,
    mates: Mateset,
    device: torch.device,
    chunk: int = vaa.VAA_CHUNK,
    epsilon: float = rules.DEFAULT_EPSILON,
    tick: Callable[[], None] | None = None,
) -> dict[str, float | int]:
    return rates(value_mode_choices(model, mates.probe, device, chunk, epsilon, tick), mates)
