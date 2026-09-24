"""games10k top-1: the policy's agreement with Stockfish on real-game positions (arm a07's metric).

games10k.npy (blink.data.games10k) holds 10,000 unique positions from held-out Lichess games, which
the blocklist keeps out of training, each labelled with SF19's best move at 1M nodes: ROOT_DTYPE
records with one PV. Val is the eval-DB distribution by construction, so only this set can show what
rebalancing toward the game distribution buys. Top-1 is legal-masked and fp32, exactly as the val
top1 in blink.train.telemetry, so the two read on one scale. The legal masks are rebuilt from the
square codes once, when the set is loaded (about 2.3 s for 10,000 positions on the build machine).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from blink.board import encode, moves
from blink.train.telemetry import EVAL_CHUNK, legal_mask_from_codes

REQUIRED = ("board", "move")


@dataclass(frozen=True)
class GameSet:
    board: np.ndarray  # uint8 [N, 32], packed side-to-move codes
    best: np.ndarray  # uint16 [N], Stockfish's best move in the 1880 vocabulary
    legal: np.ndarray  # bool [N, 1880]

    def __post_init__(self) -> None:
        n = len(self.board)
        if self.board.shape != (n, 32) or len(self.best) != n or self.legal.shape != (n, moves.NUM_MOVES):
            raise ValueError(
                f"games10k arrays disagree: board {self.board.shape}, best {self.best.shape}, "
                f"legal {self.legal.shape}"
            )

    @property
    def n(self) -> int:
        return len(self.board)


def from_records(records: np.ndarray) -> GameSet:
    """A scorable set from ROOT_DTYPE records (its legal masks rebuilt from the square codes)."""
    boards = np.ascontiguousarray(records["board"], dtype=np.uint8).reshape(-1, 32)
    masks = [legal_mask_from_codes(codes) for codes in encode.unpack(boards)]
    legal = np.stack(masks) if masks else np.zeros((0, moves.NUM_MOVES), dtype=bool)
    return GameSet(boards, records["move"].astype(np.uint16), legal)


def load(path: Path) -> GameSet:
    records = np.load(path, allow_pickle=False)
    names = records.dtype.names or ()
    missing = [name for name in REQUIRED if name not in names]
    if missing:
        raise ValueError(f"{path} is not games10k records: it has no {', '.join(missing)} field")
    return from_records(records)


@torch.no_grad()
def legal_argmax(
    model: torch.nn.Module,
    games: GameSet,
    device: torch.device,
    chunk: int = EVAL_CHUNK,
    tick: Callable[[], None] | None = None,
) -> np.ndarray:
    """The policy's argmax over legal moves for every position, in bounded fp32 chunks (fp32 rows cost
    twice VAA's bf16 ones, so callers keep `chunk` at the val top1's EVAL_CHUNK or below)."""
    codes = encode.unpack(games.board)
    picks = []
    for start in range(0, games.n, chunk):
        tokens = torch.from_numpy(codes[start : start + chunk].astype(np.int64)).to(device)
        legal = torch.from_numpy(games.legal[start : start + chunk]).to(device)
        policy, _ = model(tokens)
        picks.append(policy.float().masked_fill(~legal, float("-inf")).argmax(-1).cpu())
        if tick is not None:
            tick()
    return torch.cat(picks).numpy() if picks else np.zeros(0, dtype=np.int64)


def top1(
    model: torch.nn.Module,
    games: GameSet,
    device: torch.device,
    chunk: int = EVAL_CHUNK,
    tick: Callable[[], None] | None = None,
) -> dict[str, float | int]:
    """{"top1", "n"}: the share of positions whose legal argmax is Stockfish's move."""
    was_training = model.training
    model.eval()
    try:
        picks = legal_argmax(model, games, device, chunk, tick)
    finally:
        model.train(was_training)
    n = games.n
    return {"top1": float((picks == games.best).mean()) if n else float("nan"), "n": n}
