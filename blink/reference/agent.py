"""DeepMindAgent: DeepMind's released ActionValueEngine as a Blink agent (plan E0 and E7).

It plays like searchless_chess's ActionValueEngine.play at temperature None:
  - one row per legal move, in action order: L rows in one network call;
  - the expected win of each move over the 128 return buckets;
  - the released repetition rule (a move after which a threefold can be claimed, or a fivefold stands,
    is worth 0.5), which reads the real move stack of the board it is given;
  - the first argmax.
There is no Stockfish fallback: the paper describes one (all top-5 moves above 99%), the released code
does not have it. Blink's rules R1-R5 do not apply either (no mate-in-one shortcut, no clock guard).
The Decision reports n_rows = L and n_calls = 1, so the UCI info line, the match PGN comments and
`blink audit no-search` count DeepMind's model like any other player (L <= L + 1).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chess
import numpy as np
import torch

from blink.play.agents import Decision
from blink.play.budget import DecisionRecord
from blink.reference import deepmind

MODE = "action-value"
REPETITION_RULE = "repetition"  # written to DecisionRecord.rule when the released rule scored a move 0.5

Sink = Callable[[DecisionRecord], None]


class Scorer(Protocol):
    def __call__(self, rows: np.ndarray) -> np.ndarray:
        """int64 [L, 79] rows -> float32 [L, 128] return-bucket log-probs at the last position."""
        ...


@dataclass(frozen=True)
class TorchScorer:
    """The ported model as a Scorer: one fp32 forward pass over all rows."""

    model: torch.nn.Module
    device: str = "cpu"

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        batch = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.int64)).to(self.device)
        with torch.inference_mode():
            return self.model(batch).float().cpu().numpy()


@dataclass(frozen=True)
class DeepMindAgent:
    scorer: Scorer
    sink: Sink | None = None
    name: str = "DM-9M"

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        ordered = deepmind.ordered_legal_moves(board)
        if not ordered:
            raise ValueError(f"no legal moves: the game is over at {board.fen()}")
        rows = deepmind.sequences(board, ordered)
        log_probs = np.asarray(self.scorer(rows))
        if log_probs.shape != (len(ordered), deepmind.NUM_RETURN_BUCKETS):
            raise ValueError(
                f"the scorer returned {log_probs.shape}, expected [{len(ordered)}, "
                f"{deepmind.NUM_RETURN_BUCKETS}] return-bucket log-probs"
            )
        draws = deepmind.repetition_draws(board, ordered)
        scores = np.where(draws, 0.5, deepmind.win_probabilities(log_probs))
        best = int(np.argmax(scores))
        record = DecisionRecord(
            game=game,
            ply=board.ply(),
            mode=MODE,
            n_legal=len(ordered),
            n_rows=len(rows),
            n_calls=1,
            rule=REPETITION_RULE if draws.any() else "",
        )
        if self.sink is not None:
            self.sink(record)
        return Decision(ordered[best], n_rows=len(rows), n_calls=1, win=float(scores[best]), record=record)


def load_agent(
    weights: Path, size: str, device: str = "cuda", sink: Sink | None = None, name: str | None = None
) -> DeepMindAgent:
    """A DeepMindAgent on the ported model loaded from a converted npz, fp32 on `device`."""
    model = deepmind.load_model(weights, deepmind.CONFIGS[size], device=device)
    return DeepMindAgent(TorchScorer(model, device), sink=sink, name=name or f"DM-{size}")
