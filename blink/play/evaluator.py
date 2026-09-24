"""The one interface between a trained network and everything that plays or evaluates with it.

An Evaluator takes square codes for N positions (each from its own side to move's view) and returns,
for each, the 1880 policy logits and the 128-bin value distribution. Play code never sees torch,
so the same agents run on a torch module, an ONNX session or a hand-written test oracle.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from blink.board import moves, value


@dataclass(frozen=True)
class Evaluation:
    policy_logits: np.ndarray  # float32 [N, 1880]
    value_probs: np.ndarray  # float32 [N, 128], each row sums to 1

    def __post_init__(self) -> None:
        n = self.policy_logits.shape[0]
        if self.policy_logits.shape != (n, moves.NUM_MOVES):
            raise ValueError(f"policy_logits must be [N, {moves.NUM_MOVES}], got {self.policy_logits.shape}")
        if self.value_probs.shape != (n, value.NUM_BINS):
            raise ValueError(f"value_probs must be [N, {value.NUM_BINS}], got {self.value_probs.shape}")

    def win_probability(self) -> np.ndarray:
        """Expected win probability for the side to move of each row: the mean over bin centres."""
        return self.value_probs @ value.BIN_CENTERS


class Evaluator(Protocol):
    def evaluate(self, codes: np.ndarray) -> Evaluation:
        """codes: uint8 [N, 64] square codes from blink.board.encode. One call is one network call."""
        ...
