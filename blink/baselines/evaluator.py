"""BaselineEvaluator: a linear or MLP baseline behind the play-side Evaluator protocol.

The value head is a two-hot distribution over the 128 bins whose mean is exactly the model's win%
(clipped to the outer bin centres), and the policy logits are all zeros, so a baseline can only play
value mode ("one look per move"). Every baseline, material included, plays through the same
blink.play.agents.ValueAgent as Blink, with the same rules R1-R5 and the same EvalBudget.

The forward pass is torch (the model builds its bits with models.features_torch); the numpy
blink.baselines.features is the reference those bits are tested against, and its checked_codes is
the input check here.
"""

from pathlib import Path

import numpy as np
import torch

from blink import paths
from blink.baselines import features, models
from blink.board import moves, value
from blink.play.agents import ValueAgent
from blink.play.evaluator import Evaluation
from blink.play.oracles import MaterialEvaluator

AGENT_NAMES = {"material": "Material", "linear": "Linear", "mlp": "MLP"}


def two_hot(win: np.ndarray) -> np.ndarray:
    """float32 [N, 128]: mass split between the two bin centres around each win probability."""
    position = np.clip(np.asarray(win, dtype=np.float64) * value.NUM_BINS - 0.5, 0.0, value.NUM_BINS - 1)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, value.NUM_BINS - 1)
    upper = position - low
    probs = np.zeros((len(position), value.NUM_BINS), dtype=np.float64)
    rows = np.arange(len(position))
    np.add.at(probs, (rows, low), 1.0 - upper)
    np.add.at(probs, (rows, high), upper)
    return probs.astype(np.float32)


class BaselineEvaluator:
    """One evaluate() call is one forward pass of the baseline over all N rows (fp32)."""

    def __init__(self, model: torch.nn.Module, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def win_probability(self, codes: np.ndarray) -> np.ndarray:
        tokens = torch.from_numpy(features.checked_codes(codes)).to(self.device)
        return torch.sigmoid(models.logits_from_codes(self.model, tokens).float()).cpu().numpy()

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        win = self.win_probability(codes)
        policy = np.zeros((len(win), moves.NUM_MOVES), dtype=np.float32)
        return Evaluation(policy_logits=policy, value_probs=two_hot(win))


def default_path(kind: str) -> Path:
    """Where `blink baselines train --kind <kind>` writes its weights."""
    return paths.home() / "runs" / f"baseline-{kind}" / "model.pt"


def load_baseline(path: Path, device: str = "cpu") -> tuple[BaselineEvaluator, str]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"no baseline weights at {path}; run `blink baselines train` first")
    model, kind, _ = models.load(path, device=device)
    return BaselineEvaluator(model, device), kind


def baseline_agent(selector: str, device: str = "cpu") -> ValueAgent:
    """material | linear | mlp (BLINK_HOME/runs/baseline-<kind>/model.pt) | a path to a baseline .pt."""
    if selector == "material":
        return ValueAgent(MaterialEvaluator(), name=AGENT_NAMES["material"])
    path = default_path(selector) if selector in models.KINDS else Path(selector)
    evaluator, kind = load_baseline(path, device)
    return ValueAgent(evaluator, name=AGENT_NAMES[kind])
