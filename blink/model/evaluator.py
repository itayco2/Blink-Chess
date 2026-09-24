"""TorchEvaluator: a BlinkNet behind the play-side Evaluator protocol (blink.play.evaluator).

One evaluate() call is exactly one forward pass over all N rows, in fp32 (the play runtime), with the
value logits softmaxed into the 128-bin distribution the protocol expects.
"""

import numpy as np
import torch

from blink.board.encode import NUM_CODES
from blink.model.transformer import BlinkNet
from blink.play.evaluator import Evaluation


class TorchEvaluator:
    def __init__(self, model: BlinkNet, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()

    @torch.inference_mode()
    def evaluate(self, codes: np.ndarray) -> Evaluation:
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != 64:
            raise ValueError(f"codes must be [N, 64], got {codes.shape}")
        if codes.size and (codes.min() < 0 or codes.max() >= NUM_CODES):
            raise ValueError(f"square codes must be in 0..{NUM_CODES - 1}")
        tokens = torch.from_numpy(codes.astype(np.int64)).to(self.device)
        policy, value = self.model(tokens)
        probs = torch.softmax(value.float(), dim=-1)
        return Evaluation(
            policy_logits=policy.float().cpu().numpy(),
            value_probs=probs.cpu().numpy(),
        )
