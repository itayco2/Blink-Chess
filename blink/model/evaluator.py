"""TorchEvaluator: a BlinkNet behind the play-side Evaluator protocol (blink.play.evaluator).

One evaluate() call is exactly one forward pass over all N rows, with the value logits softmaxed into
the 128-bin distribution the protocol expects. The default is the play runtime, fp32 with no compile,
which calls the model exactly as it always has. Two opt-in modes (blink.play.fastmode) cut latency:

- precision="bf16" (CUDA only; refused on the CPU): the trunk runs under bf16 autocast, its output is
  cast to fp32 and BOTH heads run in fp32 with autocast off. With the heads under autocast too, a
  trained M model's policy top-1 agreed with fp32 only 96.6% of the time (its value choice 100%): a
  bf16 logit keeps 8 mantissa bits, so close logits collide. The trunk still sees every row once and
  each head once, so it is still one forward pass.
- compile=True: the trunk is wrapped with torch.compile(dynamic=True) for this evaluator only; the
  model object, its trunk attribute and its weights are untouched. dynamic=True still specialises a
  batch of 1, so warm_up() compiles the one-look (1 row) and value-mode (2+ rows) graphs before play.
"""

import numpy as np
import torch

from blink.board.encode import NUM_CODES
from blink.model.transformer import BlinkNet
from blink.play import fastmode
from blink.play.evaluator import Evaluation

WARM_ROWS = (1, 2)  # a batch of 1 gets its own graph; one symbolic graph serves every N >= 2


class TorchEvaluator:
    def __init__(
        self,
        model: BlinkNet,
        device: str | torch.device = "cpu",
        precision: str = fastmode.DEFAULT_PRECISION,
        compile: bool = False,
    ) -> None:
        self.device = torch.device(device)
        fastmode.check(precision, self.device.type)
        self.precision = precision
        self.compile = bool(compile)
        self.model = model.to(self.device).eval()
        # the default path touches nothing but model(tokens), so any module with that forward still plays
        self._compiled_trunk = torch.compile(self.model.trunk, dynamic=True) if self.compile else None

    @property
    def is_default(self) -> bool:
        return fastmode.is_default(self.precision, self.compile)

    def _forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_default:
            return self.model(tokens)
        trunk = self.model.trunk if self._compiled_trunk is None else self._compiled_trunk
        bf16 = self.precision == "bf16"
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=bf16):
            hidden = trunk(tokens)
        hidden = hidden.float()
        with torch.autocast(self.device.type, enabled=False):
            return self.model.policy(hidden), self.model.value(hidden)

    @torch.inference_mode()
    def evaluate(self, codes: np.ndarray) -> Evaluation:
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != 64:
            raise ValueError(f"codes must be [N, 64], got {codes.shape}")
        if codes.size and (codes.min() < 0 or codes.max() >= NUM_CODES):
            raise ValueError(f"square codes must be in 0..{NUM_CODES - 1}")
        tokens = torch.from_numpy(codes.astype(np.int64)).to(self.device)
        policy, value = self._forward(tokens)
        probs = torch.softmax(value.float(), dim=-1)
        return Evaluation(
            policy_logits=policy.float().cpu().numpy(),
            value_probs=probs.cpu().numpy(),
        )

    def warm_up(self) -> None:
        """Run empty boards at 1 and 2 rows once, so a compiled trunk has both graphs before the first
        decision (a recompile under R5's clock guard would cost seconds). Startup only: no decision."""
        for rows in WARM_ROWS:
            self.evaluate(np.zeros((rows, 64), dtype=np.uint8))
