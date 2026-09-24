"""Evaluators for export work: an onnxruntime CPU session and a plain torch module.

Both return blink.play.evaluator.Evaluation, so golden positions can come from either one.
"""

from pathlib import Path

import numpy as np

from blink.play.evaluator import Evaluation


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - logits.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def _evaluation(policy_logits: np.ndarray, value_logits: np.ndarray) -> Evaluation:
    value_probs = softmax(value_logits.astype(np.float64)).astype(np.float32)
    return Evaluation(policy_logits=policy_logits.astype(np.float32), value_probs=value_probs)


class OnnxEvaluator:
    """One onnxruntime CPU session over an exported file. One evaluate() is one session run."""

    def __init__(self, path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    def run(self, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feed = {"tokens": tokens.astype(np.int64)}
        policy, values = self.session.run(["policy_logits", "value_logits"], feed)
        return policy, values

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        return _evaluation(*self.run(codes))


class ModuleEvaluator:
    """A torch module on CPU, called under no_grad."""

    def __init__(self, module) -> None:
        self.module = module.eval()

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        import torch

        with torch.no_grad():
            policy, values = self.module(torch.from_numpy(codes.astype(np.int64)))
        return _evaluation(policy.numpy(), values.numpy())
