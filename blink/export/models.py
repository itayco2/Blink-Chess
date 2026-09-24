"""Model selectors for export: `stand-in`, a path to an .onnx file, or the train area's selectors.

The train area owns `blink.model.loading` (selector = run:<name>[:ema] | ship | release:<tag> | <.pt path>).
Export needs the torch module itself, which `load_model(selector, device)` returns.
"""

import importlib
from pathlib import Path

from blink.play.evaluator import Evaluator

STAND_IN = "stand-in"
LOADING_MODULE = "blink.model.loading"
MODEL_FILE = "model.onnx"


class ModelUnavailable(RuntimeError):
    pass


def _loading():
    try:
        return importlib.import_module(LOADING_MODULE)
    except ModuleNotFoundError as exc:
        raise ModelUnavailable(
            f"{LOADING_MODULE} is not importable ({exc}); only '{STAND_IN}' and .onnx paths work here"
        ) from exc


def onnx_path(selector: str) -> Path | None:
    """The .onnx file a selector names (a file, or a directory holding model.onnx), else None."""
    path = Path(selector)
    if path.suffix.lower() == ".onnx":
        return path
    if path.is_dir() and (path / MODEL_FILE).is_file():
        return path / MODEL_FILE
    return None


def load_module(selector: str):
    """The torch module a selector names, on CPU in eval mode."""
    if selector == STAND_IN:
        from blink.export import standin

        return standin.build(seed=0)
    return _loading().load_model(selector, device="cpu").eval()


def load_evaluator(selector: str) -> Evaluator:
    """An Evaluator on CPU for golden positions: an ONNX session, the stand-in, or the train area's."""
    from blink.export import evaluators

    path = onnx_path(selector)
    if path is not None:
        if not path.is_file():
            raise ModelUnavailable(f"no ONNX file at {path}")
        return evaluators.OnnxEvaluator(path)
    if selector == STAND_IN:
        return evaluators.ModuleEvaluator(load_module(selector))
    return _loading().load_evaluator(selector, device="cpu")


def slug(selector: str) -> str:
    """A directory name for a selector's export: run:skeleton:ema -> skeleton-ema."""
    body = selector.removeprefix("run:").removeprefix("release:")
    body = Path(body).stem if body.lower().endswith(".pt") else body
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in body.replace(":", "-"))
    return cleaned.strip("-.") or "model"
