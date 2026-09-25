"""Model selectors -> weights -> an Evaluator.

    run:<name>          the latest checkpoint of BLINK_HOME/runs/<name>/, raw weights
    run:<name>:ema      the same checkpoint's EMA weights
    ship                BLINK_HOME/ship/blink.pt
    release:<tag>       BLINK_HOME/ship/releases/<tag>/blink.pt (a downloaded release asset)
    <path>.pt           that file
    <path>.pt:ema       that file's EMA weights (a run selector pinned to the checkpoint it resolved to)

A weights file is either a full training checkpoint ({"config": TrainConfig dict, "model", "ema", ...})
or a slim file ({"config": TrainConfig or ModelConfig dict, "model"}). Files load with
weights_only=True, so a weights file can never run code.

load_evaluator plays fp32 with no compile unless asked (blink.play.fastmode): the mode is checked
before any weights load, and a compiled evaluator is warmed up before it is returned.
"""

from pathlib import Path

import torch

from blink import paths
from blink.model.config import ModelConfig, config_from_dict
from blink.model.evaluator import TorchEvaluator, play_evaluator
from blink.model.transformer import BlinkNet
from blink.play import fastmode
from blink.train.checkpoint import latest_checkpoint, load_checkpoint
from blink.train.status import valid_run_name

WEIGHTS_FILE = "blink.pt"
EMA_SUFFIX = ":ema"


def _resolve_run(rest: str) -> tuple[Path, str]:
    name, _, suffix = rest.partition(":")
    if not valid_run_name(name) or suffix not in ("", "ema"):
        raise ValueError(f"bad run selector 'run:{rest}' (use run:<name> or run:<name>:ema)")
    run_dir = paths.home() / "runs" / name
    path = latest_checkpoint(run_dir)
    if path is None:
        raise FileNotFoundError(f"run {name!r} has no checkpoint in {run_dir}")
    return path, "ema" if suffix else "model"


def resolve_selector(selector: str) -> tuple[Path, str]:
    """(weights file, which weights: "model" or "ema"). ValueError for a malformed selector."""
    kind, _, rest = selector.partition(":")
    if kind == "run":
        return _resolve_run(rest)
    if selector == "ship":
        return paths.home() / "ship" / WEIGHTS_FILE, "model"
    if kind == "release":
        if not valid_run_name(rest):
            raise ValueError(f"bad release tag {rest!r}")
        return paths.home() / "ship" / "releases" / rest / WEIGHTS_FILE, "model"
    if selector.endswith(".pt"):
        return Path(selector), "model"
    if selector.endswith(".pt" + EMA_SUFFIX):
        return Path(selector[: -len(EMA_SUFFIX)]), "ema"
    raise ValueError(
        f"unknown model selector {selector!r}: run:<name>[:ema] | ship | release:<tag> | <path>.pt"
    )


def pinned_selector(path: Path, which: str) -> str:
    """The selector that loads exactly these weights: resolve_selector's (path, which) back, so a run
    selector resolved once names one checkpoint however many land after it."""
    return str(path) + (EMA_SUFFIX if which == "ema" else "")


def _model_config(config: dict) -> ModelConfig:
    if isinstance(config.get("model"), dict):
        return config_from_dict(config).model
    return ModelConfig(**config)


def load_model(selector: str, device: str = "cuda") -> BlinkNet:
    path, weights = resolve_selector(selector)
    if not path.is_file():
        raise FileNotFoundError(f"{selector}: no weights file at {path}")
    state = load_checkpoint(path, map_location="cpu")
    if weights not in state:
        raise ValueError(f"{path} has no {weights!r} weights")
    model = BlinkNet(_model_config(state["config"]))
    model.load_state_dict(state[weights])
    return model.to(torch.device(device)).eval()


def load_evaluator(
    selector: str,
    device: str = "cuda",
    precision: str = fastmode.DEFAULT_PRECISION,
    compile: bool = False,
) -> TorchEvaluator:
    fastmode.check(precision, device)
    return play_evaluator(load_model(selector, device), device, precision=precision, compile=compile)
