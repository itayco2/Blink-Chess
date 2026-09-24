"""Frozen model and training configs, read from a TOML file with [model] and [train] tables.

This module is torch-free, so the dashboard, status and the torch-free CI leg can read configs.
"""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    head_dim: int = 32
    ffn_mult: int = 2
    gab: bool = False  # GAB-lite arrives after P1; the flag exists so configs can name it

    def __post_init__(self) -> None:
        for name in ("d_model", "n_layers", "n_heads", "head_dim", "ffn_mult"):
            if getattr(self, name) <= 0:
                raise ValueError(f"model.{name} must be positive, got {getattr(self, name)}")
        if self.n_heads * self.head_dim != self.d_model:
            raise ValueError(
                f"n_heads * head_dim must equal d_model: {self.n_heads} * {self.head_dim} != {self.d_model}"
            )


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    batch_size: int = 256
    steps: int = 3000
    peak_lr: float = 1e-3
    warmup_steps: int = 200
    cooldown_frac: float = 0.2  # the last 20% of steps cool down as 1 - sqrt
    weight_decay: float = 0.1  # matrices only
    beta1: float = 0.9
    beta2: float = 0.95
    clip_norm: float = 1.0
    ema_max: float = 0.9999
    alpha: float = 0.5  # soft policy target mixing weight
    tau: float = 0.05  # soft policy target temperature over win-probability gaps
    lambda_v: float = 1.0
    seed: int = 0
    metrics_every: int = 50
    eval_every: int = 250
    val_size: int = 2000
    ckpt_every_steps: int = 1000
    ckpt_every_minutes: float = 30.0  # 0 turns the wall-clock trigger off
    keep_last: int = 3
    heartbeat_s: float = 10.0

    def __post_init__(self) -> None:
        positive = ("batch_size", "steps", "metrics_every", "eval_every", "val_size", "keep_last")
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be positive, got {getattr(self, name)}")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError(f"warmup_steps must be in [0, steps), got {self.warmup_steps}")
        if not 0.0 < self.cooldown_frac <= 1.0:
            raise ValueError(f"cooldown_frac must be in (0, 1], got {self.cooldown_frac}")
        if not 0.0 <= self.alpha <= 1.0 or self.tau <= 0.0:
            raise ValueError(f"need 0 <= alpha <= 1 and tau > 0, got {self.alpha}, {self.tau}")


def config_to_dict(cfg: TrainConfig) -> dict[str, Any]:
    """Plain types only (safe for json and for torch.load(weights_only=True))."""
    return dataclasses.asdict(cfg)


def _checked(cls: type, values: dict[str, Any], table: str) -> dict[str, Any]:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown keys in [{table}]: {', '.join(unknown)}")
    return values


def config_from_dict(data: dict[str, Any]) -> TrainConfig:
    model = ModelConfig(**_checked(ModelConfig, dict(data.get("model", {})), "model"))
    train = {k: v for k, v in data.items() if k != "model"}
    return TrainConfig(model=model, **_checked(TrainConfig, train, "train"))


def load_config(path: str | Path) -> TrainConfig:
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    unknown = sorted(set(data) - {"model", "train"})
    if unknown:
        raise ValueError(f"{path}: unknown tables {', '.join(unknown)}; expected [model] and [train]")
    return config_from_dict({**data.get("train", {}), "model": data.get("model", {})})
