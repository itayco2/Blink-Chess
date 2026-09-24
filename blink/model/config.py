"""Frozen model and training configs, read from a TOML file with [model] and [train] tables.

A config may start with `base = "recipe.toml"` (a path relative to the file): the base's tables are
read first and this file's keys override them one by one, so every size config inherits Recipe D and
states only what differs. This module is torch-free, so the dashboard, status and the torch-free CI
leg can read configs.
"""

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUTO = "auto"
HOUR_S = 3600.0


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    head_dim: int = 32
    ffn_mult: int = 2
    gab: bool = False  # GAB-lite attention bias (blink.model.gab), shared by every layer

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
    batch_size: int = 256  # rows per optimizer step: roots plus children
    micro_batch: int | str = 0  # rows per forward pass; 0 = the whole batch; "auto" = sized from free VRAM
    child_frac: float = 0.0  # share of each batch that is child positions (value loss only)
    rebalance: bool = True  # per-sample weights from the pack manifest, when it has them
    steps: int = 3000
    peak_lr: float = 1e-3
    warmup_steps: int = 200
    cooldown_frac: float = 0.2  # the last 20% of steps cool down as 1 - sqrt
    weight_decay: float = 0.1  # matrices only
    beta1: float = 0.9
    beta2: float = 0.95
    clip_norm: float | str = 1.0  # or "auto": 2 x the 95th percentile of the warmup gradient norms
    ema_max: float = 0.9999
    alpha: float = 0.5  # soft policy target mixing weight
    tau: float = 0.05  # soft policy target temperature over win-probability gaps
    lambda_v: float = 1.0
    seed: int = 0
    metrics_every: int = 50
    eval_every: int = 250
    val_size: int = 2000
    vaa_subset: int = 2000  # valprobe roots scored at every eval; the full valprobe runs at the checks
    vaa_checks: bool = False  # apply the P7 check rules at 5%, 25% and 50% of the steps
    vaa_sigma: float = 0.01  # the 3-seed VAA noise floor (a fraction: 0.01 is 1 point)
    vaa_reference: str = ""  # the run whose VAA the 5% check compares against (a run name)
    film: bool = False  # save the 21 film frames under film/
    ckpt_every_steps: int = 1000
    ckpt_every_minutes: float = 30.0  # 0 turns the wall-clock trigger off
    keep_last: int = 3
    keep_every_hours: float = 0.0  # also keep one checkpoint per this many hours; 0 = off
    heartbeat_s: float = 10.0

    def __post_init__(self) -> None:
        self._check_positive()
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError(f"warmup_steps must be in [0, steps), got {self.warmup_steps}")
        if not 0.0 < self.cooldown_frac <= 1.0:
            raise ValueError(f"cooldown_frac must be in (0, 1], got {self.cooldown_frac}")
        if not 0.0 <= self.alpha <= 1.0 or self.tau <= 0.0:
            raise ValueError(f"need 0 <= alpha <= 1 and tau > 0, got {self.alpha}, {self.tau}")
        if not 0.0 <= self.child_frac < 1.0:
            raise ValueError(f"child_frac must be in [0, 1), got {self.child_frac}")
        self._check_micro_batch()
        self._check_clip()

    def _check_positive(self) -> None:
        positive = (
            "batch_size",
            "steps",
            "metrics_every",
            "eval_every",
            "val_size",
            "vaa_subset",
            "ckpt_every_steps",
            "keep_last",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"train.{name} must be positive, got {getattr(self, name)}")
        for name in ("ckpt_every_minutes", "heartbeat_s", "vaa_sigma", "keep_every_hours"):
            if getattr(self, name) < 0:  # 0 is meaningful: off, and every step
                raise ValueError(f"train.{name} must not be negative, got {getattr(self, name)}")

    def _check_micro_batch(self) -> None:
        micro = self.micro_batch
        if micro == AUTO:
            return
        if not isinstance(micro, int) or isinstance(micro, bool) or micro < 0:
            raise ValueError(f"train.micro_batch must be 0, a positive int or 'auto', got {micro!r}")
        if micro and self.batch_size % micro:
            raise ValueError(f"train.micro_batch {micro} must divide batch_size {self.batch_size}")

    def _check_clip(self) -> None:
        clip = self.clip_norm
        if clip == AUTO:
            if self.warmup_steps <= 0:
                raise ValueError("clip_norm 'auto' measures the warmup gradient norms: set warmup_steps > 0")
            return
        if isinstance(clip, bool) or not isinstance(clip, int | float) or clip <= 0:
            raise ValueError(f"train.clip_norm must be a positive number or 'auto', got {clip!r}")

    @property
    def children_per_step(self) -> int:
        return int(round(self.child_frac * self.batch_size))

    @property
    def roots_per_step(self) -> int:
        return self.batch_size - self.children_per_step

    def accumulation(self, micro: int) -> int:
        """Forward passes per optimizer step at this micro-batch size."""
        return max(1, self.batch_size // micro)


def epoch_floor_samples_per_s(train_roots: float, child_frac: float, hours: float) -> float:
    """Samples/s a size must sustain to see every train root once in `hours` at this root/child mix."""
    return train_roots / (1.0 - child_frac) / (hours * HOUR_S)


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


def _read_tables(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, dict[str, Any]]:
    """The [model] and [train] tables of a file, with its base's keys underneath its own."""
    path = path.resolve()
    if path in seen:
        chain = " -> ".join(str(p) for p in (*seen, path))
        raise ValueError(f"config base cycle: {chain}")
    with open(path, "rb") as handle:
        data = tomllib.load(handle)
    unknown = sorted(set(data) - {"model", "train", "base"})
    if unknown:
        raise ValueError(f"{path}: unknown tables {', '.join(unknown)}; expected [model] and [train]")
    tables = {"model": {}, "train": {}}
    if "base" in data:
        tables = _read_tables(path.parent / str(data["base"]), (*seen, path))
    return {name: {**tables[name], **data.get(name, {})} for name in ("model", "train")}


def load_config(path: str | Path) -> TrainConfig:
    tables = _read_tables(Path(path))
    return config_from_dict({**tables["train"], "model": tables["model"]})
