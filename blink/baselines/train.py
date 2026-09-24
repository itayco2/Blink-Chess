"""`blink baselines train`: the linear and MLP rungs, soft BCE on win%, early stop on val, <= 5 epochs.

The label is the Lichess win probability of the root's best line (cp or mate, side to move's view).
The training set is the ladder's shared fixed set: the first N root records of the train shards read
in file-name order, front to back (sequential reads only, PF14). Its shard list, size and a sha256 of
its fen_hash sequence are written beside the model, so s10m (rung 4) can prove it saw the same set.
"""

import copy
import dataclasses
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from blink.baselines import models
from blink.board.value import win_probability_array
from blink.data.record import ROOT_DTYPE
from blink.train.atomic import write_text_atomic

MAX_EPOCHS = 5
DEFAULT_LR = {"linear": 3e-3, "mlp": 1e-3}
EVAL_CHUNK = 65536
V1_TRAIN_ROOTS = "train_r*.bin"
SKELETON_TRAIN = re.compile(r"^train_\d+\.bin$")
VAL_FILES = ("val_roots.bin", "val.bin")
FIXED_SET_RULE = "the first N root records of the train root shards, in file-name order, front to back"

Log = Callable[[str], None]


@dataclass(frozen=True)
class BaselineConfig:
    kind: str
    positions: int = 10_000_000
    epochs: int = MAX_EPOCHS
    batch_size: int = 1024
    lr: float | None = None  # None: DEFAULT_LR[kind]
    weight_decay: float = 0.0
    patience: int = 1  # epochs without a val improvement before stopping
    seed: int = 0
    val_positions: int = 500_000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.kind not in models.KINDS:
            raise ValueError(f"kind must be one of {models.KINDS}, got {self.kind!r}")
        if not 1 <= self.epochs <= MAX_EPOCHS:
            raise ValueError(f"epochs must be 1 to at most {MAX_EPOCHS}, got {self.epochs}")
        for name in ("positions", "batch_size", "patience", "val_positions"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def learning_rate(self) -> float:
        return DEFAULT_LR[self.kind] if self.lr is None else self.lr


@dataclass(frozen=True)
class FitResult:
    model: torch.nn.Module
    initial: dict[str, float]
    history: tuple[dict[str, Any], ...]
    best_epoch: int
    val_mae: float
    val_bce: float


def root_shards(data_dir: Path) -> list[Path]:
    """v1 root shards (train_r000.bin ...) or the skeleton's train_000.bin ..., in name order."""
    data_dir = Path(data_dir)
    shards = sorted(data_dir.glob(V1_TRAIN_ROOTS))
    if not shards:
        shards = sorted(p for p in data_dir.glob("train_*.bin") if SKELETON_TRAIN.fullmatch(p.name))
    if not shards:
        raise FileNotFoundError(f"{data_dir} has no train root shards (train_r*.bin or train_<n>.bin)")
    return shards


def val_file(data_dir: Path) -> Path:
    for name in VAL_FILES:
        if (Path(data_dir) / name).is_file():
            return Path(data_dir) / name
    raise FileNotFoundError(f"{data_dir} has no val roots ({' or '.join(VAL_FILES)})")


def fixed_train_set(data_dir: Path, positions: int) -> tuple[np.ndarray, dict[str, Any]]:
    """The ladder's shared training set and a description that identifies it exactly."""
    parts, used, need = [], [], positions
    for shard in root_shards(data_dir):
        if need == 0:
            break
        part = np.fromfile(shard, dtype=ROOT_DTYPE, count=need)
        parts.append(part)
        used.append(shard.name)
        need -= len(part)
    if need:
        found = positions - need
        raise ValueError(
            f"{data_dir} holds only {found:,} train roots, fewer than the {positions:,} asked for"
        )
    records = np.concatenate(parts)
    description = {
        "rule": FIXED_SET_RULE,
        "dir": str(data_dir),
        "shards": used,
        "positions": positions,
        "fen_hash_sha256": hashlib.sha256(records["fen_hash"].tobytes()).hexdigest(),
    }
    return records, description


def labels(records: np.ndarray) -> np.ndarray:
    """The mover's win probability of the best line, float32."""
    return win_probability_array(records["cp"], records["mate"]).astype(np.float32)


@torch.no_grad()
def evaluate(model: torch.nn.Module, boards: torch.Tensor, win: torch.Tensor, device: torch.device) -> dict:
    """Soft BCE and win% MAE over a packed-board set, in chunks."""
    was_training = model.training
    model.eval()
    bce = mae = 0.0
    for start in range(0, len(boards), EVAL_CHUNK):
        codes = models.unpack_torch(boards[start : start + EVAL_CHUNK].to(device))
        target = win[start : start + EVAL_CHUNK].to(device)
        logits = models.logits_from_codes(model, codes).float()
        bce += F.binary_cross_entropy_with_logits(logits, target, reduction="sum").item()
        mae += (torch.sigmoid(logits) - target).abs().sum().item()
    model.train(was_training)
    n = max(len(boards), 1)
    return {"val_bce": bce / n, "val_mae": mae / n}


def _train_epoch(model, optimizer, boards, win, cfg: BaselineConfig, epoch: int, device) -> float:
    model.train()
    order = torch.randperm(len(boards), generator=torch.Generator().manual_seed(cfg.seed * 1000 + epoch))
    total = torch.zeros((), device=device)
    for start in range(0, len(order), cfg.batch_size):
        index = order[start : start + cfg.batch_size]
        codes = models.unpack_torch(boards[index].to(device, non_blocking=True))
        target = win[index].to(device, non_blocking=True)
        loss = F.binary_cross_entropy_with_logits(models.logits_from_codes(model, codes), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.detach() * len(index)
    return total.item() / len(order)


def _tensors(boards: np.ndarray, win: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(np.ascontiguousarray(boards)), torch.from_numpy(np.ascontiguousarray(win))


def fit(cfg: BaselineConfig, train_boards, train_win, val_boards, val_win, log: Log) -> FitResult:
    """Train until val BCE stops improving (patience) or cfg.epochs; keep the best epoch's weights."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = models.build(cfg.kind).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    boards, win = _tensors(train_boards, train_win)
    val = _tensors(val_boards, val_win)
    initial = evaluate(model, *val, device)
    history, best, stale = [], None, 0
    for epoch in range(1, cfg.epochs + 1):
        started = time.perf_counter()
        train_bce = _train_epoch(model, optimizer, boards, win, cfg, epoch, device)
        row = {"epoch": epoch, "train_bce": train_bce, **evaluate(model, *val, device)}
        row["seconds"] = time.perf_counter() - started
        history.append(row)
        log(
            f"{cfg.kind} epoch {epoch}: train BCE {train_bce:.5f}, val BCE {row['val_bce']:.5f}, "
            f"val win% MAE {row['val_mae']:.4f} ({row['seconds']:.1f} s)"
        )
        if best is None or row["val_bce"] < best[0]["val_bce"]:
            best, stale = (row, copy.deepcopy(model.state_dict())), 0
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"{cfg.kind}: val BCE did not improve for {stale} epoch(s), stopping")
                break
    row, state = best
    model.load_state_dict(state)
    return FitResult(model.eval(), initial, tuple(history), row["epoch"], row["val_mae"], row["val_bce"])


def train_baseline(cfg: BaselineConfig, data_dir: Path, out_dir: Path, log: Log = print) -> dict[str, Any]:
    """Train one rung on the fixed set; write out_dir/model.pt and out_dir/metrics.json."""
    started = time.perf_counter()
    records, train_set = fixed_train_set(data_dir, cfg.positions)
    val_path = val_file(data_dir)
    val = np.fromfile(val_path, dtype=ROOT_DTYPE, count=cfg.val_positions)
    if len(val) == 0:
        raise ValueError(f"{val_path} holds no records")
    log(
        f"{cfg.kind}: {len(records):,} train positions from {len(train_set['shards'])} shard(s), "
        f"{len(val):,} val positions from {val_path.name}"
    )
    result = fit(cfg, records["board"], labels(records), val["board"], labels(val), log)
    metrics = {
        "kind": cfg.kind,
        "config": dataclasses.asdict(cfg),
        "train_set": train_set,
        "val": {"file": val_path.name, "positions": len(val)},
        "initial": result.initial,
        "history": list(result.history),
        "best_epoch": result.best_epoch,
        "val_mae": result.val_mae,
        "val_bce": result.val_bce,
        "seconds": time.perf_counter() - started,
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {k: metrics[k] for k in ("val_mae", "val_bce", "best_epoch", "train_set")}
    models.save(out_dir / "model.pt", result.model, cfg.kind, meta)
    write_text_atomic(out_dir / "metrics.json", json.dumps(metrics, indent=2) + "\n")
    log(f"{cfg.kind}: best epoch {result.best_epoch}, val win% MAE {result.val_mae:.4f} -> {out_dir}")
    return metrics
