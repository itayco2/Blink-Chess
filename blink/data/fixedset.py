"""The baseline ladder's fixed training set (plan P3): the first N root records of the train root shards.

Rungs 2, 3 and 4 (linear, MLP, s10m) train on exactly these positions, so the ladder adds one idea per
rung and never a data change. The shards are read in file-name order, front to back (sequential reads
only, PF14). The set is identified by its shard list, its size and a sha256 of its fen_hash sequence:
`blink baselines train` writes that description beside each model, and `blink data ladder10m` writes it
into the manifest of the pack s10m trains on. Torch-free, so the data commands can use it.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from blink.data.record import ROOT_DTYPE

V1_TRAIN_ROOTS = "train_r*.bin"
SKELETON_TRAIN = re.compile(r"^train_\d+\.bin$")
FIXED_SET_RULE = "the first N root records of the train root shards, in file-name order, front to back"


def root_shards(data_dir: Path) -> list[Path]:
    """v1 root shards (train_r000.bin ...) or the skeleton's train_000.bin ..., in name order."""
    data_dir = Path(data_dir)
    shards = sorted(data_dir.glob(V1_TRAIN_ROOTS))
    if not shards:
        shards = sorted(p for p in data_dir.glob("train_*.bin") if SKELETON_TRAIN.fullmatch(p.name))
    if not shards:
        raise FileNotFoundError(f"{data_dir} has no train root shards (train_r*.bin or train_<n>.bin)")
    return shards


def fen_hash_sha256(records: np.ndarray) -> str:
    """The set's identity: a sha256 of its fen_hash sequence, so order counts as well as content."""
    return hashlib.sha256(np.ascontiguousarray(records["fen_hash"]).tobytes()).hexdigest()


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
        "fen_hash_sha256": fen_hash_sha256(records),
    }
    return records, description
