"""The data side of `blink train`: which records feed the trainer, and the world they belong to.

--data DIR reads a pack directory. The v1 layout (interface 2) has root shards train_r000..r255.bin,
child shards train_c000..c255.bin, val_roots.bin and manifest.json, whose "rebalance" table holds the
48 per-bucket weights applied through blink.data.rebalance.weights_for. The P1 skeleton layout has
train_000.bin shards and val.bin, roots only. Each optimizer step draws cfg.roots_per_step roots and
cfg.children_per_step children from two ShardLoaders started at that step, so a resume is exact.

--source-raw PATH parses the first --max-lines lines of a raw eval-DB file (roots only).

The checks also score two held-out sets when they exist (blink.train.checksets): games10k from
--games10k, else BLINK_HOME/data/games10k.npy, and a pack's own mateset.npz (a raw source has none).
"""

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from blink import paths
from blink.data.record import CHILD_DTYPE, ROOT_DTYPE
from blink.model.config import TrainConfig
from blink.train.world import NO_BLOCKLIST, SPLIT_RULE, world_id

DEFAULT_MAX_LINES = 1_000_000
DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))
SKELETON_SHARD = re.compile(r"^train_\d+\.bin$")
CHILD_SEED_OFFSET = 1  # the child stream shuffles independently of the root stream


class CommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class DataPlan:
    source: Any  # blink.train.source.BatchSource
    val: np.ndarray | None
    world: str
    description: dict[str, Any]
    probe: Any = None  # blink.train.vaa.Probe | None
    games10k: Path | None = None  # scored at the checks when the file exists
    mateset: Path | None = None


def _probe(args: argparse.Namespace, root: Path | None):
    from blink.train import vaa

    path = Path(args.valprobe) if args.valprobe else None
    if path is None and root is not None and (root / "valprobe.npz").is_file():
        path = root / "valprobe.npz"
    if path is None:
        return None
    probe = vaa.load_probe(path)
    print(f"valprobe: {probe.n_roots:,} roots, {len(probe.child_board):,} children from {path}", flush=True)
    return probe


def _games10k(args: argparse.Namespace) -> Path:
    from blink.data import games10k

    chosen = getattr(args, "games10k", None)
    return Path(chosen) if chosen else games10k.default_path()


def _mateset(root: Path) -> Path:
    from blink.data import mateset

    return root / mateset.OUTPUT


def _require_roots_only(cfg: TrainConfig, where: str) -> None:
    if cfg.children_per_step:
        raise CommandError(
            f"the config puts {cfg.children_per_step} children in every step but {where} has no child "
            "records; use a pack with train_c*.bin shards or a config with child_frac = 0"
        )


def _raw_plan(args: argparse.Namespace, cfg: TrainConfig) -> DataPlan:
    from blink.train import rawsource
    from blink.train.source import InMemorySource

    _require_roots_only(cfg, "--source-raw")
    raw = rawsource.load_or_build(
        Path(args.source_raw), args.max_lines, paths.home() / "data" / "raw-cache", args.workers
    )
    train, val, _ = rawsource.split(raw.records)
    if len(train) < cfg.batch_size:
        raise CommandError(f"only {len(train)} train records for batch size {cfg.batch_size}")
    description = {
        "source": "raw",
        "raw": str(args.source_raw),
        "max_lines": args.max_lines,
        "records": len(raw.records),
        "train": len(train),
        "val": len(val),
        "cache": str(raw.cache),
        "sha1": raw.sha1,
    }
    print(f"source-raw: {len(train):,} train, {len(val):,} val records (hash split {SPLIT_RULE})", flush=True)
    batches = InMemorySource(train, cfg.batch_size, cfg.seed).batches
    world = world_id(f"raw:{raw.sha1}")
    probe = _probe(args, None)
    return DataPlan(batches, val if len(val) else None, world, description, probe, _games10k(args))


def _weigher(manifest: dict[str, Any], cfg: TrainConfig):
    table = manifest.get("rebalance")
    if not cfg.rebalance or not table:
        print("rebalance: off" if not cfg.rebalance else "rebalance: no weights in the manifest", flush=True)
        return None
    try:
        from blink.data import rebalance
    except ImportError as exc:
        raise CommandError(
            f"the manifest has rebalancing weights but blink.data.rebalance is missing: {exc}"
        ) from exc
    weights = np.asarray(table["weights"], dtype=np.float32)
    if len(weights) != table.get("buckets", len(weights)):
        raise CommandError(f"manifest rebalance: {len(weights)} weights for {table['buckets']} buckets")
    print(
        f"rebalance: {len(weights)} bucket weights in [{weights.min():.2f}, {weights.max():.2f}]", flush=True
    )
    return partial(rebalance.weights_for, weights=weights)


def _layout(root: Path) -> tuple[list[Path], list[Path], Path]:
    """(root shards, child shards, val roots file) of a v1 or a skeleton pack."""
    roots = sorted(root.glob("train_r*.bin"))
    if roots:
        return roots, sorted(root.glob("train_c*.bin")), root / "val_roots.bin"
    skeleton = sorted(p for p in root.glob("train_*.bin") if SKELETON_SHARD.match(p.name))
    return skeleton, [], root / "val.bin"


def pack_world(manifest_bytes: bytes, manifest: dict[str, Any]) -> str:
    """WORLD of a pack: sha1(contract, manifest sha, blocklist sha, split rule and salt)[:12].

    A v1 manifest names its blocklist as {"sha256": ...} and its test_grouped salt under "grouped";
    the P1 skeleton manifest has neither (blocklist null), so its world is unchanged."""
    if manifest.get("world"):
        return manifest["world"]
    blocklist = manifest.get("blocklist")
    blocklist_sha = blocklist.get("sha256", NO_BLOCKLIST) if isinstance(blocklist, dict) else NO_BLOCKLIST
    split_rule = manifest.get("split_rule", SPLIT_RULE)
    salt = (manifest.get("grouped") or {}).get("salt")
    if salt is not None:
        split_rule = f"{split_rule}; grouped salt {salt}"
    return world_id(hashlib.sha1(manifest_bytes).hexdigest(), blocklist_sha, split_rule)


def _shard_plan(args: argparse.Namespace, cfg: TrainConfig) -> DataPlan:
    from blink.train.source import mixed_source, shard_source

    root = Path(args.data)
    manifest_path = root / "manifest.json"
    roots, children, val_path = _layout(root)
    if not manifest_path.is_file() or not roots:
        raise CommandError(f"{root} needs manifest.json and train_r*.bin (or train_*.bin) shards")
    if not children:
        _require_roots_only(cfg, str(root))
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    val = np.fromfile(val_path, dtype=ROOT_DTYPE, count=cfg.val_size) if val_path.is_file() else None
    root_stream = shard_source(roots, cfg.roots_per_step, cfg.seed)
    child_stream = None
    if cfg.children_per_step:
        child_stream = shard_source(
            children, cfg.children_per_step, cfg.seed + CHILD_SEED_OFFSET, CHILD_DTYPE
        )
    description = {
        "source": "shards",
        "dir": str(root),
        "train_shards": len(roots),
        "child_shards": len(children),
        "roots_per_step": cfg.roots_per_step,
        "children_per_step": cfg.children_per_step,
        "rebalance": bool(cfg.rebalance and manifest.get("rebalance")),
        "val_records": 0 if val is None else len(val),
    }
    source = mixed_source(root_stream, child_stream, _weigher(manifest, cfg))
    world = pack_world(manifest_bytes, manifest)
    probe = _probe(args, root)
    return DataPlan(source, val, world, description, probe, _games10k(args), _mateset(root))


def plan(args: argparse.Namespace, cfg: TrainConfig) -> DataPlan:
    return _raw_plan(args, cfg) if args.source_raw else _shard_plan(args, cfg)
