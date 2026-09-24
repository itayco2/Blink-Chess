"""`blink train` and `blink status`.

blink train --config configs/t.toml --run NAME (--data DIR | --source-raw PATH [--max-lines N]) [--resume]
writes BLINK_HOME/runs/NAME/. `blink status --run NAME` prints the run's state and exits 1 when the
run is stale, crashed or has a NaN loss. Torch is imported only when a command runs, so `blink --help`
stays fast and works on the torch-free CI leg.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blink import paths
from blink.data.record import ROOT_DTYPE
from blink.model.config import TrainConfig, load_config
from blink.train import status
from blink.train.world import NO_BLOCKLIST, SPLIT_RULE, world_id

DEFAULT_MAX_LINES = 1_000_000
DEFAULT_WORKERS = max(1, min(6, (os.cpu_count() or 2) - 2))
EXIT_REFUSED = 2


class CommandError(RuntimeError):
    pass


@dataclass(frozen=True)
class DataPlan:
    source: Any  # blink.train.source.BatchSource
    val: np.ndarray | None
    world: str
    description: dict[str, Any]


def _raw_plan(args: argparse.Namespace, cfg: TrainConfig) -> DataPlan:
    from blink.train import rawsource
    from blink.train.source import InMemorySource

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
    print(f"source-raw: {len(train):,} train, {len(val):,} val records (hash split {SPLIT_RULE})")
    batches = InMemorySource(train, cfg.batch_size, cfg.seed).batches
    return DataPlan(batches, val if len(val) else None, world_id(f"raw:{raw.sha1}"), description)


def _shard_plan(args: argparse.Namespace, cfg: TrainConfig) -> DataPlan:
    from blink.train.source import shard_source

    root = Path(args.data)
    manifest_path = root / "manifest.json"
    train_paths = sorted(root.glob("train_*.bin"))
    if not manifest_path.is_file() or not train_paths:
        raise CommandError(f"{root} needs manifest.json and train_*.bin shards")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    world = manifest.get("world") or world_id(
        hashlib.sha1(manifest_bytes).hexdigest(),
        manifest.get("blocklist_sha", NO_BLOCKLIST),
        manifest.get("split_rule", SPLIT_RULE),
    )
    val_path = root / "val.bin"
    val = np.fromfile(val_path, dtype=ROOT_DTYPE, count=cfg.val_size) if val_path.is_file() else None
    description = {
        "source": "shards",
        "dir": str(root),
        "train_shards": len(train_paths),
        "val_records": 0 if val is None else len(val),
    }
    return DataPlan(shard_source(train_paths, cfg.batch_size, cfg.seed), val, world, description)


def _device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def cmd_train(args: argparse.Namespace) -> int:
    from blink.train import loop
    from blink.train.world import WorldMismatch

    if not status.valid_run_name(args.run):
        print(f"blink train: bad run name {args.run!r} (letters, digits, _ - . only)", file=sys.stderr)
        return EXIT_REFUSED
    try:
        cfg = load_config(args.config)
        plan = _raw_plan(args, cfg) if args.source_raw else _shard_plan(args, cfg)
        spec = loop.RunSpec(
            run_dir=paths.home() / "runs" / args.run,
            world=plan.world,
            device=_device(args.device),
            resume=args.resume,
            max_steps=args.max_steps,
            data=plan.description,
        )
        result = loop.train(cfg, spec, plan.source, plan.val)
    except (CommandError, WorldMismatch, loop.RunExists, FileNotFoundError, ValueError) as exc:
        print(f"blink train: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    state = "finished" if result.step >= cfg.steps else "stopped"
    print(f"{args.run}: {state} at step {result.step} in {result.wall_s:.1f} s")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not status.valid_run_name(args.run):
        print(f"blink status: bad run name {args.run!r}", file=sys.stderr)
        return EXIT_REFUSED
    report = status.run_status(paths.home() / "runs" / args.run)
    print(status.format_status(report))
    return status.exit_code(report)


def register(sub: argparse._SubParsersAction) -> None:
    train = sub.add_parser("train", help="train a model into BLINK_HOME/runs/<run>/")
    train.add_argument("--config", required=True, help="a TOML config, e.g. configs/t.toml")
    train.add_argument("--run", required=True, help="run name (letters, digits, _ - .)")
    data = train.add_mutually_exclusive_group(required=True)
    data.add_argument("--data", help="a packed shard directory: train_*.bin, val.bin, manifest.json")
    data.add_argument("--source-raw", help="parse the first --max-lines of a raw eval-DB .zst (cached)")
    train.add_argument("--max-lines", type=int, default=DEFAULT_MAX_LINES)
    train.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="parser processes")
    train.add_argument("--resume", action="store_true", help="continue from the run's latest checkpoint")
    train.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda when available")
    train.add_argument("--max-steps", type=int, help="stop early at this step (the schedule is unchanged)")
    train.set_defaults(func=cmd_train)

    run_status = sub.add_parser("status", help="a run's state; exits 1 when stale, crashed or NaN")
    run_status.add_argument("--run", required=True)
    run_status.set_defaults(func=cmd_status)
