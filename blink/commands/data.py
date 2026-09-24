"""`blink data probe` and `blink data pack`.

Exit codes: 0 done; 1 done but a line raised something other than a documented parse.Rejected
(a parser bug: see error_samples in the report); 2 refused before starting (bad source, existing pack).
"""

import argparse
import os
import sys
from pathlib import Path

from blink import paths

MAX_DEFAULT_WORKERS = 10
FREE_THREADS = 2
DEFAULT_PROBE_FRAMES = 20
PREFIX_NAME = "prefix-342M.jsonl.zst"


def default_workers(cpus: int | None = None) -> int:
    """Leave two hardware threads for the desktop and the GPU feeder; the plan's pack uses Pool(10)."""
    count = cpus if cpus is not None else os.cpu_count() or 1
    return max(1, min(MAX_DEFAULT_WORKERS, count - FREE_THREADS))


def default_source() -> Path:
    return paths.home() / "data" / "raw" / PREFIX_NAME


def _source_problem(source: Path) -> str | None:
    if not source.is_file():
        return f"no eval DB file at {source} (pass --source)"
    return None


def _cmd_probe(args: argparse.Namespace) -> int:
    from blink.data import probe

    source = Path(args.source)
    if problem := _source_problem(source):
        print(f"blink data probe: {problem}", file=sys.stderr)
        return 2
    check_every = args.check_every or probe.DEFAULT_CHECK_EVERY
    report = probe.probe(source, frames=args.frames, workers=args.workers, check_every=check_every)
    out = Path(args.out)
    probe.write_report(report, out)
    print(probe.summary(report))
    print(f"wrote {out}")
    return 1 if report["errors"] else 0


def _rate(lines: int, seconds: float) -> str:
    return f"{lines / seconds:,.0f}" if seconds > 0 else "n/a"


def _pack_summary(m: dict, timing: dict) -> str:
    lines = m["lines"]
    return "\n".join(
        [
            f"source      {m['source']['path']} ({m['frames']} frames, end: {m['end']})",
            f"lines       {lines:,} parsed {m['parsed']:,} rejects {m['rejects']}",
            f"errors      {m['errors'] or 'none'}",
            f"records     {m['records_written']:,} written, {m['dropped_blocklisted']:,} blocklisted, "
            f"splits {m['splits']}",
            f"shards      {len(m['shards'])} files, {m['shard_rule']}, seed {m['seed']}",
            f"throughput  {_rate(lines, timing['wall_s'])} lines/s wall over {timing['wall_s']:.1f} s with "
            f"{timing['workers']} workers; per worker: decode {_rate(lines, timing['decode_s'])} lines/s, "
            f"parse {_rate(lines, timing['parse_s'])} lines/s; write {timing['write_s']:.2f} s",
        ]
    )


def _cmd_pack(args: argparse.Namespace) -> int:
    from blink.data import pack

    source = Path(args.source)
    if problem := _source_problem(source):
        print(f"blink data pack: {problem}", file=sys.stderr)
        return 2
    cfg = pack.PackConfig(
        source=source,
        out=Path(args.out),
        shards=args.shards,
        frames=args.frames,
        workers=args.workers,
        seed=pack.DEFAULT_SEED if args.seed is None else args.seed,
        blocklist=Path(args.blocklist) if args.blocklist else None,
        overwrite=args.overwrite,
    )
    try:
        manifest = pack.pack(cfg)
    except FileExistsError as exc:
        print(f"blink data pack: {exc}", file=sys.stderr)
        return 2
    timing = pack.read_timing(cfg.out)
    print(_pack_summary(manifest, timing))
    print(f"wrote {cfg.out / pack.MANIFEST}")
    return 1 if manifest["errors"] else 0


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def _add_common(parser: argparse.ArgumentParser, frames_default: int | None) -> None:
    parser.add_argument(
        "--source", default=str(default_source()), help="pzstd eval DB file (default: %(default)s)"
    )
    parser.add_argument(
        "--frames", type=_positive, default=frames_default, help="frames to read (default: %(default)s)"
    )
    parser.add_argument("--workers", type=_positive, default=default_workers(), help="decode processes")


def register(subparsers: argparse._SubParsersAction) -> None:
    data = subparsers.add_parser("data", help="probe and pack the Lichess eval DB")
    sub = data.add_subparsers(dest="data_command", required=True)

    probe = sub.add_parser("probe", help="parse N frames and report what the data looks like")
    _add_common(probe, DEFAULT_PROBE_FRAMES)
    probe.add_argument("--out", default=str(paths.home() / "data" / "probe.json"), help="report path")
    probe.add_argument("--check-every", type=_positive, help="re-check 1 line in K with python-chess (16)")
    probe.set_defaults(func=_cmd_probe)

    pack = sub.add_parser("pack", help="pack N frames into permuted train/val/test_iid shards")
    _add_common(pack, None)
    pack.add_argument("--shards", type=_positive, required=True, help="train shards (fen_hash %% K)")
    pack.add_argument("--out", required=True, help="output directory")
    pack.add_argument("--blocklist", help="1-D uint64 .npy of fen_hash values never to pack")
    pack.add_argument("--seed", type=int, help="shard permutation seed (default: pack.DEFAULT_SEED)")
    pack.add_argument("--overwrite", action="store_true", help="replace an existing pack in --out")
    pack.set_defaults(func=_cmd_pack)
