"""`blink data probe|pack` (P1), `blink data bigpack|rebalance|valprobe|mateset|verify|stats` (P2) and
`blink data ladder10m` (the P3 ladder's fixed set as the pack s10m trains on).

Exit codes: 0 done; 1 done but a line raised something other than a documented parse.Rejected
(a parser bug: see error_samples in the report), or verify found a failed check; 2 refused: a missing
source or file, an existing pack without --overwrite or --resume, more records than the in-RAM pack
holds, or a source that changed under a resume (stop and ask).
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
FULL_NAME = "lichess_db_eval.jsonl.zst"
GAMES_PREFIX = "lichess_db_standard_rated_2026-08.prefix300M.pgn.zst"


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
    except (FileExistsError, MemoryError) as exc:
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


def default_full_source() -> Path:
    return paths.home() / "data" / "raw" / FULL_NAME


def default_pack() -> Path:
    return paths.home() / "data" / "v1"


class Refused(Exception):
    """A P2 command cannot start: the message says what to pass or fix (exit 2)."""


def _probe_roots(probe: Path):
    """Every root record of a finished pack (P1 skeleton or v1): the probe the grouped salt is chosen on."""
    import numpy as np

    from blink.data import bigpack
    from blink.data.record import ROOT_DTYPE

    if not (probe / bigpack.MANIFEST).is_file():
        raise Refused(f"no probe pack at {probe}: pass --salt N or --salt-probe DIR (a packed directory)")
    manifest = bigpack.read_manifest(probe)
    names = [n for n, e in manifest["shards"].items() if e.get("kind", "roots") == "roots"]
    return np.concatenate([np.fromfile(probe / name, dtype=ROOT_DTYPE) for name in sorted(names)])


def _salt(args: argparse.Namespace) -> tuple[int, dict | None]:
    from blink.data import grouped

    if args.salt is not None:
        return args.salt, None
    probe = Path(args.salt_probe)
    try:
        choice = grouped.choose_salt(_probe_roots(probe)["board"], giant_share=args.giant_share)
    except ValueError as exc:
        raise Refused(f"{exc}; pass --salt N") from exc
    return choice.salt, {**choice.as_dict(), "probe_dir": str(probe)}


def _bigpack_config(args: argparse.Namespace):
    from blink.data import bigpack

    source = Path(args.source)
    if problem := _source_problem(source):
        raise Refused(problem)
    blocklist = None if args.no_blocklist else Path(args.blocklist)
    if blocklist is not None and not blocklist.is_file():
        raise Refused(f"no blocklist at {blocklist} (pass --blocklist PATH or --no-blocklist)")
    salt, probe = _salt(args)
    return bigpack.BigPackConfig(
        source=source,
        out=Path(args.out),
        salt=salt,
        workers=args.workers,
        limit_frames=args.limit_frames,
        blocklist=blocklist,
        seed=args.seed,
        buckets=args.buckets,
        buffer_bytes=args.buffer_mb << 20,
        resume=args.resume,
        overwrite=args.overwrite,
        grouped_probe=probe,
    )


def _bigpack_summary(m: dict) -> str:
    p1, p2 = m["pass1"], m["pass2"]
    parsed = m["parsed_roots"]
    kids = sum(p1["children"].values())
    t1, t2 = p1["timing"], p2["timing"]
    disk = sum(e["bytes"] for e in m["shards"].values())
    return "\n".join(
        [
            f"source      {m['source']['path']} ({m['frames']} frames, end: {m['end']})",
            f"lines       {m['lines']:,} parsed {parsed:,} rejects {m['rejects']}",
            f"errors      {m['errors'] or 'none'}",
            f"children    {kids:,} ({kids / parsed if parsed else 0:.3f} per root); "
            f"{p1['grouped_children_dropped']:,} train children in held-out groups dropped",
            f"roots       {m['splits']['roots']}",
            f"children    {m['splits']['children']}",
            f"dropped     pass 2 {p2['dropped']}; eval {m['eval_dropped']}",
            f"pass 1      {t1['lines_per_s']:,.0f} lines/s over {t1['wall_s']:.1f} s, "
            f"{t1['workers']} workers; flush {t1['flush_mb_per_s']:,.0f} MB/s; "
            f"eval files {t1['eval_files_s']:.1f} s",
            f"pass 2      {t2['records_per_s']:,.0f} records/s over {t2['seconds']:.1f} s "
            f"({m['lines'] / t2['seconds'] if t2['seconds'] else 0:,.0f} source lines/s)",
            f"disk        {disk:,} B in {len(m['shards'])} files; grouped salt {m['grouped']['salt']}",
        ]
    )


def _cmd_bigpack(args: argparse.Namespace) -> int:
    from blink.data import bigpack

    try:
        manifest = bigpack.bigpack(_bigpack_config(args))
    except (Refused, FileExistsError, FileNotFoundError, ValueError, bigpack.SourceChanged) as exc:
        print(f"blink data bigpack: {exc}", file=sys.stderr)
        return 2
    print(_bigpack_summary(manifest))
    print(f"wrote {Path(args.out) / bigpack.MANIFEST}")
    return 1 if manifest["errors"] else 0


def _cmd_rebalance(args: argparse.Namespace) -> int:
    import json

    from blink.data import bigpack, pack, rebalance

    pack_dir = Path(args.pack)
    try:
        if args.games_hist:
            games = rebalance.GamesHistogram.from_dict(
                json.loads(Path(args.games_hist).read_text(encoding="utf-8"))
            )
        else:
            sites = rebalance.heldout_sites(Path(args.heldout))
            games = rebalance.games_histogram(Path(args.games), sites, args.max_games, args.workers)
            text = json.dumps(games.as_dict(), indent=1).encode("utf-8")
            pack.write_atomic(pack_dir / "games_hist.json", text)
        block = bigpack.write_rebalance(pack_dir, games)
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink data rebalance: {exc}", file=sys.stderr)
        return 2
    weights = block["weights"]
    print(
        f"rebalance   {games.games_with_eval:,} evaluated games, {int(games.counts.sum()):,} positions, "
        f"{games.heldout_skipped:,} held-out games skipped; weights {min(weights):.3f} to {max(weights):.3f}"
    )
    return 0


def _cmd_evalset(args: argparse.Namespace) -> int:
    from blink.data import mateset, valprobe

    module = valprobe if args.data_command == "valprobe" else mateset
    try:
        got = module.run(Path(args.pack), args.n)
    except FileNotFoundError as exc:
        print(f"blink data {args.data_command}: {exc}", file=sys.stderr)
        return 2
    short = "" if got["written"] == got["requested"] else f" (only {got['written']:,} val roots qualify)"
    print(f"{args.data_command:<11} {got['written']:,} of {got['requested']:,} roots{short}")
    print(f"children    {got['children']:,}")
    print(f"wrote {got['path']}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from blink.data import verify

    cfg = verify.VerifyConfig(
        pack_dir=Path(args.pack),
        blocklist=Path(args.blocklist) if args.blocklist else None,
        legality_sample=args.legality_sample,
        seed=args.seed,
    )
    try:
        report = verify.verify(cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink data verify: {exc}", file=sys.stderr)
        return 2
    for name, check in report["checks"].items():
        print(f"{'ok  ' if check['ok'] else 'FAIL'} {name:<34} {check['value']} ({check['limit']})")
    verdict = "ok" if report["ok"] else "FAILED"
    print(f"verify      {verdict} in {report['seconds']:.1f} s; valprobe {report['valprobe']}")
    return 0 if report["ok"] else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    from blink.data import stats

    try:
        got = stats.write(Path(args.pack), Path(args.out) if args.out else None)
    except (FileNotFoundError, KeyError) as exc:
        print(f"blink data stats: {exc}", file=sys.stderr)
        return 2
    print(f"lines       {got['lines']:,}, children per root {got['children_per_root']:.3f}")
    print(f"roots       {got['roots']}")
    print(f"disk        {got['disk_bytes']:,} B; verify {got['verify']}")
    return 0


def _cmd_ladder(args: argparse.Namespace) -> int:
    from blink.data import ladder

    cfg = ladder.LadderConfig(
        pack=Path(args.pack), out=Path(args.out), positions=args.positions, overwrite=args.overwrite
    )
    try:
        manifest = ladder.build(cfg, log=lambda line: print(line, flush=True))
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        print(f"blink data ladder10m: {exc}", file=sys.stderr)
        return 2
    print(ladder.summary(manifest, cfg.out))
    return 0


def _register_ladder(sub: argparse._SubParsersAction) -> None:
    lad = sub.add_parser(
        "ladder10m", help="the s10m pack: the ladder's fixed train roots and their children (plan P3, P5)"
    )
    lad.add_argument("--pack", default=str(default_pack()), help="finished v1 pack (default: %(default)s)")
    lad.add_argument(
        "--positions",
        type=_positive,
        default=10_000_000,
        help="the first N train roots (default: %(default)s)",
    )
    lad.add_argument(
        "--out",
        default=str(paths.home() / "data" / "ladder10m"),
        help="output directory (default: %(default)s)",
    )
    lad.add_argument("--overwrite", action="store_true", help="replace an earlier ladder pack in --out")
    lad.set_defaults(func=_cmd_ladder)


def _register_bigpack(sub: argparse._SubParsersAction) -> None:
    from blink.data import bigpack, grouped

    big = sub.add_parser("bigpack", help="pack the whole eval DB: 256 root and 256 child shards (plan P2)")
    big.add_argument("--out", required=True, help="output directory, e.g. D:\\blink\\data\\v1")
    big.add_argument(
        "--source", default=str(default_full_source()), help="pzstd eval DB (default: %(default)s)"
    )
    big.add_argument("--workers", type=_positive, default=default_workers(), help="decode processes")
    big.add_argument("--limit-frames", type=_positive, help="read only the first N frames (a probe run)")
    big.add_argument(
        "--blocklist", default=str(paths.home() / "data" / "blocklist_v1.npy"), help="blocklist .npy"
    )
    big.add_argument("--no-blocklist", action="store_true", help="pack without a blocklist (tests only)")
    big.add_argument("--salt", type=int, help="test_grouped salt; default: chosen on --salt-probe")
    big.add_argument(
        "--salt-probe", default=str(paths.home() / "data" / "skeleton"), help="pack to choose it on"
    )
    big.add_argument(
        "--giant-share", type=float, default=grouped.GIANT_SHARE, help="largest selectable group"
    )
    big.add_argument("--seed", type=int, default=bigpack.DEFAULT_SEED, help="shard permutation seed")
    big.add_argument("--buckets", type=_positive, default=bigpack.BUCKETS, help="train shards per kind")
    big.add_argument(
        "--buffer-mb", type=_positive, default=bigpack.BUFFER_BYTES >> 20, help="per-bucket buffer"
    )
    big.add_argument("--resume", action="store_true", help="finish an interrupted pack (pass 1 restarts)")
    big.add_argument("--overwrite", action="store_true", help="replace an existing pack in --out")
    big.set_defaults(func=_cmd_bigpack)


def _register_p2(sub: argparse._SubParsersAction) -> None:
    _register_bigpack(sub)
    downloads, evaldir = paths.home() / "downloads", paths.home() / "eval"
    reb = sub.add_parser("rebalance", help="fill manifest['rebalance'] from the %%eval games of the prefix")
    reb.add_argument("--pack", default=str(default_pack()), help="finished pack (default: %(default)s)")
    reb.add_argument("--games", default=str(downloads / GAMES_PREFIX), help="pzstd PGN of games")
    reb.add_argument("--heldout", default=str(evaldir / "heldout_games.pgn"), help="games to leave out")
    reb.add_argument("--max-games", type=_positive, help="stop after N evaluated games")
    reb.add_argument("--workers", type=_positive, default=default_workers(), help="PGN parsing processes")
    reb.add_argument("--games-hist", help="reuse a saved games_hist.json instead of reading the games")
    reb.set_defaults(func=_cmd_rebalance)
    for name, n, what in (
        ("valprobe", 20_000, "val roots with every legal child"),
        ("mateset", 2_000, "mate-in-2..5 val roots"),
    ):
        cmd = sub.add_parser(name, help=f"{what} (npz beside the pack)")
        cmd.add_argument("--pack", default=str(default_pack()), help="finished pack (default: %(default)s)")
        cmd.add_argument("--n", type=_positive, default=n, help="roots (default: %(default)s)")
        cmd.set_defaults(func=_cmd_evalset)
    ver = sub.add_parser("verify", help="scan the pack for leakage, legality and balance (verify.json)")
    ver.add_argument("--pack", default=str(default_pack()), help="finished pack (default: %(default)s)")
    ver.add_argument("--blocklist", help="blocklist .npy (default: the one the manifest names)")
    ver.add_argument("--legality-sample", type=float, default=0.01, help="share of records re-checked")
    ver.add_argument("--seed", type=int, default=0, help="sample seed")
    ver.set_defaults(func=_cmd_verify)
    stat = sub.add_parser("stats", help="write data_stats.json from the manifest and verify.json")
    stat.add_argument("--pack", default=str(default_pack()), help="finished pack (default: %(default)s)")
    stat.add_argument("--out", help="output path (default: <pack>/data_stats.json)")
    stat.set_defaults(func=_cmd_stats)
    _register_ladder(sub)


def register(subparsers: argparse._SubParsersAction) -> None:
    data = subparsers.add_parser("data", help="probe and pack the Lichess eval DB")
    sub = data.add_subparsers(dest="data_command", required=True)
    _register_p2(sub)

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
