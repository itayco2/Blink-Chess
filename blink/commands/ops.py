"""Operations commands: `blink ops launch|ps`, `blink supervise`, `blink bench ...`, `blink sweep ...`.

blink ops launch --name NAME -- <blink args>    a fully detached job (Win32_Process.Create), prints its PID
blink ops ps                                    Blink processes and launched jobs, with their heartbeats
blink supervise --run NAME -- train ...         the trainer as a child, every P7 stop rule enforced
blink bench throughput|loader|play              measured rates into bench.json (plan P4)
blink sweep ablations|sizes|choose              plan P5 and P6

Torch is imported only inside the commands that need it, so `blink --help` works torch-free.
"""

import argparse
import json
import sys
from pathlib import Path

from blink import paths

EXIT_REFUSED = 2


def _rest(args: list[str]) -> list[str]:
    """The arguments after `--` (argparse keeps the separator in a REMAINDER)."""
    return args[1:] if args and args[0] == "--" else list(args)


def _say(line: str) -> None:
    print(line, flush=True)


# ---------------------------------------------------------------- ops


def cmd_launch(args: argparse.Namespace) -> int:
    from blink.ops import launch

    blink_args = _rest(args.blink_args)
    if not blink_args:
        print("blink ops launch: give the blink command after --", file=sys.stderr)
        return EXIT_REFUSED
    try:
        plan = launch.plan_launch(args.name, blink_args)
        if args.dry_run:
            _say(plan.command_line)
            return 0
        result = launch.launch(plan)
    except (ValueError, launch.LaunchError) as exc:
        print(f"blink ops launch: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    _say(f"launched {args.name}: pid {result.pid} (cmd.exe), python {list(result.python_pids)}")
    _say(f"  logs {plan.out} and {plan.err}; heartbeat {plan.heartbeat or 'none'}; record {result.record}")
    return 0


def cmd_ps(args: argparse.Namespace) -> int:
    from blink.ops import launch

    rows = launch.blink_processes()
    records = launch.launch_records()
    if args.json:
        _say(json.dumps({"processes": rows, "launched": records}, indent=2))
        return 0
    _say(launch.format_ps(rows))
    for record in records[: args.recent]:
        state = "alive" if record["alive"] else "gone"
        _say(f"launched {record['name']}: pid {record['pid']} {state}, heartbeat {record['beat']}")
    return 0


# ---------------------------------------------------------------- supervise


def _supervise_config(args: argparse.Namespace):
    from blink.train.supervise import SuperviseConfig

    return SuperviseConfig(
        interval_s=args.interval,
        poll_s=min(1.0, args.interval),
        heartbeat_stale_s=args.stale,
        startup_grace_s=args.grace,
        bench_rate=args.bench_rate,
        backoff_s=args.backoff,
        disabled=tuple(args.disable or ()),
    )


def cmd_supervise(args: argparse.Namespace) -> int:
    from blink.train import status, supervise

    if not status.valid_run_name(args.run):
        print(f"blink supervise: bad run name {args.run!r}", file=sys.stderr)
        return EXIT_REFUSED
    try:
        cfg = _supervise_config(args)
        argv = supervise.child_argv(supervise.train_argv(_rest(args.train_args), args.run))
    except ValueError as exc:
        print(f"blink supervise: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        _say(" ".join(argv))
        return 0
    deadline = None if args.max_hours is None else args.max_hours * 3600
    outcome = supervise.supervise(
        cfg,
        paths.home() / "runs" / args.run,
        argv,
        log=_say,
        deadline_s=deadline,
        launch_command=" ".join(sys.argv),
    )
    _say(f"supervise {args.run}: {outcome.status}")
    return outcome.exit_code


def _register_ops(sub: argparse._SubParsersAction) -> None:
    ops = sub.add_parser("ops", help="detached launches and the processes they started")
    actions = ops.add_subparsers(dest="ops_command", required=True)
    launch = actions.add_parser("launch", help="start `blink <args>` fully detached; prints its PID")
    launch.add_argument("--name", required=True, help="names the logs: BLINK_HOME/logs/NAME.out and .err")
    launch.add_argument("--dry-run", action="store_true", help="print the command line and stop")
    launch.add_argument("blink_args", nargs=argparse.REMAINDER, help="-- then the blink command")
    launch.set_defaults(func=cmd_launch)
    ps = actions.add_parser("ps", help="Blink processes with their heartbeats")
    ps.add_argument("--json", action="store_true")
    ps.add_argument("--recent", type=int, default=5, help="how many launch records to show")
    ps.set_defaults(func=cmd_ps)


def _register_supervise(sub: argparse._SubParsersAction) -> None:
    from blink.train.supervise import RULES

    sup = sub.add_parser("supervise", help="run `train ...` as a child and enforce every stop rule")
    sup.add_argument("--run", required=True)
    sup.add_argument("--bench-rate", type=float, help="benchmark samples/s; turns the throughput rule on")
    sup.add_argument("--interval", type=float, default=60.0, help="seconds between rule checks")
    sup.add_argument("--stale", type=float, default=60.0, help="heartbeat age that stops the run")
    sup.add_argument("--grace", type=float, default=600.0, help="seconds allowed before the first step")
    sup.add_argument("--backoff", type=float, default=60.0, help="seconds before a crash resume")
    sup.add_argument("--disable", action="append", choices=RULES, help="turn one rule off (recorded)")
    sup.add_argument("--max-hours", type=float, help="stop after this much wall-clock time")
    sup.add_argument("--dry-run", action="store_true", help="print the child command and stop")
    sup.add_argument("train_args", nargs=argparse.REMAINDER, help="-- train --config ... --run NAME ...")
    sup.set_defaults(func=cmd_supervise)


# ---------------------------------------------------------------- bench


def _ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]


def _bench_out(args: argparse.Namespace) -> Path:
    return Path(args.out) if args.out else paths.home() / "eval" / "bench.json"


def cmd_bench_throughput(args: argparse.Namespace) -> int:
    from blink.train import bench

    try:
        sizes = [bench.resolve_size(size) for size in args.sizes.split(",") if size]
        specs = [
            bench.ThroughputSpec(
                name, path, micro, mode, args.steps, args.warmup, args.effective, args.device
            )
            for name, path in sizes
            for micro in _ints(args.micro)
            for mode in args.compile.split(",")
        ]
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink bench throughput: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    machine = bench.machine_facts(args.device)
    rows = bench.run_throughput(specs, log=_say)
    bench.update_bench(_bench_out(args), "throughput", rows, machine)
    _say(f"{len(rows)} throughput rows -> {_bench_out(args)}")
    return 0


def cmd_bench_loader(args: argparse.Namespace) -> int:
    from blink.baselines.train import root_shards
    from blink.train import bench

    data = Path(args.data) if args.data else paths.home() / "data" / "skeleton"
    try:
        shards = root_shards(data)
    except FileNotFoundError as exc:
        print(f"blink bench loader: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    result = bench.measure_loader(shards, batch_size=args.batch, passes=args.passes)
    for row in result["passes"]:
        _say(
            f"loader pass {row['pass']}: {row['records']:,} records in {row['seconds']:.2f} s, "
            f"{row['samples_per_s']:,.0f} samples/s, {row['read_mb_per_s']:,.1f} MB/s"
        )
    bench.update_bench(_bench_out(args), "loader", {str(data): result})
    return 0


def cmd_bench_play(args: argparse.Namespace) -> int:
    from blink.train import bench

    try:
        sizes = [bench.resolve_size(size) for size in args.sizes.split(",") if size]
    except FileNotFoundError as exc:
        print(f"blink bench play: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    specs = [
        bench.PlaySpec(name, path, rows, concurrency, args.iters, args.warmup, args.device)
        for name, path in sizes
        for rows in _ints(args.rows)
        for concurrency in _ints(args.concurrency)
    ]
    machine = bench.machine_facts(args.device)
    rows = bench.run_play(specs, log=_say)
    bench.update_bench(_bench_out(args), "play", rows, machine)
    return 0


def _register_bench(sub: argparse._SubParsersAction) -> None:
    bench = sub.add_parser("bench", help="measured throughput, loader and play latency into bench.json")
    actions = bench.add_subparsers(dest="bench_command", required=True)
    throughput = actions.add_parser("throughput", help="training samples/s and peak VRAM per size")
    throughput.add_argument("--sizes", default="s,m,m12,l", help="configs/<size>.toml names or .toml paths")
    throughput.add_argument("--micro", default="256,512,1024")
    throughput.add_argument("--compile", default="off,inductor,cudagraphs")
    throughput.add_argument("--steps", type=int, default=20, help="timed optimizer steps per row")
    throughput.add_argument(
        "--warmup", type=int, default=5, help="untimed steps first (compile happens here)"
    )
    throughput.add_argument("--effective", type=int, default=1024, help="effective batch (accumulation)")
    throughput.set_defaults(func=cmd_bench_throughput)
    loader = actions.add_parser("loader", help="ShardLoader samples/s and read MB/s")
    loader.add_argument("--data", help="a shard directory (default BLINK_HOME/data/skeleton)")
    loader.add_argument("--batch", type=int, default=1024)
    loader.add_argument("--passes", type=int, default=2)
    loader.set_defaults(func=cmd_bench_loader)
    play = actions.add_parser("play", help="value-mode latency at 1 and L+1 rows, 1/2/5 processes")
    play.add_argument("--sizes", default="s,m,m12,l")
    play.add_argument("--rows", default="1,219")
    play.add_argument("--concurrency", default="1,2,5")
    play.add_argument("--iters", type=int, default=200)
    play.add_argument("--warmup", type=int, default=20)
    for parser in (throughput, loader, play):
        parser.add_argument("--out", help="bench.json path (default BLINK_HOME/eval/bench.json)")
    for parser in (throughput, play):
        parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    play.set_defaults(func=cmd_bench_play)


def register(sub: argparse._SubParsersAction) -> None:
    _register_ops(sub)
    _register_supervise(sub)
    _register_bench(sub)
