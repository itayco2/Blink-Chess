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


def _bench_rate(args: argparse.Namespace) -> tuple[float | None, str]:
    """The throughput rule's benchmark: --bench-rate, or the best bench.json row of --bench-size."""
    if args.bench_rate is not None:
        return args.bench_rate, "--bench-rate"
    if args.bench_size is None:
        return None, "no --bench-rate or --bench-size"
    from blink.train.bench import best_rates

    mode = _train_compile_mode(_rest(args.train_args))
    bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
    best = best_rates(bench, compile=mode).get(args.bench_size)
    if best is None:
        raise ValueError(f"bench.json has no usable throughput row for size {args.bench_size}")
    return float(best["samples_per_s"]), f"size {args.bench_size} in bench.json"


def _train_compile_mode(train_args: list[str]) -> str | None:
    """The compile mode of the child's `--config` (None without one: any mode's best row)."""
    from blink.model.config import compile_mode, read_tables

    if "--config" not in train_args[:-1]:
        return None
    return compile_mode(read_tables(train_args[train_args.index("--config") + 1]))


def _supervise_config(args: argparse.Namespace):
    from blink.train.supervise import SuperviseConfig

    rate, source = _bench_rate(args)
    cfg = SuperviseConfig(
        interval_s=args.interval,
        poll_s=min(1.0, args.interval),
        heartbeat_stale_s=args.stale,
        startup_grace_s=args.grace,
        bench_rate=rate,
        backoff_s=args.backoff,
        disabled=tuple(args.disable or ()),
    )
    if rate is None or not cfg.on("throughput"):
        return cfg, f"throughput rule: off ({source if rate is None else 'disabled'})"
    floor = (1.0 - cfg.slow_frac) * rate
    share = round(100 * (1.0 - cfg.slow_frac))
    return cfg, f"throughput rule: floor {floor:,.0f} samples/s ({share}% of {rate:,.0f}, {source})"


def cmd_supervise(args: argparse.Namespace) -> int:
    from blink.train import status, supervise

    try:
        run = supervise.run_of(_rest(args.train_args), args.run)
        if not status.valid_run_name(run):
            raise ValueError(f"bad run name {run!r} (letters, digits, _ - . only)")
        cfg, throughput = _supervise_config(args)
        argv = supervise.child_argv(supervise.train_argv(_rest(args.train_args), run))
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink supervise: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        _say(" ".join(argv))
        _say(throughput)
        return 0
    _say(f"supervise {run}: {throughput}")
    deadline = None if args.max_hours is None else args.max_hours * 3600
    outcome = supervise.supervise(
        cfg,
        paths.home() / "runs" / run,
        argv,
        log=_say,
        deadline_s=deadline,
        launch_command=" ".join(sys.argv),
    )
    _say(f"supervise {run}: {outcome.status}")
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
    sup.add_argument("--run", help="the run name (default: the --run of the train command)")
    sup.add_argument("--bench-rate", type=float, help="benchmark samples/s; turns the throughput rule on")
    sup.add_argument("--bench-size", help="take the benchmark from bench.json's best row for this size")
    sup.add_argument("--bench", help="bench.json for --bench-size (default BLINK_HOME/eval/bench.json)")
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


# ---------------------------------------------------------------- sweep


def _read_json(path: Path, what: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{what} not found at {path}") from exc


def _home_eval(name: str, given: str | None) -> Path:
    return Path(given) if given else paths.home() / "eval" / name


def _repo_config(given: str | None, default: str) -> Path:
    """A --plan or --config path, or the tracked default under the repo's configs/ folder."""
    from blink.train.sweep import CONFIG_DIR

    return Path(given) if given else CONFIG_DIR / default


def _plan_rate(args: argparse.Namespace, size: str, mode: str) -> float:
    """--rate, or bench.json's best row for `size` measured in the compile mode the runs train in."""
    from blink.train.bench import best_rates

    if args.rate:
        return args.rate
    bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
    best = best_rates(bench, compile=mode).get(size)
    if best is None:
        why = f"bench.json has no usable throughput row for {size} at compile {mode}"
        raise ValueError(f"{why}; run `blink bench throughput`")
    return float(best["samples_per_s"])


def _print_arm_plan(plan, rate: float) -> None:
    from blink.model.config import read_tables
    from blink.train import sweep

    base = read_tables(plan.recipe)
    for arm in plan.arms:
        if arm.combine:
            _say(f"abl-{arm.name}: {arm.change}, chosen when it starts by the adopt rule")
            continue
        steps = sweep.steps_for(plan.hours, rate, sweep.batch_size_of(base, arm))
        peak = sweep.merged_config(base, arm, steps)["train"]["peak_lr"]
        _say(f"abl-{arm.name}: {arm.change}; steps {steps:,}; peak_lr {peak:g}; {arm.overrides}")


def cmd_sweep_ablations(args: argparse.Namespace) -> int:
    from blink.model.config import compile_mode, read_tables
    from blink.train import sweep

    try:
        plan = sweep.load_plan(_repo_config(args.plan, "ablations/plan.toml"))
        rate = _plan_rate(args, plan.size, compile_mode(read_tables(plan.recipe)))
        if args.dry_run:
            _print_arm_plan(plan, rate)
            return 0
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"blink sweep ablations: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    out = _home_eval("ablations.json", args.out)
    report = sweep.run_ablations(plan, out, rate, sweep.supervised_runner(_say), log=_say, slip=args.slip)
    noise = report.get("noise") or {}
    sigma = noise.get("vaa", {}).get("sigma")
    _say(f"sigma VAA {sigma}, sigma ok {noise.get('sigma_ok')}; recipe {report['recipe']}")
    for name, decision in report["decisions"].items():
        _say(f"  {name}: {'ADOPT' if decision['adopt'] else 'keep D'} ({decision['reason']})")
    _say(f"-> {out}")
    return 0


def cmd_sweep_sizes(args: argparse.Namespace) -> int:
    from blink.train import sweep

    try:
        sizes = [s for s in (args.sizes or "").split(",") if s] or None
        setup = sweep.load_size_sweep(_repo_config(args.config, "sweep.toml"), sizes=sizes, hours=args.hours)
        bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"blink sweep sizes: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        _say(f"sizes {setup.sizes} ({setup.conditional} only above the epoch floor), {setup.hours} h each")
        return 0
    out = _home_eval("sweep.json", args.out)
    rules = sweep.load_rules(_repo_config(args.config, "sweep.toml"))
    report = sweep.run_sizes(setup, bench, out, sweep.supervised_runner(_say), log=_say, rules=rules)
    for size, entry in report["sizes"].items():
        _say(f"  {size}: {entry['status']}, VAA {entry.get('vaa')}, {entry.get('samples_per_s')} samples/s")
    return 0


def _sigma(args: argparse.Namespace) -> float:
    if args.sigma is not None:
        return args.sigma
    noise = _read_json(_home_eval("ablations.json", args.ablations), "ablations.json").get("noise") or {}
    if "vaa" not in noise:
        raise ValueError("ablations.json has no noise floor yet (a01-a03); pass --sigma")
    return float(noise["vaa"]["sigma"])


def cmd_sweep_choose(args: argparse.Namespace) -> int:
    from blink.model.config import compile_mode, read_tables
    from blink.train import sweep

    sweep_path = _home_eval("sweep.json", args.sweep)
    try:
        bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
        state = _read_json(sweep_path, "sweep.json")
        config = _repo_config(args.config, "sweep.toml")
        rules = sweep.load_rules(config) if config.is_file() else sweep.ChooseRules()
        recipe = sweep.CONFIG_DIR / "recipe.toml"  # the long run trains in the recipe's compile mode
        mode = compile_mode(read_tables(recipe)) if recipe.is_file() else None
        choice = sweep.choose(bench, state.get("sizes", {}), _sigma(args), rules, compile=mode)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"blink sweep choose: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    from blink.train.atomic import write_text_atomic

    write_text_atomic(sweep_path, json.dumps({**state, "choice": choice}, indent=2) + "\n")
    for size, entry in choice["sizes"].items():
        _say(
            f"  {size}: {'eligible' if entry['eligible'] else 'out'} ({entry['reason']}), VAA {entry['vaa']}"
        )
    _say(f"N* = {choice['n_star']} ({choice['reason']})")
    return 0 if choice["n_star"] else 1


def _register_sweep(sub: argparse._SubParsersAction) -> None:
    sweep = sub.add_parser("sweep", help="recipe ablations (P5), the size sweep and the N* choice (P6)")
    actions = sweep.add_subparsers(dest="sweep_command", required=True)
    abl = actions.add_parser("ablations", help="run the ablation arms in order (resumable)")
    abl.add_argument("--plan", help="default: configs/ablations/plan.toml")
    abl.add_argument(
        "--rate", type=float, help="samples/s instead of bench.json's best row for the plan's size"
    )
    abl.add_argument(
        "--slip", action="store_true", help="apply the P5 slip rule: drop the plan's slip_cut arms"
    )
    sizes = actions.add_parser("sizes", help="S, M, M12 (and L above the epoch floor) at equal hours")
    sizes.add_argument("--sizes", help="default: the sizes and conditional sizes of configs/sweep.toml")
    sizes.add_argument("--hours", type=float)
    choose = actions.add_parser("choose", help="N*: epoch floor > best 6 h VAA > default M")
    choose.add_argument("--sweep", help="sweep.json (default BLINK_HOME/eval/sweep.json)")
    choose.add_argument(
        "--ablations", help="ablations.json, for sigma (default BLINK_HOME/eval/ablations.json)"
    )
    choose.add_argument("--sigma", type=float, help="the a01-a03 VAA sigma, instead of ablations.json")
    for parser in (abl, sizes, choose):
        parser.add_argument("--bench", help="bench.json (default BLINK_HOME/eval/bench.json)")
    for parser in (sizes, choose):
        parser.add_argument("--config", help="default: configs/sweep.toml")
    for parser in (abl, sizes):
        parser.add_argument("--out", help="the resumable state file (default under BLINK_HOME/eval)")
        parser.add_argument("--dry-run", action="store_true", help="print what would run and stop")
    abl.set_defaults(func=cmd_sweep_ablations)
    sizes.set_defaults(func=cmd_sweep_sizes)
    choose.set_defaults(func=cmd_sweep_choose)


def register(sub: argparse._SubParsersAction) -> None:
    _register_ops(sub)
    _register_supervise(sub)
    _register_bench(sub)
    _register_sweep(sub)
