"""Operations commands: `blink ops launch|ps`, `blink supervise`, `blink bench ...`, `blink sweep ...`
(the Lichess bot's commands are blink.commands.lichess).

blink ops launch --name NAME -- <blink args>    a fully detached job (Win32_Process.Create), prints its PID
blink ops ps                                    Blink processes and launched jobs, with their heartbeats
blink supervise --run NAME -- train ...         the trainer as a child, every P7 stop rule enforced
blink bench throughput|loader|play              measured rates into bench.json (plan P4)
blink bench parity                              a fast play mode's moves against fp32's, on val roots
blink sweep ablations|sizes|choose              plan P5 and P6
blink sweep rescore                             score finished arms post hoc (games10k, mateset)

Torch is imported only inside the commands that need it, so `blink --help` works torch-free.
"""

import argparse
import json
import sys
from pathlib import Path

from blink import paths
from blink.play import fastmode

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

    mode, pin = _train_compile_mode(_rest(args.train_args)), _train_micro_pin(_rest(args.train_args))
    pins = None if pin is None else {args.bench_size: pin}
    bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
    best = best_rates(bench, compile=mode, pins=pins).get(args.bench_size)
    if best is None:
        at = "" if pin is None else f" at the config's micro-batch {pin}"
        raise ValueError(f"bench.json has no usable throughput row for size {args.bench_size}{at}")
    return float(best["samples_per_s"]), f"size {args.bench_size} in bench.json"


def _train_tables(train_args: list[str]) -> dict | None:
    """The [model] and [train] tables of the child's `--config`, or None without one."""
    from blink.model.config import read_tables

    if "--config" not in train_args[:-1]:
        return None
    return read_tables(train_args[train_args.index("--config") + 1])


def _train_compile_mode(train_args: list[str]) -> str | None:
    """The compile mode of the child's `--config` (None without one: any mode's best row)."""
    from blink.model.config import compile_mode

    tables = _train_tables(train_args)
    return None if tables is None else compile_mode(tables)


def _train_micro_pin(train_args: list[str]) -> int | None:
    """The micro-batch the child's `--config` pins (None: "auto", or no config: the fastest row)."""
    from blink.model.config import micro_batch_pin

    tables = _train_tables(train_args)
    return None if tables is None else micro_batch_pin(tables)


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
        mode = {"precision": args.precision, "compile": args.compile}
        specs = [
            bench.PlaySpec(name, path, rows, concurrency, args.iters, args.warmup, args.device, **mode)
            for name, path in sizes
            for rows in _ints(args.rows)
            for concurrency in _ints(args.concurrency)
        ]
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink bench play: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    machine = bench.machine_facts(args.device)
    rows = bench.run_play(specs, log=_say)
    bench.update_bench(_bench_out(args), "play", rows, machine)
    return 0


def _parity_out(args: argparse.Namespace) -> Path:
    from blink.eval.fastchess import NAME_UNSAFE

    if args.out:
        return Path(args.out)
    name = NAME_UNSAFE.sub("_", args.model).strip("_") + fastmode.tag(args.precision, args.compile)
    return paths.home() / "eval" / "parity" / f"{name}.json"


def _parity_evaluators(args: argparse.Namespace):
    """fp32 and the fast mode on one model object: the fast one compiles its own trunk wrapper."""
    from blink.model.evaluator import TorchEvaluator, play_evaluator
    from blink.model.loading import load_model

    model = load_model(args.model, device=args.device)
    fast = play_evaluator(model, args.device, precision=args.precision, compile=args.compile)
    return TorchEvaluator(model, args.device), fast


def cmd_bench_parity(args: argparse.Namespace) -> int:
    import time

    from blink.eval import parity
    from blink.train.atomic import write_text_atomic
    from blink.train.posthoc import gpu_refusal

    data = Path(args.data) if args.data else paths.home() / "data" / "v1"
    refusal = fastmode.refusal(args.precision, args.compile, args.device) or gpu_refusal(args.device)
    try:
        if refusal:
            raise ValueError(refusal)
        boards = parity.val_positions(data, args.positions)
        reference, fast = _parity_evaluators(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink bench parity: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    mode = fastmode.describe(args.precision, args.compile)
    _say(f"parity of {args.model} {mode} against fp32 on {len(boards)} val roots of {data}")
    started = time.perf_counter()
    report = parity.compare(reference, fast, boards, epsilon=args.epsilon, log=_say)
    result = {
        "model": args.model,
        "device": args.device,
        "precision": args.precision,
        "compile": args.compile,
        "reference": {"precision": fastmode.DEFAULT_PRECISION, "compile": False},
        "data": str(data),
        "requested": args.positions,
        "epsilon": args.epsilon,
        "seconds": round(time.perf_counter() - started, 1),
        **report,
    }
    out = _parity_out(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(result, indent=2) + "\n")
    _say(
        f"policy top-1 agreement {report['policy_top1_agreement']}, value choice agreement "
        f"{report['value_choice_agreement']}, max |d win%| {report['max_abs_dwin_pct']} pt "
        f"({report['scored']} positions scored, {report['mate_now']} mates in one by R2) -> {out}"
    )
    return 0


def _register_parity(actions: argparse._SubParsersAction) -> None:
    from blink.play import rules

    parity = actions.add_parser(
        "parity", help="policy top-1, value choice and win% of a fast mode against fp32 on val roots"
    )
    parity.add_argument("--model", required=True, help="run:<name>[:ema] | ship | release:<tag> | <path>")
    parity.add_argument("--positions", type=int, default=2000, help="val roots, taken as VAA takes them")
    parity.add_argument("--data", help="the pack holding val_roots.bin (default BLINK_HOME/data/v1)")
    parity.add_argument("--epsilon", type=float, default=rules.DEFAULT_EPSILON, help="R4 tie window")
    parity.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parity.add_argument("--out", help="JSON report (default BLINK_HOME/eval/parity/<model><mode>.json)")
    fastmode.add_arguments(parity)
    parity.set_defaults(func=cmd_bench_parity)


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
    play.add_argument("--warmup", type=int, default=20, help="untimed calls first (compile happens here)")
    fastmode.add_arguments(play)
    for parser in (throughput, loader, play):
        parser.add_argument("--out", help="bench.json path (default BLINK_HOME/eval/bench.json)")
    for parser in (throughput, play):
        parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    play.set_defaults(func=cmd_bench_play)
    _register_parity(actions)


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
    if args.rate:
        return args.rate
    return _row_rate(args, size, mode)


def _row_rate(args: argparse.Namespace, size: str, mode: str) -> float:
    """bench.json's best row for `size` in `mode`. An arm with its own row (a10's s-muon) is always
    planned from it: --rate pins only the plan size, never an arm that costs more per step."""
    from blink.train.bench import best_rates

    bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
    best = best_rates(bench, compile=mode).get(size)
    if best is None:
        why = f"bench.json has no usable throughput row for {size} at compile {mode}"
        # the plain command measures s, m, m12 and l in three modes, and would re-measure rows in use
        raise ValueError(f"{why}; run `blink bench throughput --sizes {size} --micro 1024 --compile {mode}`")
    return float(best["samples_per_s"])


def cmd_sweep_ablations(args: argparse.Namespace) -> int:
    from blink.data import games10k
    from blink.model.config import compile_mode, read_tables
    from blink.train import sweep

    out = _home_eval("ablations.json", args.out)
    try:
        plan = sweep.load_plan(_repo_config(args.plan, "ablations/plan.toml"))
        rate = _plan_rate(args, plan.size, compile_mode(read_tables(plan.recipe)))
        # only the arms this launch plans afresh: not one the slip rule cuts, nor one already recorded
        fresh = sweep.fresh_rate_arms(plan, out, args.slip)
        arm_rates = sweep.own_bench_rates(plan, lambda size, mode: _row_rate(args, size, mode), fresh)
        if args.dry_run:
            for line in sweep.preview(plan, out, rate, arm_rates, args.slip):
                _say(line)
            return 0
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"blink sweep ablations: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    # the arms' own inputs (blink train's defaults); the GPU is the sweep's and idle between arms
    scorer = _child_scorer(games10k.default_path(), plan.data)
    runner = sweep.supervised_runner(_say)
    report = sweep.run_ablations(
        plan, out, rate, runner, log=_say, slip=args.slip, arm_rates=arm_rates, scorer=scorer
    )
    noise = report.get("noise") or {}
    sigma = noise.get("vaa", {}).get("sigma")
    _say(f"sigma VAA {sigma}, sigma ok {noise.get('sigma_ok')}; recipe {report['recipe']}")
    for name, decision in report["decisions"].items():
        _say(f"  {name}: {'ADOPT' if decision['adopt'] else 'keep D'} ({decision['reason']})")
    _say(f"-> {out}")
    return 0


def _child_scorer(games: Path, pack: Path):
    """The sweep's post-hoc scorer: `blink eval arm-metrics` in a child process on the GPU, which takes
    its CUDA context and cached blocks with it when it exits. Scored in the sweep's own process, they
    would stay held while the next arm's trainer (a15) sizes its micro-batch from the free VRAM."""
    import subprocess

    from blink.train.supervise import child_argv

    def score(run: str) -> None:
        args = ["eval", "arm-metrics", "--run", run, "--data", str(pack), "--games10k", str(games)]
        code = subprocess.run(child_argv([*args, "--device", "cuda"]), check=False).returncode
        if code != 0:
            raise RuntimeError(f"`blink eval arm-metrics --run {run}` exited {code}")

    return score


def _posthoc_scorer(games: Path, mates: Path, device: str, force: bool = False):
    """posthoc.score_run for one arm's run under BLINK_HOME/runs, on these games10k and mateset files."""
    from blink.train import posthoc

    def score(run: str) -> None:
        posthoc.score_run(paths.home() / "runs" / run, games, mates, device, _say, force)

    return score


def cmd_sweep_rescore(args: argparse.Namespace) -> int:
    """Arms trained before the checks scored games10k and the mateset get them from their final
    checkpoints, then every arm is judged again, so a07 and a08 meet an a01-a03 floor of their metrics.

    Refuses the GPU while a run trains or starts (posthoc.gpu_refusal). Exits 1 unless every finished
    arm was scored, since a07 and a15 leave `held` only once the seeds are."""
    from blink.data import games10k, mateset
    from blink.train import posthoc, sweep

    out = _home_eval("ablations.json", args.out)
    try:
        plan = sweep.load_plan(_repo_config(args.plan, "ablations/plan.toml"))
        if not out.is_file():
            raise FileNotFoundError(f"no ablations.json at {out}: nothing has run yet")
        refusal = posthoc.gpu_refusal(args.device)
        if refusal:
            raise ValueError(refusal)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"blink sweep rescore: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    games = Path(args.games10k) if args.games10k else games10k.default_path()
    mates = (Path(args.data) if args.data else plan.data) / mateset.OUTPUT
    judged = sweep.rescore_ablations(plan, out, _posthoc_scorer(games, mates, args.device, args.force), _say)
    for name, decision in judged["decisions"].items():
        _say(f"  {name}: {'ADOPT' if decision['adopt'] else 'keep D'} ({decision['reason']})")
    _say(f"recipe {judged['recipe']}")
    return _rescore_outcome(judged["scored"], judged["not_scored"], out)


def _rescore_outcome(scored: list[str], failed: dict[str, str], out: Path) -> int:
    if not scored and not failed:
        _say(f"nothing to score: no arm has finished in {out}")
        return 1
    if scored:
        _say(
            f"posthoc.json holds the scores of {', '.join(scored)}; {out} is the sweep's: a running sweep "
            "reads them for a15 and its final report, else `blink sweep ablations` records them (and runs "
            "any pending or held arm)"
        )
    if failed:
        _say(f"not scored: {', '.join(failed)}" + ("" if scored else "; no arm was scored"))
        return 1
    return 0


def cmd_sweep_sizes(args: argparse.Namespace) -> int:
    from blink.train import size_sweep, sweep

    try:
        sizes = [s for s in (args.sizes or "").split(",") if s] or None
        setup = size_sweep.load_size_sweep(
            _repo_config(args.config, "sweep.toml"), sizes=sizes, hours=args.hours
        )
        bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"blink sweep sizes: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if args.dry_run:
        _say(f"sizes {setup.sizes} ({setup.conditional} only above the epoch floor), {setup.hours} h each")
        return 0
    from blink.train import nstar

    out = _home_eval("sweep.json", args.out)
    rules = nstar.load_rules(_repo_config(args.config, "sweep.toml"))
    report = size_sweep.run_sizes(setup, bench, out, sweep.supervised_runner(_say), log=_say, rules=rules)
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


def _size_pins(sizes, recipe: Path | None) -> dict[str, int]:
    """The micro-batch each size's repo config pins (M and M12: 256); a size without a config file
    under configs/ (or with "auto") is judged at its fastest row, as before."""
    from blink.train import size_sweep, sweep

    arm = sweep.load_arm(recipe) if recipe is not None else None
    pins = {}
    for size in sizes:
        if (sweep.CONFIG_DIR / f"{size}.toml").is_file():
            pin = size_sweep.size_micro_pin(sweep.CONFIG_DIR, size, arm)
            if pin is not None:
                pins[size] = pin
    return pins


def cmd_sweep_choose(args: argparse.Namespace) -> int:
    from blink.model.config import compile_mode, read_tables
    from blink.train import nstar, sweep

    sweep_path = _home_eval("sweep.json", args.sweep)
    try:
        bench = _read_json(_home_eval("bench.json", args.bench), "bench.json")
        state = _read_json(sweep_path, "sweep.json")
        config = _repo_config(args.config, "sweep.toml")
        rules = nstar.load_rules(config) if config.is_file() else nstar.ChooseRules()
        recipe = sweep.CONFIG_DIR / "recipe.toml"  # the long run trains in the recipe's compile mode
        mode = compile_mode(read_tables(recipe)) if recipe.is_file() else None
        sizes = state.get("sizes", {})
        pins = _size_pins(sizes, recipe if recipe.is_file() else None)
        choice = nstar.choose(bench, sizes, _sigma(args), rules, compile=mode, pins=pins)
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
        "--rate",
        type=float,
        help="samples/s for the plan size instead of bench.json's best row (a sweep under way keeps "
        "the rate it pinned in ablations.json)",
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
    rescore = actions.add_parser(
        "rescore", help="score finished arms' final checkpoints on games10k and the mateset, then judge again"
    )
    rescore.add_argument("--plan", help="default: configs/ablations/plan.toml")
    rescore.add_argument("--out", help="ablations.json (default BLINK_HOME/eval/ablations.json)")
    rescore.add_argument("--data", help="the pack whose mateset.npz is scored (default: the plan's data)")
    rescore.add_argument("--games10k", help="default: BLINK_HOME/data/games10k.npy")
    rescore.add_argument(
        "--device", choices=("cuda", "cpu"), default="cuda", help="cuda is refused while a run is training"
    )
    rescore.add_argument("--force", action="store_true", help="score again even when already scored")
    rescore.set_defaults(func=cmd_sweep_rescore)
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
