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


def register(sub: argparse._SubParsersAction) -> None:
    _register_ops(sub)
    _register_supervise(sub)
