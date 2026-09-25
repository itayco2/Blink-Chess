"""`blink eval strength`: PR-6's strength check of a run (blink.eval.strength).

blink eval strength --run long [--limit 2000] [--again]    score the latest EMA checkpoint, print the trend
blink eval strength --show [--run long] [--last N]         print the recorded trend, score nothing
blink eval strength --run long --record-pr3 PARITY_JSON    record PR-3's parity report for the run
"""

import argparse
import sys
import time
from pathlib import Path

from blink import paths

EXIT_REFUSED = 2


def _show(args: argparse.Namespace) -> int:
    from blink.eval import strength

    for line in strength.trend(strength.read_checks(paths.home()), args.run, args.last):
        print(line)
    return 0


def _record_pr3(args: argparse.Namespace) -> int:
    from blink.eval import strength

    row = strength.pr3_record(args.run, Path(args.record_pr3), time.time())
    strength.append_check(paths.home(), row)
    print(f"PR-3 parity of runs/{args.run} recorded from {args.record_pr3}")
    return 0


def _check(args: argparse.Namespace) -> int:
    from blink.eval import strength

    home = paths.home()
    row = strength.run_check(args.run, home, strength.child_scorer(home), args.limit, args.again)
    rows = strength.read_checks(home)
    latest = row or next(r for r in reversed(rows) if r.get("kind") == strength.KIND and r["run"] == args.run)
    if row is None:
        print(f"runs/{args.run} was already checked at step {latest['step']:,} (--again scores it again)")
    for line in strength.trend(rows, args.run):
        print(line)
    if strength.pr3_due(rows, args.run, latest["hours"]):
        for line in strength.pr3_lines(args.run, latest["step"], latest["hours"]):
            print(line)
    return 0


def cmd_strength(args: argparse.Namespace) -> int:
    from blink.train.status import valid_run_name

    try:
        if args.run is not None and not valid_run_name(args.run):
            raise ValueError(f"bad run name {args.run!r}")
        if args.show:
            return _show(args)
        if args.run is None:
            raise ValueError("--run names the run to check (or --show prints the recorded checks)")
        return _record_pr3(args) if args.record_pr3 else _check(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"blink eval strength: {exc}", file=sys.stderr)
        return EXIT_REFUSED


def register(ev_sub: argparse._SubParsersAction) -> None:
    from blink.eval.strength import CHECK_PUZZLES

    st = ev_sub.add_parser(
        "strength", help="PR-6: the latest EMA checkpoint on the first 2,000 DeepMind puzzles, beside DM-9M"
    )
    st.add_argument("--run", help="the run to check (the flagship: long)")
    st.add_argument("--limit", type=int, default=CHECK_PUZZLES, help="puzzles, from the front of dm10k")
    st.add_argument("--again", action="store_true", help="score a step that was already checked again")
    st.add_argument("--show", action="store_true", help="print the recorded checks and score nothing")
    st.add_argument("--last", type=int, default=None, help="with --show: only the last N checks")
    st.add_argument("--record-pr3", help="record PR-3's parity report (blink bench parity's JSON) for --run")
    st.set_defaults(func=cmd_strength)
