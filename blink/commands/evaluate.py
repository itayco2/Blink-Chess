"""`blink eval puzzles|signcheck` and `blink audit no-search`."""

import argparse
import json
from pathlib import Path

from blink import paths
from blink.eval import fastchess, nosearch, puzzles, signcheck
from blink.play import factory, rules

MODES_OR_BOTH = (*factory.MODES, "both")


def _set_label(name: str) -> str:
    return name if name in puzzles.PUZZLE_SETS else Path(name).stem


def _print_puzzles(summary: dict, agent_name: str) -> None:
    low, high = summary["wilson95"]
    print(
        f"{agent_name} on {summary['set']}: {summary['correct']}/{summary['n']} = "
        f"{100 * summary['accuracy']:.1f}% (Wilson 95% {100 * low:.1f} to {100 * high:.1f}), "
        f"illegal moves {summary['illegal_moves']}"
    )
    for band, row in summary["bands"].items():
        print(f"  {band:>10}: {row['correct']}/{row['n']} = {100 * row['accuracy']:.1f}%")


def _cmd_puzzles(args: argparse.Namespace) -> int:
    source = puzzles.resolve_set(args.set)
    evaluator = factory.load_evaluator(args.model, device=args.device)
    label = f"{_set_label(args.set)}_{fastchess.NAME_UNSAFE.sub('_', args.model).strip('_')}"
    out_dir = args.out or paths.home() / "eval" / "puzzles"
    illegal = 0
    for mode in factory.MODES if args.mode == "both" else (args.mode,):
        agent = factory.make_agent(mode, evaluator, epsilon=args.epsilon)
        summary = puzzles.run_puzzle_set(source, agent, mode, out_dir, limit=args.limit, label=label)
        _print_puzzles(summary, f"{agent.name} ({args.model})")
        illegal += summary["illegal_moves"]
    print(f"per-puzzle CSVs in {out_dir}")
    return 0 if illegal == 0 else 1


def _cmd_signcheck(args: argparse.Namespace) -> int:
    source = (args.data or paths.home() / "data" / "skeleton") / f"{args.split}.bin"
    records = signcheck.read_records(source, limit=args.limit)
    result = {
        "model": args.model,
        "records": str(source),
        **signcheck.signcheck(factory.load_evaluator(args.model, device=args.device), records),
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0 if result["passes_3x"] else 1


def _cmd_audit_no_search(args: argparse.Namespace) -> int:
    files = nosearch.pgn_files(args.pgn)
    report = nosearch.audit(files, engine=args.engine)
    out = nosearch.write_report(report, args.out)
    print(
        f"{report['games']} games in {report['files']} files; {report['decisions']} decisions by "
        f"{sorted(report['players'])}; rows histogram {report['histogram']}; "
        f"violations {len(report['violations'])}; missing counts {report['missing_counts']}; "
        f"forfeits {report['forfeits']}; adjudications {report['adjudications']}"
    )
    for violation in report["violations"][:10]:
        print(f"  VIOLATION {violation}")
    if report["decisions"] == 0:
        print(f"no moves by players whose name contains {args.engine!r}: nothing was audited")
    print(f"report: {out}")
    return 0 if report["compliant"] else 1


def register(subparsers: argparse._SubParsersAction) -> None:
    ev = subparsers.add_parser("eval", help="static evaluations: puzzles, sign check")
    ev_sub = ev.add_subparsers(dest="eval_command", required=True)

    pz = ev_sub.add_parser("puzzles", help="DeepMind's puzzle scorer on a puzzle set")
    pz.add_argument("--set", default="dm10k", help="dm10k or a CSV with PuzzleId, Rating, PGN, Moves")
    pz.add_argument("--limit", type=int, default=None)
    pz.add_argument(
        "--model", required=True, help="run:<name>[:ema] | ship | release:<tag> | <path> | random"
    )
    pz.add_argument("--mode", choices=MODES_OR_BOTH, default="both")
    pz.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    pz.add_argument("--epsilon", type=float, default=rules.DEFAULT_EPSILON)
    pz.add_argument("--out", type=Path, default=None, help="folder (default BLINK_HOME/eval/puzzles)")
    pz.set_defaults(func=factory.friendly(_cmd_puzzles))

    sc = ev_sub.add_parser(
        "signcheck", help="value-mode top-1 with the true child sign against a flipped one"
    )
    sc.add_argument("--model", required=True)
    sc.add_argument("--data", type=Path, default=None, help="shard folder (default BLINK_HOME/data/skeleton)")
    sc.add_argument("--split", default="val", help="reads <data>/<split>.bin")
    sc.add_argument("--limit", type=int, default=10_000)
    sc.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    sc.add_argument("--out", type=Path, default=None, help="also write the result JSON here")
    sc.set_defaults(func=factory.friendly(_cmd_signcheck))

    audit = subparsers.add_parser("audit", help="compliance audits")
    audit_sub = audit.add_subparsers(dest="audit_command", required=True)
    ns = audit_sub.add_parser(
        "no-search", help="rebuild the rows-per-move histogram from PGNs and check NSC-1"
    )
    ns.add_argument("--pgn", type=Path, required=True, help="a PGN file or a folder of them")
    ns.add_argument(
        "--engine", default=nosearch.DEFAULT_ENGINE, help="audit players whose name contains this"
    )
    ns.add_argument("--out", type=Path, default=Path("results") / "nosearch.json")
    ns.set_defaults(func=_cmd_audit_no_search)
