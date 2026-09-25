"""`blink match` (two agents, or a round robin, in process) and `blink gauntlet` (the UCI engine against
SF19 via fastchess).

The ladder's rungs are sides like any model: random (rung 0), material (rung 1) and the learned
baselines linear, mlp (BLINK_HOME/runs/baseline-<kind>/model.pt) or baseline:<path> (rungs 2 and 3).
Material and the learned rungs play value mode through the one ValueAgent and rules R1-R5 Blink's value
mode uses; their policy is flat, so their R4 ties are drawn with the side's seed (ValueAgent.tie_seed).
The learned rungs import torch only when one plays, so `blink --help` stays torch-free.
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from blink import paths
from blink.eval import books, fastchess, match, roundrobin
from blink.play import agents, factory, rules
from blink.reference import gauntlet as dm_gauntlet
from blink.reference import registry

SIDES_HELP = "random | material | linear | mlp | baseline:<path> | random-net | dm:<size> | a model selector"
is_baseline = factory.is_baseline  # the one baseline constructor, shared with E6 (blink.eval.ladder)
baseline_side = factory.baseline_side


def side_agent(side: str, mode: str, device: str, seed: int, epsilon: float) -> agents.Agent:
    """random and the baselines are the ladder's rungs; any other side is a model selector (random-net:
    random logits). The baselines always play value mode, whatever `mode` says."""
    if side == "random":
        return agents.RandomAgent(seed=seed)
    if is_baseline(side):
        return baseline_side(side, device, epsilon, seed)
    if registry.is_dm(side):
        return registry.load_agent(side, device=device)
    evaluator = factory.load_evaluator(side, device=device, seed=seed)
    return replace(
        factory.make_agent(mode, evaluator, epsilon=epsilon), name=fastchess.engine_name(side, mode)
    )


def _default_pgn(folder: str, a: str, b: str) -> Path:
    tag = f"{fastchess.NAME_UNSAFE.sub('_', a)}_vs_{fastchess.NAME_UNSAFE.sub('_', b)}"
    return paths.home() / "games" / folder / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}.pgn"


def _refuse(message: object) -> int:
    print(f"blink match: {message}", file=sys.stderr)
    return 2


def _match_line(a_name: str, b_name: str, summary: dict) -> str:
    return (
        f"{a_name} vs {b_name}: +{summary['a_wins']} ={summary['draws']} -{summary['a_losses']} "
        f"({100 * summary['a_score']:.1f}%), illegal moves {summary['illegal_moves']}, "
        f"crashes {summary['crashes']}, adjudicated {summary['adjudications']}; endings {summary['reasons']}"
    )


def _clean(summary: dict) -> bool:
    return summary["illegal_moves"] == 0 and summary["crashes"] == 0


def _cmd_match(args: argparse.Namespace) -> int:
    if args.round_robin is not None:
        if args.a or args.b:
            return _refuse("pass either --round-robin or --a and --b, not both")
        return _cmd_round_robin(args)
    if not (args.a and args.b):
        return _refuse("pass --a and --b (or --round-robin A,B,...)")
    try:
        a = side_agent(args.a, args.mode, args.device, args.seed, args.epsilon)
        b = side_agent(args.b, args.b_mode or args.mode, args.device, args.seed + 1, args.epsilon)
        openings = books.openings_for(args.book, (args.games + 1) // 2)
    except (OSError, ValueError) as exc:
        return _refuse(exc)
    pgn = args.out or _default_pgn("match", args.a, args.b)
    summary = match.run_match(a, b, openings, args.games, pgn, max_plies=args.max_plies)
    pgn.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(_match_line(a.name, b.name, summary))
    print(f"PGN: {pgn}")
    return 0 if _clean(summary) else 1


def _entrants(args: argparse.Namespace, labels: list[str]) -> list[roundrobin.Entrant]:
    """One agent per side, built once; side k's random tie-breaks are seeded seed + k."""
    return [
        roundrobin.Entrant(label, side_agent(label, args.mode, args.device, args.seed + k, args.epsilon))
        for k, label in enumerate(labels)
    ]


def _cmd_round_robin(args: argparse.Namespace) -> int:
    labels = [label.strip() for label in args.round_robin.split(",") if label.strip()]
    out = args.out or paths.home() / "games" / "round_robin" / time.strftime("%Y%m%d-%H%M%S")
    try:
        roundrobin.check_labels(labels)
        roundrobin.check_out(out, labels)
        openings = books.openings_for(args.book, (args.games + 1) // 2)
        entrants = _entrants(args, labels)
    except (OSError, ValueError) as exc:
        return _refuse(exc)
    pairs = len(roundrobin.pairs(len(labels)))
    print(f"round robin: {len(labels)} sides, {pairs} pairs, {args.games} games each on book {args.book}")
    record = roundrobin.run_round_robin(
        entrants,
        openings,
        args.games,
        out,
        max_plies=args.max_plies,
        on_pair=lambda row: print(_match_line(row["a"], row["b"], row["summary"]), flush=True),
    )
    print(roundrobin.format_table(labels, record["table"], record["scores"]))
    print(f"games and table: {out / roundrobin.RECORD}")
    return 0 if all(_clean(pair) for pair in record["pairs"]) else 1


def _gauntlet_ok(report: dict) -> bool:
    audit = report["audit"]
    return report["returncode"] == 0 and not report["blink_forfeits"] and audit["compliant"]


def _cmd_gauntlet(args: argparse.Namespace) -> int:
    out_dir = args.out or paths.home() / "games" / "gauntlet"
    is_deepmind = registry.is_dm(args.model)
    if not args.dry_run:
        (registry.check_available if is_deepmind else factory.check_available)(args.model)
    # DeepMind's engine plays under its own name and has its own moves audited (plan E7).
    runner = dm_gauntlet if is_deepmind else fastchess
    # Blink plays E2b's epsilon, as E5 does: one engine name, one configuration.
    epsilon = args.epsilon if args.epsilon is not None else match.read_epsilon(args.results_dir)
    ok = True
    for anchor in args.anchor or [1320]:
        gauntlet = runner.prepare_gauntlet(
            model=args.model,
            mode=args.mode,
            device=args.device,
            anchor=anchor,
            games=args.games,
            book=args.book,
            out_dir=out_dir,
            concurrency=args.concurrency,
            max_moves=args.max_moves,
            tc=args.tc,
            epsilon=epsilon,
        )
        if args.dry_run:
            print(subprocess.list2cmdline(gauntlet.command()))
            continue
        report = runner.execute(gauntlet)
        Path(report["pgn"]).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        audit, summary, engine = report["audit"], report["summary"] or {}, report["blink"]
        print(
            f"{engine} vs {report['anchor']}: games {summary.get('games')}, W {summary.get('wins')} "
            f"D {summary.get('draws')} L {summary.get('losses')}; "
            f"{engine} forfeits {report['blink_forfeits']}; "
            f"no-search decisions {audit['decisions']}, violations {len(audit['violations'])}; "
            f"{report['seconds']} s"
        )
        print(f"PGN: {report['pgn']}")
        ok = ok and _gauntlet_ok(report)
    return 0 if ok else 1


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mode", choices=factory.MODES, default="policy")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--book", default="dev", help="dev | final (slices of 8moves_v3) or a PGN path")


def register(subparsers: argparse._SubParsersAction) -> None:
    m = subparsers.add_parser("match", help="play two agents, or every pair of several, in process")
    m.add_argument("--a", help=SIDES_HELP)
    m.add_argument("--b", help=SIDES_HELP)
    m.add_argument(
        "--round-robin", metavar="SIDES", help="comma-separated sides: play every pair, print the cross table"
    )
    _add_model_args(m)
    m.add_argument("--b-mode", choices=factory.MODES, default=None, help="B's mode when it differs from A's")
    m.add_argument("--games", type=int, default=20)
    m.add_argument("--max-plies", type=int, default=match.MAX_ENGINE_PLIES)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--epsilon", type=float, default=rules.DEFAULT_EPSILON)
    m.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PGN path, or the round robin's folder (default under BLINK_HOME/games/match or round_robin)",
    )
    m.set_defaults(func=factory.friendly(_cmd_match))

    g = subparsers.add_parser("gauntlet", help="Blink's UCI engine against Stockfish 19 anchors (fastchess)")
    g.add_argument("--model", required=True, help="a model selector, or random for the random-logit network")
    _add_model_args(g)
    g.add_argument(
        "--anchor", type=int, action="append", help="SF19 UCI_Elo; repeat for several (default 1320)"
    )
    g.add_argument(
        "--games", type=int, default=20, help="games per anchor (even: each opening once per colour)"
    )
    g.add_argument("--concurrency", type=int, default=5)
    g.add_argument("--max-moves", type=int, default=fastchess.MAX_MOVES)
    g.add_argument("--tc", default=None, help="a cutechess time control for both engines instead of st")
    g.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Blink's R4 tie window (default: E2b's choice in <results-dir>/epsilon.json, 0 before E2b)",
    )
    g.add_argument("--results-dir", type=Path, default=Path("results"), help="where E2b wrote epsilon.json")
    g.add_argument(
        "--out", type=Path, default=None, help="folder for PGNs (default BLINK_HOME/games/gauntlet)"
    )
    g.add_argument("--dry-run", action="store_true", help="print the fastchess command and stop")
    g.set_defaults(func=factory.friendly(_cmd_gauntlet))
