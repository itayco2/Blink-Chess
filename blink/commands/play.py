"""`blink match` (two agents, in process) and `blink gauntlet` (the UCI engine against SF19 via fastchess)."""

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from blink import paths
from blink.eval import books, fastchess, match
from blink.play import agents, factory, rules


def side_agent(side: str, mode: str, device: str, seed: int, epsilon: float) -> agents.Agent:
    """random | material are baselines; any other side is a model selector (random-net: random logits)."""
    if side == "random":
        return agents.RandomAgent(seed=seed)
    if side == "material":
        return agents.MaterialAgent(seed=seed)
    evaluator = factory.load_evaluator(side, device=device, seed=seed)
    return replace(
        factory.make_agent(mode, evaluator, epsilon=epsilon), name=fastchess.engine_name(side, mode)
    )


def _default_pgn(folder: str, a: str, b: str) -> Path:
    tag = f"{fastchess.NAME_UNSAFE.sub('_', a)}_vs_{fastchess.NAME_UNSAFE.sub('_', b)}"
    return paths.home() / "games" / folder / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}.pgn"


def _cmd_match(args: argparse.Namespace) -> int:
    a = side_agent(args.a, args.mode, args.device, args.seed, args.epsilon)
    b = side_agent(args.b, args.b_mode or args.mode, args.device, args.seed + 1, args.epsilon)
    pgn = args.out or _default_pgn("match", args.a, args.b)
    try:
        openings = books.openings_for(args.book, (args.games + 1) // 2)
    except (OSError, ValueError) as exc:
        print(f"blink match: {exc}")
        return 2
    summary = match.run_match(a, b, openings, args.games, pgn, max_plies=args.max_plies)
    pgn.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{a.name} vs {b.name}: +{summary['a_wins']} ={summary['draws']} -{summary['a_losses']} "
        f"({100 * summary['a_score']:.1f}%), illegal moves {summary['illegal_moves']}, "
        f"crashes {summary['crashes']}, adjudicated {summary['adjudications']}; endings {summary['reasons']}"
    )
    print(f"PGN: {pgn}")
    return 0 if summary["illegal_moves"] == 0 and summary["crashes"] == 0 else 1


def _gauntlet_ok(report: dict) -> bool:
    audit = report["audit"]
    return report["returncode"] == 0 and not report["blink_forfeits"] and audit["compliant"]


def _cmd_gauntlet(args: argparse.Namespace) -> int:
    out_dir = args.out or paths.home() / "games" / "gauntlet"
    if not args.dry_run:
        factory.check_available(args.model)
    ok = True
    for anchor in args.anchor or [1320]:
        gauntlet = fastchess.prepare_gauntlet(
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
        )
        if args.dry_run:
            print(subprocess.list2cmdline(gauntlet.command()))
            continue
        report = fastchess.execute(gauntlet)
        Path(report["pgn"]).with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        audit, summary = report["audit"], report["summary"] or {}
        print(
            f"{report['blink']} vs {report['anchor']}: games {summary.get('games')}, W {summary.get('wins')} "
            f"D {summary.get('draws')} L {summary.get('losses')}; Blink forfeits {report['blink_forfeits']}; "
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
    m = subparsers.add_parser("match", help="play two agents against each other in process")
    m.add_argument("--a", required=True, help="random | material | random-net | a model selector")
    m.add_argument("--b", required=True, help="random | material | random-net | a model selector")
    _add_model_args(m)
    m.add_argument("--b-mode", choices=factory.MODES, default=None, help="B's mode when it differs from A's")
    m.add_argument("--games", type=int, default=20)
    m.add_argument("--max-plies", type=int, default=match.MAX_ENGINE_PLIES)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--epsilon", type=float, default=rules.DEFAULT_EPSILON)
    m.add_argument("--out", type=Path, default=None, help="PGN path (default under BLINK_HOME/games/match)")
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
        "--out", type=Path, default=None, help="folder for PGNs (default BLINK_HOME/games/gauntlet)"
    )
    g.add_argument("--dry-run", action="store_true", help="print the fastchess command and stop")
    g.set_defaults(func=factory.friendly(_cmd_gauntlet))
