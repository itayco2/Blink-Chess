"""`blink eval ...` (puzzles, signcheck, the P8 suite), `blink rate` and `blink audit no-search`.

P8 (plan section 6): `eval books` writes the dev/final slices once; `eval endgames` screens the E8 set;
`eval sprt` runs an in-process SPRT; `eval static` is E2 alone; `eval block <E?>` runs one block;
`eval all --model ship --protocol EVAL.md` runs E0-E9 and writes results/results.json; `rate` fits Ordo.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from blink import paths
from blink.eval import books, endgames, fastchess, match, nosearch, puzzles, rating, sflabel, signcheck, sprt
from blink.play import factory, rules
from blink.play.agents import Agent
from blink.reference import registry

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


def _missing(path: Path, what: str) -> bool:
    if path.is_file():
        return False
    print(f"blink eval: {what} {path} does not exist")
    return True


def _puzzle_agents(args: argparse.Namespace) -> list[tuple[str, Agent]]:
    """(mode, agent) pairs to score; a DeepMind selector has its one mode, action-value."""
    if registry.is_dm(args.model):
        return [(registry.MODE, registry.load_agent(args.model, device=args.device))]
    evaluator = factory.load_evaluator(args.model, device=args.device)
    modes = factory.MODES if args.mode == "both" else (args.mode,)
    return [(mode, factory.make_agent(mode, evaluator, epsilon=args.epsilon)) for mode in modes]


def _cmd_puzzles(args: argparse.Namespace) -> int:
    source = puzzles.resolve_set(args.set)
    if _missing(source, "puzzle set"):
        return 2
    label = f"{_set_label(args.set)}_{fastchess.NAME_UNSAFE.sub('_', args.model).strip('_')}"
    out_dir = args.out or paths.home() / "eval" / "puzzles"
    illegal = 0
    for mode, agent in _puzzle_agents(args):
        started = time.perf_counter()
        summary = puzzles.run_puzzle_set(source, agent, mode, out_dir, limit=args.limit, label=label)
        seconds = time.perf_counter() - started
        _print_puzzles(summary, f"{agent.name} ({args.model})")
        per_puzzle_ms = 1000 * seconds / max(summary["n"], 1)
        print(f"  {summary['n']} puzzles in {seconds:.1f} s ({per_puzzle_ms:.0f} ms each)")
        illegal += summary["illegal_moves"]
    print(f"per-puzzle CSVs in {out_dir}")
    return 0 if illegal == 0 else 1


def _cmd_signcheck(args: argparse.Namespace) -> int:
    source = (args.data or paths.home() / "data" / "skeleton") / f"{args.split}.bin"
    if _missing(source, "record file"):
        return 2
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
        "--model",
        required=True,
        help="run:<name>[:ema] | ship | release:<tag> | <path> | random | dm:9M[:ema] (DeepMind)",
    )
    pz.add_argument("--mode", choices=MODES_OR_BOTH, default="both", help="ignored for dm: selectors")
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

    _register_p8(ev_sub, subparsers)


# ---------------------------------------------------------------------- P8: books, endgames, SPRT, rate


def _fail(prefix: str, exc: BaseException) -> int:
    print(f"{prefix}: {exc}", file=sys.stderr)
    return 2


def _cmd_books(args: argparse.Namespace) -> int:
    source = args.source or books.book_file()
    out = args.out or books.books_dir()
    try:
        manifest = books.write_slices(source, out, slices=books.SLICES)
    except (OSError, ValueError) as exc:
        return _fail("blink eval books", exc)
    for name in books.SLICES:
        row = manifest[name]
        span = f"{row['first']}-{row['last']} ({row['openings']:,})"
        print(f"{row['file']}: openings {span}, sha256 {row['sha256']}")
    print(
        f"source {manifest['source']['file']} sha256 {manifest['source']['sha256']}; "
        f"move sequences in both slices: {manifest['overlap_move_sequences']}"
    )
    return 0


def _cmd_endgames(args: argparse.Namespace) -> int:
    source = args.epd or endgames.epd_path()
    if _missing(source, "endgame file"):
        return 2
    exe = fastchess.stockfish_exe()
    started = time.perf_counter()
    with (
        sflabel.SfLabeler(args.screen_nodes, exe=exe) as screen,
        sflabel.SfLabeler(args.confirm_nodes, exe=exe) as confirm,
    ):
        result = endgames.screen(endgames.read_positions(source, args.limit), screen, confirm, want=args.want)
        searched = screen.searched + confirm.searched
    summary = endgames.write_sets(result, args.out or endgames.out_dir())
    seconds = time.perf_counter() - started
    print(
        f"screened {result.screened:,} positions: {result.passed_screen} at +5.00 after "
        f"{args.screen_nodes:,} nodes, {len(result.kept)} confirmed at {args.confirm_nodes:,}; "
        f"dev {summary['dev']}, final {summary['final']}; {searched} new searches in {seconds:.0f} s"
    )
    for game in result.kept:
        scores = f"+{game.screen_pawns:.2f} / +{game.confirm_pawns:.2f}"
        print(f"  line {game.line}: {game.winner} {scores}  {game.fen}")
    return 0


def _sprt_agent(side: str, mode: str, args: argparse.Namespace) -> Agent:
    from blink.commands.play import side_agent

    return side_agent(side, mode, args.device, args.seed, args.epsilon)


def _cmd_sprt(args: argparse.Namespace) -> int:
    config = sprt.SprtConfig(args.elo0, args.elo1, args.alpha, args.beta, args.games)
    a = _sprt_agent(args.a, args.a_mode, args)
    b = _sprt_agent(args.b, args.b_mode or args.a_mode, args)
    try:
        openings = books.openings_for(args.book, config.cap_games // 2)
    except (OSError, ValueError) as exc:
        return _fail("blink eval sprt", exc)
    tag = "_vs_".join(fastchess.NAME_UNSAFE.sub("_", agent.name) for agent in (a, b))
    pgn = args.out or paths.home() / "eval" / "sprt" / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}.pgn"
    result = sprt.run_sprt(match.pair_player(a, b, openings, pgn, max_plies=args.max_plies), config)
    pgn.with_suffix(".json").write_text(
        json.dumps({"a": a.name, "b": b.name, **result.as_dict()}, indent=2), encoding="utf-8"
    )
    verdict = result.verdict or ("cap" if result.capped else "running")
    print(
        f"{a.name} vs {b.name}: {result.games} games, pentanomial {list(result.penta)}, LLR {result.llr:.3f} "
        f"(bounds {result.lower:.3f}, {result.upper:.3f}), verdict {verdict}, "
        f"Elo {result.elo:+.1f} +- {result.elo_ci95:.1f}"
    )
    print(f"PGN: {pgn}")
    return 0


def _cmd_rate(args: argparse.Namespace) -> int:
    files = [f for target in args.pgn for f in nosearch.pgn_files(target)]
    if not files:
        print("blink rate: no PGN files", file=sys.stderr)
        return 2
    out = args.out or paths.home() / "eval" / "ordo" / time.strftime("%Y%m%d-%H%M%S")
    try:
        fit = rating.run_ordo(files, rating.read_anchors(args.anchors), out, args.simulations)
    except (OSError, RuntimeError, ValueError) as exc:
        return _fail("blink rate", exc)
    (out / "rating.json").write_text(json.dumps(fit.as_dict(), indent=2), encoding="utf-8")
    for row in fit.rows:
        error = "fixed" if row.error is None else f"+- {row.error:.1f}"
        flag = " (extrapolated)" if row.extrapolated and row.error is not None else ""
        print(f"  {row.player:<40} {row.rating:7.1f} {error:>9}  {row.points:g}/{row.played}{flag}")
    for player, why in fit.excluded.items():
        entry = fit.tally[player]
        print(f"  {player:<40} not rated: {why} ({entry['points']:g}/{int(entry['games'])})")
    print(f"white advantage and draw rate: {fit.extras or 'not fitted'}; report {out / 'rating.json'}")
    return 0


# ------------------------------------------------------------------------------ P8: static, blocks, all


def _context(args: argparse.Namespace):
    from blink.eval import orchestrate

    stamp = time.strftime("%Y%m%d-%H%M%S")
    return orchestrate.EvalContext(
        model=args.model,
        device=args.device,
        out_dir=args.out or paths.home() / "eval" / "p8" / stamp,
        results_dir=args.results,
        protocol=args.protocol,
        games=args.games,
        positions=args.positions,
        concurrency=args.concurrency,
        mode=args.mode,
        film_run=args.film_run,
        side_models=tuple(args.side_model or ()),
        data_dir=args.data,
        selfcheck_tc=args.selfcheck_tc,
    )


def _run_guarded(prefix: str, action) -> int:
    from blink.eval import orchestrate

    try:
        action()
    except (orchestrate.ProtocolMismatch, orchestrate.TrainingLive, FileNotFoundError, ValueError) as exc:
        return _fail(prefix, exc)
    return 0


def _cmd_block(args: argparse.Namespace) -> int:
    from blink.eval import orchestrate

    ctx = _context(args)
    return _run_guarded(
        "blink eval block",
        lambda: orchestrate.run_blocks(ctx, orchestrate.default_runners(), only=[args.block]),
    )


def _cmd_all(args: argparse.Namespace) -> int:
    from blink.eval import orchestrate

    ctx = _context(args)
    only = args.only.split(",") if args.only else None
    if args.dry_run:

        def show() -> None:
            print(json.dumps(orchestrate.check_protocol(ctx.protocol), indent=2))
            print(
                orchestrate.game_table(
                    [b for b in orchestrate.BLOCK_ORDER if not only or b in only], ctx.games
                )
            )
            print(f"live training runs: {orchestrate.live_training_runs() or 'none'}")

        return _run_guarded("blink eval all", show)
    return _run_guarded("blink eval all", lambda: orchestrate.run_all(ctx, only=only))


def _cmd_static(args: argparse.Namespace) -> int:
    from blink.eval import orchestrate, static

    ctx = _context(args)
    label = fastchess.NAME_UNSAFE.sub("_", args.model).strip("_")
    limits = static.StaticLimits(
        args.roots, args.value_roots, args.val_roots, args.games10k, args.mateset, args.band_puzzles
    )
    agents = match.blink_agents(args.model, args.device)
    inputs = orchestrate.static_inputs(ctx, label)
    started = time.perf_counter()
    with sflabel.SfLabeler(args.sf_nodes, exe=fastchess.stockfish_exe()) as labeler:
        e2 = static.run_e2(
            agents["policy"].evaluator, agents, inputs, limits, None if args.no_sf else labeler
        )
    rows = static.diagnostics_rows(e2, f"Blink-{label}")
    out = ctx.out_dir / "E2.json"
    orchestrate._write_json(
        out,
        {"model": args.model, "inputs": inputs.__dict__, "e2": e2, "diagnostics": [r.__dict__ for r in rows]},
    )
    for row in rows:
        print({k: v for k, v in row.__dict__.items() if v not in (None, {})})
    print(f"E2 in {time.perf_counter() - started:.0f} s; report {out}")
    return 0


def _add_block_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True, help="the model under test: run:<name>[:ema] | ship | <path>"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--mode", choices=factory.MODES, default=None, help="the shipped mode, before E3 chose"
    )
    parser.add_argument("--games", type=int, default=None, help="smoke: every match (and SPRT cap) this long")
    parser.add_argument(
        "--positions", type=int, default=None, help="smoke: cap for static sets and SF labels"
    )
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--film-run", default=None, help="the run whose film frames E1 and E4b use")
    parser.add_argument("--side-model", action="append", help="a 6 GPU-h size or s10m for E5's side rows")
    parser.add_argument("--data", type=Path, default=None, help="pack folder (default BLINK_HOME/data/v1)")
    parser.add_argument("--out", type=Path, default=None, help="folder (default BLINK_HOME/eval/p8/<time>)")
    parser.add_argument("--results", type=Path, default=Path("results"), help="where results.json goes")
    parser.add_argument("--protocol", type=Path, default=Path("EVAL.md"))
    parser.add_argument("--selfcheck-tc", default="120+1", help="E0: the slow side of SF's self-check")


def _register_p8(ev_sub: argparse._SubParsersAction, subparsers: argparse._SubParsersAction) -> None:
    bk = ev_sub.add_parser("books", help="write dev.pgn and final.pgn (8moves_v3 slices) once, with sha256")
    bk.add_argument("--source", type=Path, default=None, help="default BLINK_HOME/books/8moves_v3.pgn")
    bk.add_argument("--out", type=Path, default=None, help="default BLINK_HOME/books")
    bk.set_defaults(func=_cmd_books)

    eg = ev_sub.add_parser("endgames", help="screen endgames.epd with SF19 for the E8 conversion set")
    eg.add_argument("--epd", type=Path, default=None, help="default BLINK_HOME/books/endgames.epd")
    eg.add_argument("--limit", type=int, default=None, help="screen at most this many lines")
    eg.add_argument("--want", type=int, default=endgames.WANT)
    eg.add_argument("--screen-nodes", type=int, default=endgames.SCREEN_NODES)
    eg.add_argument("--confirm-nodes", type=int, default=endgames.CONFIRM_NODES)
    eg.add_argument("--out", type=Path, default=None, help="default BLINK_HOME/eval/endgames")
    eg.set_defaults(func=_cmd_endgames)

    sp = ev_sub.add_parser("sprt", help="an in-process SPRT between two agents (pentanomial, fishtest LLR)")
    sp.add_argument("--a", required=True, help="random | material | random-net | a model selector")
    sp.add_argument("--b", required=True, help="random | material | random-net | a model selector")
    sp.add_argument("--a-mode", choices=factory.MODES, default="value")
    sp.add_argument("--b-mode", choices=factory.MODES, default=None)
    sp.add_argument("--games", type=int, default=sprt.MODE_SPRT.cap_games, help="the cap, in games")
    sp.add_argument("--elo0", type=float, default=sprt.MODE_SPRT.elo0)
    sp.add_argument("--elo1", type=float, default=sprt.MODE_SPRT.elo1)
    sp.add_argument("--alpha", type=float, default=sprt.MODE_SPRT.alpha)
    sp.add_argument("--beta", type=float, default=sprt.MODE_SPRT.beta)
    sp.add_argument("--book", default="dev")
    sp.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--epsilon", type=float, default=rules.DEFAULT_EPSILON)
    sp.add_argument("--max-plies", type=int, default=match.MAX_ENGINE_PLIES)
    sp.add_argument("--out", type=Path, default=None, help="PGN path (default BLINK_HOME/eval/sprt)")
    sp.set_defaults(func=factory.friendly(_cmd_sprt))

    st = ev_sub.add_parser("static", help="E2 alone: the static metrics of one model")
    _add_block_args(st)
    for flag, default in (
        ("--roots", 1_000_000),
        ("--value-roots", 200_000),
        ("--val-roots", 50_000),
        ("--games10k", 10_000),
        ("--mateset", 2_000),
        ("--band-puzzles", 500),
    ):
        st.add_argument(flag, type=int, default=default)
    st.add_argument("--sf-nodes", type=int, default=1_000_000)
    st.add_argument(
        "--no-sf", action="store_true", help="skip the SF19-labelled numbers (regret, mate-preserving)"
    )
    st.set_defaults(func=factory.friendly(_cmd_static))

    blk = ev_sub.add_parser("block", help="run one P8 block alone (E0 ... E9)")
    blk.add_argument("block", help="E0 | E1 | E2 | E2b | E3 | E4 | E4b | E5 | E6 | E7 | E8 | E9")
    _add_block_args(blk)
    blk.set_defaults(func=factory.friendly(_cmd_block))

    al = ev_sub.add_parser("all", help="every P8 block in the plan's order, then results/results.json")
    _add_block_args(al)
    al.add_argument("--only", default=None, help="comma-separated blocks, still run in the plan's order")
    al.add_argument("--dry-run", action="store_true", help="print the protocol check and the game table")
    al.set_defaults(func=factory.friendly(_cmd_all))

    rt = subparsers.add_parser("rate", help="Ordo with fixed SF19 anchors over PGNs (-W -D -s 1000)")
    rt.add_argument(
        "--pgn", type=Path, action="append", required=True, help="a PGN file or folder; repeatable"
    )
    rt.add_argument("--anchors", type=Path, default=Path("configs") / "anchors.csv")
    rt.add_argument("--simulations", type=int, default=rating.ORDO_SIMULATIONS)
    rt.add_argument("--out", type=Path, default=None, help="default BLINK_HOME/eval/ordo/<time>")
    rt.set_defaults(func=_cmd_rate)
