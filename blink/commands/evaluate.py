"""`blink eval ...` (puzzles, signcheck, arm-metrics, the P8 suite), `blink rate`, `blink audit no-search`.

`eval arm-metrics` scores a finished P5 arm's final checkpoint on games10k and its pack's mateset.
P8 (plan section 6): `eval books` writes the dev/final slices once; `eval endgames` screens the E8 set;
`eval sprt` runs an in-process SPRT; `eval static` is E2 alone; `eval block <E?>` runs one block;
`eval all --model ship --protocol EVAL.md` runs E0-E9 and writes results/results.json; `rate` fits Ordo.

CPU budget of the SF19 labels (plan P8): `eval all` searches E2's win% regret and mate-preserving labels
and E9's failure labels on P8_SF_PROCS = 5 Stockfish processes, one thread each. P8 runs on an idle machine
(the i7-8700 has 6 cores: 5 for Stockfish, 1 for the harness); the blocks run one at a time, so no timed
block runs beside the labels, and the pool has exited before the next block's CPU check. `eval block` and
`eval static` keep 1 unless told. While a blink train, supervise or sweep process is live, `eval all`,
`eval block` and `eval static` label on 3 processes at most whatever they were given (plan P7;
blink.eval.sfbudget), and `eval endgames` follows PR-4 (EVAL.md section 5): at most 4
before P7 and 3 during it.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from blink import paths
from blink.commands import strength as strength_command
from blink.eval import (
    books,
    endgames,
    fastchess,
    match,
    nosearch,
    puzzles,
    rating,
    sfbudget,
    sflabel,
    signcheck,
    sprt,
)
from blink.play import factory, fastmode, rules
from blink.play.agents import Agent
from blink.reference import registry

MODES_OR_BOTH = (*factory.MODES, "both")
P8_SF_PROCS = 5  # `eval all`'s Stockfish processes for SF19 labels (the module docstring's CPU budget)


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


def _puzzle_agents(args: argparse.Namespace, epsilon: float) -> list[tuple[str, Agent]]:
    """(mode, agent) pairs to score; a DeepMind selector has its one mode, action-value."""
    if registry.is_dm(args.model):
        return [(registry.MODE, registry.load_agent(args.model, device=args.device))]
    evaluator = factory.load_evaluator(
        args.model, device=args.device, precision=args.precision, compile=args.compile
    )
    modes = factory.MODES if args.mode == "both" else (args.mode,)
    return [(mode, factory.make_agent(mode, evaluator, epsilon=epsilon)) for mode in modes]


def _fast_refused(prefix: str, args: argparse.Namespace) -> bool:
    """Print why the fast play mode asked for cannot play here (bf16 off CUDA, any mode for dm:)."""
    refusal = fastmode.refusal(args.precision, args.compile, args.device, deepmind=registry.is_dm(args.model))
    if refusal:
        print(f"{prefix}: {refusal}", file=sys.stderr)
    return refusal is not None


def _weights_line(model: str) -> str | None:
    """Which checkpoint a run selector scores (the run's latest when the scoring starts), so a check of a
    run that is still training can name the step it scored (blink.eval.strength reads this line). Without
    torch (a stand-in evaluator on the torch-free leg) there is no checkpoint to name."""
    if not model.startswith("run:"):
        return None
    try:
        from blink.model.loading import resolve_selector

        path, which = resolve_selector(model)
    except (ImportError, FileNotFoundError, ValueError):
        return None
    return f"weights {path} ({which})"


def _cmd_puzzles(args: argparse.Namespace) -> int:
    source = puzzles.resolve_set(args.set)
    if _missing(source, "puzzle set") or _fast_refused("blink eval puzzles", args):
        return 2
    # The engine name's tag (model tag, then the fast mode's), so results.json finds these files
    # (blink.eval.publish._blink_puzzles): a fast mode's scores are filed apart from fp32's.
    tag = fastchess.model_tag(args.model) + fastmode.tag(args.precision, args.compile)
    label = f"{_set_label(args.set)}_{tag}"
    out_dir = args.out or paths.home() / "eval" / "puzzles"
    # The published value-mode score is the shipped configuration's: E2b's epsilon unless told otherwise.
    epsilon = args.epsilon if args.epsilon is not None else match.read_epsilon(args.results_dir)
    illegal, weights = 0, _weights_line(args.model)
    if weights:
        print(weights, flush=True)
    for mode, agent in _puzzle_agents(args, epsilon):
        started = time.perf_counter()
        tie = epsilon if mode == "value" else None
        summary = puzzles.run_puzzle_set(
            source, agent, mode, out_dir, limit=args.limit, label=label, epsilon=tie
        )
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


def _cmd_arm_metrics(args: argparse.Namespace) -> int:
    """games10k_top1 and the mate rates of a finished run's final checkpoint, into its posthoc.json."""
    from blink.data import games10k, mateset
    from blink.train import posthoc
    from blink.train.status import valid_run_name

    if not valid_run_name(args.run):
        print(f"blink eval arm-metrics: bad run name {args.run!r}", file=sys.stderr)
        return 2
    refusal = posthoc.gpu_refusal(args.device)
    if refusal:
        print(f"blink eval arm-metrics: {refusal}", file=sys.stderr)
        return 2
    run_dir = paths.home() / "runs" / args.run
    pack = args.data or posthoc.pack_of(run_dir)
    games = args.games10k or games10k.default_path()
    try:
        mates = None if pack is None else pack / mateset.OUTPUT
        posthoc.score_run(
            run_dir, games, mates, args.device, lambda line: print(line, flush=True), args.force
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"blink eval arm-metrics: {exc}", file=sys.stderr)
        return 2
    return 0


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
    pz.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="value mode's R4 tie window (default: E2b's choice in <results-dir>/epsilon.json, 0 before E2b)",
    )
    pz.add_argument("--results-dir", type=Path, default=Path("results"), help="where E2b wrote epsilon.json")
    fastmode.add_arguments(pz)
    pz.add_argument("--out", type=Path, default=None, help="folder (default BLINK_HOME/eval/puzzles)")
    pz.set_defaults(func=factory.friendly(_cmd_puzzles))
    strength_command.register(ev_sub)

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

    am = ev_sub.add_parser(
        "arm-metrics",
        help="score a finished run's final checkpoint on games10k and its pack's mateset (posthoc.json)",
    )
    am.add_argument("--run", required=True, help="a run under BLINK_HOME/runs, e.g. abl-a01")
    am.add_argument(
        "--data", type=Path, default=None, help="the pack whose mateset.npz is scored (default: the run's)"
    )
    am.add_argument("--games10k", type=Path, default=None, help="default: BLINK_HOME/data/games10k.npy")
    am.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    am.add_argument("--force", action="store_true", help="score again even when already scored")
    am.set_defaults(func=_cmd_arm_metrics)

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


def _print_look(look) -> None:
    from blink.eval import endgame_looks

    print(f"look at {endgame_looks.describe(look)}{': DECLARED' if look.declares else ''}", flush=True)


def _look_plan(positions: int | None, last_line: int):
    from blink.eval import endgame_looks

    return endgame_looks.LookPlan(
        positions, last_line, first=endgame_looks.FIRST_LOOKS, every=endgame_looks.LOOK_EVERY
    )


def _screen_epd(args: argparse.Namespace, screen, confirm) -> tuple[endgames.ScreenResult, dict]:
    """endgames.epd front to back, with PR-4's looks and declaration."""
    from blink.eval import endgame_sources

    source = args.epd or endgames.epd_path()
    described = endgame_sources.describe_epd(source)
    result = endgames.screen(
        endgames.read_positions(source, args.limit),
        screen,
        confirm,
        want=args.want,
        looks=_look_plan(described["positions"], described["lines"]),
        on_look=_print_look,
    )
    branch = "epd-declared" if result.declaration else "epd"
    return result, {
        "branch": branch,
        "screens": [endgames.screen_record("dev+final", described, result, args.limit)],
    }


def _screen_fallback(args: argparse.Namespace, screen, confirm) -> tuple[endgames.ScreenResult, dict]:
    """PR-4's fallback: dev from val (outside test_grouped's groups), then final from test_grouped."""
    from blink.eval import endgame_sources

    sources = endgame_sources.fallback_sources(args.data or paths.home() / "data" / "v1")
    print("fallback source (PR-4): used only with Itay's OK, given before any conversion game", flush=True)
    results, records = {}, []
    for name, want in (("dev", endgames.DEV_COUNT), ("final", endgames.WANT - endgames.DEV_COUNT)):
        source = sources[name]
        roots = source.read(args.limit)
        described = source.describe(roots)
        plan = _look_plan(None, len(roots))
        results[name] = endgames.screen(
            source.positions(roots), screen, confirm, want=want, looks=plan, on_look=_print_look
        )
        records.append(endgames.screen_record(name, described, results[name], args.limit))
    return endgames.combine(results["dev"], results["final"]), {"branch": "fallback", "screens": records}


def _endgames_refusal(source: str, out: Path) -> str | None:
    """Why this screen may not write into `out`: endgames.epd never replaces the fallback's sets, and the
    fallback writes only where endgames.epd's declaration is recorded, which its endgames.json then keeps."""
    if source == "epd" and endgames.recorded_branch(out) == "fallback":
        return f"{out} holds the fallback source's sets (PR-4); screen endgames.epd into another --out"
    if source == "fallback" and endgames.declaration_record(out) is None:
        return (
            f"{out}/endgames.json records no PR-4 declaration by endgames.epd: the fallback source is used "
            "only after endgames.epd, screened into this --out, is declared unable to supply 700, and only "
            "with Itay's OK"
        )
    return None


def _harness(start: dict) -> dict:
    """The harness commit read before the first search (the code that screened), and whether HEAD moved
    before the sets were written (main merged into the checkout mid-screen); None outside a checkout."""
    from blink.eval import endgame_sources

    end = endgame_sources.harness_commit()
    moved = None if start["commit"] is None or end["commit"] is None else end["commit"] != start["commit"]
    return {**start, "head_changed_during_run": moved}


def _print_branch(summary: dict) -> None:
    declared = (summary.get("declared_by") or {}).get("declaration")
    if declared:
        print(f"branch {summary['branch']}; follows endgames.epd's declaration: {declared}")
    else:
        print(f"branch {summary['branch']}; declaration: {summary['declaration'] or 'none'}")


def _cmd_endgames(args: argparse.Namespace) -> int:
    from blink.eval import endgame_sources

    out = args.out or endgames.out_dir()
    refusal = _endgames_refusal(args.source, out)
    if refusal:
        return _fail("blink eval endgames", ValueError(refusal))
    if args.source == "epd" and _missing(args.epd or endgames.epd_path(), "endgame file"):
        return 2
    harness = endgame_sources.harness_commit()  # before any search: the commit whose code screens
    carried = {"declared_by": endgames.declaration_record(out)} if args.source == "fallback" else {}
    exe = fastchess.stockfish_exe()
    started = time.perf_counter()
    run = _screen_epd if args.source == "epd" else _screen_fallback
    with (
        sflabel.SfLabeler(args.screen_nodes, exe=exe, procs=args.sf_procs) as screen,
        sflabel.SfLabeler(args.confirm_nodes, exe=exe, procs=args.sf_procs) as confirm,
    ):
        try:
            result, record = run(args, screen, confirm)
        except (OSError, ValueError, KeyError) as exc:
            return _fail("blink eval endgames", exc)
        searched = screen.searched + confirm.searched
        dropped = screen.cache.dropped + confirm.cache.dropped
    summary = endgames.write_sets(result, out, {**record, **carried, "harness": _harness(harness)})
    seconds = time.perf_counter() - started
    print(
        f"screened {result.screened:,} positions: {result.passed_screen} at +5.00 after "
        f"{args.screen_nodes:,} nodes, {len(result.kept)} confirmed at {args.confirm_nodes:,}; "
        f"dev {summary['dev']}, final {summary['final']} (sharing {summary['overlap_positions']}); "
        f"{result.repeats_skipped} repeated positions skipped; {searched} new searches in {seconds:.0f} s"
    )
    _print_branch(summary)
    if dropped:
        print(f"{dropped} cache lines cut short by a kill were skipped (their labels were searched again)")
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


def _listed_pgns(lists: list[Path]) -> list[Path]:
    """The paths in --pgn-list files (one per line; blank lines and # comments skipped); all must exist."""
    listed = [
        Path(line.strip())
        for listing in lists
        for line in listing.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    missing = [str(p) for p in listed if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"listed PGNs not found: {', '.join(missing)}")
    return listed


def _cmd_rate(args: argparse.Namespace) -> int:
    try:
        listed = _listed_pgns(args.pgn_list)
    except OSError as exc:
        return _fail("blink rate", exc)
    files = [f for target in [*args.pgn, *listed] for f in nosearch.pgn_files(target)]
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
        sf_procs=args.sf_procs,
        allow_busy_cpu=args.allow_busy_cpu,
        precision=args.precision,
        compile=args.compile,
    )


def _run_guarded(prefix: str, action) -> int:
    from blink.eval import orchestrate

    try:
        outcome = action()
    except (
        orchestrate.ProtocolMismatch,
        orchestrate.TrainingLive,
        orchestrate.MachineBusy,
        orchestrate.EpsilonChanged,
        orchestrate.WeightsChanged,
        orchestrate.PlayModeChanged,
        FileNotFoundError,
        ValueError,
    ) as exc:
        return _fail(prefix, exc)
    if not isinstance(outcome, dict):
        return 0
    gates = outcome.get("gate_failures") or []
    for line in gates:
        print(f"{prefix}: done-when gate failed: {line}", file=sys.stderr)
    if outcome.get("ordo_error"):
        return _fail(prefix, RuntimeError(f"results.json written without Elo: {outcome['ordo_error']}"))
    return 1 if gates else 0


def _cmd_block(args: argparse.Namespace) -> int:
    from blink.eval import orchestrate

    if _fast_refused("blink eval block", args):
        return 2
    ctx = _context(args)
    return _run_guarded(
        "blink eval block",
        lambda: orchestrate.run_blocks(ctx, orchestrate.default_runners(), only=[args.block]),
    )


def _cmd_all(args: argparse.Namespace) -> int:
    from blink.eval import orchestrate

    if _fast_refused("blink eval all", args):
        return 2
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

    if _fast_refused("blink eval static", args):
        return 2
    ctx = _context(args)
    label = fastchess.model_tag(args.model) + ctx.mode_tag
    limits = static.StaticLimits(
        args.roots, args.value_roots, args.val_roots, args.games10k, args.mateset, args.band_puzzles
    )
    epsilon = match.read_epsilon(args.results)
    agents = match.blink_agents(args.model, args.device, epsilon=epsilon, **ctx.play_mode)
    inputs = orchestrate.static_inputs(ctx, label, epsilon)
    started = time.perf_counter()
    procs = sfbudget.sf_procs_now(args.sf_procs)
    with sflabel.SfLabeler(args.sf_nodes, exe=fastchess.stockfish_exe(), procs=procs) as labeler:
        e2 = static.run_e2(
            agents["policy"].evaluator, agents, inputs, limits, None if args.no_sf else labeler
        )
    rows = static.diagnostics_rows(e2, f"Blink-{label}")
    out = ctx.out_dir / "E2.json"
    orchestrate._write_json(
        out,
        {
            "model": args.model,
            "inputs": inputs.__dict__,
            "e2": e2,
            "diagnostics": [r.__dict__ for r in rows],
            "value_epsilon": epsilon,
            **ctx.play_mode,
        },
    )
    for row in rows:
        print({k: v for k, v in row.__dict__.items() if v not in (None, {})})
    print(f"E2 in {time.perf_counter() - started:.0f} s; report {out}")
    return 0


def _add_block_args(parser: argparse.ArgumentParser, sf_procs: int = 1) -> None:
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
    parser.add_argument(
        "--sf-procs",
        type=int,
        default=sf_procs,
        help=f"Stockfish processes (one thread each) for E2's and E9's SF19 labels (default {sf_procs}). "
        f"eval all takes {P8_SF_PROCS}, P8's idle-machine budget; while training is live, 3 at most",
    )
    parser.add_argument(
        "--allow-busy-cpu", action="store_true", help="smoke runs only: time-based blocks on a busy machine"
    )
    fastmode.add_arguments(parser)  # every Blink player of the run plays this mode; its tag ends the names


def _register_p8(ev_sub: argparse._SubParsersAction, subparsers: argparse._SubParsersAction) -> None:
    _register_sets(ev_sub)
    _register_sprt(ev_sub)
    _register_blocks(ev_sub)
    rt = subparsers.add_parser("rate", help="Ordo with fixed SF19 anchors over PGNs (-W -D -s 1000)")
    rt.add_argument("--pgn", type=Path, action="append", default=[], help="a PGN file or folder; repeatable")
    rt.add_argument(
        "--pgn-list",
        type=Path,
        action="append",
        default=[],
        help="a file of PGN paths, one per line (results/final_slice_pgns.txt); repeatable",
    )
    rt.add_argument("--anchors", type=Path, default=Path("configs") / "anchors.csv")
    rt.add_argument("--simulations", type=int, default=rating.ORDO_SIMULATIONS)
    rt.add_argument("--out", type=Path, default=None, help="default BLINK_HOME/eval/ordo/<time>")
    rt.set_defaults(func=_cmd_rate)


def _register_sets(ev_sub: argparse._SubParsersAction) -> None:
    bk = ev_sub.add_parser("books", help="write dev.pgn and final.pgn (8moves_v3 slices) once, with sha256")
    bk.add_argument("--source", type=Path, default=None, help="default BLINK_HOME/books/8moves_v3.pgn")
    bk.add_argument("--out", type=Path, default=None, help="default BLINK_HOME/books")
    bk.set_defaults(func=_cmd_books)

    eg = ev_sub.add_parser("endgames", help="screen endgames.epd with SF19 for the E8 conversion set")
    eg.add_argument(
        "--source",
        choices=("epd", "fallback"),
        default="epd",
        help="epd: endgames.epd (the plan's set). fallback: PR-4's source from the v1 pack (EVAL.md "
        "section 5), only after endgames.epd, screened into the same --out, is declared unable to supply "
        "700, with Itay's OK given before any conversion game",
    )
    eg.add_argument("--epd", type=Path, default=None, help="default BLINK_HOME/books/endgames.epd")
    eg.add_argument(
        "--data", type=Path, default=None, help="--source fallback: the pack (default BLINK_HOME/data/v1)"
    )
    eg.add_argument("--limit", type=int, default=None, help="screen at most this many lines (records)")
    eg.add_argument(
        "--want",
        type=int,
        default=endgames.WANT,
        help="epd: stop at this many kept (the fallback: 200 + 500)",
    )
    eg.add_argument("--screen-nodes", type=int, default=endgames.SCREEN_NODES)
    eg.add_argument("--confirm-nodes", type=int, default=endgames.CONFIRM_NODES)
    eg.add_argument("--out", type=Path, default=None, help="default BLINK_HOME/eval/endgames")
    eg.add_argument(
        "--sf-procs",
        type=int,
        default=1,
        help="Stockfish processes, one thread each (PR-4: at most 4 before P7, 3 during it)",
    )
    eg.set_defaults(func=_cmd_endgames)


def _register_sprt(ev_sub: argparse._SubParsersAction) -> None:
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


def _register_blocks(ev_sub: argparse._SubParsersAction) -> None:
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
    _add_block_args(al, sf_procs=P8_SF_PROCS)
    al.add_argument("--only", default=None, help="comma-separated blocks, still run in the plan's order")
    al.add_argument("--dry-run", action="store_true", help="print the protocol check and the game table")
    al.set_defaults(func=factory.friendly(_cmd_all))
