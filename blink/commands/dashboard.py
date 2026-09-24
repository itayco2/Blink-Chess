"""`blink dashboard`, `blink film ...` and `blink report ...`: what a run shows while and after it trains.

- dashboard: the live training dashboard on http://127.0.0.1:8767 (loopback only, by design).
- film pick | extract | render: the 21-frame learning film (plan P11).
- report scoreboard | claims | compute: README numbers, the hook and the claim from results/*.json (P12).
"""

import argparse
import functools
import sys
from collections.abc import Callable
from pathlib import Path

from blink import paths
from blink.dashboard import server

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
README = REPO_ROOT / "README.md"
FILM_FRAMES = 21


def cmd_dashboard(args: argparse.Namespace) -> int:
    server.serve(paths.home() / "runs", args.port)
    return 0


# ------------------------------------------------------------------------------------------ report


def cmd_scoreboard(args: argparse.Namespace) -> int:
    from blink.report import scoreboard as sb

    try:
        block = sb.render_block(sb.load_bundle(Path(args.results)))
        if args.write:
            sb.write_readme(Path(args.readme), block)
            print(f"wrote the scoreboard into {args.readme}")
            return 0
    except sb.ScoreboardError as exc:
        print(f"scoreboard: {exc}", file=sys.stderr)
        return 1
    if not args.check:
        print(block, end="")
        return 0
    problems = sb.check_readme(Path(args.readme), block)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print(f"{args.readme}: the scoreboard equals results/*.json byte for byte")
    return 1 if problems else 0


def cmd_claims(args: argparse.Namespace) -> int:
    from blink.report import claims

    try:
        text = claims.fill_claim(Path(args.results))
    except claims.ClaimRefused as exc:
        print(f"claims refused: {exc}", file=sys.stderr)
        return 1
    mode = claims.shipped_mode(Path(args.results))
    print(f"HOOK_EN: {claims.hook('en', mode)}")
    print(f"HOOK_HE: {claims.hook('he', mode)}")
    print()
    print(text)
    return 0


def cmd_compute(args: argparse.Namespace) -> int:
    from blink.report import compute

    names = [n for n in args.runs.split(",") if n] if args.runs else None
    report = compute.project_compute(Path(args.runs_root), flagship=args.flagship, names=names)
    out = compute.write_compute(report, Path(args.out))
    print(compute.summary(report))
    print(f"wrote {out}")
    return 0


def _register_report(sub: argparse._SubParsersAction) -> None:
    report = sub.add_parser("report", help="README scoreboard, hook, claim and compute from results/*.json")
    rsub = report.add_subparsers(dest="report_command", required=True)

    score = rsub.add_parser("scoreboard", help="print, --write or --check the README scoreboard block")
    score.add_argument("--results", default=str(RESULTS_DIR))
    score.add_argument("--readme", default=str(README))
    group = score.add_mutually_exclusive_group()
    group.add_argument("--write", action="store_true", help="replace the block between the README markers")
    group.add_argument("--check", action="store_true", help="exit 1 unless the README block is byte-equal")
    score.set_defaults(func=cmd_scoreboard)

    claim = rsub.add_parser("claims", help="the hook and the pre-registered claim, or why it is refused")
    claim.add_argument("--results", default=str(RESULTS_DIR))
    claim.set_defaults(func=cmd_claims)

    comp = rsub.add_parser("compute", help="GPU-hours and GPU-board kWh from run telemetry -> compute.json")
    comp.add_argument("--runs-root", default=str(paths.home() / "runs"))
    comp.add_argument("--flagship", default="long", help="the flagship run's name (default: long)")
    comp.add_argument("--runs", help="comma-separated run names to count (default: every run folder)")
    comp.add_argument("--out", default=str(RESULTS_DIR / "compute.json"))
    comp.set_defaults(func=cmd_compute)


# ------------------------------------------------------------------------------------------ film


def _checked_run(run: str) -> str:
    from blink.film.extract import FilmError
    from blink.train.status import valid_run_name

    if not valid_run_name(run):
        raise FilmError(f"bad run name {run!r}: letters, digits, '_', '-' and '.', never a path")
    return run


def _film_dir(run: str) -> Path:
    return paths.home() / "film" / _checked_run(run)


def _run_dir(run: str) -> Path:
    return paths.home() / "runs" / _checked_run(run)


def _film_errors(func: Callable[[argparse.Namespace], int]) -> Callable[[argparse.Namespace], int]:
    """A FilmError (a leak, mixed worlds, a missing puzzle, ffmpeg failing) is a message and exit 1."""

    @functools.wraps(func)
    def run(args: argparse.Namespace) -> int:
        from blink.film.extract import FilmError

        try:
            return func(args)
        except FilmError as exc:
            print(f"film: {exc}", file=sys.stderr)
            return 1

    return run


def cmd_film_pick(args: argparse.Namespace) -> int:
    from blink.film import extract, pick

    sources = extract.frame_sources(_run_dir(args.run))
    candidates = pick.draw_candidates(Path(args.bands), args.candidates)
    ranked = pick.rank(candidates, sources, device=args.device)
    out = pick.write_ranking(ranked, _film_dir(args.run) / "candidates.json", args.run, len(sources))
    print(pick.format_top(ranked, args.top))
    print(f"scored {len(candidates)} candidates over {len(sources)} frames; wrote {out}")
    return 0


def cmd_film_extract(args: argparse.Namespace) -> int:
    from blink.film import extract

    run_dir = _run_dir(args.run)
    position = extract.film_position(Path(args.bands), args.puzzle)
    rungs = extract.ladder_rungs(
        Path(args.results), [extract.parse_milestone(text) for text in args.milestone]
    )
    film = extract.extract(
        run_dir,
        position,
        Path(args.blocklist),
        pad_to=args.pad_to,
        device=args.device,
        milestones=rungs,
        pack_dir=Path(args.pack) if args.pack else None,
    )
    out = extract.write_film(film, Path(args.out) if args.out else _film_dir(args.run) / "film.json")
    total = len(film["frames"])
    real = sum(not f["interpolated"] for f in film["frames"])
    print(f"{total} frames ({real} measured, {total - real} interpolated); wrote {out}")
    return 0


def cmd_film_render(args: argparse.Namespace) -> int:
    from blink.film import render

    film_path = Path(args.film) if args.film else _film_dir(args.run) / "film.json"
    out = Path(args.out) if args.out else film_path.with_name(f"film-{args.lang}.mp4")
    report = render.render(
        film_path, out, args.lang, fps=args.fps, mode=args.mode, results_dir=Path(args.results)
    )
    print(render.format_report(report))
    return 0 if not report["problems"] else 1


def _register_film(sub: argparse._SubParsersAction) -> None:
    film = sub.add_parser("film", help="the 21-frame learning film: pick, extract, render")
    fsub = film.add_subparsers(dest="film_command", required=True)
    bands = str(paths.home() / "eval" / "lichess_bands.csv")

    pick = fsub.add_parser("pick", help="story-score the 200 film candidates across a run's frames")
    pick.add_argument("--run", required=True)
    pick.add_argument("--top", type=int, default=5)
    pick.add_argument("--candidates", type=int, default=200)
    pick.add_argument("--bands", default=bands)
    pick.add_argument("--device", default="cpu")
    pick.set_defaults(func=_film_errors(cmd_film_pick))

    extract = fsub.add_parser("extract", help="per-frame predictions on one position -> film.json")
    extract.add_argument("--run", required=True)
    extract.add_argument("--puzzle", required=True, help="the PuzzleId Itay picked at G9")
    extract.add_argument("--bands", default=bands)
    extract.add_argument("--blocklist", default=str(paths.home() / "data" / "blocklist_v1.npy"))
    extract.add_argument("--pack", help="the run's pack directory, if it moved since config.json was written")
    extract.add_argument(
        "--pad-to", type=int, default=None, help=f"interpolate up to N frames (the film: {FILM_FRAMES})"
    )
    extract.add_argument("--device", default="cpu")
    extract.add_argument(
        "--milestone",
        action="append",
        default=[],
        help="LABEL=AGENT: flash LABEL when the run passes that ladder row's measured VAA (else top-1)",
    )
    extract.add_argument(
        "--results", default=str(RESULTS_DIR), help="results/ holding the ladder's results.json"
    )
    extract.add_argument("--out")
    extract.set_defaults(func=_film_errors(cmd_film_extract))

    render = fsub.add_parser("render", help="film.json -> a 1080x1350 4:5 mp4 (Edge + ffmpeg)")
    render.add_argument("--lang", choices=("en", "he"), required=True)
    render.add_argument("--fps", type=int, default=30)
    render.add_argument("--run", default="long")
    render.add_argument("--film", help="film.json (default: BLINK_HOME/film/<run>/film.json)")
    render.add_argument("--out")
    render.add_argument(
        "--mode", choices=("policy", "value"), help="a preview's hook mode, only before a mode ships"
    )
    render.add_argument("--results", default=str(RESULTS_DIR))
    render.set_defaults(func=_film_errors(cmd_film_render))


def register(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("dashboard", help="live training dashboard on 127.0.0.1 (never the LAN)")
    parser.add_argument("--port", type=int, default=server.DEFAULT_PORT)
    parser.set_defaults(func=cmd_dashboard)
    _register_report(sub)
    _register_film(sub)
