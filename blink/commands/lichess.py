"""The Lichess bot's commands (plan P9), moved out of blink.commands.ops when fast play made that file
too long.

blink lichess config|check-config               the bot's config.yml and config.casual.yml
blink lichess snapshot|check|pause|resume-note  public-API numbers, the stop rule, the bot pause
blink lichess watch                             the stop rule enforced every 2 minutes, no agent needed

Nothing here imports torch, so `blink --help` and the torch-free tier keep working.
"""

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from blink import paths
from blink.play import fastmode

EXIT_REFUSED = 2


def _say(line: str) -> None:
    print(line, flush=True)


def _refuse(command: str, exc: Exception) -> int:
    print(f"blink lichess {command}: {exc}", file=sys.stderr)
    return EXIT_REFUSED


def _positive(kind: type, zero_ok: bool = False) -> Callable[[str], int | float]:
    """An argparse type: an int or float above zero (a zero poll would spin on the public API).

    `zero_ok` admits zero too: a pause --timeout of 0 stops the bot without waiting for its game.
    """

    def parse(text: str) -> int | float:
        value = kind(text)
        if value < 0 or (value == 0 and not zero_ok):
            raise argparse.ArgumentTypeError(
                f"must be {'0 or more' if zero_ok else 'above zero'}, got {text}"
            )
        return value

    return parse


def cmd_lichess_config(args: argparse.Namespace) -> int:
    from blink.lichess import config

    spec = config.BotSpec(args.model, args.mode, args.sha, args.casual_model)
    kinds = (args.only,) if args.only else config.KINDS
    try:
        written = config.generate(
            spec,
            Path(args.out_dir),
            kinds=kinds,
            templates=Path(args.templates),
            results_dir=args.results_dir,
        )
    except (config.ConfigError, FileNotFoundError) as exc:
        return _refuse("config", exc)
    for kind, path in written.items():
        stamp = config.load_config(path)["blink"]
        sha = f", sha256 {stamp['sha'][:12]}" if stamp.get("sha") else ""
        tie = f", epsilon {stamp['epsilon']!r}" if "epsilon" in stamp else ""
        played = (stamp.get("precision", fastmode.DEFAULT_PRECISION), stamp.get("compile", False))
        fast = "" if fastmode.is_default(*played) else f", {fastmode.describe(*played)}"
        _say(f"{kind}: {path} (model {stamp['model']}, mode {stamp['mode']}{sha}{tie}{fast})")
    _say("check them with `blink lichess check-config --config <file>`; the token is never written")
    return 0


def cmd_lichess_check_config(args: argparse.Namespace) -> int:
    from blink.lichess import config

    results = None if args.results.lower() == "none" else Path(args.results)
    try:
        report = config.check_file(Path(args.config), kind=args.kind, results=results)
    except (config.ConfigError, FileNotFoundError, ValueError) as exc:
        return _refuse("check-config", exc)
    for note in report.notes:
        _say(f"  note: {note}")
    for problem in report.problems:
        _say(f"  problem: {problem}")
    _say(f"{args.config} ({report.kind or 'unknown kind'}): {len(report.problems)} problems")
    return 1 if report.problems else 0


def cmd_lichess_snapshot(args: argparse.Namespace) -> int:
    import datetime

    from blink.lichess import snapshot

    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    try:
        snapshot.check_name(args.bot)
        report = snapshot.take_snapshot(snapshot.default_api(), args.bot, args.max_games, today)
    except (ValueError, snapshot.ApiError, OSError) as exc:
        return _refuse("snapshot", exc)
    for line in snapshot.format_report(report, args.max_games):
        _say(line)
    if args.no_write:
        _say("--no-write: nothing written")
        return 0
    _say(f"-> {snapshot.write_snapshot(report.snapshot, Path(args.out))}")
    return 0


def _pause_bot(bot: str, api, args: argparse.Namespace, reason: str) -> None:
    from blink.lichess import pause

    deps = pause.default_deps(bot, api, Path(args.bot_root))
    record = pause.pause(bot, deps, paths.home() / "lichess", args.timeout, args.poll, reason)
    for line in pause.format_record(record):
        _say(line)


def _pgn_dir(args: argparse.Namespace) -> Path:
    return Path(args.pgn_dir) if args.pgn_dir else paths.home() / "lichess" / "pgn"


def cmd_lichess_check(args: argparse.Namespace) -> int:
    from blink.lichess import monitor, snapshot, watch

    try:
        snapshot.check_name(args.bot)
        api = snapshot.default_api()
        verdict = monitor.check(api, args.bot, window=args.window, pgn_dir=_pgn_dir(args))
    except (ValueError, snapshot.ApiError, OSError) as exc:
        return _refuse("check", exc)
    _say(monitor.format_verdict(args.bot, verdict))
    if verdict.stop and args.stop:
        try:
            action = watch.enforce(
                verdict,
                paths.home() / "lichess",
                lambda reason: _pause_bot(args.bot, api, args, reason),
                watch.utc_now(),
                _say,
            )
        except OSError as exc:
            return _refuse("check --stop", exc)
        _say(f"action: {action}")
    return 1 if verdict.stop else 0


def cmd_lichess_watch(args: argparse.Namespace) -> int:
    from blink.lichess import monitor, snapshot, watch

    try:
        snapshot.check_name(args.bot)
    except ValueError as exc:
        return _refuse("watch", exc)
    api = snapshot.default_api()
    pgn_dir, lichess_dir = _pgn_dir(args), paths.home() / "lichess"
    deps = watch.WatchDeps(
        exported=lambda: monitor.exported_games(api, args.bot, args.window),
        local=lambda: monitor.games_from_pgns(pgn_dir, args.bot),
        pause_now=lambda reason: _pause_bot(args.bot, api, args, reason),
        log=_say,
    )
    _say(
        f"watching {args.bot} every {args.every:g} s: PGNs in {pgn_dir}, heartbeat "
        f"{lichess_dir / watch.HEARTBEAT_NAME}; a firing rule pauses the bot (--timeout {args.timeout:g})"
    )
    watch.watch(args.bot, deps, lichess_dir, args.every, args.rounds, args.window)
    return 0


def cmd_lichess_pause(args: argparse.Namespace) -> int:
    from blink.lichess import snapshot

    try:
        snapshot.check_name(args.bot)
        _pause_bot(args.bot, snapshot.default_api(), args, args.reason)
    except (ValueError, OSError) as exc:
        return _refuse("pause", exc)
    return 0


def cmd_lichess_resume_note(args: argparse.Namespace) -> int:
    from blink.lichess import pause

    _say(pause.resume_note(paths.home() / "lichess" / pause.FLAG_NAME))
    return 0


def _pause_options(parser: argparse.ArgumentParser, timeout: float) -> None:
    from blink.lichess import pause

    parser.add_argument(
        "--timeout",
        type=_positive(float, zero_ok=True),
        default=timeout,
        help=f"seconds to wait for no game before stopping the bot (default {timeout:g})",
    )
    parser.add_argument(
        "--poll", type=_positive(float), default=pause.DEFAULT_POLL_S, help="seconds between status reads"
    )
    parser.add_argument("--bot-root", default=str(pause.BOT_ROOT), help="the lichess-bot folder to match")


def _register_lichess_config(actions: argparse._SubParsersAction) -> None:
    from blink.lichess import config

    gen = actions.add_parser(
        "config", help="write config.yml and config.casual.yml from the tracked templates"
    )
    gen.add_argument("--model", required=True, help="the rated bot's model selector (casual too, by default)")
    gen.add_argument("--mode", required=True, choices=config.MODES)
    gen.add_argument("--sha", help="the shipped weights' sha256 (default: hashed from the weights file)")
    gen.add_argument("--casual-model", help="the preview model for the G5 casual smoke (default: --model)")
    gen.add_argument("--only", choices=config.KINDS, help="write just one of the two configs")
    gen.add_argument("--out-dir", default=str(config.DEFAULT_OUT_DIR))
    gen.add_argument("--templates", default=str(config.TEMPLATE_DIR), help="default: deploy/lichess")
    gen.add_argument(
        "--results-dir",
        type=Path,
        default=config.DEFAULT_RESULTS_DIR,
        help="where E2b wrote epsilon.json and eval wrote results.json: a rated value-mode bot plays that "
        "epsilon, and a rated bot the fast play mode results.json ships (default results)",
    )
    gen.set_defaults(func=cmd_lichess_config)
    chk = actions.add_parser("check-config", help="check a generated config against the plan")
    chk.add_argument("--config", required=True)
    chk.add_argument("--kind", choices=config.KINDS, help="default: the kind the config records")
    chk.add_argument(
        "--results", default="results/results.json", help="the shipped model to compare, or none"
    )
    chk.set_defaults(func=cmd_lichess_check_config)


def _register_lichess(sub: argparse._SubParsersAction) -> None:
    from blink.lichess import pause

    lichess = sub.add_parser("lichess", help="the Lichess BOT: configs, public snapshot, stop rule, pause")
    actions = lichess.add_subparsers(dest="lichess_command", required=True)
    _register_lichess_config(actions)
    snap = actions.add_parser("snapshot", help="rating, RD, N and game rates from the public API (no token)")
    snap.add_argument("--bot", required=True)
    snap.add_argument("--no-write", action="store_true", help="print only; write nothing")
    snap.add_argument("--out", default=str(Path("results") / "lichess.json"))
    snap.add_argument(
        "--max-games", type=_positive(int), default=3000, help="newest rated blitz games to export"
    )
    snap.set_defaults(func=cmd_lichess_snapshot)
    check = actions.add_parser("check", help="the stop rule over the last 50 games; exits 1 when it fires")
    check.add_argument("--stop", action="store_true", help="pause the bot when the rule fires on new games")
    watcher = actions.add_parser("watch", help="the stop rule every --every seconds, pausing the bot itself")
    watcher.add_argument("--every", type=_positive(float), default=120.0, help="seconds between checks")
    watcher.add_argument("--rounds", type=_positive(int), help="stop after this many checks (default never)")
    for parser in (check, watcher):
        parser.add_argument("--bot", required=True)
        parser.add_argument("--window", type=_positive(int), default=50)
        parser.add_argument("--pgn-dir", help="the PGNs lichess-bot saves (default BLINK_HOME/lichess/pgn)")
        _pause_options(parser, timeout=0.0)  # a stop-rule pause never waits for the live game
    stop = actions.add_parser("pause", help="flag, wait for no live game, then stop the bot by PID")
    stop.add_argument("--bot", required=True)
    stop.add_argument("--reason", default="GPU window", help="recorded in pause.json and the flag")
    _pause_options(stop, timeout=pause.DEFAULT_TIMEOUT_S)
    note = actions.add_parser("resume-note", help="delete the PAUSED flag; Itay restarts the bot himself")
    note.set_defaults(func=cmd_lichess_resume_note)
    check.set_defaults(func=cmd_lichess_check)
    watcher.set_defaults(func=cmd_lichess_watch)
    stop.set_defaults(func=cmd_lichess_pause)


def register(sub: argparse._SubParsersAction) -> None:
    _register_lichess(sub)
