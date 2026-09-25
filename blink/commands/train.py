"""`blink train` and `blink status`.

blink train --config configs/s.toml --run NAME (--data DIR | --source-raw PATH [--max-lines N])
            [--valprobe FILE] [--games10k FILE] [--resume [--lr-scale F]] [--max-steps N]
            [--device cuda|cpu]
blink train --run NAME --data DIR --preview-cooldown 3h --from-step N    (writes runs/NAME-preview)
blink train --run NAME --data DIR --preview-steps K --from-step N [--preview-name BRANCH]
            (exactly K cooldown steps, written to runs/BRANCH; `blink supervise -- train ...` watches
            runs/BRANCH, so a branch crash-resumes like any run. A resumed branch reads only its own
            checkpoints, so the parent may move on once the branch has one; but a branch point that is
            not a check step, such as a --max-steps stop, is pruned once the parent resumes and saves
            keep_last newer checkpoints, so start every branch from it before resuming the parent)
blink train calibrate --config configs/long.toml [--steps 2000] [--write]    (PR-5: blink.commands.calibrate)

A v1 pack directory holds train_r*.bin roots, train_c*.bin children, val_roots.bin, valprobe.npz
and manifest.json (with the rebalancing weights), plus mateset.npz, which the checks score with
games10k (BLINK_HOME/data/games10k.npy unless --games10k names another); the P1 skeleton layout
(train_000.bin, val.bin) still works for roots-only configs. `blink status --run NAME` prints the
run's state and exits 1 when the run is stale, crashed or has a NaN loss; `blink status --live` prints
every run that trains or waits out a user pause. `blink train` honours the
user pause flag BLINK_HOME/PAUSE (blink.train.userpause): it waits to start while the flag is up, and
when the flag goes up mid-run it checkpoints the step and exits with supervise.EXIT_USER_PAUSE (75).
Torch is imported only when a command runs, so `blink --help` stays fast and works on the torch-free
CI leg.
"""

import argparse
import sys
from pathlib import Path

from blink import paths
from blink.commands import train_data
from blink.model.config import TrainConfig, config_from_dict, load_config
from blink.train import status, userpause

EXIT_REFUSED = 2
CommandError = train_data.CommandError


def pick_device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _is_branch(args: argparse.Namespace) -> bool:
    return args.preview_cooldown is not None or args.preview_steps is not None


def _branch_name(args: argparse.Namespace) -> str:
    """The run a branch writes: --preview-name, else NAME-preview."""
    from blink.train import preview

    return args.preview_name or preview.preview_name(args.run)


def _check_branch_flags(args: argparse.Namespace) -> None:
    if args.preview_cooldown is not None and args.preview_steps is not None:
        raise CommandError("--preview-cooldown and --preview-steps are two lengths for one branch: give one")
    if _is_branch(args) != (args.from_step is not None):
        raise CommandError("--from-step goes with one of --preview-cooldown or --preview-steps")
    if args.preview_steps is not None and args.preview_steps < 1:
        raise CommandError(f"--preview-steps must be at least 1, got {args.preview_steps}")
    if args.preview_name is None:
        return
    if not _is_branch(args):
        raise CommandError("--preview-name names a branch: it needs --from-step and a branch length")
    if not status.valid_run_name(args.preview_name) or args.preview_name == args.run:
        raise CommandError(f"bad branch name {args.preview_name!r} (a new run name, not --run's)")


def _check_required(args: argparse.Namespace) -> None:
    """What argparse cannot require once `train calibrate` shares the parser: --run and the data."""
    if args.run is None:
        raise CommandError("--run is required")
    if args.data is None and args.source_raw is None:
        raise CommandError("one of --data or --source-raw is required")
    calibrate_only = [f for f, v in (("--steps", args.steps), ("--from-run", args.from_run)) if v is not None]
    if calibrate_only or args.write:
        raise CommandError(f"{', '.join(calibrate_only or ['--write'])}: only for `blink train calibrate`")


def _check_flags(args: argparse.Namespace) -> None:
    _check_required(args)
    if not status.valid_run_name(args.run):
        raise CommandError(f"bad run name {args.run!r} (letters, digits, _ - . only)")
    if args.lr_scale is not None and (not args.resume or args.lr_scale <= 0):
        raise CommandError("--lr-scale needs --resume and a positive factor")
    _check_branch_flags(args)
    if not _is_branch(args) and args.config is None:
        raise CommandError("--config is required (a preview takes its config from the checkpoint)")


def _branch_point(args: argparse.Namespace, record: dict, name: str) -> Path:
    """The parent checkpoint a saved branch was cut from, which --run and --from-step must name."""
    from blink.train.checkpoint import step_of

    if not record.get("branched_from"):
        raise CommandError(f"{name} is not a branch (its config.json names no parent checkpoint)")
    source = Path(record["branched_from"])
    parent, step = source.parent.name, step_of(source)
    if (parent, step) != (args.run, args.from_step):
        raise CommandError(
            f"{name} was branched from {parent} step {step:,}; "
            f"--run {args.run} --from-step {args.from_step:,} names another branch point"
        )
    return source


def _saved_branch(args: argparse.Namespace, saved: Path) -> tuple[TrainConfig, Path]:
    """A resumed branch keeps the plan it was started with; an exact length must still be that plan.

    It continues from its own checkpoints only, so the parent's checkpoint it was cut from need not
    exist any more: a --max-steps stop is neither a check step nor a kept one, and the parent prunes
    it once it resumes and saves keep_last newer checkpoints.
    """
    import json

    record = json.loads(saved.read_text(encoding="utf-8"))
    source = _branch_point(args, record, saved.parent.name)
    cfg = config_from_dict(record["config"])
    if args.preview_steps is not None and cfg.steps != args.from_step + args.preview_steps:
        planned = cfg.steps - args.from_step
        raise CommandError(
            f"{saved.parent.name} was branched for {planned:,} cooldown steps from step {args.from_step:,}; "
            f"--preview-steps {args.preview_steps:,} would change its plan"
        )
    return cfg, source


def _branch_steps(args: argparse.Namespace, main_dir: Path, batch_size: int) -> int:
    """The branch's cooldown steps: exactly --preview-steps, or --preview-cooldown at the main run's
    measured training rate."""
    from blink.train import preview

    if args.preview_steps is not None:
        print(
            f"preview: {args.preview_steps:,} cooldown steps (--preview-steps) "
            f"from {main_dir.name} step {args.from_step:,}",
            flush=True,
        )
        return args.preview_steps
    seconds = preview.parse_duration(args.preview_cooldown)
    rate = preview.training_rate(preview.read_metrics(main_dir))
    steps = preview.preview_steps(seconds, rate, batch_size)
    print(
        f"preview: {steps:,} cooldown steps ({args.preview_cooldown} at {rate:,.0f} samples/s) "
        f"from {main_dir.name} step {args.from_step:,}",
        flush=True,
    )
    return steps


def _preview_config(args: argparse.Namespace, main_dir: Path, preview_dir: Path) -> tuple[TrainConfig, Path]:
    """(the preview's config, the main run's checkpoint it branches from: a resume never reads it)."""
    from blink.train import preview
    from blink.train.checkpoint import checkpoint_name, list_checkpoints, load_checkpoint, step_of

    saved = preview_dir / "config.json"
    if args.resume and saved.is_file():
        return _saved_branch(args, saved)
    source = main_dir / checkpoint_name(args.from_step)
    if not source.is_file():
        steps = [step_of(p) for p in list_checkpoints(main_dir)]
        raise CommandError(f"{main_dir.name} has no checkpoint at step {args.from_step} (it has {steps})")
    main_cfg = config_from_dict(load_checkpoint(source)["config"])
    if args.config is not None and load_config(args.config) != main_cfg:
        raise CommandError(f"--config differs from the config {main_dir.name} was trained with")
    steps = _branch_steps(args, main_dir, main_cfg.batch_size)
    return preview.preview_config(main_cfg, args.from_step, steps), source


def _config(args: argparse.Namespace) -> tuple[TrainConfig, Path | None]:
    """(the run's config, the checkpoint a preview branches from, or None)."""
    if not _is_branch(args):
        return load_config(args.config), None
    runs = paths.home() / "runs"
    return _preview_config(args, runs / args.run, runs / _branch_name(args))


def _spec(args: argparse.Namespace, plan: train_data.DataPlan, branch_from: Path | None):
    from blink.train import loop

    is_preview = _is_branch(args)
    name = _branch_name(args) if is_preview else args.run
    return loop.RunSpec(
        run_dir=paths.home() / "runs" / name,
        world=plan.world,
        device=pick_device(args.device),
        resume=args.resume,
        max_steps=args.max_steps,
        data=plan.description,
        lr_scale=args.lr_scale,
        init_from=None if args.resume else branch_from,
        preview=is_preview,
        games10k=plan.games10k,
        mateset=plan.mateset,
        pause_flag=userpause.flag_path(),
    )


def cmd_train(args: argparse.Namespace) -> int:
    if args.action == "calibrate":
        from blink.commands.calibrate import cmd_calibrate

        return cmd_calibrate(args)
    from blink.train import loop
    from blink.train.world import WorldMismatch

    try:
        _check_flags(args)
        cfg, branch_from = _config(args)
        plan = train_data.plan(args, cfg)
        spec = _spec(args, plan, branch_from)
        result = loop.train(cfg, spec, plan.source, plan.val, probe=plan.probe)
    except (CommandError, WorldMismatch, loop.RunExists, FileNotFoundError, ValueError) as exc:
        print(f"blink train: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if result.paused:
        from blink.train.supervise import EXIT_USER_PAUSE

        print(
            f"{spec.run_dir.name}: {userpause.PAUSED_USER} at step {result.step} ({result.checkpoint.name})"
        )
        return EXIT_USER_PAUSE
    state = "finished" if result.step >= cfg.steps else "stopped"
    print(f"{spec.run_dir.name}: {state} at step {result.step} in {result.wall_s:.1f} s")
    return 0


def _live_status() -> int:
    """Every run that trains or waits out a user pause; none is no error (the PC may be Itay's)."""
    runs = [run for run in status.list_runs(paths.home() / "runs") if status.active(run)]
    if not runs:
        print(f"no live run under {paths.home() / 'runs'}")
    for run in runs:
        print(status.format_status(run))
    return max((status.exit_code(run) for run in runs), default=0)


def cmd_status(args: argparse.Namespace) -> int:
    if args.live:
        return _live_status()
    if args.run is None:
        print("blink status: name a run with --run NAME, or --live for every live run", file=sys.stderr)
        return EXIT_REFUSED
    if not status.valid_run_name(args.run):
        print(f"blink status: bad run name {args.run!r}", file=sys.stderr)
        return EXIT_REFUSED
    run_dir = paths.home() / "runs" / args.run
    report = status.run_status(run_dir)
    print(status.format_status(report))
    warning = status.speed_warning(status.speed_check(status.read_rows(run_dir / "metrics.jsonl")))
    if warning:
        print(warning)  # a warning only: the exit code stays the stop rules' (stale, crashed, NaN)
    return status.exit_code(report)


def register(sub: argparse._SubParsersAction) -> None:
    train = sub.add_parser("train", help="train a model into BLINK_HOME/runs/<run>/")
    train.add_argument(
        "action",
        nargs="?",
        choices=("calibrate",),
        help="calibrate: PR-5's calibration of a flagship config (sets its steps with --write)",
    )
    train.add_argument(
        "--config", help="a TOML config, e.g. configs/s.toml (a preview reads its checkpoint's)"
    )
    train.add_argument("--run", help="run name (letters, digits, _ - .); required unless calibrating")
    data = train.add_mutually_exclusive_group()
    data.add_argument("--data", help="a pack directory: train_r*/train_c* (or train_*) shards, manifest.json")
    data.add_argument("--source-raw", help="parse the first --max-lines of a raw eval-DB .zst (cached)")
    train.add_argument("--max-lines", type=int, default=train_data.DEFAULT_MAX_LINES)
    train.add_argument("--workers", type=int, default=train_data.DEFAULT_WORKERS, help="parser processes")
    train.add_argument("--valprobe", help="a valprobe .npz for VAA (default: DATA/valprobe.npz when present)")
    train.add_argument(
        "--games10k", help="games10k .npy scored at the checks (default: BLINK_HOME/data/games10k.npy)"
    )
    train.add_argument("--resume", action="store_true", help="continue from the run's latest checkpoint")
    train.add_argument(
        "--lr-scale", type=float, help="with --resume: the LR scale from here on (replaces the checkpoint's)"
    )
    train.add_argument(
        "--preview-cooldown", help="branch a cooldown of this long (3h, 90m) into NAME-preview"
    )
    train.add_argument(
        "--preview-steps", type=int, help="branch a cooldown of exactly this many steps (not a duration)"
    )
    train.add_argument(
        "--preview-name", help="the branch's run name (default NAME-preview); supervise it under this name"
    )
    train.add_argument(
        "--from-step",
        type=int,
        help="with --preview-cooldown or --preview-steps: the checkpoint step to branch",
    )
    train.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda when available")
    train.add_argument("--max-steps", type=int, help="stop early at this step (the schedule is unchanged)")
    train.add_argument("--steps", type=int, help="calibrate: steps to train (default 2000)")
    train.add_argument("--write", action="store_true", help="calibrate: set the measured steps in --config")
    train.add_argument(
        "--from-run", help="calibrate: recompute from this finished run's metrics (no training)"
    )
    train.set_defaults(func=cmd_train)

    run_status = sub.add_parser("status", help="a run's state; exits 1 when stale, crashed or NaN")
    which = run_status.add_mutually_exclusive_group()
    which.add_argument("--run")
    which.add_argument("--live", action="store_true", help="every run that trains or waits out a user pause")
    run_status.set_defaults(func=cmd_status)
