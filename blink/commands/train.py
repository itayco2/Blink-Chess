"""`blink train` and `blink status`.

blink train --config configs/s.toml --run NAME (--data DIR | --source-raw PATH [--max-lines N])
            [--valprobe FILE] [--games10k FILE] [--resume [--lr-scale F]] [--max-steps N]
            [--device cuda|cpu]
blink train --run NAME --data DIR --preview-cooldown 3h --from-step N    (writes runs/NAME-preview)

A v1 pack directory holds train_r*.bin roots, train_c*.bin children, val_roots.bin, valprobe.npz
and manifest.json (with the rebalancing weights), plus mateset.npz, which the checks score with
games10k (BLINK_HOME/data/games10k.npy unless --games10k names another); the P1 skeleton layout
(train_000.bin, val.bin) still works for roots-only configs. `blink status --run NAME` prints the
run's state and exits 1 when the run is stale, crashed or has a NaN loss. Torch is imported only
when a command runs, so `blink --help` stays fast and works on the torch-free CI leg.
"""

import argparse
import sys
from pathlib import Path

from blink import paths
from blink.commands import train_data
from blink.model.config import TrainConfig, config_from_dict, load_config
from blink.train import status

EXIT_REFUSED = 2
CommandError = train_data.CommandError


def _device(requested: str | None) -> str:
    import torch

    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _check_flags(args: argparse.Namespace) -> None:
    if not status.valid_run_name(args.run):
        raise CommandError(f"bad run name {args.run!r} (letters, digits, _ - . only)")
    if args.lr_scale is not None and (not args.resume or args.lr_scale <= 0):
        raise CommandError("--lr-scale needs --resume and a positive factor")
    if (args.preview_cooldown is None) != (args.from_step is None):
        raise CommandError("--preview-cooldown and --from-step go together")
    if args.preview_cooldown is None and args.config is None:
        raise CommandError("--config is required (a preview takes its config from the checkpoint)")


def _preview_config(args: argparse.Namespace, main_dir: Path, preview_dir: Path) -> tuple[TrainConfig, Path]:
    """(the preview's config, the main run's checkpoint it branches from)."""
    import json

    from blink.train import preview
    from blink.train.checkpoint import checkpoint_name, list_checkpoints, load_checkpoint, step_of

    source = main_dir / checkpoint_name(args.from_step)
    if not source.is_file():
        steps = [step_of(p) for p in list_checkpoints(main_dir)]
        raise CommandError(f"{main_dir.name} has no checkpoint at step {args.from_step} (it has {steps})")
    saved = preview_dir / "config.json"
    if args.resume and saved.is_file():
        return config_from_dict(json.loads(saved.read_text(encoding="utf-8"))["config"]), source
    main_cfg = config_from_dict(load_checkpoint(source)["config"])
    if args.config is not None and load_config(args.config) != main_cfg:
        raise CommandError(f"--config differs from the config {main_dir.name} was trained with")
    seconds = preview.parse_duration(args.preview_cooldown)
    rate = preview.training_rate(preview.read_metrics(main_dir))
    steps = preview.preview_steps(seconds, rate, main_cfg.batch_size)
    print(
        f"preview: {steps:,} cooldown steps ({args.preview_cooldown} at {rate:,.0f} samples/s) "
        f"from {main_dir.name} step {args.from_step:,}",
        flush=True,
    )
    return preview.preview_config(main_cfg, args.from_step, steps), source


def _config(args: argparse.Namespace) -> tuple[TrainConfig, Path | None]:
    """(the run's config, the checkpoint a preview branches from, or None)."""
    if args.preview_cooldown is None:
        return load_config(args.config), None
    from blink.train import preview

    runs = paths.home() / "runs"
    return _preview_config(args, runs / args.run, runs / preview.preview_name(args.run))


def _spec(args: argparse.Namespace, plan: train_data.DataPlan, branch_from: Path | None):
    from blink.train import loop, preview

    is_preview = args.preview_cooldown is not None
    name = preview.preview_name(args.run) if is_preview else args.run
    return loop.RunSpec(
        run_dir=paths.home() / "runs" / name,
        world=plan.world,
        device=_device(args.device),
        resume=args.resume,
        max_steps=args.max_steps,
        data=plan.description,
        lr_scale=args.lr_scale,
        init_from=None if args.resume else branch_from,
        preview=is_preview,
        games10k=plan.games10k,
        mateset=plan.mateset,
    )


def cmd_train(args: argparse.Namespace) -> int:
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
    state = "finished" if result.step >= cfg.steps else "stopped"
    print(f"{spec.run_dir.name}: {state} at step {result.step} in {result.wall_s:.1f} s")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not status.valid_run_name(args.run):
        print(f"blink status: bad run name {args.run!r}", file=sys.stderr)
        return EXIT_REFUSED
    report = status.run_status(paths.home() / "runs" / args.run)
    print(status.format_status(report))
    return status.exit_code(report)


def register(sub: argparse._SubParsersAction) -> None:
    train = sub.add_parser("train", help="train a model into BLINK_HOME/runs/<run>/")
    train.add_argument(
        "--config", help="a TOML config, e.g. configs/s.toml (a preview reads its checkpoint's)"
    )
    train.add_argument("--run", required=True, help="run name (letters, digits, _ - .)")
    data = train.add_mutually_exclusive_group(required=True)
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
    train.add_argument("--from-step", type=int, help="with --preview-cooldown: the checkpoint step to branch")
    train.add_argument("--device", choices=("cuda", "cpu"), help="default: cuda when available")
    train.add_argument("--max-steps", type=int, help="stop early at this step (the schedule is unchanged)")
    train.set_defaults(func=cmd_train)

    run_status = sub.add_parser("status", help="a run's state; exits 1 when stale, crashed or NaN")
    run_status.add_argument("--run", required=True)
    run_status.set_defaults(func=cmd_status)
