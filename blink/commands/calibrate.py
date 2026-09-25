"""`blink train calibrate`: PR-5's calibration of the flagship config (EVAL.md section 5).

blink train calibrate --config configs/long.toml [--steps 2000] [--write] [--data DIR] [--run NAME]
                      [--valprobe FILE] [--games10k FILE] [--device cuda|cpu]
blink train calibrate --config configs/long.toml --from-run NAME [--write]    (no training)

Trains the real trainer on the config for --steps steps (2,000) under a throwaway run name
(calib-<config>-<time> unless --run names one), on BLINK_HOME/data/v1 unless --data names a pack. Then
it prints R_true and steps = floor(120 x 3600 x R_true / 1024) with everything that follows from them
(blink.train.calibrate), records them in the run's calibration.json and, with --write, sets steps in
the config file. --from-run recomputes the same numbers from a finished calibration run's metrics.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from blink import paths
from blink.commands import train_data
from blink.model.config import TrainConfig, load_config
from blink.train import status

EXIT_REFUSED = 2
DEFAULT_PACK = "v1"  # BLINK_HOME/data/v1: the pack the sweeps and the flagship train on
CommandError = train_data.CommandError


def _train_only_flags(args: argparse.Namespace) -> list[str]:
    given = {
        "--resume": args.resume,
        "--lr-scale": args.lr_scale is not None,
        "--max-steps": args.max_steps is not None,
        "--from-step": args.from_step is not None,
        "--preview-cooldown": args.preview_cooldown is not None,
        "--preview-steps": args.preview_steps is not None,
        "--preview-name": args.preview_name is not None,
        "--source-raw": args.source_raw is not None,
    }
    return [flag for flag, present in given.items() if present]


def _check_flags(args: argparse.Namespace) -> None:
    from blink.train.calibrate import SKIP_STEPS

    if args.config is None:
        raise CommandError("calibrate needs --config: the flagship config it calibrates")
    refused = _train_only_flags(args)
    if refused:
        raise CommandError(f"{', '.join(refused)}: not for calibrate (its run stops at --steps)")
    for flag, name in (("--run", args.run), ("--from-run", args.from_run)):
        if name is not None and not status.valid_run_name(name):
            raise CommandError(f"bad {flag} name {name!r} (letters, digits, _ - . only)")
    if args.from_run is not None:
        if args.steps is not None or args.run is not None or args.data is not None:
            raise CommandError("--from-run reads a finished run: --steps, --run and --data do not apply")
        return
    if args.steps is not None and args.steps <= SKIP_STEPS:
        raise CommandError(f"--steps must be more than the first {SKIP_STEPS} steps R_true leaves out")


def _train(args: argparse.Namespace, cfg: TrainConfig, config: Path) -> Path:
    """The calibration run: the real trainer, stopped at --steps. Returns its run directory."""
    from blink.commands.train import pick_device
    from blink.train import calibrate, loop

    run = args.run or calibrate.throwaway_name(config)
    steps = args.steps or calibrate.CALIBRATION_STEPS
    trained = calibrate.calibration_config(cfg)
    data = args.data or str(paths.home() / "data" / DEFAULT_PACK)
    plan = train_data.plan(argparse.Namespace(**{**vars(args), "data": data}), trained)
    run_dir = paths.home() / "runs" / run
    print(f"calibrate: {steps:,} steps of {config} as run {run} (film off), on {data}", flush=True)
    spec = loop.RunSpec(
        run_dir=run_dir,
        world=plan.world,
        device=pick_device(args.device),
        max_steps=steps,
        data=plan.description,
        games10k=plan.games10k,
        mateset=plan.mateset,
    )
    loop.train(trained, spec, plan.source, plan.val, probe=plan.probe)
    return run_dir


def _record(config: Path, run_dir: Path, rate, steps: int, facts: dict, cfg: TrainConfig) -> dict:
    from blink.train import calibrate

    return {
        "rule": "PR-5 (EVAL.md section 5)",
        "config": str(config),
        "run": run_dir.name,
        "r_true": rate.samples_per_s,
        "samples": rate.samples,
        "seconds": rate.seconds,
        "intervals": rate.intervals,
        "from_step": rate.first_step,
        "to_step": rate.last_step,
        "skip_steps": calibrate.SKIP_STEPS,
        "hours": calibrate.T_LONG_HOURS,
        "batch_size": cfg.batch_size,
        "steps": steps,
        "derived": facts,
        "time": time.time(),
    }


def cmd_calibrate(args: argparse.Namespace) -> int:
    from blink.train import calibrate, loop, preview
    from blink.train.atomic import write_text_atomic
    from blink.train.world import WorldMismatch

    try:
        _check_flags(args)
        config = Path(args.config)
        cfg = load_config(config)
        run_dir = paths.home() / "runs" / args.from_run if args.from_run else _train(args, cfg, config)
        rate = calibrate.true_rate(preview.read_metrics(run_dir), cfg.batch_size)
        steps = calibrate.flagship_steps(rate.samples_per_s, cfg.batch_size)
        facts = calibrate.derived(steps, cfg.cooldown_frac)
        for line in calibrate.describe(rate, steps, cfg.batch_size, facts):
            print(line, flush=True)
        record = {**_record(config, run_dir, rate, steps, facts, cfg), "written": False}
        if args.write:
            note = calibrate.steps_note(rate.samples_per_s, cfg.batch_size, run_dir.name)
            record = {**record, "written": True, "old_steps": calibrate.write_steps(config, steps, note)}
        write_text_atomic(run_dir / "calibration.json", json.dumps(record, indent=2) + "\n")
    except (CommandError, WorldMismatch, loop.RunExists, FileNotFoundError, ValueError) as exc:
        print(f"blink train calibrate: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if record["written"]:
        print(f"wrote steps = {steps} to {config} (was {record['old_steps']})")
    print(f"recorded in {run_dir / 'calibration.json'}")
    return 0
