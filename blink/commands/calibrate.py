"""`blink train calibrate`: PR-5's calibration of the flagship config (EVAL.md section 5).

blink train calibrate --config configs/long.toml [--steps 2000] [--write] [--data DIR] [--run NAME]
                      [--valprobe FILE] [--games10k FILE] [--device cuda|cpu]
blink train calibrate --config configs/long.toml --from-run NAME [--write]    (no training)

Trains the real trainer on the config for --steps steps (2,000) under a throwaway run name
(calib-<config>-<time> unless --run names one), on BLINK_HOME/data/v1 unless --data names a pack. Then
it prints R_true and steps = floor(120 x 3600 x R_true / 1024) with everything that follows from them
(blink.train.calibrate), records them in the run's calibration.json and, with --write, sets steps in
the config file. --from-run recomputes the same numbers from a finished calibration run's metrics (and its
supervisor.json events, when it was supervised).

A calibration that was paused, restarted or shared is never used: the user pause flag (BLINK_HOME/PAUSE)
stops the run at its next step and the command exits with supervise.EXIT_USER_PAUSE (75), and a run whose
counted span holds a restart, a pause or a window over 3x the median seconds per step is refused (exit
2). Either way --write writes nothing and calibration.json records the reason; calibrate again as a fresh
run.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from blink import paths
from blink.commands import train_data
from blink.model.config import TrainConfig, load_config
from blink.train import status, userpause

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


def _train(args: argparse.Namespace, cfg: TrainConfig, config: Path) -> tuple[Path, int | None]:
    """The calibration run: the real trainer, stopped at --steps. Returns its run directory and, when
    the user pause flag stopped it, the step it paused at."""
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
        pause_flag=userpause.flag_path(),
    )
    result = loop.train(trained, spec, plan.source, plan.val, probe=plan.probe)
    return run_dir, result.step if result.paused else None


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


def _events(run_dir: Path) -> list[dict]:
    """The run's supervisor.json events (a supervised --from-run), else none."""
    from blink.train.supervise import read_record

    return list((read_record(run_dir) or {}).get("events") or [])


def _refuse(config: Path, run_dir: Path, kind: str, detail: str) -> dict:
    """calibration.json for a calibration that must not be used; nothing is written to the config."""
    from blink.train.atomic import write_text_atomic

    record = {"rule": "PR-5 (EVAL.md section 5)", "config": str(config), "run": run_dir.name,
              "written": False, "refused": {"kind": kind, "detail": detail}, "time": time.time()}  # fmt: skip
    run_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(run_dir / "calibration.json", json.dumps(record, indent=2) + "\n")
    print(f"blink train calibrate: refused ({kind}): {detail}", file=sys.stderr)
    return record


def cmd_calibrate(args: argparse.Namespace) -> int:
    from blink.train import calibrate, loop, preview
    from blink.train.atomic import write_text_atomic
    from blink.train.supervise import EXIT_USER_PAUSE
    from blink.train.world import WorldMismatch

    try:
        _check_flags(args)
        config = Path(args.config)
        cfg = load_config(config)
        paused_at = None
        if args.from_run:
            run_dir = paths.home() / "runs" / args.from_run
        else:
            run_dir, paused_at = _train(args, cfg, config)
        if paused_at is not None:
            why = f"paused by the user at step {paused_at:,}: a calibration with a pause inside is never used"
            _refuse(config, run_dir, "pause", f"{why}; calibrate again as a fresh run once Blink is resumed")
            return EXIT_USER_PAUSE
        rate = calibrate.true_rate(preview.read_metrics(run_dir), cfg.batch_size, events=_events(run_dir))
        steps = calibrate.flagship_steps(rate.samples_per_s, cfg.batch_size)
        facts = calibrate.derived(steps, cfg.cooldown_frac)
        for line in calibrate.describe(rate, steps, cfg.batch_size, facts):
            print(line, flush=True)
        record = {**_record(config, run_dir, rate, steps, facts, cfg), "written": False}
        if args.write:
            note = calibrate.steps_note(rate.samples_per_s, cfg.batch_size, run_dir.name)
            record = {**record, "written": True, "old_steps": calibrate.write_steps(config, steps, note)}
        write_text_atomic(run_dir / "calibration.json", json.dumps(record, indent=2) + "\n")
    except calibrate.Refused as exc:
        _refuse(config, run_dir, exc.kind, str(exc))
        return EXIT_REFUSED
    except (CommandError, WorldMismatch, loop.RunExists, FileNotFoundError, ValueError) as exc:
        print(f"blink train calibrate: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    if record["written"]:
        print(f"wrote steps = {steps} to {config} (was {record['old_steps']})")
    print(f"recorded in {run_dir / 'calibration.json'}")
    return 0
