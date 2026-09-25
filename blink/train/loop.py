"""The resumable trainer: bf16 autocast on CUDA (fp32 on CPU), AdamW or Muon, WSD, EMA, Recipe D batches.

Each optimizer step takes batch_size rows (roots plus a child_frac share of children) in micro-batches
sized from the VRAM budget, multiplies both losses by the per-sample rebalancing weights, clips the
gradient (a fixed norm, or "auto" measured over the warmup) and updates the EMA.

Everything a run writes lives in its run directory: config.json (world id, config, parameter counts,
VRAM budget), metrics.jsonl every `metrics_every` steps (with the window's phase: train, eval or
ckpt, its wall time and its loader wait share), evals.jsonl every `eval_every` steps plus the
full-valprobe checks at 5/25/30/50/100% (which also score games10k and the pack's mateset when the run
has them), heartbeat.json every `heartbeat_s` seconds, film/ frames when `film` is on, and atomic
checkpoints ckpt_<step:09d>.pt. On CPU a resume is bitwise identical to the straight run, because the
batch source is seeked to the checkpoint's step.

A user pause (RunSpec.pause_flag, BLINK_HOME/PAUSE under `blink train`; blink.train.userpause): the flag
is looked for every PAUSE_CHECK_S seconds at step boundaries and between eval chunks. When it is up the
run checkpoints its current step, beats "paused: user" and returns with `paused` set (`blink train`
then exits with supervise.EXIT_USER_PAUSE). A pause between eval chunks leaves paused_mid_eval.json, so
the resumed run scores that step before its next one. A run that starts while the flag is up waits,
beating "paused: user", before it builds anything, so it holds no GPU memory.
"""

import json
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from blink import heartbeat
from blink.model.config import AUTO, TrainConfig, config_to_dict
from blink.model.transformer import BlinkNet, parameter_report
from blink.train import checksets, evals, film, power, resume, step, telemetry, userpause, vaa, vram
from blink.train.atomic import write_text_atomic
from blink.train.checkpoint import list_checkpoints, save_checkpoint, step_of
from blink.train.clipping import GradClip
from blink.train.ema import Ema
from blink.train.optimizers import MuonAdamW, build_optimizer
from blink.train.schedule import wsd_lr
from blink.train.source import BatchSource, StepData, as_step_data

HOUR_S = 3600.0
PAUSE_CHECK_S = 2.0  # the flag is looked for at most this often: 77 us a look on D:, 50 ns otherwise
PAUSED_MID_EVAL = "paused_mid_eval.json"


class RunExists(RuntimeError):
    pass


class UserPause(BaseException):
    """The pause flag is up. A BaseException, so no `except Exception` inside an eval swallows it."""

    def __init__(self, mid_eval: bool):
        super().__init__("paused by the user")
        self.mid_eval = mid_eval


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    world: str
    device: str = "cpu"
    resume: bool = False
    max_steps: int | None = None  # stop this invocation early; the schedule still spans cfg.steps
    data: dict[str, Any] = field(default_factory=dict)  # a description of the data, for config.json
    lr_scale: float | None = None  # on resume: the LR scale from here on (None keeps the checkpoint's)
    init_from: Path | None = None  # start a new run from another run's checkpoint (preview cooldown)
    preview: bool = False  # a preview branch checks only its own end, against the reference's final VAA
    # held-out sets the checks also score (blink.train.checksets); None scores nothing, as before
    games10k: Path | None = None  # games10k.npy for games10k_top1 (blink train: BLINK_HOME/data)
    mateset: Path | None = None  # the pack's mateset.npz for shortest_mate and mate_preserving
    pause_flag: Path | None = None  # the user pause flag (blink train: BLINK_HOME/PAUSE); None never pauses


@dataclass(frozen=True)
class TrainResult:
    step: int
    parameters: int
    last_metrics: dict[str, Any] | None
    last_eval: dict[str, Any] | None
    checkpoint: Path | None
    wall_s: float
    clip: float | None = None
    micro_batch: int | None = None
    paused: bool = False  # stopped for the user pause flag, with a checkpoint at `step`


@dataclass
class _Run:
    """The live training state. Mutable by nature: the optimizer updates the model in place."""

    cfg: TrainConfig
    spec: RunSpec
    model: BlinkNet
    ema: Ema
    optimizer: torch.optim.Optimizer | MuonAdamW
    device: torch.device
    val: telemetry.ValSet | None
    log: Callable[[str], None]
    clip: GradClip
    micro: int
    vram: dict[str, Any]
    probe: vaa.Probe | None = None
    # the training forward: `model` itself, or its torch.compile wrapper
    forward: torch.nn.Module | None = None
    reference: vaa.Reference | None = None
    sets: checksets.CheckSets = field(default_factory=checksets.CheckSets)  # each loaded at the first check
    checks: dict[int, str] = field(default_factory=dict)
    film_plan: dict[int, str] = field(default_factory=dict)
    step: int = 0
    lr_scale: float = 1.0
    check_history: list[dict[str, Any]] = field(default_factory=list)
    kept: list[int] = field(default_factory=list)
    last_kept_wall: float = field(default_factory=time.time)
    last_metrics: dict[str, Any] | None = None
    last_eval: dict[str, Any] | None = None
    last_checkpoint: Path | None = None
    last_checkpoint_time: float = field(default_factory=time.monotonic)
    last_beat: float = 0.0
    pause: userpause.FlagWatch | None = None


def _choose_micro(cfg: TrainConfig, model: BlinkNet, device: torch.device, free: int | None, log) -> tuple:
    """(micro-batch rows, the VRAM record for config.json)."""
    info: dict[str, Any] = {}
    if free is not None:
        info = {"free_gb": free / vram.GB, "budget_gb": vram.budget_bytes(free) / vram.GB}
    if cfg.micro_batch != AUTO:
        return cfg.micro_batch or cfg.batch_size, {**info, "micro_batch": cfg.micro_batch or cfg.batch_size}
    if free is None:
        log("micro_batch auto on CPU: the whole batch in one pass")
        return cfg.batch_size, {"micro_batch": cfg.batch_size}
    budget = vram.budget_bytes(free)
    micro, probes = vram.choose_micro_batch(
        cfg.batch_size, lambda m: vram.probe_peak(model, m, device), budget
    )
    torch.cuda.reset_peak_memory_stats(device)
    log(f"VRAM: {free / vram.GB:.2f} GB free, budget {budget / vram.GB:.2f} GB, micro-batch {micro}")
    return micro, {**info, "micro_batch": micro, "probes": probes}


def _training_forward(model: BlinkNet, mode: str) -> torch.nn.Module:
    """The module the training step calls. A compile wrapper shares the model's parameters, so the
    optimizer, EMA, clip, evaluation and checkpoints keep using the plain model (no `_orig_mod.` keys)."""
    return model if mode == "off" else torch.compile(model)


def _build(cfg: TrainConfig, spec: RunSpec, val, probe: vaa.Probe | None, log) -> _Run:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(spec.device)
    free = None
    if device.type == "cuda":
        torch.cuda.empty_cache()  # blocks this process cached earlier are free for this run
        free = torch.cuda.mem_get_info(device)[0]
    model = BlinkNet(cfg.model).to(device)
    has_val = val is not None and len(val) > 0
    val_set = telemetry.make_val_set(val[: cfg.val_size], device, cfg.value_mapping) if has_val else None
    optimizer = build_optimizer(model, cfg, device.type)
    if isinstance(optimizer, MuonAdamW):
        log(optimizer.summary())
    ema = Ema(model, cfg.ema_max)
    micro, vram_info = _choose_micro(cfg, model, device, free, log)
    clip = GradClip(cfg.clip_norm, cfg.warmup_steps)
    run = _Run(cfg, spec, model, ema, optimizer, device, val_set, log, clip, micro, vram_info, probe)
    run.forward = _training_forward(model, cfg.compile)
    if probe is not None:
        run.checks = vaa.check_steps(cfg.steps, preview=spec.preview)
    run.sets = checksets.for_run(spec.games10k, spec.mateset)
    if cfg.vaa_checks and cfg.vaa_reference:
        run.reference = vaa.load_reference(spec.run_dir.parent / cfg.vaa_reference)
    if cfg.film:
        run.film_plan = dict(film.frame_plan(cfg.steps))
    return run


def _write_config(run: _Run) -> None:
    report = parameter_report(run.model)
    record = {
        "run": run.spec.run_dir.name,
        "world": run.spec.world,
        "parameters": report["total"],
        "parameter_report": report,
        "device": run.spec.device,
        "created": time.time(),
        "data": run.spec.data,
        "vram": run.vram,
        "branched_from": None if run.spec.init_from is None else str(run.spec.init_from),
        "config": config_to_dict(run.cfg),
    }
    write_text_atomic(run.spec.run_dir / "config.json", json.dumps(record, indent=2) + "\n")


def _beat_payload(spec: RunSpec, cfg: TrainConfig, step_now: int, state: str, **extra: Any) -> dict:
    return {
        "kind": "train",
        "run": spec.run_dir.name,
        "pid": os.getpid(),
        "state": state,
        "step": step_now,
        "steps": cfg.steps,
        "world": spec.world,
        **extra,
    }


def _beat(run: _Run, state: str, **extra: Any) -> None:
    payload = _beat_payload(run.spec, run.cfg, run.step, state, **extra)
    heartbeat.beat_once(run.spec.run_dir / "heartbeat.json", payload)
    run.last_beat = time.monotonic()


def _beat_if_due(run: _Run, **extra: Any) -> None:
    if time.monotonic() - run.last_beat >= run.cfg.heartbeat_s:
        _beat(run, "running", **extra)


def _pause_asked(run: _Run) -> bool:
    return run.pause is not None and run.pause.requested()


def _eval_tick(run: _Run) -> None:
    _beat_if_due(run, phase="eval")
    if _pause_asked(run):
        raise UserPause(mid_eval=True)


def _evaluate(run: _Run, label: str | None = None) -> None:
    """An evals row; between VAA chunks (a full check takes minutes) the heartbeat keeps beating and
    the pause flag is looked for."""
    evals.evaluate(run, label, tick=lambda: _eval_tick(run))


def _train_step(run: _Run, data: StepData, lr: float) -> tuple[torch.Tensor, ...]:
    cfg = run.cfg
    for group in run.optimizer.param_groups:
        group["lr"] = lr
    run.optimizer.zero_grad(set_to_none=True)
    out = step.accumulate(
        run.forward, data, run.device, run.micro, cfg.alpha, cfg.tau, cfg.lambda_v, cfg.value_mapping
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(run.model.parameters(), run.clip.limit())
    if run.clip.measuring:
        message = run.clip.observe(run.step, grad_norm.detach())  # read back once, at the warmup's end
        if message:
            run.log(message)
    run.optimizer.step()
    run.ema.update(run.model, run.step)
    return out.policy, out.value, grad_norm.detach()


def _write_metrics(run: _Run, window: telemetry.MetricWindow, lr: float) -> None:
    record = {**window.flush(run.step, lr), "clip": run.clip.value}
    telemetry.append_jsonl(run.spec.run_dir / "metrics.jsonl", record)
    run.last_metrics = record
    run.log(
        f"step {run.step}/{run.cfg.steps}: policy {record['loss_policy']:.4f} "
        f"value {record['loss_value']:.4f} lr {lr:.2e} grad {record['grad_norm']:.3f} "
        f"clip {record['clip_frac']:.2f} {record['samples_per_s']:.0f} samples/s "
        f"(loader wait {100 * record['data_wait_frac']:.1f}%) [{record['phase']}]"
    )
    if not (np.isfinite(record["loss_policy"]) and np.isfinite(record["loss_value"])):
        raise FloatingPointError(f"non-finite loss at step {run.step}: {record}")


def _protected(run: _Run) -> set[int]:
    return set(run.kept) | set(run.checks)


def _write_checkpoint(run: _Run) -> None:
    hours = run.cfg.keep_every_hours
    if hours > 0 and time.time() - run.last_kept_wall >= hours * HOUR_S:
        run.kept.append(run.step)
        run.last_kept_wall = time.time()
    state = resume.checkpoint_state(run)
    run.last_checkpoint = save_checkpoint(
        run.spec.run_dir, run.step, state, keep_last=run.cfg.keep_last, protect=_protected(run)
    )
    run.last_checkpoint_time = time.monotonic()
    run.log(f"checkpoint {run.last_checkpoint.name}")


def _save_frame(run: _Run, kind: str) -> None:
    weights = run.model.state_dict() if kind == "init" else run.ema.state_dict()
    extra = {"raw": run.model.state_dict()} if kind == "final" else {}
    film.save_frame(
        run.spec.run_dir,
        run.step,
        kind,
        weights,
        run.spec.world,
        config_to_dict(run.cfg),
        samples=run.step * run.cfg.batch_size,
        **extra,
    )


def _checkpoint_due(run: _Run, end: int) -> bool:
    cfg = run.cfg
    minutes = (time.monotonic() - run.last_checkpoint_time) / 60
    by_time = cfg.ckpt_every_minutes > 0 and minutes >= cfg.ckpt_every_minutes
    return run.step == end or run.step % cfg.ckpt_every_steps == 0 or by_time or run.step in run.checks


def _after_step(run: _Run, window: telemetry.MetricWindow, lr: float, end: int) -> None:
    cfg = run.cfg
    if run.step == 1 or run.step % cfg.metrics_every == 0 or run.step == end:
        _write_metrics(run, window, lr)
    label = run.checks.get(run.step)
    if run.step % cfg.eval_every == 0 or run.step == end or label is not None:
        _evaluate(run, label)
        window.mark("eval")
    if run.step in run.film_plan:
        _save_frame(run, run.film_plan[run.step])
    if _checkpoint_due(run, end):
        _write_checkpoint(run)
        window.mark("ckpt")
    _beat_if_due(run)


def _run_steps(run: _Run, source: BatchSource, end: int) -> None:
    cfg = run.cfg
    energy, note = power.open_energy(run.device)  # GPU-board kWh for results/compute.json
    if note:
        run.log(note)
    window = telemetry.MetricWindow(run.device, energy)
    batches = source(run.step)
    while run.step < end:
        lr = wsd_lr(run.step, cfg.peak_lr, cfg.warmup_steps, cfg.steps, cfg.cooldown_frac) * run.lr_scale
        asked = time.perf_counter()
        item = next(batches, None)
        window.waited(time.perf_counter() - asked)
        if item is None:
            raise RuntimeError(f"the batch source ran out of batches at step {run.step}")
        data = as_step_data(item)
        limit = run.clip.limit()  # the clip this step uses (an auto clip may get fixed during it)
        losses = _train_step(run, data, lr)
        window.add(*losses, clip_norm=limit, samples=len(data))
        run.step += 1
        _after_step(run, window, lr, end)
        if run.step < end and _pause_asked(run):
            raise UserPause(mid_eval=False)


# ---------------------------------------------------------------- the user pause


def _wait_to_start(cfg: TrainConfig, spec: RunSpec, log: Callable[[str], None]) -> None:
    """While the pause flag is up, build nothing (so no GPU memory is taken) and beat "paused: user"."""
    flag = spec.pause_flag
    if flag is None or not flag.exists():
        return
    saved = list_checkpoints(spec.run_dir) if spec.resume else []
    payload = _beat_payload(spec, cfg, step_of(saved[-1]) if saved else 0, userpause.PAUSED_USER)

    def paused() -> None:
        spec.run_dir.mkdir(parents=True, exist_ok=True)
        heartbeat.beat_once(spec.run_dir / "heartbeat.json", payload)

    log(f"{spec.run_dir.name}: {flag} is up, so the run waits to start until it is removed")
    waited = userpause.wait_while_flagged(flag, userpause.POLL_S, paused)
    log(f"{spec.run_dir.name}: the user pause is over after {waited:,.0f} s; starting")


def _pause(run: _Run, mid_eval: bool) -> None:
    """Checkpoint the current step for the user pause. A pause between eval chunks first notes that its
    step owes an eval; a pause ended while writing resumes from an older checkpoint, which ignores it."""
    if mid_eval:
        note = json.dumps({"step": run.step, "time": time.time()}) + "\n"
        write_text_atomic(run.spec.run_dir / PAUSED_MID_EVAL, note)
    if run.last_checkpoint is None or step_of(run.last_checkpoint) != run.step:
        _write_checkpoint(run)
    _beat(run, userpause.PAUSED_USER)
    run.log(
        f"paused by the user at step {run.step} ({run.last_checkpoint.name}); "
        f"resume from it once {run.spec.pause_flag} is gone"
    )


def _finish_paused_step(run: _Run) -> None:
    """A pause between eval chunks checkpointed its step without that step's evals row (or film frame):
    the resumed run does that work before its next step, so no check is skipped or written twice."""
    note = run.spec.run_dir / PAUSED_MID_EVAL
    try:
        owed = json.loads(note.read_text(encoding="utf-8"))["step"] == run.step
    except (OSError, ValueError, KeyError, TypeError):
        return
    if owed and run.step > 0:  # step 0's eval runs again by itself
        run.log(f"step {run.step}: finishing the eval a user pause interrupted")
        _evaluate(run, run.checks.get(run.step))
        if run.step in run.film_plan:
            _save_frame(run, run.film_plan[run.step])
    note.unlink(missing_ok=True)


def _train_until(run: _Run, source: BatchSource, end: int) -> bool:
    """Train to `end`; True when the user pause flag stopped the run (checkpointed at its step)."""
    try:
        if run.spec.resume:
            _finish_paused_step(run)
        if run.step == 0:
            _evaluate(run)
            if 0 in run.film_plan:
                _save_frame(run, "init")
        _run_steps(run, source, end)
    except UserPause as pause:
        _pause(run, pause.mid_eval)
        return True
    return False


def print_now(message: str) -> None:
    """The default log: flushed per line, so a redirected detached run's log is readable live."""
    print(message, flush=True)


def _start(run: _Run) -> None:
    spec = run.spec
    if spec.resume:
        resume.restore(run)
        return
    if spec.init_from is not None:
        resume.branch(run, spec.init_from)
    elif spec.lr_scale is not None:
        raise ValueError("lr_scale applies on resume or on a branch, not to a fresh run")
    _write_config(run)
    report = parameter_report(run.model)
    run.log(
        f"run {spec.run_dir.name}: {report['total']:,} parameters ({report['non_gab']:,} non-GAB) "
        f"on {spec.device}, micro-batch {run.micro}, world {spec.world}"
    )


def train(
    cfg: TrainConfig,
    spec: RunSpec,
    source: BatchSource,
    val: np.ndarray | None,
    log: Callable[[str], None] = print_now,
    probe: vaa.Probe | None = None,
) -> TrainResult:
    started = time.monotonic()
    if not spec.resume and list_checkpoints(spec.run_dir):
        raise RunExists(f"{spec.run_dir} already has checkpoints; pass --resume or pick a new run name")
    _wait_to_start(cfg, spec, log)
    run = _build(cfg, spec, val, probe, log)
    if spec.pause_flag is not None:
        run.pause = userpause.FlagWatch(spec.pause_flag, PAUSE_CHECK_S)
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    _start(run)
    end = cfg.steps if spec.max_steps is None else min(cfg.steps, spec.max_steps)
    _beat(run, "running")
    try:
        paused = _train_until(run, source, end)
    except BaseException as exc:
        _beat(run, "crashed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    if not paused:
        _beat(run, "finished" if run.step >= cfg.steps else "stopped")
    return TrainResult(
        run.step,
        parameter_report(run.model)["total"],
        run.last_metrics,
        run.last_eval,
        run.last_checkpoint,
        time.monotonic() - started,
        clip=run.clip.value,
        micro_batch=run.micro,
        paused=paused,
    )
