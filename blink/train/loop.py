"""The resumable trainer: bf16 autocast on CUDA (fp32 on CPU), AdamW, WSD, EMA, Recipe D batches.

Each optimizer step takes batch_size rows (roots plus a child_frac share of children) in micro-batches
sized from the VRAM budget, multiplies both losses by the per-sample rebalancing weights, clips the
gradient (a fixed norm, or "auto" measured over the warmup) and updates the EMA.

Everything a run writes lives in its run directory: config.json (world id, config, parameter counts,
VRAM budget), metrics.jsonl every `metrics_every` steps (with the window's phase: train, eval or
ckpt), evals.jsonl every `eval_every` steps plus the full-valprobe checks at 5/25/30/50/100%,
heartbeat.json every `heartbeat_s` seconds, film/ frames when `film` is on, and atomic checkpoints
ckpt_<step:09d>.pt. On CPU a resume is bitwise identical to the straight run, because the batch
source is seeked to the checkpoint's step.
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
from blink.train import evals, film, resume, step, telemetry, vaa, vram
from blink.train.atomic import write_text_atomic
from blink.train.checkpoint import list_checkpoints, save_checkpoint
from blink.train.clipping import GradClip
from blink.train.ema import Ema
from blink.train.schedule import wsd_lr
from blink.train.source import BatchSource, StepData, as_step_data

HOUR_S = 3600.0


class RunExists(RuntimeError):
    pass


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    world: str
    device: str = "cpu"
    resume: bool = False
    max_steps: int | None = None  # stop this invocation early; the schedule still spans cfg.steps
    data: dict[str, Any] = field(default_factory=dict)  # a description of the data, for config.json
    lr_scale: float = 1.0  # multiplies the schedule from here on, on top of the checkpoint's scale
    init_from: Path | None = None  # start a new run from another run's checkpoint (preview cooldown)
    preview: bool = False  # a preview branch checks only its own end, against the reference's final VAA


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


@dataclass
class _Run:
    """The live training state. Mutable by nature: the optimizer updates the model in place."""

    cfg: TrainConfig
    spec: RunSpec
    model: BlinkNet
    ema: Ema
    optimizer: torch.optim.Optimizer
    device: torch.device
    val: telemetry.ValSet | None
    log: Callable[[str], None]
    clip: GradClip
    micro: int
    vram: dict[str, Any]
    probe: vaa.Probe | None = None
    reference: vaa.Reference | None = None
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


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig, device_type: str) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (ndim >= 2); fused kernels on CUDA."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2), fused=device_type == "cuda"
    )


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
    val_set = telemetry.make_val_set(val[: cfg.val_size], device) if has_val else None
    optimizer = build_optimizer(model, cfg, device.type)
    ema = Ema(model, cfg.ema_max)
    micro, vram_info = _choose_micro(cfg, model, device, free, log)
    clip = GradClip(cfg.clip_norm, cfg.warmup_steps)
    run = _Run(cfg, spec, model, ema, optimizer, device, val_set, log, clip, micro, vram_info, probe)
    if probe is not None:
        run.checks = vaa.check_steps(cfg.steps, preview=spec.preview)
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


def _beat(run: _Run, state: str, **extra: Any) -> None:
    payload = {
        "kind": "train",
        "run": run.spec.run_dir.name,
        "pid": os.getpid(),
        "state": state,
        "step": run.step,
        "steps": run.cfg.steps,
        "world": run.spec.world,
        **extra,
    }
    heartbeat.beat_once(run.spec.run_dir / "heartbeat.json", payload)
    run.last_beat = time.monotonic()


def _train_step(run: _Run, data: StepData, lr: float) -> tuple[torch.Tensor, ...]:
    cfg = run.cfg
    for group in run.optimizer.param_groups:
        group["lr"] = lr
    run.optimizer.zero_grad(set_to_none=True)
    out = step.accumulate(run.model, data, run.device, run.micro, cfg.alpha, cfg.tau, cfg.lambda_v)
    grad_norm = torch.nn.utils.clip_grad_norm_(run.model.parameters(), run.clip.limit())
    if run.clip.measuring:
        message = run.clip.observe(run.step, grad_norm.item())
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
        f"clip {record['clip_frac']:.2f} {record['samples_per_s']:.0f} samples/s [{record['phase']}]"
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
        evals.evaluate(run, label)
        window.mark("eval")
    if run.step in run.film_plan:
        _save_frame(run, run.film_plan[run.step])
    if _checkpoint_due(run, end):
        _write_checkpoint(run)
        window.mark("ckpt")
    if time.monotonic() - run.last_beat >= cfg.heartbeat_s:
        _beat(run, "running")


def _run_steps(run: _Run, source: BatchSource, end: int) -> None:
    cfg = run.cfg
    window = telemetry.MetricWindow(run.device)
    batches = source(run.step)
    while run.step < end:
        lr = wsd_lr(run.step, cfg.peak_lr, cfg.warmup_steps, cfg.steps, cfg.cooldown_frac) * run.lr_scale
        item = next(batches, None)
        if item is None:
            raise RuntimeError(f"the batch source ran out of batches at step {run.step}")
        data = as_step_data(item)
        limit = run.clip.limit()  # the clip this step uses (an auto clip may get fixed during it)
        losses = _train_step(run, data, lr)
        window.add(*losses, clip_norm=limit, samples=len(data))
        run.step += 1
        _after_step(run, window, lr, end)


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
    elif spec.lr_scale != 1.0:
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
    run = _build(cfg, spec, val, probe, log)
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    _start(run)
    end = cfg.steps if spec.max_steps is None else min(cfg.steps, spec.max_steps)
    _beat(run, "running")
    try:
        if run.step == 0:
            evals.evaluate(run)
            if 0 in run.film_plan:
                _save_frame(run, "init")
        _run_steps(run, source, end)
    except BaseException as exc:
        _beat(run, "crashed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise
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
    )
