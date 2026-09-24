"""The resumable trainer: bf16 autocast on CUDA (fp32 on CPU), AdamW, clip 1.0, WSD, EMA.

Everything a run writes lives in its run directory: config.json (world id, config, parameter count),
metrics.jsonl every `metrics_every` steps, evals.jsonl every `eval_every` steps on a fixed val sample,
heartbeat.json every `heartbeat_s` seconds, and atomic checkpoints ckpt_<step:09d>.pt holding the
model, EMA, optimizer, schedule, step, RNG states, world id and config. On CPU a resume is bitwise
identical to the straight run, because the batch source is seeked to the checkpoint's step.
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
from blink.model.config import TrainConfig, config_to_dict
from blink.model.losses import compute_losses
from blink.model.transformer import BlinkNet, count_parameters
from blink.train import telemetry
from blink.train.atomic import write_text_atomic
from blink.train.batch import Batch, make_batch
from blink.train.checkpoint import latest_checkpoint, list_checkpoints, load_checkpoint, save_checkpoint
from blink.train.ema import Ema
from blink.train.schedule import wsd_lr
from blink.train.source import BatchSource
from blink.train.world import require_same_world

CHECKPOINT_FORMAT = 1


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


@dataclass(frozen=True)
class TrainResult:
    step: int
    parameters: int
    last_metrics: dict[str, Any] | None
    last_eval: dict[str, Any] | None
    checkpoint: Path | None
    wall_s: float


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
    step: int = 0
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


def _rng_state(device: torch.device) -> dict[str, Any]:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    on_cuda = device.type == "cuda" and torch.cuda.is_initialized()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if on_cuda else [],
        "numpy": [name, keys.tolist(), int(pos), int(has_gauss), float(cached)],
        "python": random.getstate(),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    random.setstate(state["python"])


def _checkpoint_state(run: _Run) -> dict[str, Any]:
    cfg = run.cfg
    return {
        "format": CHECKPOINT_FORMAT,
        "step": run.step,
        "model": run.model.state_dict(),
        "ema": run.ema.state_dict(),
        "optimizer": run.optimizer.state_dict(),
        "schedule": {
            "kind": "wsd",
            "peak_lr": cfg.peak_lr,
            "warmup_steps": cfg.warmup_steps,
            "steps": cfg.steps,
            "cooldown_frac": cfg.cooldown_frac,
        },
        "rng": _rng_state(run.device),
        "world": run.spec.world,
        "config": config_to_dict(cfg),
    }


def _write_checkpoint(run: _Run) -> None:
    state = _checkpoint_state(run)
    run.last_checkpoint = save_checkpoint(run.spec.run_dir, run.step, state, keep_last=run.cfg.keep_last)
    run.last_checkpoint_time = time.monotonic()
    run.log(f"checkpoint {run.last_checkpoint.name}")


def _restore(run: _Run) -> None:
    path = latest_checkpoint(run.spec.run_dir)
    if path is None:
        raise FileNotFoundError(f"--resume: no checkpoint in {run.spec.run_dir}")
    state = load_checkpoint(path, map_location="cpu")
    require_same_world(found=state["world"], expected=run.spec.world)
    if state["config"] != config_to_dict(run.cfg):
        changed = sorted(k for k, v in config_to_dict(run.cfg).items() if state["config"].get(k) != v)
        run.log(f"warning: config differs from the checkpoint in {changed}; resume will not be bitwise")
    run.model.load_state_dict(state["model"])
    run.ema.load_state_dict(state["ema"])
    run.optimizer.load_state_dict(state["optimizer"])
    _restore_rng(state["rng"])
    run.step = int(state["step"])
    run.last_checkpoint = path
    for name in ("metrics.jsonl", "evals.jsonl"):
        telemetry.truncate_after(run.spec.run_dir / name, run.step)
    run.log(f"resumed from {path.name} at step {run.step}")


def _write_config(cfg: TrainConfig, spec: RunSpec, parameters: int) -> None:
    record = {
        "run": spec.run_dir.name,
        "world": spec.world,
        "parameters": parameters,
        "device": spec.device,
        "created": time.time(),
        "data": spec.data,
        "config": config_to_dict(cfg),
    }
    write_text_atomic(spec.run_dir / "config.json", json.dumps(record, indent=2) + "\n")


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


def _train_step(run: _Run, batch: Batch, lr: float) -> tuple[torch.Tensor, ...]:
    cfg = run.cfg
    for group in run.optimizer.param_groups:
        group["lr"] = lr
    with torch.autocast(run.device.type, dtype=torch.bfloat16, enabled=run.device.type == "cuda"):
        policy, value = run.model(batch.tokens)
    loss_policy, loss_value = compute_losses(policy, value, batch, cfg.alpha, cfg.tau)
    run.optimizer.zero_grad(set_to_none=True)
    (loss_policy + cfg.lambda_v * loss_value).backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(run.model.parameters(), cfg.clip_norm)
    run.optimizer.step()
    run.ema.update(run.model, run.step)
    return loss_policy.detach(), loss_value.detach(), grad_norm.detach()


def _evaluate(run: _Run) -> None:
    if run.val is None:
        return
    raw = telemetry.evaluate(run.model, run.val, run.cfg.alpha, run.cfg.tau)
    ema = telemetry.evaluate(run.ema.module, run.val, run.cfg.alpha, run.cfg.tau)
    record = {"step": run.step, **raw, **{f"ema_{k}": v for k, v in ema.items() if k != "n"}}
    telemetry.append_jsonl(run.spec.run_dir / "evals.jsonl", record)
    run.last_eval = record
    run.log(
        f"eval {run.step}: top-1 {raw['top1']:.3f} (ema {ema['top1']:.3f}), "
        f"policy CE {raw['policy_ce']:.3f}, value CE {raw['value_ce']:.3f}, win% MAE {raw['win_mae']:.4f}"
    )


def _write_metrics(run: _Run, window: telemetry.MetricWindow, lr: float) -> None:
    record = window.flush(run.step, lr)
    telemetry.append_jsonl(run.spec.run_dir / "metrics.jsonl", record)
    run.last_metrics = record
    run.log(
        f"step {run.step}/{run.cfg.steps}: policy {record['loss_policy']:.4f} "
        f"value {record['loss_value']:.4f} lr {lr:.2e} grad {record['grad_norm']:.3f} "
        f"{record['samples_per_s']:.0f} samples/s"
    )
    if not (np.isfinite(record["loss_policy"]) and np.isfinite(record["loss_value"])):
        raise FloatingPointError(f"non-finite loss at step {run.step}: {record}")


def _checkpoint_due(run: _Run, end: int) -> bool:
    cfg = run.cfg
    minutes = (time.monotonic() - run.last_checkpoint_time) / 60
    by_time = cfg.ckpt_every_minutes > 0 and minutes >= cfg.ckpt_every_minutes
    return run.step == end or run.step % cfg.ckpt_every_steps == 0 or by_time


def _run_steps(run: _Run, source: BatchSource, end: int) -> None:
    cfg = run.cfg
    window = telemetry.MetricWindow(run.device)
    batches = source(run.step)
    while run.step < end:
        lr = wsd_lr(run.step, cfg.peak_lr, cfg.warmup_steps, cfg.steps, cfg.cooldown_frac)
        records = next(batches, None)
        if records is None:
            raise RuntimeError(f"the batch source ran out of batches at step {run.step}")
        losses = _train_step(run, make_batch(records, run.device), lr)
        window.add(*losses, clip_norm=cfg.clip_norm, samples=len(records))
        run.step += 1
        if run.step == 1 or run.step % cfg.metrics_every == 0 or run.step == end:
            _write_metrics(run, window, lr)
        if run.step % cfg.eval_every == 0 or run.step == end:
            _evaluate(run)
        if _checkpoint_due(run, end):
            _write_checkpoint(run)
        if time.monotonic() - run.last_beat >= cfg.heartbeat_s:
            _beat(run, "running")


def _build(cfg: TrainConfig, spec: RunSpec, val: np.ndarray | None, log: Callable[[str], None]) -> _Run:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(spec.device)
    model = BlinkNet(cfg.model).to(device)
    has_val = val is not None and len(val) > 0
    val_set = telemetry.make_val_set(val[: cfg.val_size], device) if has_val else None
    optimizer = build_optimizer(model, cfg, device.type)
    return _Run(cfg, spec, model, Ema(model, cfg.ema_max), optimizer, device, val_set, log)


def print_now(message: str) -> None:
    """The default log: flushed per line, so a redirected detached run's log is readable live."""
    print(message, flush=True)


def train(
    cfg: TrainConfig,
    spec: RunSpec,
    source: BatchSource,
    val: np.ndarray | None,
    log: Callable[[str], None] = print_now,
) -> TrainResult:
    started = time.monotonic()
    if not spec.resume and list_checkpoints(spec.run_dir):
        raise RunExists(f"{spec.run_dir} already has checkpoints; pass --resume or pick a new run name")
    run = _build(cfg, spec, val, log)
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    parameters = count_parameters(run.model)
    if spec.resume:
        _restore(run)
    else:
        _write_config(cfg, spec, parameters)
        log(f"run {spec.run_dir.name}: {parameters:,} parameters on {spec.device}, world {spec.world}")
    end = cfg.steps if spec.max_steps is None else min(cfg.steps, spec.max_steps)
    _beat(run, "running")
    try:
        if run.step == 0:
            _evaluate(run)
        _run_steps(run, source, end)
    except BaseException as exc:
        _beat(run, "crashed", error=f"{type(exc).__name__}: {exc}"[:500])
        raise
    _beat(run, "finished" if run.step >= cfg.steps else "stopped")
    return TrainResult(
        run.step, parameters, run.last_metrics, run.last_eval, run.last_checkpoint, time.monotonic() - started
    )
