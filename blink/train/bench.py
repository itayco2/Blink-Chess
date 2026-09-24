"""`blink bench throughput|loader|play`: measured rates that replace estimates in bench.json (plan P4).

- throughput: training samples/s and peak torch.cuda.max_memory_reserved for each model size x
  micro-batch x compile mode, with gradient accumulation up to the effective batch (1024). An
  out-of-memory configuration is recorded (oom: true) and the sweep goes on; so is any other error.
- loader: ShardLoader samples/s and read MB/s over the shards, one sequential pass after another.
- play: value-mode latency (p50, p99) of one evaluate() call at a given row count (1, or L+1 up to
  219), with 1, 2 or 5 processes calling at once, as fastchess (5) and the bot (2) do. Each process
  holds its own CUDA context, exactly like separate engine processes.

Every section is merged into one JSON file by its key, so commands can run one at a time. Model
sizes are configs/<size>.toml (area P4 writes s, m, m12 and l) or any TOML path.
"""

import gc
import json
import multiprocessing
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from blink.board import encode, moves
from blink.data.record import NO_MOVE, ROOT_DTYPE
from blink.train.atomic import write_text_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
COMPILE_MODES = ("off", "inductor", "cudagraphs")
EFFECTIVE_BATCH = 1024
MIN_MICRO = 256  # a size is only eligible at micro-batch >= 256 (plan P6)
GIB = 2**30
KEYS = {"throughput": ("size", "micro", "compile"), "play": ("size", "rows", "concurrency")}
WORKER_TIMEOUT_S = 900
ABORT = -1.0  # the shared start time's value when a worker failed before the start
START_DELAY_S = 0.5  # the start time is this far after the last worker reports warm
START_POLL_S = 0.001

Log = Callable[[str], None]


def resolve_size(size: str) -> tuple[str, Path]:
    """'m' -> configs/m.toml; a path to a .toml -> (its stem, the path)."""
    path = Path(size) if size.endswith(".toml") else CONFIG_DIR / f"{size}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"no model config {path} (configs/s, m, m12 and l .toml come from area P4)")
    return path.stem, path


def random_codes(n: int, seed: int) -> np.ndarray:
    """Random square codes: throughput and latency do not depend on the position."""
    return np.random.default_rng(seed).integers(0, encode.NUM_CODES, size=(n, 64), dtype=np.uint8)


def synthetic_records(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    records = np.zeros(n, dtype=ROOT_DTYPE)
    records["board"] = encode.pack(random_codes(n, seed))
    records["move"] = rng.integers(0, moves.NUM_MOVES, size=n)
    records["cp"] = rng.integers(-1000, 1001, size=n)
    records["alt_move"] = NO_MOVE
    records["alt_move"][:, 0] = rng.integers(0, moves.NUM_MOVES, size=n)
    records["alt_cp"][:, 0] = records["cp"] - rng.integers(0, 200, size=n)
    return records


# ---------------------------------------------------------------- throughput


@dataclass(frozen=True)
class ThroughputSpec:
    size: str
    config: Path
    micro: int
    compile: str = "off"
    steps: int = 20
    warmup: int = 5  # includes compilation for inductor and cudagraphs
    effective_batch: int = EFFECTIVE_BATCH
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.compile not in COMPILE_MODES:
            raise ValueError(f"compile must be one of {COMPILE_MODES}, got {self.compile!r}")
        if self.micro <= 0 or self.steps <= 0 or self.warmup < 0:
            raise ValueError("need micro > 0, steps > 0 and warmup >= 0")


def _compiled(model, mode: str):
    import torch

    if mode == "off":
        return model
    return torch.compile(model) if mode == "inductor" else torch.compile(model, backend="cudagraphs")


def _train_step(model, optimizer, batch, accum: int, cfg, device_type: str, clip: float, graphs: bool):
    """One optimizer step over `accum` micro-batches, as the trainer does: bf16 autocast, clip, step.

    `clip` is GradClip.limit(): with clip_norm = "auto" that is infinity, as in the trainer's warmup,
    and clip_grad_norm_ does the same work for any limit. Under CUDA graphs every replay reuses the
    graph's memory: a .grad left as None would keep the graph's own output, which the next
    micro-batch overwrites. So the gradients are our own zeroed buffers that backward adds into, and
    each micro-batch is marked as a new step before it runs.
    """
    import torch

    from blink.model.losses import compute_losses

    if graphs:
        for param in model.parameters():
            if param.grad is None:
                param.grad = torch.zeros_like(param)
    optimizer.zero_grad(set_to_none=not graphs)
    for _ in range(accum):
        if graphs:
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            policy, value = model(batch.tokens)
        loss_policy, loss_value = compute_losses(policy, value, batch, cfg.alpha, cfg.tau)
        ((loss_policy + cfg.lambda_v * loss_value) / accum).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    optimizer.step()
    return loss_policy.detach() + cfg.lambda_v * loss_value.detach()


def _sync(device) -> None:
    import torch

    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_steps(spec: ThroughputSpec) -> dict[str, Any]:
    import torch

    from blink.model.config import load_config
    from blink.model.transformer import BlinkNet, count_parameters
    from blink.train.batch import make_batch
    from blink.train.clipping import GradClip
    from blink.train.loop import build_optimizer

    cfg = load_config(spec.config)
    device = torch.device(spec.device)
    model = BlinkNet(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg, device.type)
    step_model = _compiled(model, spec.compile)
    batch = make_batch(synthetic_records(spec.micro, seed=spec.micro), device)
    accum = max(1, spec.effective_batch // spec.micro)
    step_args = (
        accum,
        cfg,
        device.type,
        GradClip(cfg.clip_norm, cfg.warmup_steps).limit(),
        spec.compile == "cudagraphs",
    )
    started = time.perf_counter()
    for _ in range(spec.warmup):
        _train_step(step_model, optimizer, batch, *step_args)
    _sync(device)
    warm = time.perf_counter()
    for _ in range(spec.steps):
        loss = _train_step(step_model, optimizer, batch, *step_args)
    _sync(device)
    elapsed = time.perf_counter() - warm
    samples = spec.steps * accum * spec.micro
    return {
        "parameters": count_parameters(model),
        "accum": accum,
        "samples": samples,
        "seconds": elapsed,
        "warmup_s": warm - started,
        "samples_per_s": samples / elapsed,
        "loss": float(loss.item()),
    }


def _is_oom(exc: BaseException) -> bool:
    import torch

    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _release() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch._dynamo.reset()


def spilled(peak_gb: float | None, free_gb: float) -> bool:
    """Reserved past the VRAM that was free when the row started (the desktop and other processes hold
    the rest): the driver's sysmem fallback paged the excess to system RAM instead of raising OOM (PF64)."""
    return peak_gb is not None and peak_gb > free_gb


def measure_throughput(spec: ThroughputSpec) -> dict[str, Any]:
    """One row. Never raises for a failed configuration: the failure is the row's result."""
    import torch

    on_cuda = spec.device == "cuda"
    row: dict[str, Any] = {
        "size": spec.size,
        "config": str(spec.config),
        "micro": spec.micro,
        "compile": spec.compile,
        "effective_batch": spec.effective_batch,
        "device": spec.device,
        "oom": False,
        "error": None,
    }
    if on_cuda:
        _release()
        torch.cuda.reset_peak_memory_stats()
        row["vram_free_gb"] = torch.cuda.mem_get_info()[0] / GIB
    try:
        row.update(_timed_steps(spec))
    except Exception as exc:  # noqa: BLE001 - every failure is recorded in its row; the sweep goes on
        row.update(oom=_is_oom(exc), error=f"{type(exc).__name__}: {exc}"[:300], samples_per_s=0.0)
    finally:
        row["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / GIB if on_cuda else None
        row["spilled"] = on_cuda and spilled(row["peak_reserved_gb"], row["vram_free_gb"])
        _release()
    return row


def run_throughput(specs: Sequence[ThroughputSpec], log: Log = print) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        row = measure_throughput(spec)
        rows.append(row)
        state = "OOM" if row["oom"] else (row["error"] or f"{row['samples_per_s']:,.0f} samples/s")
        peak = row["peak_reserved_gb"]
        log(
            f"{spec.size} micro {spec.micro} compile {spec.compile}: {state}"
            + ("" if peak is None else f", peak reserved {peak:.2f} GiB")
            + (" (SPILLED past VRAM into system RAM: not a usable rate)" if row["spilled"] else "")
        )
    return rows


# ---------------------------------------------------------------- loader


def measure_loader(paths: Sequence[Path], batch_size: int = 1024, passes: int = 2, seed: int = 0) -> dict:
    """ShardLoader over every shard, `passes` times; the second pass shows the warm-cache rate."""
    from blink.data.loader import ShardLoader

    total_bytes = sum(Path(p).stat().st_size for p in paths)
    results = []
    for index in range(passes):
        loader = ShardLoader(list(paths), batch_size, seed + index, loop=False)
        started, records = time.perf_counter(), 0
        for batch in loader:
            records += len(batch)
        elapsed = max(time.perf_counter() - started, 1e-9)
        results.append(
            {
                "pass": index + 1,
                "records": records,
                "seconds": elapsed,
                "samples_per_s": records / elapsed,
                "read_mb_per_s": total_bytes / elapsed / 1e6,
            }
        )
    return {"shards": len(paths), "bytes": total_bytes, "batch_size": batch_size, "passes": results}


# ---------------------------------------------------------------- play latency


@dataclass(frozen=True)
class PlaySpec:
    size: str
    config: Path
    rows: int
    concurrency: int = 1
    iters: int = 200
    warmup: int = 20
    device: str = "cuda"


def _latencies(config: str, rows: int, iters: int, warmup: int, device: str, start=None) -> list[float]:
    import torch

    from blink.model.config import load_config
    from blink.model.evaluator import TorchEvaluator
    from blink.model.transformer import BlinkNet

    torch.manual_seed(0)
    evaluator = TorchEvaluator(BlinkNet(load_config(config).model), device)
    codes = random_codes(rows, seed=rows)
    for _ in range(warmup):
        evaluator.evaluate(codes)
    if start is not None:
        start()  # returns when every worker is warm
    out = []
    for _ in range(iters):
        started = time.perf_counter()
        evaluator.evaluate(codes)  # returns numpy: the device work is complete when it returns
        out.append(time.perf_counter() - started)
    return out


def _wait_for_start(start_at, timeout_s: float = WORKER_TIMEOUT_S) -> None:
    """Poll the parent's shared start time (wall clock, the same in every process); ABORT stops."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = start_at.value
        if value == ABORT:
            raise RuntimeError("another play worker failed before the start")
        if value > 0 and time.time() >= value:
            return
        if time.monotonic() > deadline:
            raise TimeoutError("no start signal from the parent")
        time.sleep(START_POLL_S)


def _play_worker(config: str, rows: int, iters: int, warmup: int, device: str, start_at, messages) -> None:
    def start() -> None:
        messages.put(("ready", os.getpid()))
        _wait_for_start(start_at)

    try:
        messages.put(("ok", _latencies(config, rows, iters, warmup, device, start)))
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, which raises it
        messages.put(("error", f"{type(exc).__name__}: {exc}"))


def _collect_replies(messages, start_at, workers: int, timeout_s: float = WORKER_TIMEOUT_S) -> list:
    """Every worker's final reply. The start time goes out once all are warm; a failure first aborts."""
    ready, replies = 0, []
    while len(replies) < workers:
        kind, detail = messages.get(timeout=timeout_s)
        if kind == "ready":
            ready += 1
            if ready == workers and start_at.value == 0.0:
                start_at.value = time.time() + START_DELAY_S
            continue
        replies.append((kind, detail))
        if kind == "error" and start_at.value == 0.0:
            start_at.value = ABORT
    return replies


def _concurrent_latencies(spec: PlaySpec) -> list[float]:
    """`concurrency` processes, each with its own model and CUDA context, timed from one start time.

    A multiprocessing.Barrier failed here with WinError 5 on its semaphore at 5 CUDA processes and
    left the parent waiting out its timeout (PF67); a queue and a shared start time need no semaphore.
    """
    ctx = multiprocessing.get_context("spawn")
    messages, start_at = ctx.Queue(), ctx.Value("d", 0.0, lock=False)
    args = (str(spec.config), spec.rows, spec.iters, spec.warmup, spec.device, start_at, messages)
    workers = [ctx.Process(target=_play_worker, args=args, daemon=True) for _ in range(spec.concurrency)]
    for worker in workers:
        worker.start()
    try:
        replies = _collect_replies(messages, start_at, spec.concurrency)
    finally:
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                worker.kill()
    errors = [detail for state, detail in replies if state == "error"]
    if errors:
        raise RuntimeError(f"a play worker failed: {errors[0]}")
    return [latency for _, latencies in replies for latency in latencies]


def measure_play(spec: PlaySpec) -> dict[str, Any]:
    if spec.concurrency == 1:
        latencies = _latencies(str(spec.config), spec.rows, spec.iters, spec.warmup, spec.device)
    else:
        latencies = _concurrent_latencies(spec)
    ms = np.asarray(latencies) * 1000.0
    return {
        "size": spec.size,
        "config": str(spec.config),
        "rows": spec.rows,
        "concurrency": spec.concurrency,
        "device": spec.device,
        "latencies": len(ms),
        "p50_ms": float(np.percentile(ms, 50)),
        "p99_ms": float(np.percentile(ms, 99)),
        "mean_ms": float(ms.mean()),
        "max_ms": float(ms.max()),
    }


def run_play(specs: Sequence[PlaySpec], log: Log = print) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        try:
            row = measure_play(spec)
        except Exception as exc:  # noqa: BLE001 - recorded in its row; the sweep goes on
            row = {
                "size": spec.size,
                "rows": spec.rows,
                "concurrency": spec.concurrency,
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }
        rows.append(row)
        summary = row.get("error") or f"p50 {row['p50_ms']:.2f} ms, p99 {row['p99_ms']:.2f} ms"
        log(f"{spec.size} value mode, {spec.rows} rows, concurrency {spec.concurrency}: {summary}")
    return rows


# ---------------------------------------------------------------- bench.json


def machine_facts(device: str = "cuda") -> dict[str, Any]:
    import torch

    from blink.doctor import vram_budget_bytes

    facts: dict[str, Any] = {"torch": torch.__version__, "measured": time.time()}
    if device == "cuda" and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        facts.update(
            gpu=torch.cuda.get_device_name(0),
            vram_free_gb=free / GIB,
            vram_total_gb=total / GIB,
            vram_budget_gb=vram_budget_bytes(free) / GIB,
        )
    return facts


def _merge_rows(old: list[dict], new: list[dict], key: tuple[str, ...]) -> list[dict]:
    fresh = {tuple(row.get(k) for k in key) for row in new}
    return [row for row in old if tuple(row.get(k) for k in key) not in fresh] + list(new)


def update_bench(path: Path, section: str, value: Any, machine: dict[str, Any] | None = None) -> dict:
    """Merge one section into bench.json: rows replace rows with the same key; dicts merge by key."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        data = {}
    if section in KEYS:
        merged = _merge_rows(data.get(section, []), list(value), KEYS[section])
    else:
        merged = {**data.get(section, {}), **value}
    updated = {**data, section: merged, **({"machine": machine} if machine else {})}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, json.dumps(updated, indent=2) + "\n")
    return updated


def best_rates(data: dict[str, Any], compile: str | None = None) -> dict[str, dict[str, Any]]:
    """Per size, the fastest measured row that fits the VRAM budget at micro-batch >= 256, unspilled.

    With `compile`, only rows measured in that mode count: a run is planned and policed at the rate of
    the mode it actually trains in (PF66).
    """
    budget = (data.get("machine") or {}).get("vram_budget_gb")
    best: dict[str, dict[str, Any]] = {}
    for row in data.get("throughput", []):
        if compile is not None and row.get("compile", "off") != compile:
            continue
        peak = row.get("peak_reserved_gb")
        fits = not row.get("spilled") and (budget is None or peak is None or peak <= budget)
        usable = not row.get("oom") and not row.get("error") and row.get("micro", 0) >= MIN_MICRO and fits
        if usable and row["samples_per_s"] > best.get(row["size"], {}).get("samples_per_s", -1.0):
            best[row["size"]] = row
    return best
