"""The VRAM budget and the micro-batch it allows.

Free VRAM on this machine is not a constant (desktop apps hold part of the 8 GB, PF02), so it is
measured at launch with torch.cuda.mem_get_info() and never hard-coded. The budget is free - 0.8 GB,
and the training micro-batch is the largest halving of the batch whose measured peak (a real forward
and backward in bf16, plus the AdamW state the probe does not allocate) stays within it.
"""

from collections.abc import Callable
from typing import Any

HEADROOM_BYTES = int(0.8 * 2**30)
GB = 2**30
MIN_MICRO = 32
ADAM_SLOTS = 2  # exp_avg and exp_avg_sq, fp32


def budget_bytes(free_bytes: int) -> int:
    budget = free_bytes - HEADROOM_BYTES
    if budget <= 0:
        raise MemoryError(f"only {free_bytes / GB:.2f} GB of VRAM free: the budget keeps 0.8 GB headroom")
    return budget


def candidates(batch_size: int, floor: int = MIN_MICRO) -> list[int]:
    """batch, batch/2, batch/4, ... while the halving divides evenly and stays >= floor."""
    out, micro = [batch_size], batch_size
    while micro % 2 == 0 and micro // 2 >= floor:
        micro //= 2
        out.append(micro)
    return out


def _probe(micro: int, peak_of, budget: int, predicted: float | None, probes: list) -> bool:
    peak = peak_of(micro)
    fits = peak is not None and peak <= budget
    probes.append(
        {
            "micro": micro,
            "peak_gb": None if peak is None else peak / GB,
            "predicted_gb": None if predicted is None else predicted / GB,
            "fits": fits,
        }
    )
    return fits


def choose_micro_batch(
    batch_size: int, peak_of: Callable[[int], float | None], budget: int, floor: int = MIN_MICRO
) -> tuple[int, list[dict[str, Any]]]:
    """The largest candidate whose measured peak (bytes; None = out of memory) fits, and every probe.

    Only sizes predicted to fit are ever run: the two smallest candidates are measured, a line through
    them predicts the rest, and the largest predicted fit is measured to confirm it (stepping down on a
    miss). Running a size that overflows would not fail cleanly on Windows: the driver can spill it into
    system memory (the sysmem fallback) and crawl instead of raising out-of-memory.
    """
    sizes = sorted(candidates(batch_size, floor))
    probes: list[dict[str, Any]] = []
    measured = {}
    for micro in sizes[:2]:
        if not _probe(micro, peak_of, budget, None, probes):
            break
        measured[micro] = probes[-1]["peak_gb"] * GB
    if not measured:
        raise MemoryError(
            f"no micro-batch of at least {floor} fits the {budget / GB:.2f} GB budget: {probes}"
        )
    if len(measured) == 1:  # a single candidate, or the second smallest already overflows
        return sizes[0], probes
    (m1, p1), (m2, p2) = sorted(measured.items())
    per_row = (p2 - p1) / (m2 - m1)
    for micro in reversed(sizes[2:]):
        predicted = p1 + per_row * (micro - m1)
        if predicted <= budget and _probe(micro, peak_of, budget, predicted, probes):
            return micro, probes
    return m2, probes


def probe_peak(model, micro: int, device) -> int | None:
    """Peak reserved bytes of one bf16 forward and backward at `micro` rows, plus the AdamW state."""
    import torch

    from blink.model.losses import soft_cross_entropy_rows

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator().manual_seed(0)  # never touches the global RNG streams
    tokens = torch.randint(0, 16, (micro, 64), generator=generator).to(device)
    try:
        with torch.autocast(device.type, dtype=torch.bfloat16):
            policy, value = model(tokens)
        uniform_p = torch.full_like(policy, 1.0 / policy.shape[-1], dtype=torch.float32)
        uniform_v = torch.full_like(value, 1.0 / value.shape[-1], dtype=torch.float32)
        loss = soft_cross_entropy_rows(policy, uniform_p) + soft_cross_entropy_rows(value, uniform_v)
        loss.mean().backward()
        peak = torch.cuda.max_memory_reserved(device)
    except torch.OutOfMemoryError:
        peak = None
    finally:
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    if peak is None:
        return None
    adam = ADAM_SLOTS * sum(p.numel() * 4 for p in model.parameters() if p.requires_grad)
    return peak + adam
