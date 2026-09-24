"""`blink doctor`: facts about this machine, and the checks Blink needs to pass before training.

Checks are pure functions of facts, so tests can feed them any machine. `collect_*` functions
gather the real facts. torch is imported lazily, because the Windows CI leg runs without it.
"""

import shutil
import subprocess
import sys
from dataclasses import dataclass

from blink.checks import CheckResult

GB = 1_000_000_000
VRAM_HEADROOM_BYTES = int(0.8 * GB)
VRAM_COMFORT_BYTES = int(6.0 * GB)
NVIDIA_SMI_TIMEOUT_S = 5
TRAIN_SYNC_FIX = "uv sync (torch comes from the cu130 index via the default 'train' group)"


@dataclass(frozen=True)
class GpuFacts:
    name: str
    capability: tuple[int, int]
    bf16_supported: bool
    efficient_sdpa_bf16: bool
    flash_sdpa: bool
    free_bytes: int
    total_bytes: int


def check_torch_build(torch_version: str | None, cuda_available: bool) -> CheckResult:
    if torch_version is None:
        return CheckResult("torch", "skip", "torch is not installed (torch-free leg)")
    if "+cpu" in torch_version or not cuda_available:
        return CheckResult(
            "torch",
            "FAIL",
            f"{torch_version} cannot use the GPU",
            fix=f"{TRAIN_SYNC_FIX}; check uv.lock has +cu130",
        )
    return CheckResult("torch", "ok", torch_version)


def check_disk(label: str, free_bytes: int, floor_bytes: int) -> CheckResult:
    detail = f"{free_bytes / GB:.1f} GB free (floor {floor_bytes / GB:.1f} GB)"
    if free_bytes < floor_bytes:
        return CheckResult(f"disk {label}", "FAIL", detail, fix=f"free space on {label}")
    return CheckResult(f"disk {label}", "ok", detail)


def disk_checks(
    platform: str,
    disk_c: int,
    disk_d: int,
    usage=lambda root: shutil.disk_usage(root).free,
) -> list[CheckResult]:
    """C: and D: floors exist only on the Windows build machine; elsewhere there is nothing to check."""
    if platform != "win32":
        return []
    return [check_disk("C:", usage("C:\\"), disk_c), check_disk("D:", usage("D:\\"), disk_d)]


def vram_budget_bytes(free_bytes: int) -> int:
    """The most a run may reserve: measured free VRAM minus headroom. Never hard-coded (PF02)."""
    return max(0, free_bytes - VRAM_HEADROOM_BYTES)


def check_vram(free_bytes: int, holders: tuple[str, ...]) -> CheckResult:
    detail = f"{free_bytes / GB:.2f} GB free, budget {vram_budget_bytes(free_bytes) / GB:.2f} GB"
    if free_bytes < VRAM_COMFORT_BYTES:
        named = ", ".join(holders) if holders else "unknown processes"
        return CheckResult("vram", "WARN", f"{detail}; held by: {named}")
    return CheckResult("vram", "ok", detail)


def _kernel_runs(torch, backend) -> bool:
    from torch.nn.attention import sdpa_kernel

    q = torch.randn(2, 4, 64, 32, device="cuda", dtype=torch.bfloat16)
    try:
        with sdpa_kernel(backend):
            torch.nn.functional.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()
        return True
    except RuntimeError:
        return False


def collect_gpu_facts() -> GpuFacts:
    import torch
    from torch.nn.attention import SDPBackend

    free, total = torch.cuda.mem_get_info()
    return GpuFacts(
        name=torch.cuda.get_device_name(0),
        capability=torch.cuda.get_device_capability(0),
        bf16_supported=torch.cuda.is_bf16_supported(),
        efficient_sdpa_bf16=_kernel_runs(torch, SDPBackend.EFFICIENT_ATTENTION),
        flash_sdpa=_kernel_runs(torch, SDPBackend.FLASH_ATTENTION),
        free_bytes=int(free),
        total_bytes=int(total),
    )


def vram_holders() -> tuple[str, ...]:
    """Names of processes holding GPU memory, as nvidia-smi reports them."""
    if shutil.which("nvidia-smi") is None:
        return ()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=NVIDIA_SMI_TIMEOUT_S,
        ).stdout
    except (subprocess.TimeoutExpired, OSError):
        return ()  # nvidia-smi hangs when the driver is wedged; doctor must still finish
    names = {line.strip().split("\\")[-1] for line in out.splitlines() if line.strip()}
    return tuple(sorted(names))


def check_energy(available: bool, note: str | None) -> CheckResult:
    """The NVML energy counter the trainer logs as gpu_power_w (results/compute.json's kWh)."""
    if available:
        return CheckResult("gpu-board energy", "ok", note or "NVML energy counter reads")
    return CheckResult(
        "gpu-board energy",
        "WARN",
        note or "NVML energy counter unavailable",
        fix="a run started now logs no gpu_power_w, so compute.json will have no kWh for it",
    )


def energy_check() -> CheckResult:
    from blink.train import power

    reader, note = power.open_energy("cuda")
    return check_energy(reader is not None, note)


def torch_facts() -> tuple[str | None, bool]:
    try:
        import torch
    except ImportError:
        return None, False
    return torch.__version__, torch.cuda.is_available()


def triton_smoke() -> CheckResult:
    """Compile one tiny function with inductor. Failure is recorded, never blocking (PF04)."""
    try:
        import torch

        fn = torch.compile(lambda x: torch.nn.functional.gelu(x) * 2)
        fn(torch.randn(64, device="cuda"))
        torch.cuda.synchronize()
        return CheckResult("torch.compile", "ok", "inductor + triton compiled a kernel")
    except Exception as exc:  # any failure means "compile unavailable", which does not block
        return CheckResult("torch.compile", "WARN", f"compile unavailable: {type(exc).__name__}: {exc}"[:200])


def run(disk_c: int, disk_d: int) -> list[CheckResult]:
    """Every doctor check, in the order a human would read them."""
    version, cuda = torch_facts()
    results = [
        CheckResult("python", "ok", f"{sys.version.split()[0]} at {sys._base_executable}"),
        check_torch_build(version, cuda),
        *disk_checks(sys.platform, disk_c, disk_d),
    ]
    if version is None or not cuda:
        return results
    facts = collect_gpu_facts()
    results.append(
        CheckResult(
            "gpu",
            "ok",
            f"{facts.name}, capability {facts.capability}, bf16 {facts.bf16_supported}, "
            f"efficient SDPA bf16 {facts.efficient_sdpa_bf16}, flash SDPA {facts.flash_sdpa}",
        )
    )
    results.append(check_vram(facts.free_bytes, vram_holders()))
    results.append(energy_check())
    results.append(triton_smoke())
    return results
