"""GPU-board energy for results/compute.json: NVML's total-energy counter, read with ctypes.

nvmlDeviceGetTotalEnergyConsumption gives the board's energy in millijoules since the driver loaded
(Volta and newer; the RTX 3070 is Ampere). telemetry.MetricWindow reads it when a window opens and
when it closes, so every metrics.jsonl row carries gpu_power_w, the window's mean board power, and
blink.report.compute multiplies it back by the window's seconds. The library is the driver's own
(nvml.dll on Windows, libnvidia-ml.so.1 on Linux), loaded with ctypes: nothing is installed and no
process is started. The NVML index is the CUDA index (one GPU on this machine; a CUDA_VISIBLE_DEVICES
remapping is not handled). Any failure (a CPU run, no driver, an older board, a failed read) means no
reading and one log line, never an exception: power telemetry must never stop a training run.
"""

import ctypes
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

NVML_SUCCESS = 0
LINUX_LIBRARY = "libnvidia-ml.so.1"
MILLIJOULES_PER_JOULE = 1000.0

EnergyReader = Callable[[], float | None]


class NvmlError(RuntimeError):
    pass


def _windows_paths() -> tuple[Path, ...]:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    program_files = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    return windir / "System32" / "nvml.dll", program_files / "NVIDIA Corporation" / "NVSMI" / "nvml.dll"


def load_nvml() -> Any:
    """The driver's NVML library (raises OSError when there is none)."""
    if sys.platform == "win32":
        for path in _windows_paths():
            if path.is_file():
                return ctypes.CDLL(str(path))
        raise OSError("nvml.dll was not found in System32 or NVSMI")
    return ctypes.CDLL(LINUX_LIBRARY)


def _check(code: int, call: str) -> None:
    if code != NVML_SUCCESS:
        raise NvmlError(f"{call} returned NVML error {code}")


class NvmlEnergy:
    """Joules the board has used since the driver loaded, or None when a read fails."""

    def __init__(self, lib: Any, handle: ctypes.c_void_p) -> None:
        self._lib = lib
        self._handle = handle

    def __call__(self) -> float | None:
        millijoules = ctypes.c_ulonglong(0)
        try:
            code = self._lib.nvmlDeviceGetTotalEnergyConsumption(self._handle, ctypes.byref(millijoules))
        except (OSError, AttributeError):
            return None
        return millijoules.value / MILLIJOULES_PER_JOULE if code == NVML_SUCCESS else None


def open_energy(device: Any, loader: Callable[[], Any] = load_nvml) -> tuple[EnergyReader | None, str | None]:
    """(reader, log line) for a CUDA device; (None, None) on the CPU, (None, reason) when NVML fails."""
    kind = getattr(device, "type", str(device).split(":")[0])  # a torch.device or a name like "cuda:0"
    if kind != "cuda":
        return None, None
    index = device.index if isinstance(getattr(device, "index", None), int) else 0
    try:
        lib = loader()
        _check(lib.nvmlInit_v2(), "nvmlInit_v2")
        handle = ctypes.c_void_p()
        _check(
            lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle)), "GetHandleByIndex"
        )
    except (OSError, AttributeError, NvmlError) as exc:
        return None, f"GPU-board energy: unavailable ({exc}); compute.json will have no kWh for this run"
    reader = NvmlEnergy(lib, handle)
    if reader() is None:
        return None, "GPU-board energy: the NVML energy counter is not supported on this board; no kWh"
    return reader, f"GPU-board energy: NVML total-energy counter on GPU {index}, logged as gpu_power_w"
