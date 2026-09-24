"""GPU-board energy per metrics window, from NVML's total-energy counter (the kWh producer for compute.json).

No test here touches a GPU: the NVML library is a fake with the same three calls, and the windows run on
the CPU with an injected energy reader.
"""

import ctypes
import json

import pytest

torch = pytest.importorskip("torch")

from blink.report import compute  # noqa: E402
from blink.train import power, telemetry  # noqa: E402

pytestmark = pytest.mark.torch


class FakeNvml:
    """nvmlInit_v2, nvmlDeviceGetHandleByIndex_v2 and nvmlDeviceGetTotalEnergyConsumption, in millijoules."""

    def __init__(self, millijoules, init_code=0, handle_code=0, energy_code=0):
        self.readings = iter(millijoules)
        self.init_code, self.handle_code, self.energy_code = init_code, handle_code, energy_code
        self.index = None

    def nvmlInit_v2(self):
        return self.init_code

    def nvmlDeviceGetHandleByIndex_v2(self, index, handle_ref):
        self.index = index.value
        handle_ref._obj.value = 0xB1
        return self.handle_code

    def nvmlDeviceGetTotalEnergyConsumption(self, handle, energy_ref):
        assert isinstance(handle, ctypes.c_void_p) and handle.value == 0xB1
        energy_ref._obj.value = next(self.readings)
        return self.energy_code


def _window(readings):
    energy = iter(readings)
    return telemetry.MetricWindow(torch.device("cpu"), energy=lambda: next(energy))


def _add(window, samples=1024):
    zero = torch.zeros(())
    window.add(zero, zero, zero, clip_norm=1.0, samples=samples)


def test_each_metrics_row_carries_the_windows_mean_board_power():
    """Joules between the window's open and close over its seconds; the next window opens with a new read."""
    window = _window([1_000.0, 1_600.0, 1_650.0, 2_000.0, 2_000.0])
    _add(window)
    first = window.flush(step=1, lr=1e-3)
    seconds = 1024 / first["samples_per_s"]
    assert first[compute.POWER_FIELD] == pytest.approx(600.0 / seconds)
    _add(window)
    second = window.flush(step=2, lr=1e-3)
    assert second[compute.POWER_FIELD] == pytest.approx(350.0 * second["samples_per_s"] / 1024)


def test_a_window_without_a_reading_logs_no_power_rather_than_a_guess():
    zero = telemetry.MetricWindow(torch.device("cpu"))
    _add(zero)
    assert compute.POWER_FIELD not in zero.flush(step=1, lr=1e-3)
    failed = _window([1_000.0, None, 5.0])
    _add(failed)
    assert compute.POWER_FIELD not in failed.flush(step=1, lr=1e-3)
    reset = _window([5_000.0, 10.0, 10.0])  # the driver reloaded: the counter went backwards
    _add(reset)
    assert compute.POWER_FIELD not in reset.flush(step=1, lr=1e-3)


def test_rows_from_the_window_give_compute_its_metrics_kwh(tmp_path):
    run = tmp_path / "long"
    run.mkdir()
    window = _window([0.0, 36_000.0, 36_000.0, 72_000.0, 72_000.0])
    rows = []
    for step in (1, 2):
        _add(window, samples=256)
        rows.append(window.flush(step=step, lr=1e-3))
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    config = {"device": "cuda", "config": {"batch_size": 256}}
    (run / "config.json").write_text(json.dumps(config), encoding="utf-8")
    result = compute.run_compute(run)
    assert result.kwh_source == "metrics"
    assert result.kwh == pytest.approx(72_000.0 / 3.6e6)


def test_the_nvml_reader_returns_joules_from_the_millijoule_counter():
    lib = FakeNvml([5_000_000, 5_600_000, 5_600_000])
    reader, note = power.open_energy(torch.device("cuda", 0), loader=lambda: lib)
    assert reader is not None and "NVML" in note and lib.index == 0
    assert reader() == pytest.approx(5_600.0)


def test_nvml_failures_disable_the_reading_with_a_reason_and_never_raise():
    def missing():
        raise OSError("nvml.dll was not found")

    assert power.open_energy(torch.device("cpu")) == (None, None)
    reader, note = power.open_energy(torch.device("cuda"), loader=missing)
    assert reader is None and "nvml.dll" in note and "no kWh" in note
    reader, note = power.open_energy(torch.device("cuda"), loader=lambda: FakeNvml([], init_code=9))
    assert reader is None and "nvmlInit_v2" in note
    unsupported = FakeNvml([0], energy_code=3)  # NVML_ERROR_NOT_SUPPORTED on an older board
    reader, note = power.open_energy(torch.device("cuda"), loader=lambda: unsupported)
    assert reader is None and "not supported" in note
    flaky = FakeNvml([1_000, 0], energy_code=0)
    reader, _ = power.open_energy(torch.device("cuda"), loader=lambda: flaky)
    flaky.energy_code = 999
    assert reader() is None
