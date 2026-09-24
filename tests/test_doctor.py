import pytest

from blink import doctor

GB = 1_000_000_000


def test_doctor_fails_when_torch_is_the_cpu_wheel():
    result = doctor.check_torch_build(torch_version="2.14.0+cpu", cuda_available=False)
    assert result.status == "FAIL"
    assert "uv sync" in result.fix and "cu130" in result.fix


def test_doctor_accepts_the_cu130_wheel():
    result = doctor.check_torch_build(torch_version="2.14.0+cu130", cuda_available=True)
    assert result.status == "ok"


def test_doctor_skips_torch_on_the_torch_free_leg():
    assert doctor.check_torch_build(torch_version=None, cuda_available=False).status == "skip"


def test_doctor_fails_below_the_disk_floor_and_names_it():
    result = doctor.check_disk("C:", free_bytes=10 * GB, floor_bytes=12 * GB)
    assert result.status == "FAIL"
    assert "12.0 GB" in result.detail


def test_doctor_passes_above_the_disk_floor():
    assert doctor.check_disk("D:", free_bytes=333 * GB, floor_bytes=300 * GB).status == "ok"


def test_disk_checks_only_run_on_windows():
    """Off Windows there is no C: or D:, so doctor must not report them as full (it used to fail)."""
    assert doctor.disk_checks(platform="linux", disk_c=12 * GB, disk_d=240 * GB) == []


def test_disk_checks_cover_c_and_d_on_windows():
    names = [r.name for r in doctor.disk_checks(platform="win32", disk_c=0, disk_d=0, usage=lambda _: 1)]
    assert names == ["disk C:", "disk D:"]


def test_a_hung_nvidia_smi_gives_no_holders_instead_of_hanging(monkeypatch):
    def hang(*args, **kwargs):
        raise doctor.subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(doctor.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(doctor.subprocess, "run", hang)
    assert doctor.vram_holders() == ()


def test_vram_budget_is_measured_free_minus_0_8_gb():
    assert doctor.vram_budget_bytes(free_bytes=int(6.3 * GB)) == int(5.5 * GB)


def test_low_free_vram_warns_and_names_the_processes_holding_it():
    result = doctor.check_vram(free_bytes=int(4.9 * GB), holders=("NVIDIA Broadcast.exe", "chrome.exe"))
    assert result.status == "WARN"
    assert "NVIDIA Broadcast.exe" in result.detail and "chrome.exe" in result.detail


def test_enough_free_vram_is_ok():
    assert doctor.check_vram(free_bytes=int(6.5 * GB), holders=()).status == "ok"


@pytest.mark.cuda
def test_doctor_reports_capability_8_6_bf16_and_the_efficient_sdpa_kernel():
    facts = doctor.collect_gpu_facts()
    assert facts.capability == (8, 6)
    assert facts.bf16_supported is True
    assert facts.efficient_sdpa_bf16 is True
    assert facts.flash_sdpa is False  # torch 2.14 on Windows builds no flash kernel (PF05)


def test_doctor_warns_when_the_gpu_board_energy_counter_is_unavailable():
    """Before a run that will be published, the NVML energy counter (compute.json's kWh) must read."""
    ok = doctor.check_energy(True, "GPU-board energy: NVML total-energy counter on GPU 0")
    assert ok.status == "ok" and "NVML" in ok.detail
    missing = doctor.check_energy(False, "GPU-board energy: unavailable (nvml.dll was not found)")
    assert missing.status == "WARN" and "nvml.dll" in missing.detail and "kWh" in missing.fix


def test_the_energy_reader_accepts_a_device_name_as_well_as_a_torch_device():
    from blink.train import power

    calls = []

    class Lib:
        def nvmlInit_v2(self):
            return 0

        def nvmlDeviceGetHandleByIndex_v2(self, index, ref):
            calls.append(index.value)
            return 0

        def nvmlDeviceGetTotalEnergyConsumption(self, handle, ref):
            ref._obj.value = 42_000
            return 0

    reader, _ = power.open_energy("cuda", loader=Lib)
    assert reader() == 42.0 and calls == [0]
