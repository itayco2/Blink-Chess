import pytest

from blink.train import vram

GB = 2**30


def test_the_budget_is_measured_free_vram_minus_0_8_gb():
    assert vram.budget_bytes(int(6.3 * GB)) == pytest.approx(5.5 * GB, abs=1)
    with pytest.raises(MemoryError, match="0.8 GB"):
        vram.budget_bytes(int(0.5 * GB))


def test_candidates_halve_the_batch_down_to_the_floor():
    assert vram.candidates(1024) == [1024, 512, 256, 128, 64, 32]
    assert vram.candidates(96, floor=16) == [96, 48, 24]


def _linear(base_gb: float, mb_per_row: float):
    calls = []

    def peak(micro: int) -> float:
        calls.append(micro)
        return (base_gb + micro * mb_per_row / 1024) * GB

    return peak, calls


def test_micro_batch_is_the_largest_candidate_predicted_and_measured_to_fit():
    """PF02: 0.6 GB + 5 MB a row fits 980 rows in 5.5 GB, so 512; only 32, 64 and 512 are ever run."""
    peak, calls = _linear(0.6, 5.0)
    micro, probes = vram.choose_micro_batch(1024, peak, budget=int(5.5 * GB))
    assert micro == 512 and calls == [32, 64, 512]
    assert probes[-1]["fits"] and probes[-1]["predicted_gb"] == pytest.approx(3.1)


def test_no_probe_ever_runs_a_size_predicted_to_overflow_the_budget():
    peak, calls = _linear(0.6, 5.0)
    vram.choose_micro_batch(1024, peak, budget=int(1.5 * GB))
    assert max(calls) == 128  # 0.6 + 0.625 GB; 256 rows would need 1.85 GB


def test_a_verify_that_runs_out_of_memory_or_over_budget_steps_down():
    def peak(micro: int):
        if micro >= 512:
            return None  # out of memory despite the prediction
        if micro == 256:
            return 2 * GB  # worse than predicted: over the 1.5 GB budget
        return (0.1 + micro * 0.002) * GB

    micro, probes = vram.choose_micro_batch(1024, peak, budget=int(1.5 * GB))
    assert micro == 128
    assert [p["micro"] for p in probes] == [32, 64, 512, 256, 128]


def test_a_batch_that_is_its_own_only_candidate_is_measured_directly():
    peak, calls = _linear(0.1, 1.0)
    assert vram.choose_micro_batch(16, peak, budget=GB)[0] == 16 and calls == [16]


def test_no_micro_batch_that_fits_is_an_explicit_error():
    with pytest.raises(MemoryError, match="no micro-batch"):
        vram.choose_micro_batch(256, lambda m: 10 * GB, budget=GB)


@pytest.mark.cuda
def test_a_real_probe_on_cuda_reports_a_peak_that_grows_with_the_micro_batch():
    torch = pytest.importorskip("torch")
    from train_helpers import tiny_model_config

    from blink.model.transformer import BlinkNet

    model = BlinkNet(tiny_model_config(gab=True)).cuda()
    small = vram.probe_peak(model, 32, torch.device("cuda"))
    large = vram.probe_peak(model, 512, torch.device("cuda"))
    assert 0 < small < large
    assert all(p.grad is None for p in model.parameters())


def test_a_batch_below_the_floor_is_its_own_only_candidate():
    assert vram.candidates(16) == [16]
