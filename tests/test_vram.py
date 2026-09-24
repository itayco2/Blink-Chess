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


def test_micro_batch_is_the_largest_candidate_whose_peak_fits_the_budget():
    peaks = {1024: 9 * GB, 512: 5.6 * GB, 256: 3.1 * GB, 128: 1.9 * GB}
    micro, probes = vram.choose_micro_batch(1024, lambda m: peaks[m], budget=int(5.5 * GB))
    assert micro == 256
    assert [p["micro"] for p in probes] == [1024, 512, 256]
    assert probes[-1]["fits"] and not probes[0]["fits"]


def test_an_out_of_memory_probe_moves_to_a_smaller_micro_batch():
    micro, probes = vram.choose_micro_batch(512, lambda m: None if m > 128 else m * 0.005 * GB, budget=GB)
    assert micro == 128
    assert probes[0]["peak_gb"] is None


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
