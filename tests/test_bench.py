"""`blink bench throughput|loader|play`: measured numbers into bench.json (plan P4)."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from blink import cli  # noqa: E402
from blink.board import encode  # noqa: E402
from blink.data.record import ROOT_DTYPE  # noqa: E402
from blink.train import bench  # noqa: E402

pytestmark = pytest.mark.torch

TINY = """
[model]
d_model = 64
n_layers = 1
n_heads = 2
head_dim = 32
[train]
batch_size = 16
steps = 10
warmup_steps = 1
"""


@pytest.fixture
def tiny_config(tmp_path) -> Path:
    path = tmp_path / "tiny.toml"
    path.write_text(TINY, encoding="utf-8")
    return path


def test_sizes_resolve_to_the_configs_folder_or_to_a_path(tmp_path, tiny_config):
    assert bench.resolve_size("t") == ("t", bench.CONFIG_DIR / "t.toml")
    assert bench.resolve_size(str(tiny_config)) == ("tiny", tiny_config)
    with pytest.raises(FileNotFoundError, match="xl.toml"):
        bench.resolve_size("xl")


def test_a_throughput_row_measures_samples_per_second(tiny_config):
    spec = bench.ThroughputSpec(
        "tiny", tiny_config, micro=8, compile="off", steps=2, warmup=1, effective_batch=16, device="cpu"
    )
    row = bench.measure_throughput(spec)
    assert row["oom"] is False and row["error"] is None
    assert row["samples_per_s"] > 0 and row["accum"] == 2 and row["samples"] == 2 * 16
    assert row["size"] == "tiny" and row["micro"] == 8 and row["compile"] == "off"
    assert np.isfinite(row["loss"])


def test_an_out_of_memory_configuration_is_recorded_and_the_sweep_goes_on(tiny_config, monkeypatch):
    def boom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    monkeypatch.setattr(bench, "_train_step", boom)
    specs = [
        bench.ThroughputSpec("tiny", tiny_config, micro=m, compile="off", steps=1, warmup=1, device="cpu")
        for m in (8, 16)
    ]
    rows = bench.run_throughput(specs, log=lambda _: None)
    assert [row["oom"] for row in rows] == [True, True]
    assert all("out of memory" in row["error"] for row in rows)


def test_bench_json_sections_merge_by_their_keys(tmp_path):
    out = tmp_path / "bench.json"
    first = [{"size": "s", "micro": 256, "compile": "off", "samples_per_s": 1.0}]
    bench.update_bench(out, "throughput", first, machine={"gpu": "x"})
    second = [
        {"size": "s", "micro": 256, "compile": "off", "samples_per_s": 2.0},
        {"size": "s", "micro": 512, "compile": "off", "samples_per_s": 3.0},
    ]
    bench.update_bench(out, "throughput", second)
    bench.update_bench(out, "loader", {"skeleton": {"samples_per_s": 9.0}})
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert [r["samples_per_s"] for r in saved["throughput"]] == [2.0, 3.0]
    assert saved["loader"]["skeleton"]["samples_per_s"] == 9.0 and saved["machine"]["gpu"] == "x"


def _row(size, micro, rate, peak, compile="off", oom=False, error=None):
    keys = ("size", "micro", "samples_per_s", "peak_reserved_gb", "compile", "oom", "error")
    return dict(zip(keys, (size, micro, rate, peak, compile, oom, error), strict=True))


def test_a_recipe_with_clip_auto_benchmarks_without_error(tmp_path):
    """PF64: the recipe's clip_norm = "auto" reached clip_grad_norm_ as a string and every row errored."""
    config = tmp_path / "auto.toml"
    config.write_text(TINY + 'clip_norm = "auto"\n', encoding="utf-8")
    spec = bench.ThroughputSpec("auto", config, micro=8, compile="off", steps=1, warmup=1, device="cpu")
    row = bench.measure_throughput(spec)
    assert row["error"] is None and row["samples_per_s"] > 0


@pytest.mark.cuda
def test_the_cudagraphs_backend_trains_several_steps_in_the_bench(tiny_config):
    """PF64: with accumulation, a .grad that keeps the graph's output is overwritten by the next replay."""
    spec = bench.ThroughputSpec(
        "tiny", tiny_config, micro=16, compile="cudagraphs", steps=3, warmup=2, effective_batch=64
    )
    row = bench.measure_throughput(spec)
    assert row["error"] is None, row["error"]
    assert row["spilled"] is False and 0 < row["peak_reserved_gb"] < row["vram_total_gb"]


@pytest.mark.cuda
def test_cuda_graph_gradients_over_accumulated_micro_batches_match_eager_ones(tiny_config):
    """Our own .grad buffers must hold this step's sum, not a replay's overwritten output."""
    import copy

    from blink.model.config import load_config
    from blink.model.transformer import BlinkNet
    from blink.train.batch import make_batch

    cfg = load_config(tiny_config)
    torch.manual_seed(0)
    eager = BlinkNet(cfg.model).cuda()
    graphed = copy.deepcopy(eager)
    batch = make_batch(bench.synthetic_records(16, seed=1), torch.device("cuda"))
    frozen = torch.optim.SGD(graphed.parameters(), lr=0.0)  # the step leaves both models equal
    step_model = bench._compiled(graphed, "cudagraphs")
    for _ in range(3):
        bench._train_step(step_model, frozen, batch, 2, cfg, "cuda", float("inf"), graphs=True)
    bench._train_step(
        eager, torch.optim.SGD(eager.parameters(), lr=0.0), batch, 2, cfg, "cuda", float("inf"), False
    )
    for (name, a), b in zip(eager.named_parameters(), graphed.parameters(), strict=True):
        torch.testing.assert_close(b.grad, a.grad, rtol=2e-2, atol=1e-4, msg=name)


def test_a_peak_above_the_cards_vram_is_a_spill_into_system_memory():
    """PF64: with the driver's sysmem fallback a row that should OOM runs slowly instead (16.7 GiB on 8)."""
    assert bench.spilled(peak_gb=16.66, total_gb=8.0) is True
    assert bench.spilled(peak_gb=7.9, total_gb=8.0) is False
    assert bench.spilled(peak_gb=None, total_gb=8.0) is False


def test_best_rates_never_pick_a_spilled_row_even_without_a_budget():
    rows = [_row("m", 256, 900.0, 3.0), {**_row("m", 512, 5000.0, 16.7), "spilled": True}]
    assert bench.best_rates({"throughput": rows})["m"]["micro"] == 256


def test_best_rates_skip_oom_errors_small_micro_batches_and_rows_over_the_vram_budget():
    rows = [
        _row("m", 128, 9000.0, 1.0),
        _row("m", 256, 2000.0, 3.0),
        _row("m", 512, 2600.0, 7.0, compile="inductor"),
        _row("m", 1024, 0.0, None, oom=True, error="oom"),
        _row("l", 256, 800.0, 5.0),
    ]
    best = bench.best_rates({"machine": {"vram_budget_gb": 5.5}, "throughput": rows})
    assert best["m"]["samples_per_s"] == 2000.0 and best["m"]["micro"] == 256
    assert best["l"]["samples_per_s"] == 800.0


def _write_shard(path: Path, n: int, start: int) -> None:
    records = np.zeros(n, dtype=ROOT_DTYPE)
    records["cp"] = np.arange(start, start + n)
    records.tofile(path)


def test_the_loader_bench_reads_every_record_once_per_pass(tmp_path):
    for i, n in enumerate((300, 200, 250)):
        _write_shard(tmp_path / f"train_{i:03d}.bin", n, 1000 * i)
    result = bench.measure_loader(sorted(tmp_path.glob("train_*.bin")), batch_size=50, passes=2)
    assert [p["records"] for p in result["passes"]] == [750, 750]
    assert result["bytes"] == 750 * ROOT_DTYPE.itemsize
    assert all(p["samples_per_s"] > 0 and p["read_mb_per_s"] > 0 for p in result["passes"])


def test_the_play_bench_reports_p50_and_p99_at_each_row_count_and_concurrency(tiny_config):
    specs = [
        bench.PlaySpec("tiny", tiny_config, rows=r, concurrency=c, iters=10, warmup=2, device="cpu")
        for r, c in ((1, 1), (5, 2))
    ]
    rows = bench.run_play(specs, log=lambda _: None)
    assert [(r["rows"], r["concurrency"]) for r in rows] == [(1, 1), (5, 2)]
    assert rows[1]["latencies"] == 2 * 10
    assert all(0 < r["p50_ms"] <= r["p99_ms"] for r in rows)


def test_random_play_codes_are_legal_square_codes():
    codes = bench.random_codes(100, seed=0)
    assert codes.shape == (100, 64) and codes.dtype == np.uint8 and codes.max() < encode.NUM_CODES


def test_bench_loader_command_writes_the_loader_section(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    data = tmp_path / "data"
    data.mkdir()
    _write_shard(data / "train_000.bin", 400, 0)
    out = tmp_path / "bench.json"
    assert cli.main(["bench", "loader", "--data", str(data), "--batch", "100", "--out", str(out)]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["loader"][str(data)]["passes"][0]["records"] == 400
    assert "samples/s" in capsys.readouterr().out


def test_bench_throughput_command_on_cpu_writes_rows(tmp_path, tiny_config, capsys):
    out = tmp_path / "bench.json"
    flags = ["--micro", "8", "--compile", "off", "--steps", "1", "--warmup", "1", "--device", "cpu"]
    argv = ["bench", "throughput", "--sizes", str(tiny_config), *flags, "--out", str(out)]
    assert cli.main(argv) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["throughput"][0]["size"] == "tiny" and saved["throughput"][0]["samples_per_s"] > 0


@pytest.mark.cuda
def test_the_t_model_trains_on_the_gpu_in_the_throughput_bench():
    spec = bench.ThroughputSpec("t", bench.CONFIG_DIR / "t.toml", micro=64, compile="off", steps=2, warmup=1)
    row = bench.measure_throughput(spec)
    assert row["error"] is None and row["samples_per_s"] > 0 and row["peak_reserved_gb"] > 0
