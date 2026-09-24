"""Play latency by play mode: bench play rows keyed by precision and compile, and N* judged at one mode."""

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from blink import cli  # noqa: E402
from blink.train import bench, nstar  # noqa: E402

pytestmark = pytest.mark.torch

TINY = """
[model]
d_model = 64
n_layers = 1
n_heads = 2
head_dim = 32
"""


@pytest.fixture
def tiny_config(tmp_path) -> Path:
    path = tmp_path / "tiny.toml"
    path.write_text(TINY, encoding="utf-8")
    return path


@pytest.fixture
def compile_spy(monkeypatch):
    """torch.compile replaced by a pass-through that records dynamic=; no compiler runs on the CPU."""
    calls = []

    def spy(module, **kwargs):
        calls.append(kwargs)
        return module

    monkeypatch.setattr(torch, "compile", spy)
    return calls


def _play(size, rows, concurrency, p99, precision=None, compile=None) -> dict:
    row = {"size": size, "rows": rows, "concurrency": concurrency, "p99_ms": p99, "p50_ms": p99 / 2}
    if precision is not None:
        row["precision"] = precision
    if compile is not None:
        row["compile"] = compile
    return row


# ---------------------------------------------------------------- bench play


def test_a_play_spec_defaults_to_fp32_uncompiled_and_refuses_bf16_off_cuda(tiny_config):
    spec = bench.PlaySpec("tiny", tiny_config, rows=3, device="cpu")
    assert (spec.precision, spec.compile) == ("fp32", False)
    with pytest.raises(ValueError, match="CUDA only"):
        bench.PlaySpec("tiny", tiny_config, rows=3, device="cpu", precision="bf16")


def test_play_rows_record_the_mode_they_were_timed_in(tiny_config, compile_spy):
    specs = [
        bench.PlaySpec("tiny", tiny_config, rows=3, iters=3, warmup=1, device="cpu"),
        bench.PlaySpec("tiny", tiny_config, rows=3, iters=3, warmup=1, device="cpu", compile=True),
    ]
    lines = []
    rows = bench.run_play(specs, log=lines.append)
    assert [(r["precision"], r["compile"]) for r in rows] == [("fp32", False), ("fp32", True)]
    assert all(r["p99_ms"] > 0 for r in rows)
    assert compile_spy == [{"dynamic": True}]
    assert "fp32 compiled" in lines[1]


def test_a_failed_play_row_still_carries_its_key(tiny_config, monkeypatch):
    def fail(spec):
        raise RuntimeError("boom")

    monkeypatch.setattr(bench, "measure_play", fail)
    spec = bench.PlaySpec("tiny", tiny_config, rows=3, device="cpu", compile=True)
    (row,) = bench.run_play([spec], log=lambda _: None)
    assert row["error"] == "RuntimeError: boom" and (row["precision"], row["compile"]) == ("fp32", True)


def test_play_rows_of_different_modes_never_overwrite_each_other(tmp_path):
    out = tmp_path / "bench.json"
    legacy = _play("m", 219, 5, 102.0)  # timed before the mode existed: fp32, uncompiled
    fast = _play("m", 219, 5, 31.0, "bf16", True)
    bench.update_bench(out, "play", [legacy, fast])
    bench.update_bench(out, "play", [_play("m", 219, 5, 54.0, "bf16", False)])
    bench.update_bench(out, "play", [_play("m", 219, 5, 99.0, "fp32", False)])  # replaces the legacy row
    saved = json.loads(out.read_text(encoding="utf-8"))["play"]
    by_mode = {bench.play_mode(row): row["p99_ms"] for row in saved}
    assert by_mode == {("bf16", True): 31.0, ("bf16", False): 54.0, ("fp32", False): 99.0}
    assert len(saved) == 3


def test_bench_play_command_times_the_mode_it_is_given(tmp_path, tiny_config, compile_spy):
    out = tmp_path / "bench.json"
    base = ["bench", "play", "--sizes", str(tiny_config), "--rows", "3", "--concurrency", "1"]
    base += ["--iters", "2", "--warmup", "1", "--device", "cpu", "--out", str(out)]
    assert cli.main(base) == 0
    assert cli.main([*base, "--compile"]) == 0
    saved = json.loads(out.read_text(encoding="utf-8"))["play"]
    assert sorted(bench.play_mode(row) for row in saved) == [("fp32", False), ("fp32", True)]


def test_bench_play_command_refuses_bf16_on_the_cpu(tmp_path, tiny_config, capsys):
    argv = ["bench", "play", "--sizes", str(tiny_config), "--device", "cpu", "--precision", "bf16"]
    assert cli.main([*argv, "--out", str(tmp_path / "bench.json")]) == 2
    assert "CUDA only" in capsys.readouterr().err
    assert not (tmp_path / "bench.json").exists()


# ---------------------------------------------------------------- N*


def _bench(play: list[dict]) -> dict:
    throughput = [
        {"size": s, "micro": 256, "compile": "off", "samples_per_s": 3000.0, "oom": False, "error": None,
         "peak_reserved_gb": 3.0, "parameters": p}
        for s, p in (("s", 4e6), ("m", 21e6), ("m12", 31e6))
    ]  # fmt: skip
    return {"machine": {"vram_budget_gb": 5.5}, "throughput": throughput, "play": play}


def _rows(size: str, p99: float, precision=None, compile=None) -> list[dict]:
    return [_play(size, 219, c, p99, precision, compile) for c in (2, 5)]


FAST = nstar.ChooseRules(p99_precision="bf16", p99_compile=True)


def test_p99_is_read_from_rows_of_the_rules_play_mode_only():
    data = _bench(_rows("m", 102.0) + _rows("m", 54.0, "bf16", False) + _rows("m", 31.0, "bf16", True))
    assert nstar.p99_of(data, "m", nstar.ChooseRules()) == {"5": 102.0, "2": 102.0}
    assert nstar.p99_of(data, "m", FAST) == {"5": 31.0, "2": 31.0}
    bf16 = nstar.ChooseRules(p99_precision="bf16")
    assert nstar.p99_of(data, "m", bf16) == {"5": 54.0, "2": 54.0}


def test_a_mode_without_rows_is_not_measured_rather_than_borrowed_from_another():
    data = _bench(_rows("m", 40.0))
    assert nstar.p99_of(data, "m", FAST) == {"5": None, "2": None}


def test_choose_judges_every_size_at_the_configured_play_mode():
    play = _rows("s", 30.0) + _rows("m", 102.0) + _rows("m12", 140.0)
    play += _rows("s", 12.0, "bf16", True) + _rows("m", 31.0, "bf16", True) + _rows("m12", 45.0, "bf16", True)
    sizes = {"s": {"vaa": 0.50}, "m": {"vaa": 0.54}, "m12": {"vaa": 0.58}}
    today = nstar.choose(_bench(play), sizes, 0.005, nstar.ChooseRules())
    assert today["n_star"] == "s" and "in fp32" in today["sizes"]["m"]["reason"]
    fast = nstar.choose(_bench(play), sizes, 0.005, FAST)
    assert fast["n_star"] == "m12" and fast["rules"]["p99_precision"] == "bf16"
    assert fast["sizes"]["m12"]["p99_ms"] == {"5": 45.0, "2": 45.0}


def test_the_reason_names_the_mode_that_was_not_measured():
    choice = nstar.choose(_bench(_rows("m", 40.0)), {"m": {"vaa": 0.5}}, 0.005, FAST)
    assert (
        "not measured" in choice["sizes"]["m"]["reason"] and "bf16 compiled" in choice["sizes"]["m"]["reason"]
    )


def test_the_rules_default_to_todays_mode_and_the_repo_config_keeps_it():
    assert nstar.ChooseRules().play_mode == ("fp32", False)
    repo = nstar.load_rules(Path(__file__).resolve().parents[1] / "configs" / "sweep.toml")
    assert repo.play_mode == ("fp32", False)
    assert (repo.p99_ms_max, repo.p99_rows, repo.p99_concurrency) == (100.0, 219, (5, 2))


def test_the_rules_read_the_play_mode_from_sweep_toml(tmp_path):
    path = tmp_path / "sweep.toml"
    path.write_text('[choose]\np99_precision = "bf16"\np99_compile = true\n', encoding="utf-8")
    assert nstar.load_rules(path).play_mode == ("bf16", True)


@pytest.mark.parametrize("bad", [{"p99_precision": "fp16"}, {"p99_compile": "yes"}])
def test_a_bad_play_mode_in_the_rules_is_refused(bad):
    with pytest.raises(ValueError):
        nstar.ChooseRules(**bad)
