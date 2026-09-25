"""Sizes that pin their micro-batch (M and M12 pin 256): planned, trained and judged at that bench row.

The trainer runs a pinned micro-batch as it is, with no VRAM probe, so the size sweep, `sweep choose`
and the supervisor must read the bench row measured at that micro-batch, not the fastest row that fits.
"""

import json
import tomllib

from test_sweep import SIZE_WITH_BASE, SizeRunner, _bench

from blink import cli
from blink.train import nstar, size_sweep, sweep

PINNED_RECIPE = """
[model]
head_dim = 32

[train]
batch_size = 1024
micro_batch = "auto"
warmup_steps = 10
compile = "inductor"
"""


def _pinned_bench(m_rates: dict[int, float]) -> dict:
    ok = {"oom": False, "error": None, "peak_reserved_gb": 3.0, "compile": "inductor"}
    rows = [{**ok, "size": "m", "micro": micro, "samples_per_s": rate} for micro, rate in m_rates.items()]
    rows += [{**ok, "size": "s", "micro": 512, "samples_per_s": 9000.0}]
    rows += [{**ok, "size": "s", "micro": 1024, "samples_per_s": 9938.0}]
    return {"machine": {"vram_budget_gb": 6.2}, "throughput": rows, "play": []}


def test_the_size_sweep_plans_and_trains_a_size_at_its_pinned_micro_batch(tmp_path, monkeypatch):
    """M pins micro-batch 256: its run is planned and policed at the 256 row, and the recipe's "auto"
    (arm-style overrides merged over each size) must not undo the pin in the config it trains."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    (tmp_path / "recipe.toml").write_text(PINNED_RECIPE, encoding="utf-8")
    (tmp_path / "s.toml").write_text(SIZE_WITH_BASE, encoding="utf-8")
    pinned = SIZE_WITH_BASE.replace("steps = 100", "steps = 100\nmicro_batch = 256")
    (tmp_path / "m.toml").write_text(pinned, encoding="utf-8")
    setup = size_sweep.SizeSweep(
        sizes=("s", "m"),
        conditional=(),
        hours=0.01,
        recipe=tmp_path / "recipe.toml",
        data=tmp_path,
        config_dir=tmp_path,
    )
    runner = SizeRunner(tmp_path / "home")
    bench = _pinned_bench({256: 2695.0, 512: 2803.0})
    report = size_sweep.run_sizes(setup, bench, tmp_path / "sweep.json", runner=runner, log=lambda _: None)
    assert (report["sizes"]["m"]["micro"], report["sizes"]["m"]["samples_per_s"]) == (256, 2695.0)
    assert report["sizes"]["s"]["micro"] == 1024  # no pin: the fastest row that fits, as before
    trains = {r.run: tomllib.loads(r.config.read_text(encoding="utf-8"))["train"] for r in runner.requests}
    assert (trains["size-m"]["micro_batch"], trains["size-s"]["micro_batch"]) == (256, "auto")
    assert [r.bench_rate for r in runner.requests] == [9938.0, 2695.0]


def test_the_size_sweep_does_not_run_a_pinned_size_without_a_row_at_its_pin(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    (tmp_path / "recipe.toml").write_text(PINNED_RECIPE, encoding="utf-8")
    pinned = SIZE_WITH_BASE.replace("steps = 100", "steps = 100\nmicro_batch = 256")
    (tmp_path / "m.toml").write_text(pinned, encoding="utf-8")
    setup = size_sweep.SizeSweep(
        sizes=("m",), conditional=(), hours=0.01, recipe=None, data=tmp_path, config_dir=tmp_path
    )
    runner = SizeRunner(tmp_path / "home")
    report = size_sweep.run_sizes(
        setup, _pinned_bench({512: 2803.0}), tmp_path / "sweep.json", runner=runner, log=lambda _: None
    )
    assert "micro-batch 256" in report["sizes"]["m"]["status"] and not runner.requests


def test_choose_judges_a_size_that_pins_its_micro_batch_at_that_rows_rate():
    bench = _bench({"s": 9000.0, "m": 1600.0}, {"s": 10.0, "m": 20.0})
    bench["throughput"].append({**bench["throughput"][1], "micro": 512, "samples_per_s": 3000.0})
    sizes = {"s": {"vaa": 0.50}, "m": {"vaa": 0.54}}
    assert nstar.choose(bench, sizes, 0.005, nstar.ChooseRules())["n_star"] == "m"
    pinned = nstar.choose(bench, sizes, 0.005, nstar.ChooseRules(), pins={"m": 256})
    assert pinned["n_star"] == "s" and pinned["sizes"]["m"]["failed"] == "floor"
    assert pinned["sizes"]["m"]["samples_per_s"] == 1600.0


def test_sweep_choose_command_judges_m_at_the_micro_batch_configs_m_toml_pins(tmp_path, capsys):
    from blink.model.config import compile_mode, micro_batch_pin, read_tables

    assert micro_batch_pin(read_tables(sweep.CONFIG_DIR / "m.toml")) == 256
    mode = compile_mode(read_tables(sweep.CONFIG_DIR / "recipe.toml"))
    data = _bench({"s": 9000.0, "m": 1600.0}, {"s": 10.0, "m": 20.0})
    data["throughput"].append({**data["throughput"][1], "micro": 512, "samples_per_s": 3000.0})
    data["throughput"] = [{**row, "compile": mode} for row in data["throughput"]]
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(data), encoding="utf-8")
    sizes = tmp_path / "sweep.json"
    sizes.write_text(json.dumps({"sizes": {"s": {"vaa": 0.50}, "m": {"vaa": 0.54}}}), encoding="utf-8")
    argv = ["sweep", "choose", "--bench", str(bench), "--sweep", str(sizes), "--sigma", "0.005"]
    assert cli.main(argv) == 0
    assert "N* = s" in capsys.readouterr().out  # m's pinned 256 row fails the epoch floor
