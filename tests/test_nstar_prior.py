"""P6 v2 (EVAL.md PR-2, proposed): N* = M by the prior, overridden only by the epoch floor and VRAM; and
PR-3 (proposed): the value-mode p99 is reported, never gating, under both N* rules."""

import json
from pathlib import Path

import pytest

from blink import cli
from blink.train import nstar, size_sweep, sweep

REPO = Path(__file__).resolve().parent.parent
PARAMS = {"t": 0.3e6, "s": 4e6, "s-muon": 4e6, "m": 21e6, "m12": 31e6, "l": 60e6}
PRIOR = nstar.ChooseRules(rule="prior")


def _row(size: str, rate: float, compile: str = "off", micro: int = 256, peak: float = 3.0) -> dict:
    return {
        "size": size,
        "micro": micro,
        "compile": compile,
        "samples_per_s": rate,
        "oom": False,
        "error": None,
        "peak_reserved_gb": peak,
        "parameters": PARAMS[size],
    }


def _bench(rates: dict[str, float], p99: dict[str, float] | None = None, compile: str = "off") -> dict:
    p99 = p99 if p99 is not None else dict.fromkeys(rates, 40.0)
    play = [
        {"size": s, "rows": 219, "concurrency": c, "p99_ms": v, "p50_ms": v / 2}
        for s, v in p99.items()
        for c in (2, 5)
    ]
    throughput = [_row(s, r, compile) for s, r in rates.items()]
    return {"machine": {"vram_budget_gb": 5.5}, "throughput": throughput, "play": play}


# ---------------------------------------------------------------- the prior rule


def test_the_prior_rule_takes_m_even_when_s_has_the_best_vaa_and_m_is_over_the_p99_bar():
    bench = _bench({"s": 9900.0, "m": 2800.0, "m12": 2290.0}, p99={"s": 158.0, "m": 583.0, "m12": 721.0})
    sizes = {"s": {"vaa": 0.60}, "m": {"vaa": 0.54}}
    choice = nstar.choose(bench, sizes, 0.003, PRIOR)
    assert choice["n_star"] == "m"
    assert "prior" in choice["reason"] and "epoch floor" in choice["reason"]
    m = choice["sizes"]["m"]
    assert m["eligible"] is True and m["failed"] is None
    assert m["p99_ms"] == {"5": 583.0, "2": 583.0} and "over 100 ms" in m["p99_note"]
    assert choice["rules"]["rule"] == "prior"


def test_the_prior_rule_needs_no_6h_vaa_at_all():
    choice = nstar.choose(_bench({"s": 9900.0, "m": 2800.0}), {}, 0.003, PRIOR)
    assert choice["n_star"] == "m"
    assert choice["sizes"]["m"]["vaa"] is None and choice["sizes"]["s"]["eligible"] is True


def test_the_prior_rule_takes_the_largest_passing_size_when_m_fails_the_floor():
    bench = _bench({"s": 9900.0, "m": 1500.0, "m12": 1100.0, "l": 900.0})
    choice = nstar.choose(bench, {}, 0.003, PRIOR)
    assert choice["n_star"] == "s"
    assert "m fails the epoch floor" in choice["reason"] and "largest size that passes" in choice["reason"]
    assert choice["sizes"]["m"]["failed"] == "floor"


def test_the_prior_rule_judges_the_floor_at_the_training_compile_mode():
    """M passes the floor compiled (2,803) and fails it eager (1,567): the training mode decides."""
    bench = _bench({"s": 9900.0, "m": 2803.0})
    bench["throughput"] = [{**row, "compile": "inductor"} for row in bench["throughput"]] + [
        _row("s", 6200.0, "off"),
        _row("m", 1567.0, "off"),
    ]
    assert nstar.choose(bench, {}, 0.003, PRIOR, compile="inductor")["n_star"] == "m"
    eager = nstar.choose(bench, {}, 0.003, PRIOR, compile="off")
    assert eager["n_star"] == "s" and eager["sizes"]["m"]["samples_per_s"] == 1567.0


def test_the_prior_rule_sets_no_n_star_and_names_the_gate_when_m_has_no_vram_row():
    """No micro-batch >= 256 of M fits the budget: a measurement to redo, never a reason to shrink."""
    bench = _bench({"s": 9900.0, "m12": 2290.0})
    bench["throughput"].append(_row("m", 2803.0, peak=7.05))  # over the 5.5 GB budget
    choice = nstar.choose(bench, {}, 0.003, PRIOR)
    assert choice["n_star"] is None
    assert "P6-N*" in choice["reason"] and "VRAM" in choice["reason"]
    assert choice["sizes"]["m"]["failed"] == "vram"


def test_the_prior_rule_falls_back_only_to_its_candidate_sizes():
    """a10's s-muon bench row is an arm, not a size; it is neither a candidate nor listed."""
    bench = _bench({"t": 60000.0, "s": 9900.0, "s-muon": 9500.0, "m": 1500.0})
    choice = nstar.choose(bench, {}, 0.003, PRIOR)
    assert choice["n_star"] == "s"
    assert "s-muon" not in choice["sizes"] and "t" not in choice["sizes"]
    assert list(choice["sizes"]) == ["s", "m", "m12", "l"]  # smallest to largest
    only_t = nstar.ChooseRules(rule="prior", candidates=("t", "s", "m"))
    assert nstar.choose(bench, {}, 0.003, only_t)["n_star"] == "s"  # the largest passing, not the fastest


def test_an_unknown_rule_is_refused():
    with pytest.raises(ValueError, match="rule must be one of"):
        nstar.ChooseRules(rule="fastest")


def test_the_rule_default_is_the_v1_vaa_rule():
    assert nstar.ChooseRules().rule == "vaa"


# ---------------------------------------------------------------- configs/sweep.toml


def test_load_rules_reads_the_rule_and_the_candidates(tmp_path):
    path = tmp_path / "sweep.toml"
    path.write_text('[choose]\nrule = "prior"\ncandidates = ["s", "m"]\n', encoding="utf-8")
    rules = nstar.load_rules(path)
    assert (rules.rule, rules.candidates) == ("prior", ("s", "m"))


def test_the_repo_sweep_toml_is_p6_v2():
    import tomllib

    table = tomllib.loads((REPO / "configs" / "sweep.toml").read_text(encoding="utf-8"))
    assert table["sizes"]["sizes"] == [] and table["sizes"]["conditional"] == []
    rules = nstar.load_rules(REPO / "configs" / "sweep.toml")
    assert (rules.rule, rules.default, rules.epoch_floor) == ("prior", "m", 1658.0)
    assert rules.candidates == ("s", "m", "m12", "l")
    assert (
        size_sweep.load_size_sweep(REPO / "configs" / "sweep.toml").sizes == ()
    )  # `sweep sizes` runs nothing


# ---------------------------------------------------------------- blink sweep choose


def _choose_files(tmp_path: Path, rule: str) -> tuple[Path, Path]:
    from blink.model.config import compile_mode, read_tables

    mode = compile_mode(read_tables(sweep.CONFIG_DIR / "recipe.toml"))  # the command's training mode
    bench = tmp_path / "bench.json"
    bench.write_text(json.dumps(_bench({"s": 9900.0, "m": 2803.0}, p99={"s": 158.0}, compile=mode)), "utf-8")
    config = tmp_path / "sweep.toml"
    config.write_text(f'[choose]\nrule = "{rule}"\n', encoding="utf-8")
    return bench, config


def test_sweep_choose_under_the_prior_rule_runs_without_sweep_json(tmp_path, capsys):
    bench, config = _choose_files(tmp_path, "prior")
    out = tmp_path / "sweep.json"
    argv = ["sweep", "choose", "--bench", str(bench), "--config", str(config), "--sweep", str(out)]
    assert cli.main([*argv, "--sigma", "0.003"]) == 0
    printed = capsys.readouterr().out
    assert "N* = m" in printed and "p99 (reported, not gating)" in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["choice"]["n_star"] == "m" and written["choice"]["sigma"] == 0.003


def test_sweep_choose_under_the_prior_rule_needs_no_noise_floor(tmp_path, capsys):
    bench, config = _choose_files(tmp_path, "prior")
    out = tmp_path / "sweep.json"
    missing = tmp_path / "ablations.json"
    argv = ["sweep", "choose", "--bench", str(bench), "--config", str(config), "--sweep", str(out)]
    assert cli.main([*argv, "--ablations", str(missing)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["choice"]["sigma"] is None
    assert "sigma not recorded" in capsys.readouterr().out


def test_sweep_choose_under_the_vaa_rule_still_needs_sweep_json(tmp_path, capsys):
    bench, config = _choose_files(tmp_path, "vaa")
    out = tmp_path / "sweep.json"
    argv = ["sweep", "choose", "--bench", str(bench), "--config", str(config), "--sweep", str(out)]
    assert cli.main([*argv, "--sigma", "0.003"]) == 2
    assert "sweep.json not found" in capsys.readouterr().err
    assert not out.exists()


def test_sweep_choose_under_the_prior_rule_keeps_what_sweep_json_already_holds(tmp_path):
    bench, config = _choose_files(tmp_path, "prior")
    out = tmp_path / "sweep.json"
    out.write_text(json.dumps({"sizes": {"s": {"vaa": 0.61}}, "hours": 6.0}), encoding="utf-8")
    argv = ["sweep", "choose", "--bench", str(bench), "--config", str(config), "--sweep", str(out)]
    assert cli.main([*argv, "--sigma", "0.003"]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["hours"] == 6.0 and written["choice"]["n_star"] == "m"
    assert written["choice"]["sizes"]["s"]["vaa"] == 0.61  # reported beside the choice, never deciding it


# ---------------------------------------------------------------- with p7prep's micro-batch pins


def test_the_prior_rule_judges_a_pinned_size_at_its_pinned_micro_batch_row():
    bench = _bench({"s": 9900.0})
    bench["throughput"] += [_row("m", 2803.0, micro=512), _row("m", 2695.0, micro=256)]
    choice = nstar.choose(bench, {}, 0.003, PRIOR, pins={"m": 256})
    assert choice["n_star"] == "m" and choice["sizes"]["m"]["samples_per_s"] == 2695.0
    unpinned = [row for row in bench["throughput"] if row.get("micro") != 256 or row["size"] != "m"]
    missing = nstar.choose({**bench, "throughput": unpinned}, {}, 0.003, PRIOR, pins={"m": 256})
    assert missing["n_star"] is None and "pinned micro-batch 256" in missing["sizes"]["m"]["reason"]


def test_sweep_choose_under_the_prior_pins_its_candidates_without_sweep_json(tmp_path):
    """configs/m.toml pins micro-batch 256: M's facts are its 256 row even when sweep.json lists no size."""
    bench, config = _choose_files(tmp_path, "prior")
    data = json.loads(bench.read_text(encoding="utf-8"))
    mode = data["throughput"][0]["compile"]
    data["throughput"] = [r for r in data["throughput"] if r["size"] != "m"]
    data["throughput"] += [_row("m", 2803.0, mode, micro=512), _row("m", 2695.0, mode, micro=256)]
    bench.write_text(json.dumps(data), encoding="utf-8")
    out = tmp_path / "sweep.json"
    argv = ["sweep", "choose", "--bench", str(bench), "--config", str(config), "--sweep", str(out)]
    assert cli.main([*argv, "--sigma", "0.003"]) == 0
    choice = json.loads(out.read_text(encoding="utf-8"))["choice"]
    assert choice["n_star"] == "m" and choice["sizes"]["m"]["samples_per_s"] == 2695.0
