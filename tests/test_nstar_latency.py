"""EVAL.md PR-3 (proposed): latency never changes N*. The value-mode p99 is reported beside each size,
with a note when it is over the bar or not measured in the configured play mode, and never gates."""

import pytest

from blink.train import nstar

PARAMS = {"s": 4e6, "m": 21e6, "m12": 31e6, "l": 60e6}


def _bench(rates: dict[str, float], p99: dict[str, float] | None = None) -> dict:
    p99 = p99 if p99 is not None else dict.fromkeys(rates, 40.0)
    throughput = [
        {"size": s, "micro": 256, "compile": "off", "samples_per_s": r, "oom": False, "error": None,
         "peak_reserved_gb": 3.0, "parameters": PARAMS[s]}
        for s, r in rates.items()
    ]  # fmt: skip
    play = [
        {"size": s, "rows": 219, "concurrency": c, "p99_ms": v, "p50_ms": v / 2}
        for s, v in p99.items()
        for c in (2, 5)
    ]
    return {"machine": {"vram_budget_gb": 5.5}, "throughput": throughput, "play": play}


def test_a_size_over_the_p99_bar_stays_eligible_and_the_note_says_why():
    """The measured fp32 rows of 2026-09-24: no size meets 100 ms at 5 games at once, and N* still exists."""
    bench = _bench({"s": 9900.0, "m": 2800.0, "m12": 2290.0}, p99={"s": 158.0, "m": 583.0, "m12": 721.0})
    sizes = {"s": {"vaa": 0.50}, "m": {"vaa": 0.54}, "m12": {"vaa": 0.58}}
    choice = nstar.choose(bench, sizes, 0.005, nstar.ChooseRules())
    assert choice["n_star"] == "m12" and choice["reason"].startswith("best 6 h VAA")
    assert all(choice["sizes"][s]["eligible"] for s in sizes)
    assert "over 100 ms" in choice["sizes"]["m12"]["p99_note"]
    assert choice["sizes"]["m12"]["p99_ms"] == {"5": 721.0, "2": 721.0}


def test_the_2_sigma_tie_to_m_holds_whatever_m_s_latency():
    bench = _bench({"s": 9900.0, "m": 2800.0}, p99={"s": 20.0, "m": 583.0})
    choice = nstar.choose(bench, {"s": {"vaa": 0.545}, "m": {"vaa": 0.54}}, 0.005, nstar.ChooseRules())
    assert choice["n_star"] == "m" and "within 2 sigma" in choice["reason"]


def test_a_p99_not_measured_in_the_play_mode_is_noted_not_gating():
    choice = nstar.choose(_bench({"m": 2800.0}, p99={}), {"m": {"vaa": 0.54}}, 0.005, nstar.ChooseRules())
    assert choice["n_star"] == "m"
    assert "not measured" in choice["sizes"]["m"]["p99_note"]
    assert choice["sizes"]["m"]["p99_ms"] == {"5": None, "2": None}


def test_a_p99_within_the_bar_leaves_no_note():
    choice = nstar.choose(_bench({"m": 2800.0}), {"m": {"vaa": 0.54}}, 0.005, nstar.ChooseRules())
    assert choice["sizes"]["m"]["p99_note"] is None


def test_the_floor_and_the_6h_vaa_still_gate():
    bench = _bench({"s": 9900.0, "m": 2800.0, "l": 1598.0})
    sizes = {"s": {"vaa": 0.5}, "m": {}, "l": {"vaa": 0.7}}
    choice = nstar.choose(bench, sizes, 0.005, nstar.ChooseRules())
    assert choice["n_star"] == "s"
    assert choice["sizes"]["m"]["failed"] == "vaa" and choice["sizes"]["l"]["failed"] == "floor"


@pytest.mark.parametrize("concurrency", [(5,), (5, 2)])
def test_the_note_covers_every_concurrency_the_rules_name(concurrency):
    rules = nstar.ChooseRules(p99_concurrency=concurrency)
    choice = nstar.choose(_bench({"m": 2800.0}, p99={"m": 150.0}), {"m": {"vaa": 0.5}}, 0.005, rules)
    assert all(c in choice["sizes"]["m"]["p99_note"] for c in map(str, concurrency))
