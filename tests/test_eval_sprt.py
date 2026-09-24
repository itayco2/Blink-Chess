"""The mode-choice SPRT (plan section 1 and E3): pentanomial, logistic, fishtest parity, every rule branch.

Reference LLRs were produced by fishtest's own `LLRcalc.LLR_logistic` (server/fishtest/stats, commit
93fe81eb8256b870ab759ec4470f6251fe985827), run with a pure-Python brentq (bisection to machine precision).
"""

import math

import pytest

from blink.eval import sprt

# (pentanomial counts, elo0, elo1, fishtest LLR_logistic)
FISHTEST_LLR = [
    ((3, 20, 50, 22, 5), 0, 20, 0.04032037241670679),
    ((3, 20, 50, 22, 5), 0, 5, 0.17838022438188125),
    ((3, 20, 50, 22, 5), -10, 10, 0.9344729113889829),
    ((0, 10, 30, 12, 0), 0, 20, -0.27041453628885204),
    ((0, 10, 30, 12, 0), 0, 5, 0.085226848211516),
    ((12, 40, 60, 30, 8), 0, 20, -2.9973011351582723),
    ((12, 40, 60, 30, 8), -10, 10, -2.0579732004260216),
]


@pytest.mark.parametrize(("penta", "elo0", "elo1", "expected"), FISHTEST_LLR)
def test_sprt_llr_matches_fishtest_on_fixed_pentanomial(penta, elo0, elo1, expected):
    assert sprt.llr_logistic(elo0, elo1, penta) == pytest.approx(expected, abs=1e-9)


def test_the_trinomial_llr_matches_fishtest_too():
    assert sprt.llr_logistic(0, 20, (30, 40, 50)) == pytest.approx(1.4333717997704136, abs=1e-9)


def test_bounds_are_walds_for_alpha_and_beta_005():
    lower, upper = sprt.bounds(0.05, 0.05)
    assert lower == pytest.approx(-2.9444389791664403, abs=1e-12)
    assert upper == pytest.approx(2.9444389791664403, abs=1e-12)


def test_the_pre_registered_config_is_elo0_0_elo1_20_alpha_beta_005_cap_6000():
    config = sprt.MODE_SPRT
    assert (config.elo0, config.elo1, config.alpha, config.beta, config.cap_games) == (
        0,
        20,
        0.05,
        0.05,
        6000,
    )


def test_a_state_accumulates_pairs_immutably():
    state = sprt.SprtState()
    later = state.with_pair(1.0, 0.5).with_pair(0.0, 0.0)
    assert state.penta == (0, 0, 0, 0, 0)
    assert later.penta == (1, 0, 0, 1, 0) and later.games == 4


def run(scores, config=sprt.MODE_SPRT):
    """run_sprt over a fixed list of game-pair scores (it must stop before running out)."""
    pairs = iter(scores)
    return sprt.run_sprt(lambda index: next(pairs), config)


def test_a_dominant_first_player_accepts_h1():
    result = run([(1.0, 1.0)] * 400)
    assert result.verdict == "H1" and result.llr >= result.upper
    assert result.games == 106  # fishtest's regularised LLR grows by about 0.056 per won pair


def test_a_dominated_first_player_accepts_h0():
    result = run([(0.0, 0.0)] * 400)
    assert result.verdict == "H0" and result.llr <= result.lower


def test_the_cap_stops_an_undecided_test():
    config = sprt.SprtConfig(elo0=0, elo1=20, alpha=0.05, beta=0.05, cap_games=20)
    result = run([(1.0, 0.0), (0.5, 0.5)] * 50, config)
    assert result.verdict is None and result.capped and result.games == 20


def test_the_cap_must_be_an_even_number_of_games():
    with pytest.raises(ValueError, match="even"):
        sprt.SprtConfig(cap_games=21)


def result(verdict, elo=0.0, capped=False):
    return sprt.SprtResult(
        penta=(0, 0, 1, 0, 0),
        llr=0.0,
        lower=-2.94,
        upper=2.94,
        verdict=verdict,
        capped=capped,
        elo=elo,
        elo_ci95=10.0,
        games=2,
    )


def test_value_ships_when_the_forward_sprt_accepts_h1():
    choice = sprt.choose_mode(result("H1", elo=30.0))
    assert choice.mode == "value" and choice.rule == "forward H1"


def test_h0_in_the_forward_sprt_calls_for_the_reverse_sprt():
    assert sprt.needs_reverse(result("H0"))
    assert not sprt.needs_reverse(result("H1"))
    with pytest.raises(ValueError, match="reverse"):
        sprt.choose_mode(result("H0"))


def test_policy_ships_when_the_reverse_sprt_accepts_h1():
    choice = sprt.choose_mode(result("H0"), result("H1", elo=25.0))
    assert choice.mode == "policy" and choice.rule == "reverse H1"


def test_a_second_h0_is_a_statistical_tie_and_policy_ships():
    choice = sprt.choose_mode(result("H0"), result("H0"))
    assert choice.mode == "policy" and choice.rule == "tied"


def test_a_capped_forward_sprt_ships_the_higher_point_estimate():
    assert sprt.choose_mode(result(None, elo=6.0, capped=True)).mode == "value"
    assert sprt.choose_mode(result(None, elo=-6.0, capped=True)).mode == "policy"


def test_a_capped_forward_sprt_with_an_exact_tie_ships_policy():
    choice = sprt.choose_mode(result(None, elo=0.0, capped=True))
    assert choice.mode == "policy" and choice.rule == "cap"


def test_a_capped_reverse_sprt_ships_the_higher_point_estimate():
    forward = result("H0")
    assert sprt.choose_mode(forward, result(None, elo=4.0, capped=True)).mode == "policy"
    assert sprt.choose_mode(forward, result(None, elo=-4.0, capped=True)).mode == "value"


def test_the_mode_choice_runs_the_reverse_only_after_h0():
    calls = []

    def forward(index):
        calls.append("forward")
        return (0.0, 0.0)

    def reverse(index):
        calls.append("reverse")
        return (1.0, 1.0) if index % 2 else (0.0, 0.0)

    config = sprt.SprtConfig(cap_games=200)
    choice = sprt.run_mode_choice(forward, reverse, config)
    assert choice.forward.verdict == "H0"
    assert choice.reverse is not None and choice.reverse.capped
    assert "reverse" in calls and calls.index("reverse") > calls.index("forward")
    assert choice.mode == "policy" and choice.rule == "cap"


def test_the_mode_choice_skips_the_reverse_after_h1():
    choice = sprt.run_mode_choice(lambda i: (1.0, 1.0), lambda i: pytest.fail("no reverse"), sprt.MODE_SPRT)
    assert choice.mode == "value" and choice.reverse is None


def test_the_elo_estimate_uses_fishtests_formula_on_the_pentanomial():
    outcome = run([(1.0, 0.5), (0.5, 0.5), (0.0, 0.5)] * 3, sprt.SprtConfig(cap_games=18))
    assert outcome.penta == (0, 3, 3, 3, 0)
    assert math.isclose(outcome.elo, 0.0, abs_tol=1e-9)
    assert outcome.as_dict()["penta"] == [0, 3, 3, 3, 0]
