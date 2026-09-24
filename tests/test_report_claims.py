"""The hook and the pre-registered claim: chosen by the shipped mode, filled only from results/*.json."""

from dataclasses import replace

import pytest
from test_report_fixtures import compute, lichess, results, strength_rows, write_bundle

from blink.report import claims

POLICY_HE = "בלינק: בינה מלאכותית לשחמט שלא מחפשת אף פעם."
VALUE_HE = "בלינק: בינה מלאכותית לשחמט שמסתכלת מהלך אחד קדימה, אף פעם לא שניים."

VALUE_CLAIM = (
    "Blink is a 22.6M-parameter transformer that looks exactly one move ahead, never two: each move is at "
    "most one batched forward pass that scores the position after every legal move once. It never evaluates "
    "an opponent's reply, and it uses no opening book, tablebase or engine at play time. It was trained by "
    "supervised learning on 573,400,000 positions drawn from the 409,710,113-position Lichess evaluation "
    "database (CC0). Training took 120.3 GPU-hours for the flagship run (181.2 for the whole project), with "
    "no cloud GPU and no paid data: one home RTX 3070 (39.8 GPU-board kWh). Against pinned Stockfish 19 "
    "UCI_Elo anchors it rates 1850 +/- 35 (95% CI, 4,100 games). That is a CCRL-Blitz-anchored engine "
    "scale, not FIDE, and puts it about level with Stockfish 19 at 4,096 nodes per move. It solves 80.1% "
    "(Wilson 95% 79.3 to 80.9) of DeepMind's 10K puzzles; no exact position (or its colour mirror) from any "
    "puzzle line, or from its source game after ply 16, was in training. As a Lichess BOT it is rated 1950 "
    "in blitz (RD 62, 231 rated games, 12% against humans, snapshot 2026-10-11). DeepMind's 9M model, "
    "re-measured in the same harness, rates 1790 +/- 30."
)


def test_the_hooks_are_exactly_the_plans_words():
    assert claims.hook("en", "policy") == "Blink: a chess AI that never searches."
    assert claims.hook("en", "value") == "Blink: a chess AI that looks exactly one move ahead, never two."
    assert claims.hook("he", "policy") == POLICY_HE
    assert claims.hook("he", "value") == VALUE_HE


def test_an_unknown_mode_or_language_has_no_hook():
    with pytest.raises(ValueError, match="mode"):
        claims.hook("en", "both")
    with pytest.raises(ValueError, match="language"):
        claims.hook("fr", "policy")


def test_hook_en_and_hook_he_follow_the_shipped_mode(tmp_path, monkeypatch):
    shipped_policy = replace(results().shipped, mode="policy")
    write_bundle(tmp_path, results_obj=results(shipped=shipped_policy))
    monkeypatch.setattr(claims, "RESULTS_DIR", tmp_path)
    assert claims.HOOK_EN == "Blink: a chess AI that never searches."
    assert claims.HOOK_HE == POLICY_HE
    write_bundle(tmp_path)
    assert claims.HOOK_EN == "Blink: a chess AI that looks exactly one move ahead, never two."
    assert claims.HOOK_HE == VALUE_HE


def test_the_hook_does_not_exist_until_a_mode_ships(tmp_path, monkeypatch):
    monkeypatch.setattr(claims, "RESULTS_DIR", tmp_path)
    with pytest.raises(AttributeError, match="shipped mode"):
        _ = claims.HOOK_EN
    write_bundle(tmp_path, results_obj=results(shipped=None))
    with pytest.raises(AttributeError, match="shipped mode"):
        _ = claims.HOOK_HE
    with pytest.raises(AttributeError):
        _ = claims.NOT_A_NAME


def test_the_claim_is_filled_only_from_results_files(tmp_path):
    write_bundle(tmp_path)
    assert claims.fill_claim(tmp_path) == VALUE_CLAIM


def test_the_policy_claim_changes_only_the_mode_clause_and_the_numbers_it_reads(tmp_path):
    shipped = replace(results().shipped, mode="policy")
    write_bundle(tmp_path, results_obj=results(shipped=shipped))
    text = claims.fill_claim(tmp_path)
    assert text.startswith(
        "Blink is a 22.6M-parameter transformer that never searches: each move is at most one forward pass. "
    )
    assert "rates 1702 +/- 38 (95% CI, 4,100 games)" in text and "at 1,024 nodes per move" in text


def test_the_claim_refuses_when_any_value_is_missing_and_names_each(tmp_path):
    rows = tuple(
        replace(r, sf_nodes_equiv=None) if r.agent == "Blink-M (value)" else r for r in strength_rows()
    )
    write_bundle(
        tmp_path, results_obj=results(strength=rows), compute_obj={**compute(), "gpu_board_kwh": None}
    )
    with pytest.raises(claims.ClaimRefused) as refused:
        claims.fill_claim(tmp_path)
    message = str(refused.value)
    assert "sf_nodes_equiv" in message and "gpu_board_kwh" in message


def test_the_claim_refuses_a_lichess_rating_that_is_not_publishable(tmp_path):
    write_bundle(tmp_path, lichess_obj=lichess(n=150, rd=90))
    with pytest.raises(claims.ClaimRefused, match="publishable"):
        claims.fill_claim(tmp_path)


def test_the_claim_refuses_missing_results_files(tmp_path):
    write_bundle(tmp_path, skip=("lichess.json",))
    with pytest.raises(claims.ClaimRefused, match="lichess.json"):
        claims.fill_claim(tmp_path)


def test_the_claim_refuses_a_kwh_that_covers_too_few_gpu_hours(tmp_path):
    write_bundle(tmp_path, compute_obj={**compute(), "kwh_coverage": 0.9})
    with pytest.raises(claims.ClaimRefused, match="kWh"):
        claims.fill_claim(tmp_path)


def test_the_claim_refuses_without_a_shipped_model_or_a_single_measured_dm_row(tmp_path):
    write_bundle(tmp_path, results_obj=results(shipped=None))
    with pytest.raises(claims.ClaimRefused, match="shipped"):
        claims.fill_claim(tmp_path)
    no_dm = tuple(r for r in strength_rows() if r.agent != "DM-9M")
    write_bundle(tmp_path, results_obj=results(strength=no_dm))
    with pytest.raises(claims.ClaimRefused, match="DeepMind"):
        claims.fill_claim(tmp_path)


def test_an_elo_below_the_lowest_anchor_is_labelled_extrapolated_in_the_claim(tmp_path):
    rows = tuple(replace(r, elo=1_210.0) if r.agent == "Blink-M (value)" else r for r in strength_rows())
    write_bundle(tmp_path, results_obj=results(strength=rows))
    assert "rates 1210 +/- 35 (95% CI, 4,100 games; extrapolated below the 1320 anchor)" in claims.fill_claim(
        tmp_path
    )


def test_blink_report_claims_prints_the_claim_or_exits_1_naming_what_is_missing(tmp_path, capsys):
    from blink import cli

    write_bundle(tmp_path)
    assert cli.main(["report", "claims", "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert VALUE_CLAIM in out and claims.hook("en", "value") in out and VALUE_HE in out
    (tmp_path / "compute.json").unlink()
    assert cli.main(["report", "claims", "--results", str(tmp_path)]) == 1
    assert "compute.json" in capsys.readouterr().err


def test_a_shipped_mode_outside_policy_and_value_is_refused(tmp_path):
    rows = tuple(
        replace(r, agent="Blink-M (both)") if r.agent == "Blink-M (value)" else r for r in strength_rows()
    )
    write_bundle(
        tmp_path, results_obj=results(strength=rows, shipped=replace(results().shipped, mode="both"))
    )
    with pytest.raises(claims.ClaimRefused, match="both"):
        claims.fill_claim(tmp_path)
