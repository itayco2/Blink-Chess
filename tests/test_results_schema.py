"""The results/*.json contract: evaluation writes it, the README scoreboard and the claims read it."""

import dataclasses
import json

import pytest

from blink.report import results_schema as rs


def a_strength_row(**overrides) -> rs.StrengthRow:
    base = dict(
        agent="Blink-M (value)",
        kind="blink",
        reproduce="uv run blink rate --model ship",
        params_total=22_550_000,
        elo=1850.0,
        elo_ci95=35.0,
        elo_games=2_000,
        dm_puzzles_pct=80.1,
        dm_puzzles_ci=(79.3, 80.9),
    )
    base.update(overrides)
    return rs.StrengthRow(**base)


def test_results_round_trip_through_json_unchanged():
    results = rs.Results(
        strength=(a_strength_row(),),
        diagnostics=(rs.DiagnosticsRow(agent="Blink-M", mode="value", top1=0.51, band_pct={"<1000": 97.0}),),
        shipped=rs.Shipped(agent="Blink-M", mode="value", sha="abc123"),
        eval_md_sha="def456",
        generated_at="2026-10-07T12:00:00+03:00",
    )
    text = rs.to_json(results)
    assert rs.from_json(text) == results
    assert json.loads(text)["schema_version"] == rs.SCHEMA_VERSION


def test_an_unknown_row_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        a_strength_row(kind="guess")


def test_paper_numbers_never_sit_in_the_measured_elo_column():
    """A reference row copied from a paper may fill only paper_reported, never the measured Elo cells."""
    with pytest.raises(ValueError, match="paper"):
        a_strength_row(kind="reference", paper_reported="2895 Lichess vs humans", elo=2895.0)
    row = a_strength_row(
        kind="reference", paper_reported="2895 Lichess vs humans", elo=None, elo_ci95=None, elo_games=None
    )
    assert row.elo is None and row.paper_reported


def test_an_elo_needs_its_interval_and_its_game_count():
    with pytest.raises(ValueError, match="interval"):
        a_strength_row(elo_ci95=None)
    with pytest.raises(ValueError, match="games"):
        a_strength_row(elo_games=None)


def test_a_lichess_snapshot_is_publishable_only_at_200_games_and_rd_below_75():
    early = rs.LichessSnapshot(bot="BlinkBot", rating=2400, rd=120, n=40, snapshot_date="2026-10-05")
    ready = rs.LichessSnapshot(bot="BlinkBot", rating=1950, rd=62, n=231, snapshot_date="2026-10-11")
    assert not early.publishable and ready.publishable
    assert rs.lichess_from_json(rs.lichess_to_json(ready)) == ready


def test_rows_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        a_strength_row().elo = 1.0  # type: ignore[misc]
