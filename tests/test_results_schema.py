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
        diagnostics=(
            rs.DiagnosticsRow(
                agent="Blink-M", mode="value", top1=0.51, band_pct={"<1000": 97.0}, band_n={"<1000": 1_203}
            ),
        ),
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
        kind="reference",
        paper_reported="2895 Lichess vs humans",
        elo=None,
        elo_ci95=None,
        elo_games=None,
        dm_puzzles_pct=None,
        dm_puzzles_ci=None,
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


MEASURED_BY_A_PAPER = dict(
    dm_puzzles_pct=95.4,
    dm_puzzles_ci=(94.9, 95.8),
    gpu_hours=12_000.0,
    sf_nodes_equiv=100_000,
    positions_seen=15_000_000_000,
    training_positions=400_000_000,
    evals_per_move_median=1,
    ms_per_move_p50=3.0,
    dm_puzzles_clean_pct=95.0,
    dm_puzzles_clean_n=9_000,
)


@pytest.mark.parametrize("field", sorted(MEASURED_BY_A_PAPER))
def test_a_paper_row_fills_no_measured_column(field):
    """A row that quotes a paper holds its numbers only in paper_reported; every measured cell stays empty."""
    extra = {field: MEASURED_BY_A_PAPER[field]}
    if field == "dm_puzzles_pct":
        extra["dm_puzzles_ci"] = MEASURED_BY_A_PAPER["dm_puzzles_ci"]
    if field == "dm_puzzles_clean_pct":
        extra["dm_puzzles_clean_n"] = MEASURED_BY_A_PAPER["dm_puzzles_clean_n"]
    with pytest.raises(ValueError, match="paper"):
        rs.StrengthRow("DM-270M (paper)", "reference", "arXiv 2402.04494", paper_reported="2895", **extra)
    assert rs.StrengthRow("DM-270M", "reference", "arXiv", params_total=270_000_000, paper_reported="2895")


def test_a_puzzle_percentage_needs_its_wilson_interval_and_a_clean_subset_its_n():
    with pytest.raises(ValueError, match="Wilson"):
        a_strength_row(dm_puzzles_ci=None)
    with pytest.raises(ValueError, match="clean"):
        a_strength_row(dm_puzzles_clean_pct=79.8)
    assert a_strength_row(dm_puzzles_clean_pct=79.8, dm_puzzles_clean_n=9_412).dm_puzzles_clean_n == 9_412


def test_diagnostics_percentages_carry_their_sample_size_or_interval():
    with pytest.raises(ValueError, match="puzzle_rating_ci"):
        rs.DiagnosticsRow("Blink-M", "value", puzzle_rating_equiv=1905.0)
    with pytest.raises(ValueError, match="conversion_n"):
        rs.DiagnosticsRow("Blink-M", "value", conversion_pct=88.4)
    with pytest.raises(ValueError, match="band_n"):
        rs.DiagnosticsRow("Blink-M", "value", band_pct={"<1000": 97.2})
    with pytest.raises(ValueError, match="band_n"):
        rs.DiagnosticsRow("Blink-M", "value", band_pct={"<1000": 97.2}, band_n={"<1000": 0})
    row = rs.DiagnosticsRow(
        "Blink-M",
        "value",
        band_pct={"<1000": 97.2},
        band_n={"<1000": 1_203},
        conversion_pct=88.4,
        conversion_n=500,
    )
    assert row.band_n == {"<1000": 1_203} and row.conversion_n == 500


def test_training_positions_can_never_exceed_the_database_or_the_samples_seen():
    """The claim's 'positions drawn from the 409,710,113-position database' counts distinct DB positions."""
    with pytest.raises(ValueError, match="409,710,113"):
        a_strength_row(training_positions=573_400_000)
    with pytest.raises(ValueError, match="positions_seen"):
        a_strength_row(training_positions=400_000_000, positions_seen=300_000_000)
    row = a_strength_row(training_positions=398_100_000, positions_seen=573_400_000)
    assert row.training_positions == 398_100_000


def test_training_positions_are_the_packs_train_roots_capped_by_the_roots_the_run_consumed():
    v1 = {"splits": {"roots": {"train": 398_100_000, "val": 1_000_000}, "children": {"train": 2_000_000_000}}}
    skeleton = {"splits": {"train": 950_000, "val": 50_000}}
    assert rs.training_positions(v1, roots_consumed=401_000_000) == 398_100_000
    assert rs.training_positions(v1, roots_consumed=12_000_000) == 12_000_000
    assert rs.training_positions(skeleton, roots_consumed=5_000_000) == 950_000
