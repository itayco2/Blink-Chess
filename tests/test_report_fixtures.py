"""One complete, clearly invented results/ folder for the report tests, and the tracked copy of it.

The numbers are made up (a fixture, not a measurement). tests/fixtures/report holds the same files so
`blink report scoreboard --results tests/fixtures/report --readme <file> --check` can be run by hand;
the test below keeps that copy byte-equal to these builders.
"""

import json
from dataclasses import replace
from pathlib import Path

from blink.report import results_schema as rs

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "report"
GENERATED_AT = "2026-10-07T12:00:00+03:00"


def strength_rows() -> tuple[rs.StrengthRow, ...]:
    blink = dict(kind="blink", params_total=22_550_000, params_non_gab=21_878_256, positions_seen=573_400_000)
    return (
        rs.StrengthRow(
            "random",
            "ladder",
            "uv run blink match --round-robin",
            elo=512.0,
            elo_ci95=61.0,
            elo_games=1_000,
            evals_per_move_median=0,
            evals_per_move_max=0,
        ),
        rs.StrengthRow(
            "MLP",
            "ladder",
            "uv run blink match --round-robin",
            params_total=525_313,
            elo=1_104.0,
            elo_ci95=44.0,
            elo_games=1_000,
            evals_per_move_median=35,
            evals_per_move_max=219,
        ),
        rs.StrengthRow(
            "Blink-M (value)",
            reproduce="uv run blink rate --model ship",
            gpu_hours=120.3,
            evals_per_move_median=35,
            evals_per_move_max=219,
            ms_per_move_p50=38.2,
            elo=1_850.0,
            elo_ci95=35.0,
            elo_games=4_100,
            sf_nodes_equiv=4_096,
            dm_puzzles_pct=80.1,
            dm_puzzles_ci=(79.3, 80.9),
            dm_puzzles_clean_pct=79.8,
            dm_puzzles_clean_n=9_412,
            **blink,
        ),
        rs.StrengthRow(
            "Blink-M (policy)",
            reproduce="uv run blink rate --model ship --mode policy",
            gpu_hours=120.3,
            evals_per_move_median=1,
            evals_per_move_max=1,
            ms_per_move_p50=6.1,
            elo=1_702.0,
            elo_ci95=38.0,
            elo_games=4_100,
            sf_nodes_equiv=1_024,
            dm_puzzles_pct=71.4,
            dm_puzzles_ci=(70.5, 72.3),
            **blink,
        ),
        rs.StrengthRow(
            "DM-9M",
            "reference",
            "uv run blink gauntlet --model dm:9M",
            params_total=8_954_240,
            elo=1_790.0,
            elo_ci95=30.0,
            elo_games=1_000,
            dm_puzzles_pct=86.1,
            dm_puzzles_ci=(85.4, 86.8),
        ),
        rs.StrengthRow(
            "DM-270M",
            "reference",
            "arXiv 2402.04494, Table 1",
            params_total=270_000_000,
            paper_reported="2895 Lichess blitz vs humans (paper, 2024)",
        ),
        rs.StrengthRow(
            "SF19 UCI_Elo 1800", "anchor", "configs/anchors.csv", elo=1_800.0, elo_ci95=0.0, elo_games=800
        ),
    )


def diagnostics_rows() -> tuple[rs.DiagnosticsRow, ...]:
    bands = {"<1000": 97.2, "1000-1500": 91.0, "1500-2000": 80.3, "2000-2500": 62.4, "2500+": 41.0}
    return (
        rs.DiagnosticsRow(
            "Blink-M",
            "value",
            top1=0.512,
            top3=0.781,
            top5=0.874,
            vaa=0.603,
            near_best=0.771,
            kendall_tau_b=0.412,
            brier=0.121,
            ece_before=0.031,
            ece_after=0.012,
            regret_games10k=0.043,
            grouped_gap=0.021,
            band_pct=bands,
            mate_shortest=0.62,
            mate_preserving=0.95,
            conversion_pct=88.4,
            puzzle_rating_equiv=1_905.0,
            puzzle_rating_ci=(1_880.0, 1_931.0),
        ),
        rs.DiagnosticsRow(
            "Blink-M",
            "policy",
            top1=0.512,
            top3=0.781,
            top5=0.874,
            band_pct={"<1000": 95.1, "1000-1500": 84.2, "1500-2000": 70.0, "2000-2500": 48.8, "2500+": 30.2},
        ),
    )


def results(**overrides) -> rs.Results:
    base = rs.Results(
        strength=strength_rows(),
        diagnostics=diagnostics_rows(),
        shipped=rs.Shipped(agent="Blink-M", mode="value", sha="0123abcd"),
        eval_md_sha="5f0c9e2a7b1d",
        generated_at=GENERATED_AT,
    )
    return replace(base, **overrides)


def lichess(**overrides) -> rs.LichessSnapshot:
    base = rs.LichessSnapshot(
        bot="BlinkBot", rating=1_950, rd=62, n=231, snapshot_date="2026-10-11", human_share=0.12
    )
    return replace(base, **overrides)


def nosearch() -> dict:
    return {
        "files": 41,
        "games": 2_051,
        "decisions": 123_456,
        "compliant": True,
        "violations": [],
        "missing_counts": 0,
        "histogram": {"0": 812, "1": 20_100, "38": 102_544},
        "value_mode_full_batches": 102_544,
        "max_rows": 38,
        "max_legal": 52,
    }


def compute() -> dict:
    return {
        "schema_version": 1,
        "generated_at": GENERATED_AT,
        "scope": "fixture",
        "flagship": "long",
        "flagship_gpu_hours": 120.3,
        "flagship_kwh": 26.4,
        "total_gpu_hours": 181.2,
        "gpu_board_kwh": 39.8,
        "kwh_gpu_hours": 181.2,
        "kwh_coverage": 1.0,
        "runs": [],
        "skipped": [],
    }


def _dump(data: dict) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def bundle_texts(results_obj=None, lichess_obj=None, nosearch_obj=None, compute_obj=None) -> dict[str, str]:
    snapshot = lichess() if lichess_obj is None else lichess_obj
    return {
        "results.json": rs.to_json(results() if results_obj is None else results_obj) + "\n",
        "lichess.json": rs.lichess_to_json(snapshot) + "\n",
        "nosearch.json": _dump(nosearch() if nosearch_obj is None else nosearch_obj),
        "compute.json": _dump(compute() if compute_obj is None else compute_obj),
    }


def write_bundle(folder: Path, skip: tuple[str, ...] = (), **kwargs) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in bundle_texts(**kwargs).items():
        if name not in skip:
            (folder / name).write_text(text, encoding="utf-8")
    return folder


def test_the_tracked_report_fixture_equals_the_builders():
    expected = bundle_texts()
    tracked = {p.name: p.read_text(encoding="utf-8") for p in sorted(FIXTURE_DIR.glob("*.json"))}
    assert tracked == expected
