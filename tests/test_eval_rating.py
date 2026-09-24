"""Ratings (plan P8, section 6): fishtest's Elo interval, Ordo with fixed anchors, anchor choice.

The reference numbers below were produced by running fishtest's own `stat_util.get_elo` (server/fishtest/
stats, commit 93fe81eb8256b870ab759ec4470f6251fe985827) with a pure-Python stand-in for its two scipy
calls (the normal quantile from statistics.NormalDist, brentq by bisection to machine precision).
"""

import math
from pathlib import Path

import pytest

from blink.eval import rating

# (counts, fishtest get_elo -> (elo, elo95, los))
FISHTEST_GET_ELO = [
    ((30, 40, 50), (58.451214271295235, 51.512405991999486, 0.9888189635733637)),
    ((3, 20, 50, 22, 5), (10.426196175570936, 29.313787131023396, 0.7577829239560048)),
    ((0, 10, 30, 12, 0), (6.6820203369767235, 30.75363268702891, 0.6653361587518766)),
    ((12, 40, 60, 30, 8), (-20.871204666028625, 27.7585063431545, 0.06938442339467632)),
]

ORDO_CSV = """"#","PLAYER","RATING","ERROR","POINTS","PLAYED","(%)"
1,"SF1500",1500.0,"-",18.50,37,50.00
2,"Blink-value",1406.1,93.3,19.50,41,47.56
3,"SF1320",1320.0,"-",22.00,42,52.38
"""
ORDO_TEXT = """
   # PLAYER         :  RATING  ERROR  POINTS  PLAYED   (%)
   1 SF1500         :  1500.0   ----    18.5      37    50

White advantage = -81.13 +/- 40.59
Draw rate (equal opponents) = 29.92 % +/- 6.92
"""


@pytest.mark.parametrize(("counts", "expected"), FISHTEST_GET_ELO)
def test_elo_ci_matches_fishtest_get_elo_on_fixed_counts(counts, expected):
    estimate = rating.elo_ci(counts)
    assert estimate.elo == pytest.approx(expected[0], abs=1e-9)
    assert estimate.ci95 == pytest.approx(expected[1], abs=1e-9)
    assert estimate.los == pytest.approx(expected[2], abs=1e-9)


def test_elo_ci_counts_games_not_pairs():
    assert rating.elo_ci((30, 40, 50)).games == 120
    assert rating.elo_ci((3, 20, 50, 22, 5)).games == 200


def test_elo_ci_refuses_counts_that_are_neither_trinomial_nor_pentanomial():
    with pytest.raises(ValueError, match="3 .* or 5"):
        rating.elo_ci((1, 2, 3, 4))


def test_a_perfect_score_is_clamped_like_fishtest_not_infinite():
    estimate = rating.elo_ci((0, 0, 20))
    assert math.isfinite(estimate.elo) and estimate.elo > 1000


def test_wdl_counts_orders_losses_draws_wins():
    assert rating.wdl_counts(wins=5, draws=3, losses=2) == (2, 3, 5)


def test_pentanomial_from_consecutive_pairs_of_scores():
    scores = [1.0, 0.0, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.5, 1.0]
    assert rating.pentanomial(scores) == (1, 0, 2, 1, 1)


def test_pentanomial_refuses_an_unfinished_pair():
    with pytest.raises(ValueError, match="pairs"):
        rating.pentanomial([1.0, 0.0, 0.5])


def test_anchors_file_holds_sf19_uci_elo_names_and_ratings(repo_root):
    anchors = rating.read_anchors(repo_root / "configs" / "anchors.csv")
    assert anchors[0] == rating.Anchor("SF1320", 1320)
    assert anchors[-1] == rating.Anchor("SF3190", 3190)
    ratings = [a.rating for a in anchors]
    assert ratings == sorted(ratings) and len(set(ratings)) == len(ratings)
    assert all(a.name == f"SF{a.rating}" for a in anchors)
    assert all(b - a <= 100 for a, b in zip(ratings, ratings[1:], strict=False))


def test_five_anchors_are_centred_on_the_estimate_at_100_point_steps():
    grid = tuple(rating.Anchor(f"SF{r}", r) for r in (1320, 1400, 1500, 1600, 1700, 1800, 1900, 2000))
    assert [a.rating for a in rating.centred_anchors(1640, grid)] == [1400, 1500, 1600, 1700, 1800]
    assert [a.rating for a in rating.centred_anchors(1100, grid)] == [1320, 1400, 1500, 1600, 1700]
    assert [a.rating for a in rating.centred_anchors(2500, grid)] == [1600, 1700, 1800, 1900, 2000]
    assert rating.nearest_anchor(1640, grid).rating == 1600


def test_the_ordo_command_uses_fixed_anchors_white_and_draw_auto_and_1000_simulations(tmp_path):
    command = rating.ordo_command(Path("ordo.exe"), tmp_path / "pgns.txt", tmp_path / "a.csv", tmp_path / "o")
    assert command[command.index("-P") + 1] == str(tmp_path / "pgns.txt")
    assert command[command.index("-m") + 1] == str(tmp_path / "a.csv")
    assert command[command.index("-s") + 1] == "1000"
    assert "-W" in command and "-D" in command
    assert "-y" not in command and "-a" not in command  # no loose anchors, no pool average


def test_ordo_csv_is_parsed_into_strength_row_numbers():
    rows = rating.parse_ordo_csv(ORDO_CSV)
    blink = next(r for r in rows if r.player == "Blink-value")
    assert (blink.rating, blink.error, blink.played, blink.points) == (1406.1, 93.3, 41, 19.5)
    assert rating.strength_fields(blink) == {"elo": 1406.1, "elo_ci95": 93.3, "elo_games": 41}
    anchor = next(r for r in rows if r.player == "SF1500")
    assert anchor.error is None and anchor.is_anchor


def test_ordo_white_advantage_and_draw_rate_are_parsed():
    assert rating.parse_ordo_text(ORDO_TEXT) == {
        "white_advantage": -81.13,
        "white_advantage_error": 40.59,
        "draw_rate_pct": 29.92,
        "draw_rate_error": 6.92,
    }


def test_ratings_below_1320_are_labelled_extrapolated():
    assert rating.parse_ordo_csv(ORDO_CSV)[1].extrapolated is False
    low = rating.OrdoRow("x", 1200.0, 50.0, 1.0, 10, 10.0)
    assert low.extrapolated


def write_games(path: Path, results: list[tuple[str, str, str]]) -> None:
    games = [f'[White "{w}"]\n[Black "{b}"]\n[Result "{r}"]\n\n1. e4 e5 {r}\n' for w, b, r in results]
    path.write_text("\n".join(games), encoding="utf-8")


def test_all_win_and_all_loss_players_are_reported_not_fitted(tmp_path):
    pgn = tmp_path / "g.pgn"
    write_games(pgn, [("Blink-policy", "SF1320", "0-1"), ("SF1320", "Blink-policy", "1-0")] * 3)
    tally = rating.tally_players([pgn])
    assert tally["Blink-policy"] == {"games": 6, "points": 0.0}
    assert rating.unfittable(tally, (rating.Anchor("SF1320", 1320),)) == {"Blink-policy": "all losses"}
    assert rating.unfittable(tally) == {"Blink-policy": "all losses", "SF1320": "all wins"}


def test_a_player_left_with_only_losses_after_an_exclusion_is_excluded_too(tmp_path):
    pgn = tmp_path / "g.pgn"
    games = [("Blink-policy", "Random", "1-0")] * 4 + [("Blink-policy", "SF1320", "0-1")] * 4
    write_games(pgn, games)
    anchors = (rating.Anchor("SF1320", 1320),)
    assert rating.unfittable(rating.tally_players([pgn]), anchors) == {"Random": "all losses"}
    assert rating.exclusions([pgn], anchors) == {"Random": "all losses", "Blink-policy": "all losses"}


def test_only_anchors_that_played_are_passed_to_ordo(tmp_path):
    anchors = (rating.Anchor("SF1320", 1320), rating.Anchor("SF1400", 1400))
    kept = rating.anchors_present(anchors, {"SF1320": {"games": 2, "points": 1.0}})
    assert kept == (rating.Anchor("SF1320", 1320),)


ORDO = rating.ordo_exe()


@pytest.mark.local
@pytest.mark.skipif(not ORDO.is_file(), reason="Ordo is not installed here")
def test_real_ordo_fits_a_small_connected_pool(tmp_path):
    results = []
    for i in range(12):
        results.append(("Blink-value", "SF1320", ("1-0", "1/2-1/2", "0-1")[i % 3]))
        results.append(("SF1400", "Blink-value", ("1-0", "0-1", "1/2-1/2", "1-0")[i % 4]))
        results.append(("SF1320", "SF1400", ("0-1", "1/2-1/2")[i % 2]))
    write_games(tmp_path / "g.pgn", results)
    anchors = (rating.Anchor("SF1320", 1320), rating.Anchor("SF1400", 1400), rating.Anchor("SF1500", 1500))
    fit = rating.run_ordo([tmp_path / "g.pgn"], anchors, tmp_path / "ordo", simulations=50)
    blink = fit.row("Blink-value")
    assert 1200 < blink.rating < 1600 and blink.error > 0 and blink.played == 24
    assert [a.name for a in fit.anchors] == ["SF1320", "SF1400"]
