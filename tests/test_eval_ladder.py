"""E4 node ladder, E4b film checkpoints, E6 round robin, E5 anchors and E7 DeepMind (plan P8)."""

from pathlib import Path

import pytest

from blink.eval import anchors, ladder, rating

GRID = tuple(rating.Anchor(f"SF{r}", r) for r in (1320, *range(1400, 3101, 100), 3190))


def report(a, b, games, score, pgn="g.pgn"):
    wins = round(score * games)
    return {
        "a": a,
        "b": b,
        "games": games,
        "wins": wins,
        "draws": 0,
        "losses": games - wins,
        "score": wins / games,
        "penta": None,
        "pgn": pgn,
    }


def test_the_rungs_are_2_to_the_4_through_2_to_the_16_in_steps_of_2():
    assert ladder.NODE_RUNGS == (16, 64, 256, 1024, 4096, 16384, 65536)
    assert (ladder.RUNG_GAMES, ladder.BRACKET_GAMES) == (100, 500)


def test_the_bracketing_rungs_get_500_games_and_the_crossover_is_interpolated():
    true_score = {16: 0.95, 64: 0.9, 256: 0.7, 1024: 0.6, 4096: 0.3, 16384: 0.1, 65536: 0.05}
    calls = []

    def play(nodes, games, skip):
        calls.append((nodes, games, skip))
        return report("Blink", f"SF19-n{nodes}", games, true_score[nodes])

    result = ladder.run_node_ladder(play)
    assert calls[:7] == [(n, 100, 0) for n in ladder.NODE_RUNGS]
    assert calls[7:] == [(1024, 400, 50), (4096, 400, 50)]
    games = {r["nodes"]: r["games"] for r in result["rungs"]}
    assert games[1024] == games[4096] == 500 and games[16] == 100
    cross = result["crossover"]
    assert cross["bracket"] == [1024, 4096] and 1024 < cross["nodes"] < 4096
    assert result["games"] == 5 * 100 + 2 * 500


def test_a_ladder_without_a_crossing_reports_its_bound():
    rungs = [{"nodes": n, "score": 0.9} for n in ladder.NODE_RUNGS]
    assert ladder.crossover(rungs) == {"nodes": None, "bound": "above the top rung", "bracket": None}
    rungs = [{"nodes": n, "score": 0.1} for n in ladder.NODE_RUNGS]
    assert ladder.crossover(rungs)["bound"] == "below the bottom rung"


def test_the_crossover_sits_where_the_logit_line_crosses_zero():
    rungs = [{"nodes": 1024, "score": 0.75}, {"nodes": 4096, "score": 0.25}]
    assert ladder.crossover(rungs)["nodes"] == 2048


def test_six_film_checkpoints_are_evenly_spaced_from_first_to_last(tmp_path):
    film = tmp_path / "film"
    film.mkdir()
    for step in (0, 250, 400, 700, 1200, 2000, 3500, 6000, 10000, 16000, 25000):
        (film / f"frame_{step:09d}.pt").write_bytes(b"")
    frames = ladder.film_frames(tmp_path)
    picked = ladder.pick_checkpoints(frames)
    assert len(picked) == 6 and picked[0] == frames[0] and picked[-1] == frames[-1]
    assert ladder.pick_checkpoints(frames[:3]) == frames[:3]


def test_film_checkpoints_play_at_the_crossover_node_count():
    seen = []

    def play(selector, nodes, games):
        seen.append((Path(selector).name, nodes, games))
        return report(selector, f"SF19-n{nodes}", games, 0.5)

    out = ladder.run_film_checkpoints(play, [Path("a.pt"), Path("b.pt")], nodes=2048, games=4)
    assert seen == [("a.pt", 2048, 4), ("b.pt", 2048, 4)] and out["games"] == 8


def test_the_ladder_round_robin_is_15_pairs_of_200_games():
    pairs = ladder.round_robin_pairs(ladder.LADDER_PLAYERS)
    assert len(pairs) == 15 and ("random", "SF1320") in pairs
    out = ladder.run_round_robin(lambda a, b, games: report(a, b, games, 0.5), games=2)
    assert out["games"] == 30 and len(out["pgns"]) == 15


def test_the_locator_estimate_is_the_anchor_plus_the_logistic_elo_of_the_score():
    anchor = rating.Anchor("SF1800", 1800)
    assert anchors.locator_estimate(anchor, report("B", "SF1800", 50, 0.5)) == pytest.approx(1800)
    assert anchors.locator_estimate(anchor, report("B", "SF1800", 50, 0.76)) == pytest.approx(
        1800 + rating.logistic_elo(0.76)
    )
    top = anchors.locator_estimate(anchor, report("B", "SF1800", 50, 1.0))
    assert top == pytest.approx(1800 + rating.logistic_elo(0.99))


def play_at(true_rating):
    calls = []

    def play(anchor, games, book, skip):
        calls.append((anchor.name, games, book))
        return report("Blink", anchor.name, games, rating.logistic_score(true_rating - anchor.rating))

    return play, calls


def test_e5_locates_then_plays_400_games_against_5_centred_anchors():
    play, calls = play_at(2050)
    out = anchors.run_anchor_block(play, GRID)
    assert calls[0] == ("SF1800", 50, "dev")
    assert [c[0] for c in calls[1:]] == ["SF1800", "SF1900", "SF2000", "SF2100", "SF2200"]
    assert all(c[1:] == (400, "final") for c in calls[1:])
    assert out["games"] == 50 + 5 * 400 and out["inner_in_band"]


def test_a_side_row_plays_200_games_against_the_nearest_anchor():
    play, calls = play_at(1500)
    out = anchors.run_side_row(play, GRID)
    assert calls[-1] == ("SF1500", 200, "final") and out["games"] == 250


def test_e7_runs_deepminds_gauntlet_then_blink_against_it():
    play, calls = play_at(2400)
    penta_report = {**report("Blink-value", "DM-9M", 1000, 0.55), "penta": [50, 100, 150, 120, 80]}
    out = anchors.run_dm_block(play, lambda games: penta_report, GRID)
    assert out["gauntlet"]["games"] == 50 + 5 * 200
    assert out["blink_vs_dm"]["elo"]["games"] == 1000
    assert out["games"] == 50 + 1000 + 1000


def captured_anchor_play(tmp_path, monkeypatch, epsilon=None):
    """anchors.fastchess_player with fastchess stubbed: returns (play, the engines each game got)."""
    import json
    from types import SimpleNamespace

    from blink.eval import fastchess

    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    if epsilon is not None:
        (results / "epsilon.json").write_text(json.dumps({"epsilon": epsilon}), encoding="utf-8")
    seen = []

    def prepare_pair(first, second, games, book, out_dir, concurrency, skip=0):
        seen.append({"first": first, "second": second, "book": book, "out_dir": out_dir})
        return SimpleNamespace(pgn=str(out_dir / "g.pgn"))

    monkeypatch.setattr(fastchess, "prepare_pair", prepare_pair)
    monkeypatch.setattr(fastchess, "execute", lambda gauntlet: gauntlet)
    monkeypatch.setattr(
        fastchess, "match_report", lambda r: {"games": 2, "score": 0.5, "pgn": r.pgn, "penta": None}
    )
    ctx = SimpleNamespace(device="cpu", out_dir=tmp_path / "out", concurrency=1, results_dir=results)
    return ctx, seen


def test_e5s_fastchess_blink_plays_with_the_epsilon_e2b_chose(tmp_path, monkeypatch):
    ctx, seen = captured_anchor_play(tmp_path, monkeypatch, epsilon=1 / 256)
    play = anchors.fastchess_player(ctx, "ship", "value", "E5")
    play(GRID[0], 2, "final", 0)
    assert "--epsilon=0.00390625" in seen[0]["first"].args
    assert seen[0]["first"].name == "Blink-value-ship"


def test_locator_games_go_to_their_own_folder_so_a_folder_of_final_slice_games_holds_no_dev_game(
    tmp_path, monkeypatch
):
    ctx, seen = captured_anchor_play(tmp_path, monkeypatch)
    play = anchors.fastchess_player(ctx, "ship", "value", "E5")
    play(GRID[0], 2, "dev", 0)
    play(GRID[0], 2, "final", 0)
    assert [(s["book"], s["out_dir"].name) for s in seen] == [("dev", "E5-locator"), ("final", "E5")]
