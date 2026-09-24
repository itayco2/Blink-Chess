"""`blink lichess snapshot`: the bot's public numbers, read from the public Lichess API with no token."""

import builtins
import io
import json
import math
import os
import urllib.error
import urllib.parse
from collections.abc import Mapping
from pathlib import Path

import pytest

from blink import cli
from blink.lichess import snapshot
from blink.report import results_schema

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "lichess"
SNAPSHOT_SOURCE = Path(snapshot.__file__)


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class FakeOpener:
    """Serves the recorded fixtures by URL path and remembers every request it was handed."""

    def __init__(self, routes: Mapping[str, list[bytes | int]]) -> None:
        self.routes = {path: list(replies) for path, replies in routes.items()}
        self.requests: list = []

    def __call__(self, request, timeout: float):
        self.requests.append(request)
        path = urllib.parse.urlsplit(request.full_url).path
        reply = self.routes[path].pop(0) if len(self.routes[path]) > 1 else self.routes[path][0]
        if isinstance(reply, int):
            raise urllib.error.HTTPError(request.full_url, reply, "error", {}, None)
        return io.BytesIO(reply)


def fixture_routes(bot: str = "BlinkBot") -> dict[str, list[bytes | int]]:
    return {
        f"/api/user/{bot}": [fixture_bytes("user.json")],
        f"/api/user/{bot}/rating-history": [fixture_bytes("rating_history.json")],
        f"/api/games/user/{bot}": [fixture_bytes("games.ndjson")],
    }


def fixture_games() -> tuple[snapshot.BotGame, ...]:
    return snapshot.parse_ndjson(fixture_bytes("games.ndjson").splitlines(), "BlinkBot")


def an_api(opener: FakeOpener, sleeps: list[float] | None = None) -> snapshot.PublicApi:
    record = sleeps if sleeps is not None else []
    return snapshot.PublicApi(opener=opener, sleep=record.append, clock=lambda: 0.0, min_gap_s=0.0)


def test_games_parse_from_the_bots_side_in_time_order():
    games = fixture_games()
    assert [g.id for g in games] == [f"g{i:02d}" for i in range(1, 13)]
    first, draw, flagged = games[0], games[3], games[4]
    assert (first.bot_color, first.result, first.opponent, first.opponent_is_bot) == (
        "white",
        "loss",
        "stockbot3000",
        True,
    )
    assert first.opponent_rating == 2950 and first.plies == 24 and len(first.opening) == 20
    assert (draw.result, draw.opponent_is_bot) == ("draw", False)
    assert flagged.time_loss and not games[5].time_loss  # g06: the opponent ran out of time


def test_a_game_the_bot_did_not_play_is_refused():
    record = json.loads(fixture_bytes("games.ndjson").splitlines()[0])
    with pytest.raises(ValueError, match="not a player"):
        snapshot.parse_game(record, "SomeoneElse")


def test_abort_time_loss_human_share_and_duplicate_rates_on_the_fixture():
    games = fixture_games()
    assert snapshot.abort_rate(games) == pytest.approx(2 / 12)  # aborted and noStart
    assert snapshot.time_loss_rate(games) == pytest.approx(2 / 12)  # outoftime and timeout, both lost
    assert snapshot.human_share(games) == pytest.approx(4 / 10)  # of the 10 games that were played
    # g03 repeats g01's first 20 plies and g11 repeats g02's, each against StockBot3000; the same
    # openings against other opponents, and the 4-ply and aborted games, are not duplicates.
    assert snapshot.duplicate_rate(games) == pytest.approx(2 / 9)


def test_performance_is_the_rating_whose_expected_score_equals_the_real_score():
    assert snapshot.performance([]) is None
    assert snapshot.performance([(2000, 1.0), (2000, 0.0)]) == pytest.approx(2000, abs=0.01)
    three_of_four = [(2000, 1.0), (2000, 1.0), (2000, 1.0), (2000, 0.0)]
    assert snapshot.performance(three_of_four) == pytest.approx(2000 + 400 * math.log10(3), abs=0.01)
    mixed = [(2950, 0.0), (2900, 1.0), (2880, 0.0), (2500, 1.0), (2860, 0.0), (2850, 1.0)]
    rating = snapshot.performance(mixed)
    expected = sum(1 / (1 + 10 ** ((r - rating) / 400)) for r, _ in mixed)
    assert expected == pytest.approx(3.0, abs=1e-6)


def test_a_perfect_score_is_capped_800_above_the_strongest_opponent():
    assert snapshot.performance([(1500, 1.0), (1700, 1.0)]) == pytest.approx(2500)
    assert snapshot.performance([(1500, 0.0), (1700, 0.0)]) == pytest.approx(700)


def test_performance_is_split_between_humans_and_bots():
    humans, bots = snapshot.performance_split(fixture_games())
    assert humans == pytest.approx(snapshot.performance([(1800, 0.5), (1810, 0.0), (1620, 1.0), (1805, 1.0)]))
    assert bots == pytest.approx(
        snapshot.performance([(2950, 0.0), (2900, 1.0), (2880, 0.0), (2500, 1.0), (2860, 0.0), (2850, 1.0)])
    )


def test_the_snapshot_reads_rating_rd_and_n_from_the_public_profile():
    report = snapshot.take_snapshot(an_api(FakeOpener(fixture_routes())), "BlinkBot", 500, "2026-10-08")
    shot = report.snapshot
    assert (shot.bot, shot.rating, shot.rd, shot.n, shot.snapshot_date) == (
        "BlinkBot",
        2012,
        62,
        214,
        "2026-10-08",
    )
    assert shot.publishable
    assert shot.human_share == pytest.approx(0.4)
    assert shot.abort_rate == pytest.approx(2 / 12, abs=1e-4)
    assert shot.duplicate_rate == pytest.approx(2 / 9, abs=1e-4)
    assert report.exported == 12 and report.title == "BOT"
    assert (
        report.history["points"] == 4 and report.history["first"] == 2890 and report.history["last"] == 2012
    )


def test_the_snapshot_makes_three_public_get_requests_in_order():
    opener = FakeOpener(fixture_routes())
    snapshot.take_snapshot(an_api(opener), "BlinkBot", 500, "2026-10-08")
    parts = [urllib.parse.urlsplit(r.full_url) for r in opener.requests]
    assert [p.path for p in parts] == [
        "/api/user/BlinkBot",
        "/api/user/BlinkBot/rating-history",
        "/api/games/user/BlinkBot",
    ]
    assert all(
        r.get_method() == "GET" and r.full_url.startswith("https://lichess.org/api/") for r in opener.requests
    )
    query = urllib.parse.parse_qs(parts[2].query)
    assert (query["perfType"], query["rated"], query["max"]) == (["blitz"], ["true"], ["500"])
    assert opener.requests[2].get_header("Accept") == "application/x-ndjson"


def test_a_429_waits_a_full_minute_and_retries(monkeypatch):
    routes = fixture_routes()
    routes["/api/user/BlinkBot"] = [429, 429, fixture_bytes("user.json")]
    sleeps: list[float] = []
    api = an_api(FakeOpener(routes), sleeps)
    assert api.user("BlinkBot")["username"] == "BlinkBot"
    assert sleeps == [snapshot.RETRY_AFTER_429_S, snapshot.RETRY_AFTER_429_S]


def test_repeated_429s_give_up_with_a_clear_error():
    routes = fixture_routes()
    routes["/api/user/BlinkBot"] = [429] * 10
    with pytest.raises(snapshot.ApiError, match="429"):
        an_api(FakeOpener(routes)).user("BlinkBot")


def test_requests_go_one_at_a_time_with_a_gap_between_them():
    sleeps: list[float] = []
    now = iter([100.0, 100.2, 100.2, 100.9])
    api = snapshot.PublicApi(
        opener=FakeOpener(fixture_routes()), sleep=sleeps.append, clock=lambda: next(now), min_gap_s=1.0
    )
    api.user("BlinkBot")
    api.rating_history("BlinkBot")
    assert sleeps == [pytest.approx(0.8)]


def test_a_bad_bot_name_never_reaches_the_network():
    opener = FakeOpener(fixture_routes())
    with pytest.raises(ValueError, match="name"):
        an_api(opener).user("../account")
    assert opener.requests == []


class TokenGuard(dict):
    """An environment that fails the test the moment anything asks for the bot token."""

    def __getitem__(self, key):
        if key == "LICHESS_BOT_TOKEN":
            raise AssertionError("the snapshot asked the environment for the bot token")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "LICHESS_BOT_TOKEN":
            raise AssertionError("the snapshot asked the environment for the bot token")
        return super().get(key, default)


def test_snapshot_never_reads_the_token_value(monkeypatch, tmp_path):
    sentinel = "lip_" + "Q" * 20
    monkeypatch.setattr(os, "environ", TokenGuard({**os.environ, "LICHESS_BOT_TOKEN": sentinel}))
    real_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        assert "token" not in str(file).lower(), f"the snapshot opened {file}"
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    opener = FakeOpener(fixture_routes())
    report = snapshot.take_snapshot(an_api(opener), "BlinkBot", 500, "2026-10-08")
    snapshot.write_snapshot(report.snapshot, tmp_path / "lichess.json")
    for request in opener.requests:
        headers = {k.lower(): v for k, v in request.header_items()}
        assert "authorization" not in headers and "cookie" not in headers
        assert sentinel not in request.full_url and sentinel not in json.dumps(headers)
    assert sentinel not in (tmp_path / "lichess.json").read_text(encoding="utf-8")
    source = SNAPSHOT_SOURCE.read_text(encoding="utf-8")
    for banned in ("LICHESS_BOT_TOKEN", "token.dpapi", "Authorization", "environ", "getenv"):
        assert banned not in source


def test_the_written_snapshot_round_trips_through_the_results_schema(tmp_path):
    report = snapshot.take_snapshot(an_api(FakeOpener(fixture_routes())), "BlinkBot", 500, "2026-10-08")
    out = tmp_path / "results" / "lichess.json"
    snapshot.write_snapshot(report.snapshot, out)
    text = out.read_text(encoding="utf-8")
    assert results_schema.lichess_from_json(text) == report.snapshot
    assert json.loads(text)["publishable"] is True


def test_the_cli_snapshot_with_no_write_prints_and_writes_nothing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(snapshot, "default_api", lambda: an_api(FakeOpener(fixture_routes())))
    out = tmp_path / "lichess.json"
    assert cli.main(["lichess", "snapshot", "--bot", "BlinkBot", "--no-write", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "blitz 2012" in printed and "RD 62" in printed and "publishable" in printed
    assert not out.exists()


def test_the_cli_snapshot_writes_results_lichess_json(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(snapshot, "default_api", lambda: an_api(FakeOpener(fixture_routes())))
    out = tmp_path / "lichess.json"
    assert cli.main(["lichess", "snapshot", "--bot", "BlinkBot", "--out", str(out)]) == 0
    assert results_schema.lichess_from_json(out.read_text(encoding="utf-8")).n == 214
