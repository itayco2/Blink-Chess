"""The bot's stop rule: time losses > 2% or aborts > 1% over the last 50 games (plan P9 operations)."""

import json

import pytest

from blink import cli
from blink.lichess import monitor, pause, snapshot


def a_game(index: int, status: str = "mate", result: str = "win") -> snapshot.BotGame:
    return snapshot.BotGame(
        id=f"g{index:03d}",
        created_at=index,
        status=status,
        rated=True,
        bot_color="white",
        result=result,
        opponent="stockbot3000",
        opponent_is_bot=True,
        opponent_rating=2900,
        opening=("e4", "e5"),
        plies=2,
    )


def clean(count: int, start: int = 0) -> list[snapshot.BotGame]:
    return [a_game(start + i) for i in range(count)]


TIME_LOSS = {"status": "outoftime", "result": "loss"}
ABORT = {"status": "aborted", "result": "draw"}


def test_fifty_clean_games_pass_the_stop_rule():
    verdict = monitor.stop_rule(clean(50))
    assert not verdict.stop and verdict.games == 50 and verdict.reasons == ()


def test_one_time_loss_in_50_games_is_exactly_2_percent_and_does_not_stop():
    verdict = monitor.stop_rule([*clean(49), a_game(49, **TIME_LOSS)])
    assert verdict.time_losses == 1 and verdict.time_loss_rate == pytest.approx(0.02)
    assert not verdict.stop


def test_two_time_losses_in_the_last_50_games_stop_the_bot():
    verdict = monitor.stop_rule([*clean(48), a_game(48, **TIME_LOSS), a_game(49, **TIME_LOSS)])
    assert verdict.stop
    assert verdict.reasons == ("time losses 2/50 = 4.0% > 2%",)


def test_one_abort_in_50_games_is_over_1_percent_and_stops_the_bot():
    verdict = monitor.stop_rule([*clean(49), a_game(49, **ABORT)])
    assert verdict.stop and verdict.reasons == ("aborts 1/50 = 2.0% > 1%",)


def test_a_time_loss_the_opponent_suffered_is_not_counted():
    verdict = monitor.stop_rule([*clean(48), a_game(48, "outoftime", "win"), a_game(49, "outoftime", "win")])
    assert verdict.time_losses == 0 and not verdict.stop


def test_only_the_last_50_games_count():
    old_trouble = [a_game(0, **TIME_LOSS), a_game(1, **TIME_LOSS), a_game(2, **ABORT)]
    verdict = monitor.stop_rule([*old_trouble, *clean(50, start=3)])
    assert verdict.games == 50 and not verdict.stop


def test_with_fewer_than_50_games_the_rates_are_over_the_games_played():
    verdict = monitor.stop_rule([*clean(9), a_game(9, **TIME_LOSS)])
    assert verdict.games == 10 and verdict.time_loss_rate == pytest.approx(0.1) and verdict.stop


def test_no_games_yet_is_not_a_stop():
    verdict = monitor.stop_rule([])
    assert verdict.games == 0 and not verdict.stop


class FakeApi:
    def __init__(self, records: list[dict], playing: bool = False) -> None:
        self.records = records
        self.playing = playing
        self.calls: list[tuple] = []

    def games(self, name, max_games, perf_type=None, rated=None):
        self.calls.append(("games", name, max_games, perf_type, rated))
        return list(reversed(self.records[-max_games:]))  # the API streams newest first

    def is_playing(self, name):
        self.calls.append(("is_playing", name))
        return self.playing


def a_record(index: int, status: str = "mate", winner: str | None = "white") -> dict:
    record = {
        "id": f"r{index:03d}",
        "rated": True,
        "speed": "blitz",
        "createdAt": 1790000000000 + index,
        "status": status,
        "players": {
            "white": {"user": {"name": "BlinkBot", "title": "BOT", "id": "blinkbot"}, "rating": 2000},
            "black": {"user": {"name": "OtherBot", "title": "BOT", "id": "otherbot"}, "rating": 2100},
        },
        "moves": "e4 e5",
    }
    return {**record, "winner": winner} if winner else record


def test_check_reads_the_last_50_games_of_every_kind_from_the_public_api():
    api = FakeApi([a_record(i) for i in range(60)])
    verdict = monitor.check(api, "BlinkBot")
    assert api.calls == [("games", "BlinkBot", 50, None, None)]
    assert verdict.games == 50 and not verdict.stop


def test_the_check_command_exits_1_and_names_the_rule_when_it_fires(monkeypatch, capsys):
    records = [a_record(i) for i in range(48)] + [a_record(48, "outoftime", "black")] * 2
    monkeypatch.setattr(snapshot, "default_api", lambda: FakeApi(records))
    assert cli.main(["lichess", "check", "--bot", "BlinkBot"]) == 1
    printed = capsys.readouterr().out
    assert "STOP" in printed and "time losses 2/50" in printed


def test_the_check_command_passes_quietly_on_clean_games(monkeypatch, capsys):
    monkeypatch.setattr(snapshot, "default_api", lambda: FakeApi([a_record(i) for i in range(50)]))
    assert cli.main(["lichess", "check", "--bot", "BlinkBot"]) == 0
    assert "ok" in capsys.readouterr().out


def test_the_check_command_with_stop_pauses_the_bot_when_the_rule_fires(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    records = [a_record(i) for i in range(49)] + [a_record(49, "aborted", None)]
    api = FakeApi(records)
    monkeypatch.setattr(snapshot, "default_api", lambda: api)
    stopped: list[int] = []
    bot = pause.BotProcess(pid=4242, ppid=1, create_time=1.0, cmdline=("python.exe", "lichess-bot.py"))

    def fake_deps(name, api_, root):
        return pause.PauseDeps(
            is_playing=lambda: api_.is_playing(name),
            find_bot=lambda: (bot,),
            stop_tree=lambda proc: stopped.append(proc.pid) or (proc.pid,),
            sleep=lambda s: None,
            log=lambda line: None,
        )

    monkeypatch.setattr(pause, "default_deps", fake_deps)
    assert cli.main(["lichess", "check", "--bot", "BlinkBot", "--stop"]) == 1
    assert stopped == [4242]
    record = json.loads((tmp_path / "lichess" / "pause.json").read_text(encoding="utf-8"))
    assert record["reason"].startswith("stop rule: aborts 1/50")
    assert (tmp_path / "lichess" / "PAUSED").exists()
