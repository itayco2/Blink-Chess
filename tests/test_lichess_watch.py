"""`blink lichess watch`: the bot's stop rule enforced with no agent session alive (plan P9, section 4)."""

import dataclasses
import json

import pytest

from blink import cli
from blink.lichess import pause, snapshot, watch

NOW = "2026-10-08T10:00:00+00:00"


@pytest.fixture(autouse=True)
def _no_real_home_api_or_processes(monkeypatch, tmp_path):
    """No test here may touch the real BLINK_HOME, the live Lichess API or the machine's processes."""
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "guarded-blink-home"))
    monkeypatch.setattr(snapshot, "default_api", lambda: pytest.fail("a test reached the live Lichess API"))
    monkeypatch.setattr(
        pause, "default_deps", lambda *args: pytest.fail("a test built the real pause dependencies")
    )


def a_game(index: int, **fields) -> snapshot.BotGame:
    game = snapshot.BotGame(
        id=f"g{index:03d}",
        created_at=index,
        status="mate",
        rated=True,
        bot_color="white",
        result="win",
        opponent="stockbot3000",
        opponent_is_bot=True,
        opponent_rating=2900,
        opening=(),
        plies=40,
    )
    return dataclasses.replace(game, **fields)


def clean(count: int, start: int = 0) -> list[snapshot.BotGame]:
    return [a_game(start + i) for i in range(count)]


BOT_ABORT = {"status": "aborted", "result": "draw", "plies": 0}  # the bot is white and never moved
TIME_LOSS = {"status": "outoftime", "result": "loss"}


class Bot:
    """Fakes for one watcher: the games it sees, and a record of every pause, sleep and log line."""

    def __init__(self, local=(), exported=(), api_error: Exception | None = None) -> None:
        self.local = list(local)
        self.exported = list(exported)
        self.api_error = api_error
        self.paused: list[str] = []
        self.slept: list[float] = []
        self.lines: list[str] = []

    def read_export(self):
        if self.api_error is not None:
            raise self.api_error
        return tuple(self.exported)

    def deps(self) -> watch.WatchDeps:
        return watch.WatchDeps(
            exported=self.read_export,
            local=lambda: tuple(self.local),
            pause_now=self.paused.append,
            sleep=self.slept.append,
            now=lambda: NOW,
            log=self.lines.append,
        )


def heartbeat(lichess_dir) -> dict:
    return json.loads((lichess_dir / watch.HEARTBEAT_NAME).read_text(encoding="utf-8"))


def test_the_watcher_pauses_the_bot_at_once_when_the_rule_fires_on_a_new_game(tmp_path):
    bot = Bot(local=[*clean(49), a_game(49, **BOT_ABORT)])
    result = watch.watch_round("BlinkBot", bot.deps(), tmp_path)
    assert bot.paused == ["stop rule: aborts 1/50 = 2.0% > 1%"]
    assert result.action.startswith("paused the bot")
    acted = json.loads((tmp_path / watch.ACTED_NAME).read_text(encoding="utf-8"))
    assert acted["games"] == ["g049"] and acted["stopped_at"] == NOW
    beat = heartbeat(tmp_path)
    assert beat["bot"] == "BlinkBot" and beat["at"] == NOW and beat["action"] == result.action
    assert beat["verdict"].startswith("STOP BlinkBot")


def test_a_restarted_bot_is_not_stopped_again_for_games_an_earlier_stop_acted_on(tmp_path):
    bot = Bot(local=[*clean(49), a_game(49, **BOT_ABORT)])
    watch.watch_round("BlinkBot", bot.deps(), tmp_path)
    # Itay restarts the bot; the aborted game is still among the last 50
    again = watch.watch_round("BlinkBot", bot.deps(), tmp_path)
    assert len(bot.paused) == 1 and "earlier stop" in again.action
    bot.local += [a_game(50, **TIME_LOSS), a_game(51, **TIME_LOSS)]  # new evidence
    watch.watch_round("BlinkBot", bot.deps(), tmp_path)
    assert len(bot.paused) == 2 and bot.paused[1].startswith("stop rule: time losses 2/50")
    acted = json.loads((tmp_path / watch.ACTED_NAME).read_text(encoding="utf-8"))
    assert acted["games"] == ["g049", "g050", "g051"]


def test_while_the_bot_is_paused_the_watcher_only_writes_its_heartbeat(tmp_path):
    (tmp_path / pause.FLAG_NAME).write_text("paused for a GPU window", encoding="utf-8")
    deps = dataclasses.replace(
        Bot().deps(),
        exported=lambda: pytest.fail("read the export while paused"),
        local=lambda: pytest.fail("read the PGNs while paused"),
    )
    result = watch.watch_round("BlinkBot", deps, tmp_path)
    assert result.paused_flag and heartbeat(tmp_path)["paused_flag"] is True


def test_when_the_public_api_fails_the_watcher_judges_the_local_pgns_alone(tmp_path):
    bot = Bot(local=[*clean(49), a_game(49, **BOT_ABORT)], api_error=snapshot.ApiError("HTTP 503"))
    result = watch.watch_round("BlinkBot", bot.deps(), tmp_path)
    assert bot.paused and "HTTP 503" in result.api and "local PGNs alone" in result.api


def test_the_public_record_and_the_local_pgns_are_merged_by_game_id(tmp_path):
    exported = [*clean(49), a_game(49, status="noStart", result="win")]  # the opponent's no-show
    local = [a_game(49, status="noStart", result="loss")]  # a stale local copy of the same game
    bot = Bot(local=local, exported=exported)
    result = watch.watch_round("BlinkBot", bot.deps(), tmp_path)
    assert bot.paused == [] and result.verdict.startswith("ok BlinkBot")


def test_the_watcher_checks_once_per_period_until_its_rounds_are_done(tmp_path):
    bot = Bot(local=clean(50))
    assert watch.watch("BlinkBot", bot.deps(), tmp_path, every_s=120, rounds=3) == 3
    assert bot.slept == [120, 120] and bot.paused == []
    assert heartbeat(tmp_path)["round"] == 3 and heartbeat(tmp_path)["every_s"] == 120


def test_a_round_that_fails_to_pause_is_logged_and_the_next_round_tries_again(tmp_path):
    bot = Bot(local=[*clean(49), a_game(49, **BOT_ABORT)])
    calls: list[str] = []

    def failing_pause(reason: str) -> None:
        calls.append(reason)
        if len(calls) == 1:
            raise OSError("the flag folder is not writable")

    deps = dataclasses.replace(bot.deps(), pause_now=failing_pause)
    assert watch.watch("BlinkBot", deps, tmp_path, every_s=60, rounds=2) == 2
    assert len(calls) == 2 and any("not writable" in line for line in bot.lines)


# ---------------------------------------------------------------- the commands


class FakeApi:
    def __init__(self, records: list[dict], playing: bool = True) -> None:
        self.records = records
        self.playing = playing

    def games(self, name, max_games, perf_type=None, rated=None):
        return list(reversed(self.records[-max_games:]))

    def is_playing(self, name):
        return self.playing


def a_record(index: int) -> dict:
    return {
        "id": f"r{index:03d}",
        "rated": True,
        "createdAt": 1790000000000 + index,
        "status": "mate",
        "winner": "white",
        "players": {
            "white": {"user": {"name": "BlinkBot", "title": "BOT", "id": "blinkbot"}, "rating": 2000},
            "black": {"user": {"name": "OtherBot", "title": "BOT", "id": "otherbot"}, "rating": 2100},
        },
        "moves": "e4 e5",
    }


ABORTED_PGN = (
    '[Event "Rated blitz game"]\n[Site "https://lichess.org/ab0rt777"]\n[White "BlinkBot"]\n'
    '[Black "OtherBot"]\n[Result "*"]\n[UTCDate "2026.10.08"]\n[UTCTime "10:00:00"]\n'
    '[Termination "Abandoned"]\n\n*\n\n'
)


def fake_pause_deps(stopped: list[int], polls: list[bool]):
    def build(name, api, root):
        def is_playing() -> bool:
            polls.append(True)
            return api.is_playing(name)

        return pause.PauseDeps(
            is_playing=is_playing,
            find_bot=lambda: (pause.BotProcess(4242, 1, 1.0, ("python.exe", "lichess-bot.py")),),
            stop_tree=lambda proc: stopped.append(proc.pid) or (proc.pid,),
            sleep=lambda s: pytest.fail("a stop-rule pause must not wait for the live game"),
            log=lambda line: None,
        )

    return build


def a_bot_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    (tmp_path / "lichess" / "pgn").mkdir(parents=True)
    (tmp_path / "lichess" / "pgn" / "BlinkBot vs OtherBot - ab0rt777.pgn").write_text(
        ABORTED_PGN, encoding="utf-8"
    )
    monkeypatch.setattr(snapshot, "default_api", lambda: FakeApi([a_record(i) for i in range(49)]))


def test_the_watch_command_pauses_the_bot_without_waiting_for_the_live_game(monkeypatch, tmp_path, capsys):
    a_bot_home(monkeypatch, tmp_path)
    stopped: list[int] = []
    polls: list[bool] = []
    monkeypatch.setattr(pause, "default_deps", fake_pause_deps(stopped, polls))
    assert cli.main(["lichess", "watch", "--bot", "BlinkBot", "--rounds", "1"]) == 0
    record = json.loads((tmp_path / "lichess" / "pause.json").read_text(encoding="utf-8"))
    assert stopped == [4242] and len(polls) == 1
    assert record["timeout_s"] == 0 and record["live_game_at_stop"] is True
    assert record["reason"] == "stop rule: aborts 1/50 = 2.0% > 1%"
    assert (tmp_path / "lichess" / "PAUSED").exists()
    assert json.loads((tmp_path / "lichess" / watch.HEARTBEAT_NAME).read_text(encoding="utf-8"))["round"] == 1


def test_check_stop_cuts_off_the_live_game_and_never_stops_twice_for_the_same_games(
    monkeypatch, tmp_path, capsys
):
    a_bot_home(monkeypatch, tmp_path)
    stopped: list[int] = []
    monkeypatch.setattr(pause, "default_deps", fake_pause_deps(stopped, []))
    assert cli.main(["lichess", "check", "--bot", "BlinkBot", "--stop"]) == 1
    assert stopped == [4242]
    assert json.loads((tmp_path / "lichess" / "pause.json").read_text(encoding="utf-8"))["timeout_s"] == 0
    (tmp_path / "lichess" / "PAUSED").unlink()  # Itay restarts the bot
    capsys.readouterr()
    assert cli.main(["lichess", "check", "--bot", "BlinkBot", "--stop"]) == 1
    assert stopped == [4242] and "earlier stop" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["lichess", "pause", "--bot", "BlinkBot", "--timeout", "-1"],
        ["lichess", "watch", "--bot", "BlinkBot", "--every", "0"],
        ["lichess", "watch", "--bot", "BlinkBot", "--rounds", "0"],
    ],
)
def test_a_negative_timeout_or_a_zero_watch_period_is_refused_by_the_parser(argv):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
