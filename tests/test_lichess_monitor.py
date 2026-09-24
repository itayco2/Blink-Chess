"""The bot's stop rule: time losses > 2% or aborts > 1% over the last 50 games (plan P9 operations)."""

import json

import pytest

from blink import cli
from blink.lichess import monitor, pause, snapshot


@pytest.fixture(autouse=True)
def _no_real_home_api_or_processes(monkeypatch, tmp_path):
    """No test here may touch the real BLINK_HOME, the live Lichess API or the machine's processes.

    A RED-phase run once sent `pause --poll 0` through the real CLI and wrote a PAUSED flag into
    the real BLINK_HOME/lichess; tests that need these pieces install their own fakes over this guard.
    """
    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "guarded-blink-home"))
    monkeypatch.setattr(snapshot, "default_api", lambda: pytest.fail("a test reached the live Lichess API"))
    monkeypatch.setattr(
        pause, "default_deps", lambda *args: pytest.fail("a test built the real pause dependencies")
    )


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


def test_check_reads_the_last_50_games_of_every_kind_from_the_public_api(tmp_path):
    api = FakeApi([a_record(i) for i in range(60)])
    verdict = monitor.check(api, "BlinkBot", pgn_dir=tmp_path / "no-pgns-yet")
    assert api.calls == [("games", "BlinkBot", 50, None, None)]
    assert verdict.games == 50 and not verdict.stop


def a_pgn(
    game_id: str, termination: str, white: str = "BlinkBot", black: str = "OtherBot", second: int = 0
) -> str:
    return (
        f'[Event "Rated Blitz game"]\n[Site "https://lichess.org/{game_id}"]\n'
        f'[White "{white}"]\n[Black "{black}"]\n[Result "*"]\n[BlackTitle "BOT"]\n'
        f'[UTCDate "2026.09.24"]\n[UTCTime "12:00:{second:02d}"]\n[Termination "{termination}"]\n\n*\n\n'
    )


def test_an_abandoned_game_in_the_bots_own_pgns_counts_as_an_abort(tmp_path):
    """lila's game export keeps only status >= mate (Query.finished), so an aborted game (status 25)
    never reaches the public API; lichess-bot still writes its PGN, with Termination "Abandoned"."""
    (tmp_path / "BlinkBot vs OtherBot - ab0rt001.pgn").write_text(
        a_pgn("ab0rt001", "Abandoned"), encoding="utf-8"
    )
    (tmp_path / "BlinkBot vs OtherBot - norm0001.pgn").write_text(
        a_pgn("norm0001", "Normal"), encoding="utf-8"
    )
    games = monitor.aborted_from_pgns(tmp_path, "BlinkBot")
    assert [(g.id, g.status, g.bot_color, g.opponent, g.opponent_is_bot) for g in games] == [
        ("ab0rt001", "aborted", "white", "otherbot", True)
    ]
    assert games[0].created_at == 1790251200000  # 2026-09-24 12:00:00 UTC


def test_pgns_of_other_players_or_no_folder_add_nothing(tmp_path):
    (tmp_path / "x.pgn").write_text(a_pgn("other001", "Abandoned", "Alice", "Bob"), encoding="utf-8")
    assert monitor.aborted_from_pgns(tmp_path, "BlinkBot") == ()
    assert monitor.aborted_from_pgns(tmp_path / "missing", "BlinkBot") == ()


def test_check_merges_the_export_with_local_aborts_and_counts_each_game_once(tmp_path):
    records = [a_record(i) for i in range(48)] + [a_record(48, "noStart", "black")]
    (tmp_path / "a.pgn").write_text(
        a_pgn("r048", "Abandoned", second=1), encoding="utf-8"
    )  # the noStart game
    (tmp_path / "b.pgn").write_text(a_pgn("ab0rt002", "Abandoned", second=2), encoding="utf-8")
    verdict = monitor.check(FakeApi(records), "BlinkBot", pgn_dir=tmp_path)
    assert (verdict.games, verdict.aborts) == (50, 2) and verdict.stop


def test_the_check_command_exits_1_and_names_the_rule_when_it_fires(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    records = [a_record(i) for i in range(48)] + [a_record(i, "outoftime", "black") for i in (48, 49)]
    monkeypatch.setattr(snapshot, "default_api", lambda: FakeApi(records))
    assert cli.main(["lichess", "check", "--bot", "BlinkBot"]) == 1
    printed = capsys.readouterr().out
    assert "STOP" in printed and "time losses 2/50" in printed


def test_the_check_command_passes_quietly_on_clean_games(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    monkeypatch.setattr(snapshot, "default_api", lambda: FakeApi([a_record(i) for i in range(50)]))
    assert cli.main(["lichess", "check", "--bot", "BlinkBot"]) == 0
    assert "ok" in capsys.readouterr().out


def test_the_check_command_reads_aborts_from_the_pgn_folder_under_blink_home(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    (tmp_path / "lichess" / "pgn").mkdir(parents=True)
    (tmp_path / "lichess" / "pgn" / "a.pgn").write_text(a_pgn("ab0rt003", "Abandoned"), encoding="utf-8")
    monkeypatch.setattr(snapshot, "default_api", lambda: FakeApi([a_record(i) for i in range(49)]))
    assert cli.main(["lichess", "check", "--bot", "BlinkBot"]) == 1
    assert "aborts 1/50" in capsys.readouterr().out


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
