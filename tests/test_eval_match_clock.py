"""In-process games keep fastchess's clocks (EVAL.md section 3): Blink and DM-9M st=1 timemargin=500,
Stockfish st=0.1 timemargin=100 (or a tc), node-limited Stockfish and the baselines unclocked."""

import chess
import chess.engine
import pytest

from blink.eval import books, match, orchestrate
from blink.play import agents

START = chess.STARTING_FEN
OPEN_E4 = books.Opening(1, START, ("e2e4", "e7e5"))


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class Timed:
    """A player whose moves take `seconds` on the fake clock (the first one `first` when given)."""

    def __init__(self, name: str, time: FakeTime, seconds: float, first: float | None = None) -> None:
        self.name, self.time, self.seconds, self.first = name, time, seconds, first
        self.games: list[str] = []

    def choose(self, board, remaining_s=None, game=""):
        cold = self.first is not None and not self.games
        self.games.append(game)
        self.time.now += self.first if cold else self.seconds
        return agents.Decision(sorted(board.legal_moves, key=chess.Move.uci)[0], n_rows=1)


@pytest.fixture
def fake_time(monkeypatch):
    time = FakeTime()
    monkeypatch.setattr(match, "move_timer", time)
    return time


def test_a_blink_move_over_1_5_seconds_loses_on_time(fake_time):
    slow = Timed("Blink-value-ship", fake_time, 1.6)
    game, record = match.play_game(slow, agents.RandomAgent(), OPEN_E4, "g1")
    assert (record.result, record.termination, record.time_forfeit_by) == ("0-1", "time forfeit", slow.name)
    assert "st=1 timemargin=500" in record.reason and game.headers["Termination"] == "time forfeit"
    assert record.engine_plies == 0


def test_a_blink_or_deepmind_move_inside_the_margin_plays_on(fake_time):
    blink = Timed("Blink-value-ship", fake_time, 1.4)
    deepmind = Timed("DM-9M", fake_time, 1.45)
    _, record = match.play_game(blink, deepmind, OPEN_E4, "g1", max_plies=6)
    assert (record.termination, record.engine_plies) == ("adjudication", 6)


def test_baselines_have_no_clock(fake_time):
    _, record = match.play_game(
        Timed("Material", fake_time, 30.0), Timed("Random", fake_time, 30.0), OPEN_E4, "g1", max_plies=4
    )
    assert record.termination == "adjudication"


def test_stockfish_agents_carry_fastchess_clocks():
    exe = "sf.exe"
    assert match.stockfish_agent(exe, elo=1500).clock == match.Clock(move_s=0.1, margin_s=0.1)
    assert match.stockfish_agent(exe, nodes=256).clock is None
    anchor = match.stockfish_agent(exe, elo=1320, tc="60+0.6")
    assert anchor.name == "SF1320" and anchor.limit is None
    assert anchor.clock == match.Clock(base_s=60.0, inc_s=0.6, margin_s=0.1)
    with pytest.raises(ValueError, match="base"):
        match.Clock.from_tc("40/60", 0.1)


class FakeEngine:
    def __init__(self) -> None:
        self.limits: list[chess.engine.Limit] = []
        self.pings = 0

    def play(self, board, limit, game=None):
        self.limits.append(limit)
        return chess.engine.PlayResult(sorted(board.legal_moves, key=chess.Move.uci)[0], None)

    def ping(self) -> None:
        self.pings += 1


def test_an_engine_on_a_clock_is_told_its_time_left_and_loses_when_it_runs_out(fake_time):
    engine = match.EngineAgent("SF1320", "sf.exe", tc="1+0.5")
    fake = FakeEngine()
    engine._engine = fake

    def play(board, limit, game=None):
        fake_time.now += 0.9
        return FakeEngine.play(fake, board, limit, game)

    fake.play = play
    _, record = match.play_game(engine, agents.RandomAgent(), OPEN_E4, "g1")
    assert [limit.white_clock for limit in fake.limits] == pytest.approx([1.0, 0.6])
    assert all(limit.white_inc == 0.5 and limit.time is None for limit in fake.limits)
    assert (record.termination, record.time_forfeit_by, record.result) == ("time forfeit", "SF1320", "0-1")


def test_a_stockfish_move_over_st_plus_margin_loses_on_time(fake_time):
    engine = match.EngineAgent("SF1800", "sf.exe", movetime=0.1)
    fake = FakeEngine()

    def play(board, limit, game=None):
        fake_time.now += 0.25
        return FakeEngine.play(fake, board, limit, game)

    fake.play = play
    engine._engine = fake
    _, record = match.play_game(agents.RandomAgent(), engine, OPEN_E4, "g1")
    assert (record.result, record.time_forfeit_by) == ("1-0", "SF1800")
    assert fake.limits[0].time == 0.1


def test_clocked_players_warm_up_before_their_first_timed_move(tmp_path, fake_time):
    cold = Timed("Blink-value-ship", fake_time, 0.01, first=5.0)
    summary = match.run_match(cold, agents.RandomAgent(), [OPEN_E4], 2, tmp_path / "m.pgn", max_plies=4)
    assert cold.games[0] == "warm-up" and summary["time_forfeits"] == 0
    engine = match.EngineAgent("SF1800", "sf.exe", movetime=0.1)
    engine._engine = FakeEngine()
    match.warm_up(engine, agents.RandomAgent())
    assert engine._engine.pings == 1


def test_in_process_time_forfeits_reach_the_summary_and_the_forfeit_table(tmp_path, fake_time):
    slow = Timed("Blink-value-ship", fake_time, 2.0)
    pgn = tmp_path / "m.pgn"
    summary = match.run_match(slow, agents.RandomAgent(), [OPEN_E4], 2, pgn)
    assert summary["time_forfeits"] == 2 and summary["a_losses"] == 2
    assert match.match_report(summary, pgn)["time_forfeits"] == 2
    table = orchestrate.forfeit_table([pgn])
    assert table["Blink-value-ship"]["time_forfeits"] == 2
    state = {"E6": {"forfeits": table, "nosearch": {}}}
    assert any("Blink-value-ship" in line and "time" in line for line in orchestrate.gate_failures(state))
