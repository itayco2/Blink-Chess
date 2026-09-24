"""The agents: one look (policy), one look per move (value), and the two baselines (NSC-1 tests)."""

import ast
import random
from pathlib import Path

import chess
import numpy as np
import pytest

from blink.board import encode, moves
from blink.play import agents, factory, oracles, rules
from blink.play.evaluator import Evaluation
from blink.play.oracles import MaterialEvaluator, RandomLogitEvaluator, one_hot_value

REPO = Path(__file__).resolve().parent.parent
KNIGHT_SHUFFLE = ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1")
TWO_BACK_RANK_MATES = {
    "6k1/5ppp/8/8/8/8/5PPP/R3R1K1 w - - 0 1": "a1a8",
    "r3r1k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1": "a8a1",
}
BANNED_IMPORTS = (
    "chess.engine",
    "chess.polyglot",
    "chess.syzygy",
    "chess.gaviota",
    "subprocess",
    "socket",
    "urllib",
    "http",
    "requests",
)


class RecordingEvaluator:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.batches: list[np.ndarray] = []

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        self.batches.append(np.array(codes))
        return self.inner.evaluate(codes)


class TableEvaluator:
    """Win probability per position (0.5 when absent) and fixed policy logits per vocab index."""

    def __init__(self, wins: dict[bytes, float], logits: dict[int, float] | None = None) -> None:
        self.wins = wins
        self.logits = logits or {}

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        win = np.array([self.wins.get(encode.pack(row).tobytes(), 0.5) for row in codes])
        policy = np.zeros((len(codes), moves.NUM_MOVES), dtype=np.float32)
        for index, logit in self.logits.items():
            policy[:, index] = logit
        return Evaluation(policy, one_hot_value(win))


def board_after(ucis, fen: str = chess.STARTING_FEN) -> chess.Board:
    board = chess.Board(fen)
    for uci in ucis:
        board.push_uci(uci)
    return board


def after(board: chess.Board, uci: str) -> chess.Board:
    child = board.copy()
    child.push_uci(uci)
    return child


def key(board: chess.Board) -> bytes:
    return encode.position_key(board)


def index_of(board: chess.Board, uci: str) -> int:
    return moves.encode_move(board, chess.Move.from_uci(uci))


def random_positions(count: int, seed: int) -> list[chess.Board]:
    rng = random.Random(seed)
    board, out = chess.Board(), []
    while len(out) < count:
        if board.is_game_over() or board.ply() > 160:
            board = chess.Board()
        board.push(rng.choice(list(board.legal_moves)))
        if not board.is_game_over():
            out.append(board.copy())
    return out


def blink_agents(evaluator) -> tuple:
    return agents.PolicyAgent(evaluator), agents.ValueAgent(evaluator)


def test_one_look_evaluates_exactly_one_position():
    spy = RecordingEvaluator(RandomLogitEvaluator())
    board = chess.Board()
    decision = agents.PolicyAgent(spy).choose(board)
    assert len(spy.batches) == 1
    assert spy.batches[0].shape == (1, 64)
    assert np.array_equal(spy.batches[0][0], encode.encode_board(board))
    assert (decision.n_rows, decision.n_calls) == (1, 1)
    assert decision.move in board.legal_moves
    assert decision.record.mode == "policy"


def test_one_look_per_move_scores_each_child_once_plus_the_root():
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    spy = RecordingEvaluator(RandomLogitEvaluator())
    decision = agents.ValueAgent(spy).choose(board)
    assert len(spy.batches) == 1
    rows = sorted(encode.pack(row).tobytes() for row in spy.batches[0])
    expected = sorted([key(board)] + [key(after(board, move.uci())) for move in board.legal_moves])
    assert rows == expected
    assert (decision.n_rows, decision.n_calls) == (board.legal_moves.count() + 1, 1)


def test_value_mode_picks_the_child_worst_for_the_opponent():
    board = chess.Board()
    wins = {key(after(board, "e2e4")): 0.2, key(after(board, "d2d4")): 0.3, key(after(board, "g1f3")): 0.9}
    decision = agents.ValueAgent(TableEvaluator(wins)).choose(board)
    assert decision.move.uci() == "e2e4"
    assert decision.win == pytest.approx(0.8, abs=1 / 128)


def test_value_mode_wins_a_free_queen_with_an_oracle_evaluator():
    for fen, capture in (
        ("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1", "d1d5"),
        ("3qk3/8/8/8/3Q4/8/8/4K3 b - - 0 1", "d8d4"),
    ):
        decision = agents.ValueAgent(MaterialEvaluator()).choose(chess.Board(fen))
        assert decision.move.uci() == capture
        assert decision.win > 0.9


def material_rows(balances) -> np.ndarray:
    """One code row per balance: that many points as own queens, then pawns (the opponent's when negative)."""
    rows = np.full((len(balances), 64), encode.EMPTY, dtype=np.uint8)
    for row, balance in zip(rows, balances, strict=True):
        queens, pawns = divmod(abs(int(balance)), 9)
        base = encode.OWN if balance > 0 else encode.OPP
        row[:queens] = base + chess.QUEEN - 1
        row[queens : queens + pawns] = base + chess.PAWN - 1
    return rows


def test_the_ladder_material_value_ranks_every_balance_a_game_can_reach():
    """The oracle's value saturates: past +10 value's 1,000 cp clamp gives every balance the same value, so
    value mode could not tell a rook up from two queens up and let random win its pieces back (15 of 100
    dev games drawn by insufficient material). Rung 1's value keeps every balance up to nine queens apart."""
    balances = np.arange(-oracles.MAX_BALANCE, oracles.MAX_BALANCE + 1)
    rows = material_rows(balances)
    assert oracles.material_balance(rows).tolist() == balances.tolist()
    ladder = oracles.MaterialEvaluator(cp_per_point=oracles.LADDER_CP_PER_POINT, exact=True)
    win = ladder.evaluate(rows).win_probability()
    assert (np.diff(win) > 0).all()
    assert win[oracles.MAX_BALANCE] == pytest.approx(0.5, abs=1e-6)  # level material is worth a rule draw
    saturated = MaterialEvaluator().evaluate(material_rows([14, 20])).win_probability()
    assert saturated[0] == saturated[1]


def test_mate_in_one_is_taken_with_a_random_network():
    for fen, mate in TWO_BACK_RANK_MATES.items():
        for agent_cls in (agents.PolicyAgent, agents.ValueAgent):
            spy = RecordingEvaluator(RandomLogitEvaluator())
            decision = agent_cls(spy).choose(chess.Board(fen))
            assert decision.move.uci() == mate
            assert (decision.n_rows, decision.n_calls) == (0, 0)
            assert "R2" in decision.rules
            assert spy.batches == []


def test_the_same_history_gives_the_same_move_regardless_of_prior_calls():
    board = board_after(("e2e4", "c7c5", "g1f3"))
    for agent in blink_agents(RandomLogitEvaluator(seed=3)):
        first = agent.choose(board)
        for other in random_positions(5, seed=11):
            agent.choose(other)
        again = agent.choose(board)
        fresh = type(agent)(RandomLogitEvaluator(seed=3)).choose(board.copy())
        assert (again.move, again.win) == (first.move, first.win)
        assert fresh.move == first.move


def test_repetition_uses_the_history_counter_not_claim_threefold(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("python-chess repetition helpers push grandchildren")

    helpers = ("can_claim_threefold_repetition", "can_claim_draw", "is_repetition", "is_fivefold_repetition")
    for name in helpers:
        monkeypatch.setattr(chess.Board, name, forbidden)
    board = board_after(KNIGHT_SHUFFLE)
    wins = {key(after(board, "f6g8")): 0.0, key(after(board, "e7e5")): 0.4}
    decision = agents.ValueAgent(TableEvaluator(wins)).choose(board)
    assert decision.move.uci() == "e7e5"
    assert "R3" in decision.rules
    no_history = chess.Board(board.fen())
    assert agents.ValueAgent(TableEvaluator(wins)).choose(no_history).move.uci() == "f6g8"


def test_policy_mode_plays_a_rule_draw_only_when_clearly_losing():
    board = board_after(KNIGHT_SHUFFLE)
    logits = {index_of(board, "e7e5"): 5.0, index_of(board, "f6g8"): 1.0}
    losing = TableEvaluator({key(board): 0.2}, logits)
    decision = agents.PolicyAgent(losing).choose(board)
    assert (decision.move.uci(), "R3" in decision.rules) == ("f6g8", True)
    level = TableEvaluator({key(board): 0.5}, logits)
    assert agents.PolicyAgent(level).choose(board).move.uci() == "e7e5"
    prefers_draw = {index_of(board, "f6g8"): 5.0, index_of(board, "e7e5"): 1.0}
    winning = TableEvaluator({key(board): 0.8}, prefers_draw)
    assert agents.PolicyAgent(winning).choose(board).move.uci() == "e7e5"


def test_value_mode_breaks_exact_ties_by_the_root_policy_logit():
    board = chess.Board()
    evaluator = TableEvaluator({}, {index_of(board, "c2c4"): 3.0, index_of(board, "e2e4"): 2.0})
    decision = agents.ValueAgent(evaluator).choose(board)
    assert decision.move.uci() == "c2c4"
    assert "R4" in decision.rules


def lowest_index_move(board: chess.Board) -> chess.Move:
    return min(board.legal_moves, key=lambda move: moves.encode_move(board, move))


def test_without_a_tie_seed_a_flat_policy_tie_goes_to_the_lowest_vocab_index():
    board = chess.Board()  # every move keeps material level, and the material policy is flat: all 20 tie
    for game in ("g1", "g2", "g3"):
        decision = agents.ValueAgent(MaterialEvaluator()).choose(board, game=game)
        assert decision.move == lowest_index_move(board) and "R4" in decision.rules


def test_a_tie_seed_draws_a_flat_policy_tie_by_seed_game_and_position():
    board = chess.Board()
    agent = agents.ValueAgent(MaterialEvaluator(), tie_seed=3)
    picks = [agent.choose(board, game=f"g{k}") for k in range(12)]
    assert len({pick.move for pick in picks}) > 1
    assert all("R4" in pick.rules and (pick.n_calls, pick.n_rows) == (1, 21) for pick in picks)
    assert [agent.choose(board, game=f"g{k}").move for k in range(12)] == [pick.move for pick in picks]
    other_seed = agents.ValueAgent(MaterialEvaluator(), tie_seed=4)
    assert [other_seed.choose(board, game=f"g{k}").move for k in range(12)] != [pick.move for pick in picks]


def test_a_tie_seed_draws_again_when_a_position_repeats_with_new_move_counters():
    """The FEN's move counters are in the draw, so a baseline shuffling pieces does not retrace its steps
    while its opponent repeats the position (what drew material against random by threefold repetition)."""
    start, again = chess.Board(), board_after(KNIGHT_SHUFFLE[:4])
    assert rules.repetition_key(start) == rules.repetition_key(again) and start.fen() != again.fen()
    agent = agents.ValueAgent(MaterialEvaluator(), tie_seed=0)
    first = [agent.choose(start, game=f"g{k}").move for k in range(12)]
    assert [agent.choose(again, game=f"g{k}").move for k in range(12)] != first


def test_a_tie_seed_never_overrides_a_better_value_or_a_higher_root_logit():
    board = chess.Board()
    better = TableEvaluator({key(after(board, "e2e4")): 0.2})
    higher = TableEvaluator({}, {index_of(board, "d2d4"): 1.0})
    for game in ("g1", "g2", "g3", "g4"):
        assert agents.ValueAgent(better, tie_seed=5).choose(board, game=game).move.uci() == "e2e4"
        assert agents.ValueAgent(higher, tie_seed=5).choose(board, game=game).move.uci() == "d2d4"


def test_blinks_own_agents_carry_no_tie_seed():
    """N4, no randomisation: only the ladder's flat-policy baselines draw their ties."""
    assert agents.ValueAgent(RandomLogitEvaluator()).tie_seed is None
    assert factory.make_agent("value", RandomLogitEvaluator()).tie_seed is None


def test_the_clock_guard_switches_value_mode_to_one_look():
    board = chess.Board()
    agent = agents.ValueAgent(RandomLogitEvaluator(), p99_s=0.05)
    decision = agent.choose(board, remaining_s=2.0)
    assert (decision.n_rows, "R5" in decision.rules, decision.record.mode) == (1, True, "policy")
    assert agent.choose(board, remaining_s=60.0).n_rows == 21


def test_decisions_reach_the_log_sink():
    logged = []
    agent = agents.ValueAgent(RandomLogitEvaluator(), sink=logged.append)
    agent.choose(board_after(("e2e4",)), game="g7")
    assert [(r.game, r.ply, r.mode, r.n_rows) for r in logged] == [("g7", 1, "value", 21)]


def test_the_random_agent_is_legal_and_reproducible_from_its_seed():
    for board in random_positions(50, seed=5):
        first = agents.RandomAgent(seed=1).choose(board, game="g")
        assert first.move in board.legal_moves
        assert agents.RandomAgent(seed=1).choose(board, game="g").move == first.move
        assert (first.n_rows, first.n_calls, first.record) == (0, 0, None)


def test_the_material_agent_takes_the_biggest_capture_and_a_mate():
    board = chess.Board("4k3/8/8/3q4/8/1p6/8/1R1QK3 w - - 0 1")
    assert agents.MaterialAgent().choose(board).move.uci() == "d1d5"
    mates = {"a1a8", "e1e8", "a8a1", "e8e1"}
    for fen in TWO_BACK_RANK_MATES:
        assert agents.MaterialAgent().choose(chess.Board(fen)).move.uci() in mates


def test_no_agent_moves_when_the_game_is_over():
    stalemate = chess.Board("7k/8/6QK/8/8/8/8/8 b - - 0 1")
    for agent in (*blink_agents(RandomLogitEvaluator()), agents.RandomAgent(), agents.MaterialAgent()):
        with pytest.raises(ValueError, match="no legal moves"):
            agent.choose(stalemate)


class DepthSpy:
    """Patches chess.Board.push and records how many plies past the decision root any push reaches."""

    def __init__(self, monkeypatch) -> None:
        self.root_ply: int | None = None
        self.deepest = 0
        self.pushes = 0
        original = chess.Board.push
        spy = self

        def push(board, move):
            if spy.root_ply is not None:
                spy.pushes += 1
                spy.deepest = max(spy.deepest, board.ply() + 1 - spy.root_ply)
            return original(board, move)

        monkeypatch.setattr(chess.Board, "push", push)

    def decide(self, agent, board: chess.Board):
        self.root_ply = board.ply()
        try:
            return agent.choose(board)
        finally:
            self.root_ply = None


def play_spied_game(spy: DepthSpy, white, black, rng: random.Random) -> int:
    board = chess.Board()
    for _ in range(4):
        board.push(rng.choice(list(board.legal_moves)))
    while not board.is_game_over() and board.halfmove_clock < 100 and board.ply() < 600:
        if board.is_repetition(3):
            break
        board.push(spy.decide(white if board.turn else black, board).move)
    return board.ply()


def test_the_depth_spy_never_sees_a_push_deeper_than_one(monkeypatch):
    spy = DepthSpy(monkeypatch)
    policy, value = blink_agents(RandomLogitEvaluator(seed=1))
    for i, board in enumerate(random_positions(1000, seed=2026)):
        spy.decide(value if i % 2 else policy, board)
    rng = random.Random(7)
    plies = 0
    for game in range(20):
        value = agents.ValueAgent(RandomLogitEvaluator(seed=game) if game % 3 == 0 else MaterialEvaluator())
        policy = agents.PolicyAgent(RandomLogitEvaluator(seed=100 + game))
        white, black = (value, policy) if game % 2 == 0 else (policy, value)
        plies += play_spied_game(spy, white, black, rng)
    assert plies > 1000
    assert spy.pushes > 20_000
    assert spy.deepest == 1


def test_the_depth_spy_catches_a_two_ply_look(monkeypatch):
    class Peeker:
        def choose(self, board, remaining_s=None, game=""):
            probe = board.copy()
            probe.push(next(iter(probe.legal_moves)))
            probe.push(next(iter(probe.legal_moves)))
            return agents.Decision(next(iter(board.legal_moves)))

    spy = DepthSpy(monkeypatch)
    spy.decide(Peeker(), chess.Board())
    assert spy.deepest == 2


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def play_time_modules() -> list[Path]:
    files = sorted((REPO / "blink" / "play").glob("*.py")) + [REPO / "blink" / "uci.py"]
    backends = REPO / "blink" / "model"
    files += sorted(p for p in backends.glob("*.py") if p.name != "loading.py") if backends.is_dir() else []
    return files


def test_play_modules_import_no_engine_book_tablebase_or_network():
    offenders = []
    for path in play_time_modules():
        for name in imported_modules(path):
            if any(name == banned or name.startswith(banned + ".") for banned in BANNED_IMPORTS):
                offenders.append(f"{path.name}: {name}")
    assert (REPO / "blink" / "uci.py").is_file()
    assert offenders == []


def test_the_import_scan_catches_every_import_form(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("import socket\nfrom chess import engine\nimport urllib.request\n", encoding="utf-8")
    assert {"socket", "chess.engine", "urllib.request"} <= imported_modules(sample)
