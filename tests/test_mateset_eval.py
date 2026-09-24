"""Mate keeping on the mateset (arm a08's guard): value mode's own choice, then shortest and kept mates."""

import chess
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_model_config  # noqa: E402

from blink.board import encode, moves  # noqa: E402
from blink.board.value import CP_NONE  # noqa: E402
from blink.data import mateset, valprobe  # noqa: E402
from blink.data.record import NO_MOVE, ROOT_DTYPE  # noqa: E402
from blink.model.evaluator import TorchEvaluator  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.play.agents import ValueAgent  # noqa: E402
from blink.train import mateset_eval, vaa  # noqa: E402
from blink.train.telemetry import board_from_codes  # noqa: E402

pytestmark = pytest.mark.torch
CPU = torch.device("cpu")


class CodeValueNet(torch.nn.Module):
    """A stand-in: a position's win probability for its side to move is (square 0's code) / 16."""

    def forward(self, tokens):
        w = tokens[:, 0].float() / 16
        bins = (w * 128).long().clamp(0, 127)
        value = torch.nn.functional.one_hot(bins, 128).float() * 60.0
        return torch.zeros(len(tokens), moves.NUM_MOVES), value


class HashNet(torch.nn.Module):
    """A stand-in whose outputs depend on each row alone: five coarse win levels, so children tie often
    and R4's policy tie-break decides, and policy logits that vary with the move and the position."""

    def forward(self, tokens):
        key = (tokens * torch.arange(1, 65)).sum(-1)
        value = torch.nn.functional.one_hot((key % 5) * 30, 128).float() * 60.0
        policy = (torch.arange(moves.NUM_MOVES)[None, :] * 7919 + key[:, None]) % 1000
        return policy.float(), value


def _board_with_code(code: int) -> np.ndarray:
    codes = np.zeros(64, dtype=np.uint8)
    codes[0] = code
    return encode.pack(codes)


def _mateset(roots: list[list[tuple[int, bool, int, int]]], with_child_mate_in: bool = True):
    """Each child is (square-0 code, is_best, terminal, the mover's mate-in after the move or 0)."""
    flat = [child for children in roots for child in children]
    probe = vaa.Probe(
        root_board=np.zeros((len(roots), 32), dtype=np.uint8),
        root_best=np.zeros(len(roots), dtype=np.uint16),
        child_offset=np.cumsum([0] + [len(c) for c in roots]).astype(np.int64),
        child_board=np.stack([_board_with_code(code) for code, *_ in flat]),
        child_move=np.arange(len(flat), dtype=np.uint16),
        child_is_best=np.array([best for _, best, _, _ in flat]),
        child_terminal=np.array([terminal for _, _, terminal, _ in flat], dtype=np.int8),
    )
    child_mate_in = np.array([mate for *_, mate in flat], dtype=np.int8) if with_child_mate_in else None
    return mateset_eval.Mateset(probe, np.full(len(roots), 3, dtype=np.int8), child_mate_in)


ROOTS = [
    [(2, True, 0, 3), (8, False, 0, 4), (12, False, 0, 0)],  # takes the shortest mate
    [(9, True, 0, 3), (3, False, 0, 4), (12, False, 0, 0)],  # a longer mate: kept, not shortest
    [(9, True, 0, 3), (1, False, 0, 0)],  # throws the mate away
    [(1, False, 0, 0), (15, True, 1, 1)],  # R2: the checkmate is played whatever the values say
    [(12, False, 0, 0), (14, False, 2, 0), (13, True, 0, 3)],  # R3: the rule draw (0.5) is worth most
]


def test_mate_preserving_and_shortest_mate_follow_value_mode_s_choice():
    result = mateset_eval.evaluate(CodeValueNet(), _mateset(ROOTS), CPU, chunk=2)
    assert result == {"n": 5, "shortest_mate": pytest.approx(0.4), "mate_preserving": pytest.approx(0.6)}


def test_without_child_mate_in_only_the_shortest_mate_rate_is_scored():
    result = mateset_eval.evaluate(CodeValueNet(), _mateset(ROOTS, with_child_mate_in=False), CPU)
    assert result == {"n": 5, "shortest_mate": pytest.approx(0.4)}


def test_one_root_s_choice_follows_r2_then_the_value_then_r4():
    values, logits = np.array([0.5, 0.7, 0.69]), np.array([0.0, 1.0, 5.0])
    listed = np.array([30, 10, 20], dtype=np.uint16)
    none = np.zeros(3, dtype=bool)
    assert mateset_eval.choose(values, logits, listed, none, epsilon=0.0) == 1  # the best value
    assert mateset_eval.choose(values, logits, listed, none, epsilon=0.05) == 2  # R4: the higher logit
    mates = np.array([True, False, True])  # the best value (index 10) does not mate
    assert mateset_eval.choose(values, logits, listed, mates, epsilon=0.0) == 2  # R2: lowest mating index
    tied = mateset_eval.choose(np.array([0.6, 0.6]), np.zeros(2), np.array([20, 10]), none[:2], 0.0)
    assert tied == 1  # value mode lists children in vocabulary order, and the first of a tie wins


def test_a_nan_value_never_wins_and_never_crashes_the_choice():
    """Diverged weights give NaN values: R4's tie set was empty and np.argmax raised on it."""
    probe = _mateset(ROOTS).probe
    logits = np.zeros((probe.n_roots, moves.NUM_MOVES), dtype=np.float32)
    w_child = np.full(len(probe.child_board), np.nan, dtype=np.float32)
    chosen = mateset_eval.choices(w_child, logits, probe)
    assert (chosen >= 0).all() and chosen[3] == 9  # R2 still plays the checkmate
    w_child[1] = 0.1  # one finite child: the mover's 0.9 beats every NaN
    assert mateset_eval.choices(w_child, logits, probe)[0] == 1


def _root(board: chess.Board, best: str) -> np.ndarray:
    record = np.zeros(1, dtype=ROOT_DTYPE)
    record["board"] = encode.pack(encode.encode_board(board))
    record["move"] = moves.encode_move(board, chess.Move.from_uci(best))
    record["cp"], record["alt_move"] = 0, NO_MOVE
    return record


def _positions_with_every_rule() -> np.ndarray:
    fools = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")  # R2: Qh4#
    stale = chess.Board("7k/8/6Q1/8/8/8/8/K7 w - - 0 1")  # R3: Qf7 and three king moves stalemate
    return np.concatenate([fixture_records()[:24], _root(fools, "d8h4"), _root(stale, "g6g7")])


@pytest.mark.parametrize("epsilon", [0.0, 0.3])
def test_the_choice_is_the_move_the_value_agent_plays(epsilon):
    probe = vaa.probe_from_roots(_positions_with_every_rule())
    net = HashNet()
    chosen = mateset_eval.value_mode_choices(net, probe, CPU, chunk=16, epsilon=epsilon)
    agent = ValueAgent(TorchEvaluator(net, CPU), epsilon=epsilon)
    rules_seen = set()
    for i, codes in enumerate(encode.unpack(probe.root_board)):
        board = board_from_codes(codes)
        decision = agent.choose(board)
        rules_seen.update(decision.rules)
        assert int(probe.child_move[chosen[i]]) == moves.encode_move(board, decision.move), i
    assert {"R2", "R3", "R4"} <= rules_seen  # every rule the stored positions allow was exercised


def _mate_roots() -> np.ndarray:
    roots = fixture_records()[:12].copy()
    roots["cp"], roots["mate"] = CP_NONE, np.array([2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 2, 3])
    return roots


def test_the_mateset_file_the_data_step_writes_loads_with_its_mate_in(tmp_path):
    arrays = mateset.build(_mate_roots(), n=100)
    valprobe.save_npz(tmp_path / "mateset.npz", arrays)
    loaded = mateset_eval.load(tmp_path / "mateset.npz")
    assert loaded.probe.n_roots == 10 and loaded.mate_in.tolist() == [2, 3, 4, 5, 2, 3, 4, 5, 2, 3]
    assert loaded.child_mate_in is None  # the data step does not label every legal move (yet)
    labelled = {**arrays, "child_mate_in": np.ones(len(arrays["child_move"]), dtype=np.int8)}
    valprobe.save_npz(tmp_path / "labelled.npz", labelled)
    assert mateset_eval.load(tmp_path / "labelled.npz").child_mate_in.sum() == len(arrays["child_move"])
    del arrays["mate_in"]
    valprobe.save_npz(tmp_path / "valprobe.npz", arrays)
    with pytest.raises(ValueError, match="mate_in"):
        mateset_eval.load(tmp_path / "valprobe.npz")


def test_a_child_mate_in_of_the_wrong_length_is_refused():
    good = _mateset(ROOTS)
    with pytest.raises(ValueError, match="child_mate_in"):
        mateset_eval.Mateset(good.probe, good.mate_in, good.child_mate_in[:-1])
    with pytest.raises(ValueError, match="mate_in"):
        mateset_eval.Mateset(good.probe, good.mate_in[:-1], None)


def test_a_real_model_scores_a_built_mateset_and_keeps_its_mode(tmp_path):
    valprobe.save_npz(tmp_path / "mateset.npz", mateset.build(_mate_roots(), n=100))
    model = BlinkNet(tiny_model_config()).train()
    result = mateset_eval.evaluate(model, mateset_eval.load(tmp_path / "mateset.npz"), CPU, chunk=64)
    assert result["n"] == 10 and 0.0 <= result["shortest_mate"] <= 1.0 and "mate_preserving" not in result
    assert model.training
