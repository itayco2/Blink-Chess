"""site/vocab.json and site/tests/golden.json: what the browser page is checked against (P1, P10, PF31)."""

import json

import chess
import numpy as np
import pytest

from blink.board import encode, moves, value
from blink.play.evaluator import Evaluation


class LinearOracle:
    """A deterministic Evaluator: policy logits and value logits are fixed linear maps of the codes."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.w_policy = rng.normal(size=(64, moves.NUM_MOVES)).astype(np.float32)
        self.w_value = rng.normal(size=(64, value.NUM_BINS)).astype(np.float32) * 0.1
        self.calls = 0

    def evaluate(self, codes: np.ndarray) -> Evaluation:
        self.calls += 1
        x = codes.astype(np.float32)
        logits = x @ self.w_value
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return Evaluation(policy_logits=x @ self.w_policy, value_probs=probs)


def test_vocab_json_is_generated_from_the_contract():
    from blink.export import vocab

    data = vocab.build()
    assert data["num_moves"] == moves.NUM_MOVES == 1880
    assert data["num_from_to"] == moves.NUM_FROM_TO == 1792
    assert [tuple(p) for p in data["from_to"]] == list(moves.FROM_TO)
    assert [tuple(p) for p in data["promo_pairs"]] == list(moves.PROMO_PAIRS)
    assert data["promo_pieces"] == ["q", "r", "b", "n"]
    assert data["codes"] == {
        "empty": encode.EMPTY,
        "own": encode.OWN,
        "opp": encode.OPP,
        "own_castling_rook": encode.OWN_CASTLING_ROOK,
        "opp_castling_rook": encode.OPP_CASTLING_ROOK,
        "ep_square": encode.EP_SQUARE,
        "num_codes": encode.NUM_CODES,
    }
    assert data["piece_order"] == ["p", "n", "b", "r", "q", "k"]
    assert data["num_bins"] == value.NUM_BINS


def test_the_tracked_vocab_json_is_current(repo_root):
    from blink.export import vocab

    tracked = repo_root / "site" / "vocab.json"
    assert tracked.read_text(encoding="utf-8") == vocab.render(vocab.build())


def test_golden_fens_cover_castling_promotion_en_passant_and_black_to_move():
    from blink.export import golden

    fens = golden.GOLDEN_FENS
    assert len(fens) == 50
    assert len(set(fens)) == 50
    tags = [set(golden.tags(chess.Board(fen))) for fen in fens]
    count = {name: sum(name in t for t in tags) for name in set().union(*tags)}
    assert all(chess.Board(fen).legal_moves.count() > 0 for fen in fens)
    assert count["castling"] >= 8
    assert count["promotion"] >= 6
    assert count["en_passant"] >= 5
    assert count["black_to_move"] >= 20
    assert count["ep_square_without_legal_capture"] >= 3
    assert count["castling_rights_without_king_or_rook"] >= 2
    assert any({"castling", "black_to_move"} <= t for t in tags)
    assert any({"promotion", "black_to_move"} <= t for t in tags)
    assert any({"en_passant", "black_to_move"} <= t for t in tags)


def test_golden_records_codes_legal_indices_top5_and_win_from_one_batch():
    from blink.export import golden

    oracle = LinearOracle()
    data = golden.build(oracle, {"selector": "oracle"})
    assert oracle.calls == 1
    assert data["top_k"] == 5
    assert len(data["positions"]) == 50
    for entry in data["positions"]:
        board = chess.Board(entry["fen"])
        assert entry["codes"] == encode.encode_board(board).tolist()
        expected = sorted(moves.encode_move(board, m) for m in board.legal_moves)
        assert [i for i, _ in entry["legal"]] == expected
        for index, uci in entry["legal"]:
            assert moves.decode_move(board, index).uci() == uci
        probs = [row["prob"] for row in entry["top5"]]
        assert probs == sorted(probs, reverse=True)
        assert len(probs) == min(5, len(entry["legal"]))
        assert {row["index"] for row in entry["top5"]} <= {i for i, _ in entry["legal"]}
        assert 0.0 < entry["win"] < 1.0


def test_golden_top5_is_the_softmax_over_legal_moves_only():
    from blink.export import golden

    data = golden.build(LinearOracle(seed=1), {"selector": "oracle"})
    entry = data["positions"][0]
    board = chess.Board(entry["fen"])
    codes = encode.encode_board(board)[None]
    logits = LinearOracle(seed=1).evaluate(codes).policy_logits[0]
    legal = [i for i, _ in entry["legal"]]
    probs = np.exp(logits[legal] - logits[legal].max())
    probs /= probs.sum()
    best = legal[int(np.argmax(probs))]
    assert entry["top5"][0]["index"] == best
    assert entry["top5"][0]["prob"] == pytest.approx(float(probs.max()), abs=1e-6)
    assert entry["top5"][0]["uci"] == moves.decode_move(board, best).uci()


def test_golden_json_round_trips_and_names_its_model(tmp_path):
    from blink.export import golden

    data = golden.build(LinearOracle(), {"selector": "oracle", "onnx_sha256": "ab" * 32})
    path = golden.write(data, tmp_path / "golden.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == data
    assert loaded["model"] == {"selector": "oracle", "onnx_sha256": "ab" * 32}


def test_the_tracked_golden_json_matches_the_golden_fens(repo_root):
    from blink.export import golden

    data = json.loads((repo_root / "site" / "tests" / "golden.json").read_text(encoding="utf-8"))
    assert [entry["fen"] for entry in data["positions"]] == list(golden.GOLDEN_FENS)
    for entry in data["positions"]:
        board = chess.Board(entry["fen"])
        assert entry["codes"] == encode.encode_board(board).tolist()
        assert sorted(entry["tags"]) == sorted(golden.tags(board))
