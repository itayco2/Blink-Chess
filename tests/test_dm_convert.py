"""tools/dm_convert.py, the parts that need no JAX: its fixed positions and its Haiku name flattening."""

import importlib.util
from pathlib import Path

import chess
import numpy as np
import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "dm_convert.py"
JAX_LOGITS = Path(r"D:\blink\dm\9M-jax-logits.npz")


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("dm_convert", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_special_positions_are_legal_and_cover_every_tokenizer_field(tool):
    boards = [chess.Board(fen) for fen in tool.SPECIAL_FENS]
    assert all(board.is_valid() for board in boards)
    fields = [board.fen().split(" ") for board in boards]
    assert {(f[1], f[3] != "-") for f in fields} >= {("w", True), ("b", True)}  # legal en passant, both sides
    assert {f[2] for f in fields} >= {"KQkq", "Kq", "Qk", "K", "-"}
    assert any(len(f[4]) == 3 for f in fields) and any(len(f[5]) == 3 for f in fields)
    assert any(move.promotion and board.is_capture(move) for board in boards for move in board.legal_moves)
    assert max(board.legal_moves.count() for board in boards) == 218
    assert len(tool.SPECIAL_FENS) < tool.N_POSITIONS == 100


def test_flatten_gives_haiku_module_slash_param_names_and_to_haiku_inverts_it(tool):
    tree = {
        "multi_head_dot_product_attention_1": {"linear_2": {"w": np.ones((2, 3))}},
        "embed": {"embeddings": [0]},
    }
    flat = dict(tool.flatten(tree))
    assert list(flat) == ["embed/embeddings", "multi_head_dot_product_attention_1/linear_2/w"]
    params = tool.to_haiku(flat)
    assert params["multi_head_dot_product_attention_1/linear_2"]["w"].shape == (2, 3)
    assert params["embed"]["embeddings"].tolist() == [0]


def test_deepmind_sources_are_pinned_by_commit_and_sha256(tool):
    assert len(tool.DM_COMMIT) == 40
    assert sorted(tool.DM_SOURCES) == ["tokenizer.py", "transformer.py", "utils.py"]
    assert all(len(digest) == 64 for digest in tool.DM_SOURCES.values())


@pytest.mark.local
@pytest.mark.skipif(not JAX_LOGITS.is_file(), reason="run tools/dm_convert.py in the D: venv first")
def test_the_saved_reference_holds_the_special_positions_then_puzzle_positions(tool):
    saved = np.load(JAX_LOGITS, allow_pickle=False)
    special = [chess.Board(fen).fen() for fen in tool.SPECIAL_FENS]
    assert saved["fens"].tolist()[: len(special)] == special
    assert len(saved["fens"]) == 100 and saved["offsets"][-1] == len(saved["moves"]) == 3118
