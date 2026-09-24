"""The DeepMind 9M port (plan P8 checklist): tokenizer, action vocabulary, weight layout, JAX parity."""

import json
import math
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from blink.reference import deepmind

JAX_LOGITS = Path(r"D:\blink\dm\9M-jax-logits.npz")
WEIGHTS = {kind: Path(rf"D:\blink\dm\9M-{kind}.npz") for kind in deepmind.KINDS}
PARITY_TOLERANCE = 1e-3
# DeepMind's tokenizer on the start position, as saved by tools/dm_convert.py from their own code.
START_TOKENS = [29, 20, 19, 11, 22, 21, 11, 19, 20, *[18] * 8, *[30] * 32, *[23] * 8]
START_TOKENS += [26, 25, 24, 27, 28, 24, 25, 26, 28, 27, 21, 22, 30, 30, 0, 30, 30, 1, 30, 30]
TINY = deepmind.DeepMindConfig(embedding_dim=16, num_layers=2, num_heads=2, output_size=7)

pytestmark = pytest.mark.torch


def played(*ucis: str) -> chess.Board:
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board


def tail(tokens: np.ndarray, n: int) -> str:
    return "".join(deepmind.CHARACTERS[t] for t in tokens[-n:])


# ------------------------------------------------------------------------------ tokenizer


def test_the_start_position_tokenizes_exactly_like_deepmind():
    tokens = deepmind.tokenize(chess.STARTING_FEN)
    assert tokens.dtype == np.uint8
    assert tokens.tolist() == START_TOKENS
    assert len(tokens) == deepmind.SEQUENCE_LENGTH == 77


def test_the_counters_come_from_the_game_not_a_blanket_zero_one():
    board = played("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "g8f6", "e1g1")
    tokens = deepmind.tokenize_board(board)
    assert board.fen().endswith(" b kq - 5 4")  # five quiet piece moves since 2...e5
    assert tail(tokens, 6) == "5..4.."
    assert tail(deepmind.tokenize("8/8/4k3/8/8/3K4/6R1/8 w - - 120 999"), 6) == "120999"
    assert not np.array_equal(tokens, deepmind.tokenize(board.fen().rsplit(" ", 2)[0] + " 0 1"))


def test_castling_rights_are_padded_to_four_characters():
    castling = slice(1 + 64, 1 + 64 + 4)
    assert "".join(deepmind.CHARACTERS[t] for t in deepmind.tokenize(chess.STARTING_FEN)[castling]) == "KQkq"
    partial = deepmind.tokenize("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1")[castling]
    assert "".join(deepmind.CHARACTERS[t] for t in partial) == "Kq.."
    none = deepmind.tokenize("4k3/8/8/8/8/8/8/4K3 b - - 0 1")[castling]
    assert "".join(deepmind.CHARACTERS[t] for t in none) == "...."


def test_en_passant_is_two_characters_and_only_when_legal():
    ep = slice(1 + 64 + 4, 1 + 64 + 4 + 2)
    capturable = played("e2e4", "d7d5", "e4e5", "f7f5")
    assert "".join(deepmind.CHARACTERS[t] for t in deepmind.tokenize_board(capturable)[ep]) == "f6"
    not_capturable = played("e2e4")  # python-chess's board.fen() hides an ep square nobody can take
    assert "".join(deepmind.CHARACTERS[t] for t in deepmind.tokenize_board(not_capturable)[ep]) == ".."


def test_black_to_move_is_the_letter_b():
    assert deepmind.CHARACTERS[deepmind.tokenize_board(played("e2e4"))[0]] == "b"


def test_the_tokenizer_refuses_what_deepmind_would_assert_on():
    with pytest.raises(ValueError, match="halfmove"):
        deepmind.tokenize("8/8/4k3/8/8/3K4/6R1/8 w - - 1000 999")
    with pytest.raises(ValueError, match="six"):
        deepmind.tokenize("8/8/4k3/8/8/3K4/6R1/8 w - -")


# ------------------------------------------------------------------------------ actions


def test_the_action_vocabulary_has_1968_moves_and_round_trips():
    assert deepmind.NUM_ACTIONS == len(deepmind.MOVE_TO_ACTION) == len(deepmind.ACTION_TO_MOVE) == 1968
    assert all(deepmind.MOVE_TO_ACTION[deepmind.ACTION_TO_MOVE[a]] == a for a in range(1968))
    assert deepmind.ACTION_TO_MOVE[0] == "a1b1"
    assert deepmind.ACTION_TO_MOVE[1967] == "h7g8n"
    assert {"e1g1", "e8c8", "a7b8q", "h2g1n"} <= set(deepmind.MOVE_TO_ACTION)


def test_every_legal_move_of_the_218_move_position_is_in_the_vocabulary():
    board = chess.Board("R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1")
    ordered = deepmind.ordered_legal_moves(board)
    assert len(ordered) == 218
    actions = [deepmind.MOVE_TO_ACTION[m.uci()] for m in ordered]
    assert actions == sorted(actions)


def test_rows_are_77_fen_tokens_then_the_action_then_a_dummy_bucket():
    board = played("e2e4", "c7c5")
    ordered = deepmind.ordered_legal_moves(board)
    rows = deepmind.sequences(board, ordered)
    assert rows.shape == (len(ordered), deepmind.SEQUENCE_LENGTH + 2)
    assert (rows[:, :77] == deepmind.tokenize_board(board)).all()
    assert rows[:, 77].tolist() == [deepmind.MOVE_TO_ACTION[m.uci()] for m in ordered]
    assert (rows[:, 78] == 0).all()


def test_bucket_values_are_the_128_uniform_centres():
    values = deepmind.bucket_values()
    assert values.shape == (128,)
    assert values[0] == pytest.approx(0.5 / 128) and values[-1] == pytest.approx(1 - 0.5 / 128)


# ------------------------------------------------------------------------------ model and weights


def haiku_layer_norm(x: np.ndarray, scale: np.ndarray, offset: np.ndarray) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(x.var(-1, keepdims=True) + 1e-5) * scale
    return inv * x + (offset - mean * inv)


def numpy_decoder(params: dict[str, np.ndarray], targets: np.ndarray, config) -> np.ndarray:
    """DeepMind's transformer_decoder written out in numpy on Haiku-layout [in, out] weights."""
    d, heads = config.embedding_dim, config.num_heads
    inputs = np.concatenate([np.zeros((len(targets), 1), dtype=targets.dtype), targets], axis=1)[:, :-1]
    h = params["embed/embeddings"][inputs] * np.sqrt(d) + params["embed_1/embeddings"][: inputs.shape[1]]
    for layer in range(config.num_layers):
        ln, attn, mlp = (deepmind.haiku_suffix(k) for k in (2 * layer, layer, 3 * layer))
        x = haiku_layer_norm(h, params[f"layer_norm{ln}/scale"], params[f"layer_norm{ln}/offset"])
        prefix = f"multi_head_dot_product_attention{attn}"
        q, k, v = (x @ params[f"{prefix}/linear{deepmind.haiku_suffix(i)}/w"] for i in range(3))
        b, t, _ = q.shape
        q, k, v = (a.reshape(b, t, heads, d // heads) for a in (q, k, v))
        logits = np.einsum("bthd,bThd->bhtT", q, k) / np.sqrt(d // heads)
        weights = np.exp(logits - logits.max(-1, keepdims=True))
        weights /= weights.sum(-1, keepdims=True)
        out = np.einsum("bhtT,bThd->bthd", weights, v).reshape(b, t, d)
        h = h + out @ params[f"{prefix}/linear_3/w"]
        ln2 = deepmind.haiku_suffix(2 * layer + 1)
        x = haiku_layer_norm(h, params[f"layer_norm{ln2}/scale"], params[f"layer_norm{ln2}/offset"])
        w1, w2, w3 = (params[f"linear{deepmind.haiku_suffix(3 * layer + i)}/w"] for i in range(3))
        a = x @ w1
        h = h + (a / (1 + np.exp(-a)) * (x @ w2)) @ w3
    final_ln = deepmind.haiku_suffix(2 * config.num_layers)
    h = haiku_layer_norm(h, params[f"layer_norm{final_ln}/scale"], params[f"layer_norm{final_ln}/offset"])
    head = deepmind.haiku_suffix(3 * config.num_layers)
    logits = h @ params[f"linear{head}/w"] + params[f"linear{head}/b"]
    logits = logits - logits.max(-1, keepdims=True)
    return (logits - np.log(np.exp(logits).sum(-1, keepdims=True)))[:, -1]


def random_haiku_params(config, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        name: rng.normal(0, 0.3, shape).astype(np.float32)
        for name, shape in deepmind.haiku_shapes(config).items()
    }


def test_haiku_names_follow_module_creation_order():
    shapes = deepmind.haiku_shapes(TINY)
    d, ffn = TINY.embedding_dim, TINY.embedding_dim * 4
    assert shapes["embed/embeddings"] == (1968, d)
    assert shapes["embed_1/embeddings"] == (79, d)
    assert shapes["multi_head_dot_product_attention/linear/w"] == (d, d)
    assert shapes["multi_head_dot_product_attention_1/linear_3/w"] == (d, d)
    assert shapes["linear/w"] == (d, ffn) and shapes["linear_2/w"] == (ffn, d)
    assert shapes["linear_5/w"] == (ffn, d)
    assert shapes["layer_norm_4/scale"] == (d,)
    assert shapes["linear_6/w"] == (d, 7) and shapes["linear_6/b"] == (7,)
    assert len(shapes) == 2 + 2 * (4 + 4 + 3) + 2 + 2  # embeddings, 2 x (2 LNs, q k v out, 3 MLP), LN, head


def test_the_9m_config_has_the_checkpoint_parameter_count():
    shapes = deepmind.haiku_shapes(deepmind.CONFIGS["9M"])
    assert len(shapes) == 94
    assert sum(math.prod(s) for s in shapes.values()) == 8_954_240


def test_weight_layout_matches_deepmind_on_a_random_tiny_config(tmp_path):
    params = random_haiku_params(TINY)
    path = tmp_path / "tiny.npz"
    np.savez(path, **params)
    model = deepmind.load_model(path, TINY, device="cpu")
    board = played("d2d4", "g8f6", "c2c4")
    rows = deepmind.sequences(board, deepmind.ordered_legal_moves(board))
    with torch.inference_mode():
        got = model(torch.from_numpy(rows).long()).numpy()
    want = numpy_decoder({k: v.astype(np.float64) for k, v in params.items()}, rows.astype(np.int64), TINY)
    assert got.shape == (len(rows), 7)
    assert np.abs(got - want).max() < 1e-4


def test_linear_weights_are_transposed_from_haiku_in_out():
    params = random_haiku_params(TINY, seed=1)
    state = deepmind.state_dict_from_haiku(params, TINY)
    w = params["multi_head_dot_product_attention/linear/w"]
    assert torch.equal(state["blocks.0.attn.q.weight"], torch.from_numpy(w.T.copy()))
    assert torch.equal(state["head.weight"], torch.from_numpy(params["linear_6/w"].T.copy()))
    assert torch.equal(state["head.bias"], torch.from_numpy(params["linear_6/b"]))


def test_one_embedding_table_serves_fen_characters_moves_and_buckets():
    model = deepmind.ActionValueTransformer(deepmind.CONFIGS["9M"])
    embeddings = [m for m in model.modules() if isinstance(m, torch.nn.Embedding)]
    assert [e.num_embeddings for e in embeddings] == [1968, 79]  # tokens (shared), positions (learned)
    assert model.embedding_scale == 16.0  # sqrt(256)
    linears = {name: m for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)}
    assert len(linears) == 8 * (4 + 3) + 1
    assert [name for name, m in linears.items() if m.bias is not None] == ["head"]  # bias-free attn, SwiGLU
    norms = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert len(norms) == 2 * 8 + 1 and all(n.weight is not None and n.bias is not None for n in norms)


def test_shift_right_prepends_a_zero_bos_and_drops_the_last_token():
    shifted = deepmind.shift_right(torch.tensor([[5, 6, 7], [8, 9, 10]]))
    assert shifted.tolist() == [[0, 5, 6], [0, 8, 9]]


def test_the_loader_refuses_missing_extra_or_misshapen_weights(tmp_path):
    params = random_haiku_params(TINY)
    for bad in (
        {k: v for k, v in params.items() if k != "linear_6/b"},
        {**params, "linear_99/w": np.zeros((2, 2), np.float32)},
        {**params, "linear/w": params["linear/w"].T.copy()},
    ):
        np.savez(tmp_path / "bad.npz", **bad)
        with pytest.raises(ValueError, match="DeepMind weights"):
            deepmind.load_model(tmp_path / "bad.npz", TINY, device="cpu")


# ------------------------------------------------------------------------------ parity (local)


@pytest.mark.local
@pytest.mark.skipif(not JAX_LOGITS.is_file(), reason="run tools/dm_convert.py in the D: venv first")
@pytest.mark.parametrize("kind", deepmind.KINDS)
def test_the_dm_port_matches_saved_jax_logits(kind):
    if not WEIGHTS[kind].is_file():
        pytest.skip(f"{WEIGHTS[kind]} is missing")
    saved = np.load(JAX_LOGITS, allow_pickle=False)
    assert json.loads(str(saved["meta"]))["positions"] == 100
    offsets, rows = saved["offsets"], []
    for i, fen in enumerate(saved["fens"]):
        board = chess.Board(str(fen))
        ordered = deepmind.ordered_legal_moves(board)
        assert [m.uci() for m in ordered] == saved["moves"][offsets[i] : offsets[i + 1]].tolist()
        assert (deepmind.tokenize_board(board) == saved["tokens"][i]).all()
        rows.append(deepmind.sequences(board, ordered))
    model = deepmind.load_model(WEIGHTS[kind], deepmind.CONFIGS["9M"], device="cpu")
    batch = torch.from_numpy(np.concatenate(rows))
    with torch.inference_mode():
        got = torch.cat([model(chunk) for chunk in batch.split(512)]).numpy()
    assert got.shape == saved[f"log_probs_{kind}"].shape == (3118, 128)
    assert np.abs(got - saved[f"log_probs_{kind}"]).max() <= PARITY_TOLERANCE
