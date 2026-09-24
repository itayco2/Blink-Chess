# Copyright 2025 DeepMind Technologies Limited
# Modifications copyright 2026 Itay Cohen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""DeepMind's action-value transformer (searchless_chess), ported to PyTorch, with its tokenizer.

Ported from google-deepmind/searchless_chess at commit 90ae0e6b (Apache-2.0):
- src/tokenizer.py `tokenize`: 77 tokens = side to move, 64 squares, castling padded to 4 characters,
  en passant as 2 characters, then the halfmove and fullmove counters as 3 characters each;
- src/utils.py `_compute_all_possible_actions` (the 1968 UCI actions) and
  `get_uniform_buckets_edges_values` (the 128 return-bucket centres);
- src/transformer.py `transformer_decoder`, with the action-value config of src/engines/constants.py;
- src/engines/neural_engines.py `ActionValueEngine.analyse` (one row per legal move, in action order)
  and `_update_scores_with_repetitions`.

Modifications: Haiku and JAX become PyTorch. Haiku's [in, out] Linear weights are transposed to torch's
[out, in] when the npz written by tools/dm_convert.py is loaded. The final layer norm and Linear run at
the last position only (both act per position, so the output is the same). The tokenizer raises
ValueError where DeepMind's asserts. The repetition rule returns a mask instead of editing scores in
place. Plan P8 checklist: one 1968-entry embedding table shared by FEN characters, moves and the
return-bucket token; sqrt(d) embedding scale; learned positions (79); a shift_right BOS token 0; pre-LN
with scale and offset (Haiku's eps 1e-5); bias-free attention with no mask; SwiGLU x4 with no biases;
a final LN plus a biased Linear; 128 buckets read at the last position.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

KINDS = ("params", "params_ema")
SEQUENCE_LENGTH = 77
NUM_RETURN_BUCKETS = 128
CHARACTERS = "0123456789abcdefghpnrkqPBNRQKw."
_INDEX = {char: index for index, char in enumerate(CHARACTERS)}
_EMPTY = "."
_CASTLING_WIDTH, _EP_WIDTH, _COUNTER_WIDTH = 4, 2, 3
_FILES = "abcdefgh"

# ------------------------------------------------------------------------------ tokenizer


def _indices(chars: str, fen: str) -> list[int]:
    try:
        return [_INDEX[char] for char in chars]
    except KeyError as exc:
        raise ValueError(f"character {exc.args[0]!r} is not in DeepMind's alphabet (FEN {fen!r})") from None


def tokenize(fen: str) -> np.ndarray:
    """DeepMind's 77 tokens for a full six-field FEN (uint8)."""
    fields = fen.split(" ")
    if len(fields) != 6:
        raise ValueError(f"a FEN needs six fields, real counters included; got {fen!r}")
    board, side, castling, en_passant, halfmoves, fullmoves = fields
    squares = "".join(_EMPTY * int(c) if c in "12345678" else c for c in board.replace("/", ""))
    castling = "" if castling == "-" else castling
    en_passant = _EMPTY * _EP_WIDTH if en_passant == "-" else en_passant
    for name, counter in (("halfmove", halfmoves), ("fullmove", fullmoves)):
        if not counter.isdigit() or len(counter) > _COUNTER_WIDTH:
            raise ValueError(f"the {name} counter {counter!r} needs 1 to 3 digits (FEN {fen!r})")
    text = side + squares + castling.ljust(_CASTLING_WIDTH, _EMPTY) + en_passant
    text += halfmoves.ljust(_COUNTER_WIDTH, _EMPTY) + fullmoves.ljust(_COUNTER_WIDTH, _EMPTY)
    if len(text) != SEQUENCE_LENGTH:
        raise ValueError(f"FEN {fen!r} gives {len(text)} tokens, not {SEQUENCE_LENGTH}")
    return np.asarray(_indices(text, fen), dtype=np.uint8)


def tokenize_board(board: chess.Board) -> np.ndarray:
    """The tokens of board.fen(): its real counters, and an en passant square only when a capture is legal."""
    return tokenize(board.fen())


# ------------------------------------------------------------------------------ actions and buckets


def _all_actions() -> tuple[str, ...]:
    """Every queen and knight move on an empty board, square by square, then the promotions."""
    moves: list[str] = []
    board = chess.BaseBoard.empty()
    for square in range(64):
        targets: list[int] = []
        for symbol in ("Q", "N"):
            board.set_piece_at(square, chess.Piece.from_symbol(symbol))
            targets += board.attacks(square)
        board.remove_piece_at(square)
        moves += [chess.square_name(square) + chess.square_name(target) for target in targets]
    for rank, next_rank in (("2", "1"), ("7", "8")):
        for index, file in enumerate(_FILES):
            sides = [_FILES[index - 1]] if index > 0 else []
            sides += [_FILES[index + 1]] if index < 7 else []
            for target in (file, *sides):
                moves += [f"{file}{rank}{target}{next_rank}{piece}" for piece in "qrbn"]
    if len(set(moves)) != len(moves):
        raise AssertionError("DeepMind's action list has duplicates")
    return tuple(moves)


ACTION_TO_MOVE = _all_actions()
MOVE_TO_ACTION = {move: action for action, move in enumerate(ACTION_TO_MOVE)}
NUM_ACTIONS = len(ACTION_TO_MOVE)


def bucket_values(num_buckets: int = NUM_RETURN_BUCKETS) -> np.ndarray:
    """The centres of num_buckets uniform win-probability buckets in [0, 1]."""
    edges = np.linspace(0.0, 1.0, num_buckets + 1)
    return (edges[:-1] + edges[1:]) / 2


BUCKET_VALUES = bucket_values()


def ordered_legal_moves(board: chess.Board) -> list[chess.Move]:
    """The legal moves in action order, as DeepMind's engines enumerate them."""
    return sorted(board.legal_moves, key=lambda move: MOVE_TO_ACTION[move.uci()])


def sequences(board: chess.Board, moves: Sequence[chess.Move]) -> np.ndarray:
    """ActionValueEngine's rows: 77 FEN tokens, the move's action, a dummy return bucket (int64 [L, 79])."""
    rows = np.zeros((len(moves), SEQUENCE_LENGTH + 2), dtype=np.int64)
    rows[:, :SEQUENCE_LENGTH] = tokenize_board(board)
    rows[:, SEQUENCE_LENGTH] = [MOVE_TO_ACTION[move.uci()] for move in moves]
    return rows


def win_probabilities(log_probs: np.ndarray) -> np.ndarray:
    """Expected win per row: the bucket probabilities times the bucket centres."""
    return np.exp(np.asarray(log_probs, dtype=np.float64)) @ BUCKET_VALUES


def repetition_draws(board: chess.Board, moves: Sequence[chess.Move]) -> np.ndarray:
    """True for each move after which a fivefold repetition stands or a threefold can be claimed.

    DeepMind's released engines score such a move 0.5. The board's move stack is the real history.
    """
    walker = board.copy()
    draws = np.zeros(len(moves), dtype=bool)
    for index, move in enumerate(moves):
        walker.push(move)
        draws[index] = walker.is_fivefold_repetition() or walker.can_claim_threefold_repetition()
        walker.pop()
    return draws


# ------------------------------------------------------------------------------ model


@dataclass(frozen=True)
class DeepMindConfig:
    vocab_size: int = NUM_ACTIONS
    output_size: int = NUM_RETURN_BUCKETS
    embedding_dim: int = 256
    num_layers: int = 8
    num_heads: int = 8
    max_sequence_length: int = SEQUENCE_LENGTH + 2
    widening_factor: int = 4
    layer_norm_eps: float = 1e-5  # hk.LayerNorm's default


# 136M (8 layers, d 1024) and 270M (16 layers, d 1024) stay paper-only rows (plan E7): not downloaded.
CONFIGS = {"9M": DeepMindConfig(embedding_dim=256, num_layers=8, num_heads=8)}


def shift_right(sequences: torch.Tensor) -> torch.Tensor:
    """Prepend the BOS token 0 and drop the last token, so position t predicts token t."""
    return torch.cat([torch.zeros_like(sequences[:, :1]), sequences[:, :-1]], dim=1)


class _Attention(nn.Module):
    def __init__(self, config: DeepMindConfig) -> None:
        super().__init__()
        d = config.embedding_dim
        self.num_heads = config.num_heads
        self.q, self.k, self.v, self.out = (nn.Linear(d, d, bias=False) for _ in range(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = (
            p(x).view(b, t, self.num_heads, d // self.num_heads).transpose(1, 2)
            for p in (self.q, self.k, self.v)
        )
        attended = F.scaled_dot_product_attention(q, k, v)  # no mask; scale 1/sqrt(d_head)
        return self.out(attended.transpose(1, 2).reshape(b, t, d))


class _SwiGLU(nn.Module):
    def __init__(self, config: DeepMindConfig) -> None:
        super().__init__()
        d, ffn = config.embedding_dim, config.embedding_dim * config.widening_factor
        self.gate, self.up = nn.Linear(d, ffn, bias=False), nn.Linear(d, ffn, bias=False)
        self.down = nn.Linear(ffn, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _Block(nn.Module):
    def __init__(self, config: DeepMindConfig) -> None:
        super().__init__()
        d, eps = config.embedding_dim, config.layer_norm_eps
        self.attn_norm, self.attn = nn.LayerNorm(d, eps=eps), _Attention(config)
        self.mlp_norm, self.mlp = nn.LayerNorm(d, eps=eps), _SwiGLU(config)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = h + self.attn(self.attn_norm(h))
        return h + self.mlp(self.mlp_norm(h))


class ActionValueTransformer(nn.Module):
    """targets [B, T] (T <= 79) -> log-probs [B, output_size] at the last position."""

    def __init__(self, config: DeepMindConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.pos = nn.Embedding(config.max_sequence_length, config.embedding_dim)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.num_layers))
        self.final_norm = nn.LayerNorm(config.embedding_dim, eps=config.layer_norm_eps)
        self.head = nn.Linear(config.embedding_dim, config.output_size)
        self.embedding_scale = math.sqrt(config.embedding_dim)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        inputs = shift_right(targets)
        length = inputs.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError(
                f"{length} tokens exceed the {self.config.max_sequence_length} learned positions"
            )
        h = self.embed(inputs) * self.embedding_scale + self.pos.weight[:length]
        for block in self.blocks:
            h = block(h)
        return F.log_softmax(self.head(self.final_norm(h[:, -1])), dim=-1)


# ------------------------------------------------------------------------------ Haiku weights


def haiku_suffix(index: int) -> str:
    """Haiku names the n-th module of a kind `kind`, `kind_1`, `kind_2`, ... in creation order."""
    return "" if index == 0 else f"_{index}"


def _haiku_map(config: DeepMindConfig) -> dict[str, tuple[str, bool]]:
    """torch parameter name -> (flattened Haiku name, whether to transpose [in, out] to [out, in])."""
    mapping = {"embed.weight": ("embed/embeddings", False), "pos.weight": ("embed_1/embeddings", False)}

    def norm(torch_name: str, index: int) -> None:
        mapping[f"{torch_name}.weight"] = (f"layer_norm{haiku_suffix(index)}/scale", False)
        mapping[f"{torch_name}.bias"] = (f"layer_norm{haiku_suffix(index)}/offset", False)

    for layer in range(config.num_layers):
        attention = f"multi_head_dot_product_attention{haiku_suffix(layer)}"
        norm(f"blocks.{layer}.attn_norm", 2 * layer)
        norm(f"blocks.{layer}.mlp_norm", 2 * layer + 1)
        for index, proj in enumerate(("q", "k", "v", "out")):
            mapping[f"blocks.{layer}.attn.{proj}.weight"] = (
                f"{attention}/linear{haiku_suffix(index)}/w",
                True,
            )
        for index, proj in enumerate(("gate", "up", "down")):
            mapping[f"blocks.{layer}.mlp.{proj}.weight"] = (
                f"linear{haiku_suffix(3 * layer + index)}/w",
                True,
            )
    norm("final_norm", 2 * config.num_layers)
    head = f"linear{haiku_suffix(3 * config.num_layers)}"
    mapping["head.weight"], mapping["head.bias"] = (f"{head}/w", True), (f"{head}/b", False)
    return mapping


def haiku_shapes(config: DeepMindConfig) -> dict[str, tuple[int, ...]]:
    """Every Haiku parameter name of this config and its shape in Haiku's layout."""
    with torch.device("meta"):
        state = ActionValueTransformer(config).state_dict()
    return {
        name: tuple(reversed(state[key].shape)) if transpose else tuple(state[key].shape)
        for key, (name, transpose) in _haiku_map(config).items()
    }


def state_dict_from_haiku(
    params: Mapping[str, np.ndarray], config: DeepMindConfig
) -> dict[str, torch.Tensor]:
    """A torch state dict from flattened Haiku parameters; refuses any missing, extra or misshapen one."""
    expected = haiku_shapes(config)
    missing, extra = sorted(set(expected) - set(params)), sorted(set(params) - set(expected))
    misshapen = {
        n: (tuple(params[n].shape), s) for n, s in expected.items() if n in params and params[n].shape != s
    }
    if missing or extra or misshapen:
        raise ValueError(
            f"DeepMind weights do not fit {config}: missing {missing}, extra {extra}, "
            f"misshapen (found, expected) {misshapen}"
        )
    state = {}
    for key, (name, transpose) in _haiku_map(config).items():
        array = np.asarray(params[name], dtype=np.float32)
        state[key] = torch.from_numpy(np.ascontiguousarray(array.T if transpose else array))
    return state


def load_model(path: Path, config: DeepMindConfig, device: str = "cpu") -> ActionValueTransformer:
    """The fp32 model from an npz of flattened Haiku names, in eval mode on `device`."""
    with np.load(path, allow_pickle=False) as data:
        params = {name: data[name] for name in data.files}
    model = ActionValueTransformer(config)
    model.load_state_dict(state_dict_from_haiku(params, config), strict=True)
    return model.to(device).eval()
