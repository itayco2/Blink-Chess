"""The two heads on top of the trunk.

Policy: an attention head. Each square emits a query (as a from-square) and a key (as a to-square);
their scaled dot product gives 64 x 64 from-to logits, gathered into the 1880 order of
blink.board.moves. A promotion's logit is its from-to pair's logit plus a learned per-piece bias
read from the to-square's key.

Value: mean-pool the 64 squares, a small MLP, then 128 bin logits.

The last projection of each head starts at zero, so a fresh network outputs uniform policy and
value distributions and the initial losses are exactly ln 1880 and ln 128.
"""

import math

import torch
from torch import nn

from blink.board import moves, value

NUM_PROMO_PIECES = len(moves.PROMO_PIECES)


def _promotion_indices() -> tuple[list[int], list[int], list[int]]:
    """For each promotion slot in vocabulary order: flat from*64+to, the to-square and the piece."""
    flat, to_square, piece = [], [], []
    for frm, to in moves.PROMO_PAIRS:
        for p in range(NUM_PROMO_PIECES):
            flat.append(frm * 64 + to)
            to_square.append(to)
            piece.append(p)
    return flat, to_square, piece


class AttentionPolicyHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.promotion = nn.Linear(d_model, NUM_PROMO_PIECES, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)
        flat, to_square, piece = _promotion_indices()
        from_to = [frm * 64 + to for frm, to in moves.FROM_TO]
        self.register_buffer("from_to_index", torch.tensor(from_to), persistent=False)
        self.register_buffer("promo_flat", torch.tensor(flat), persistent=False)
        bias_index = torch.tensor(to_square) * NUM_PROMO_PIECES + torch.tensor(piece)
        self.register_buffer("promo_bias_index", bias_index, persistent=False)
        nn.init.zeros_(self.query.weight)
        nn.init.zeros_(self.promotion.weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q, k = self.query(hidden), self.key(hidden)
        pair_logits = torch.matmul(q, k.transpose(1, 2)).flatten(1) * self.scale  # [B, 4096]
        bias = self.promotion(k).flatten(1)  # [B, 64 * 4]
        plain = pair_logits.index_select(1, self.from_to_index)
        promo = pair_logits.index_select(1, self.promo_flat) + bias.index_select(1, self.promo_bias_index)
        return torch.cat([plain, promo], dim=1)


class ValueHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.hidden = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, value.NUM_BINS)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        pooled = hidden.mean(dim=1)
        return self.out(nn.functional.gelu(self.hidden(pooled)))
