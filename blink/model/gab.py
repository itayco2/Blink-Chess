"""GAB-lite: one learned attention bias per position, computed once and shared by every layer.

After Chessformer's GAB (arXiv 2605.19091, ICLR 2026) and Leela's smolgen, reimplemented from the
descriptions. From the embeddings x [B, 64, d]:

    compress   each square to 32 numbers        d x 32          -> [B, 64 * 32]
    hidden     one dense layer to 128           2048 x 128      -> [B, 128]
    per head   64 generator inputs per head     128 x 64 H      -> [B, H, 64]
    generator  one 64 -> 4096 map shared by all heads           -> [B, H, 64, 64]

GELU and a parameter-free RMS norm follow the hidden and per-head layers, and no layer has a bias, so
the module has d*32 + 2048*128 + 128*64*H + 64*4096 parameters: 598,016 at S (d256, H8) and 671,744
at M (d512, H16). The generator starts at zero, so a fresh network gets a zero bias and starts exactly
like the plain trunk; the bias grows as the generator learns.
"""

import torch
from torch import nn
from torch.nn import functional as F

SQUARES = 64
PER_SQUARE = 32
HIDDEN = 128
GEN_PER_HEAD = 64
INIT_STD = 0.02


def parameter_count(d_model: int, n_heads: int) -> int:
    return (
        d_model * PER_SQUARE
        + SQUARES * PER_SQUARE * HIDDEN
        + HIDDEN * GEN_PER_HEAD * n_heads
        + GEN_PER_HEAD * SQUARES * SQUARES
    )


def _rms(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],))


class GabLite(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.compress = nn.Linear(d_model, PER_SQUARE, bias=False)
        self.hidden = nn.Linear(SQUARES * PER_SQUARE, HIDDEN, bias=False)
        self.heads = nn.Linear(HIDDEN, GEN_PER_HEAD * n_heads, bias=False)
        self.generator = nn.Linear(GEN_PER_HEAD, SQUARES * SQUARES, bias=False)
        for layer in (self.compress, self.hidden, self.heads):
            nn.init.normal_(layer.weight, std=INIT_STD)
        nn.init.zeros_(self.generator.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 64, d] embeddings -> the attention bias [B, H, 64, 64]."""
        batch = x.shape[0]
        squeezed = self.compress(x).flatten(1)
        hidden = _rms(F.gelu(self.hidden(squeezed)))
        per_head = _rms(F.gelu(self.heads(hidden)).view(batch, self.n_heads, GEN_PER_HEAD))
        return self.generator(per_head).view(batch, self.n_heads, SQUARES, SQUARES)
