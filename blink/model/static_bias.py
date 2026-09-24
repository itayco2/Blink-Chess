"""A static attention bias: one learned 64 x 64 matrix per head, in GAB-lite's place (P5 arm a06).

Arm a06 asks whether GAB-lite earns its 598,016 parameters at S by making the bias depend on the
position, or whether a fixed learned square-to-square bias does as well. So the static bias takes
exactly GAB-lite's slot: it is computed once per forward pass and added to the attention logits of
every layer, one 64 x 64 matrix per head. It has n_heads * 4096 parameters (32,768 at S, 65,536 at M).

It starts at zero, so a fresh network starts exactly like the plain trunk, as a fresh GAB-lite does.
The bias is returned as [1, H, 64, 64] and scaled_dot_product_attention broadcasts it over the batch,
so no [B, H, 64, 64] copy is made for a bias that is the same for every position. The optimizer's
"matrices only" rule (ndim >= 2) gives it weight decay, as it gives GAB-lite's generator, so the arm
changes the bias and nothing else.
"""

import torch
from torch import nn

SQUARES = 64


def parameter_count(n_heads: int) -> int:
    return n_heads * SQUARES * SQUARES


class StaticBias(nn.Module):
    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_heads, SQUARES, SQUARES))

    def forward(self) -> torch.Tensor:
        """The attention bias [1, H, 64, 64], the same for every position in the batch."""
        return self.bias.unsqueeze(0)
