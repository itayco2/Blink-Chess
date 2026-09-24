"""A small random network with BlinkNet's signature, so export and the page can be built before training.

tokens int64 [B, 64] -> (policy_logits [B, 1880], value_logits [B, 128]). Its weights are seeded and
untrained: its arrows exercise the pipeline, not chess skill. It uses the same op families the real
model exports (embedding, scaled dot-product attention, layer norm, GELU, linear heads).
"""

from dataclasses import dataclass

import torch
from torch import nn

from blink.board import encode, moves, value

POLICY_INIT_STD = 0.5  # a peakier random policy, so the page's three arrows differ visibly


@dataclass(frozen=True)
class StandInConfig:
    d_model: int = 128
    layers: int = 2
    heads: int = 4
    ffn: int = 256


class Block(nn.Module):
    def __init__(self, cfg: StandInConfig) -> None:
        super().__init__()
        self.heads = cfg.heads
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(nn.Linear(cfg.d_model, cfg.ffn), nn.GELU(), nn.Linear(cfg.ffn, cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, s, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.out(attn.transpose(1, 2).reshape(b, s, d))
        return x + self.ffn(self.norm2(x))


class StandInNet(nn.Module):
    def __init__(self, cfg: StandInConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(encode.NUM_CODES, cfg.d_model)
        self.pos = nn.Parameter(torch.randn(64, cfg.d_model) * 0.1)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.policy = nn.Linear(cfg.d_model, moves.NUM_MOVES)
        self.value = nn.Linear(cfg.d_model, value.NUM_BINS)
        nn.init.normal_(self.policy.weight, std=POLICY_INIT_STD)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(tokens) + self.pos
        for block in self.blocks:
            x = block(x)
        pooled = self.norm(x).mean(dim=1)
        return self.policy(pooled), self.value(pooled)


def build(seed: int = 0, cfg: StandInConfig | None = None) -> StandInNet:
    """The stand-in in eval mode, identical for the same seed. The global RNG is left untouched."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = StandInNet(cfg or StandInConfig())
    return net.eval()
