"""BlinkNet: 64 square tokens in, 1880 policy logits and 128 value logits out.

Recipe D trunk: token embedding (16 codes) plus a learned square embedding, L pre-RMSNorm blocks with
QK-norm attention (F.scaled_dot_product_attention, no QKV bias, head dim 32) and a GELU FFN of width
2d, dropout 0, then a final RMSNorm. With model.gab the GAB-lite module (blink.model.gab) turns the
embeddings into one attention bias [B, H, 64, 64], computed once and added in every layer.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from blink.board.encode import NUM_CODES
from blink.model.config import ModelConfig
from blink.model.gab import GabLite
from blink.model.heads import AttentionPolicyHead, ValueHead

INIT_STD = 0.02


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads, self.head_dim = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.q_norm = nn.RMSNorm(cfg.head_dim)
        self.k_norm = nn.RMSNorm(cfg.head_dim)
        self.out = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim).unbind(2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v.transpose(1, 2), attn_mask=bias)
        return self.out(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ffn_norm = nn.RMSNorm(cfg.d_model)
        self.ffn_in = nn.Linear(cfg.d_model, cfg.ffn_mult * cfg.d_model, bias=False)
        self.ffn_out = nn.Linear(cfg.ffn_mult * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), bias)
        return x + self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(x))))


class Trunk(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(NUM_CODES, cfg.d_model)
        self.square_embedding = nn.Parameter(torch.zeros(64, cfg.d_model))
        self.gab = GabLite(cfg.d_model, cfg.n_heads) if cfg.gab else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.RMSNorm(cfg.d_model)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(tokens) + self.square_embedding

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        bias = None if self.gab is None else self.gab(x)
        for block in self.blocks:
            x = block(x, bias)
        return self.final_norm(x)


class BlinkNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.trunk = Trunk(cfg)
        self.policy = AttentionPolicyHead(cfg.d_model)
        self.value = ValueHead(cfg.d_model)
        self._init_weights(cfg)

    def _init_weights(self, cfg: ModelConfig) -> None:
        """GPT-2 style: normal(0.02), residual projections scaled by 1/sqrt(2L); head zeros kept."""
        residual_std = INIT_STD / math.sqrt(2 * cfg.n_layers)
        nn.init.normal_(self.trunk.token_embedding.weight, std=INIT_STD)
        nn.init.normal_(self.trunk.square_embedding, std=INIT_STD)
        for block in self.trunk.blocks:
            nn.init.normal_(block.attn.qkv.weight, std=INIT_STD)
            nn.init.normal_(block.ffn_in.weight, std=INIT_STD)
            nn.init.normal_(block.attn.out.weight, std=residual_std)
            nn.init.normal_(block.ffn_out.weight, std=residual_std)
        nn.init.normal_(self.policy.key.weight, std=INIT_STD)
        nn.init.normal_(self.value.hidden.weight, std=INIT_STD)
        nn.init.zeros_(self.value.hidden.bias)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens: LongTensor [B, 64] -> (policy_logits [B, 1880], value_logits [B, 128])."""
        hidden = self.trunk(tokens)
        return self.policy(hidden), self.value(hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parameter_report(model: BlinkNet) -> dict[str, int]:
    """Total, GAB-lite and non-GAB parameter counts (README Table 1 shows non-GAB / total)."""
    total = count_parameters(model)
    gab = 0 if model.trunk.gab is None else count_parameters(model.trunk.gab)
    return {"total": total, "gab": gab, "non_gab": total - gab, "blocks": count_parameters(model.trunk.blocks)}
