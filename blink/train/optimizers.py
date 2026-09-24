"""The trainer's optimizer: Recipe D's AdamW, or arm a10's Muon on the hidden matrices plus AdamW.

Muon (torch.optim.Muon) orthogonalises each update of a 2D weight with Newton-Schulz iterations. It is
meant for the hidden layers of a network; the input embedding, the output layer, norm gains and
biases belong on a standard optimizer (Keller Jordan's Muon post and torch's own docstring). In
BlinkNet the hidden matrices are the Linear weights inside `trunk.blocks`, four per block:

    trunk.blocks.<i>.attn.qkv.weight   [3d, d]   one fused matrix, orthogonalised as one
    trunk.blocks.<i>.attn.out.weight   [d, d]
    trunk.blocks.<i>.ffn_in.weight     [2d, d]
    trunk.blocks.<i>.ffn_out.weight    [d, 2d]

That is 4,194,304 of S's 5,048,448 parameters (83%). Everything else stays on AdamW with today's
settings (betas, decay on matrices only, fused on CUDA), for these reasons:

- trunk.token_embedding and trunk.square_embedding are the input embeddings: lookup tables whose
  rows are separate vectors, not a linear map, so orthogonalising their update has no meaning.
- trunk.gab.* (GAB-lite) turns the embeddings into an additive attention bias. Its generator starts
  at zero so the bias switches on slowly; an orthogonalised update has full spectral size from the
  first step whatever the gradient's size, which would switch it on at once. Keeping it (and a06's
  static bias, which replaces it) on AdamW also means a10 changes one thing: the trunk's matrices.
- policy.query and policy.key are the policy's output layer: their product is the logits, with no
  nonlinearity after it, and the query starts at zero. policy.promotion and value.out are output
  projections. value.hidden is a hidden layer in principle, but it is the value head's and small
  (d x d on one pooled vector per board); it stays with its head so both heads train as in Recipe D.
- Norm gains and biases are 1D, which Muon refuses anyway.

Muon's decoupled weight decay is param *= 1 - lr * wd, the same form as AdamW's, so the matrices it
owns decay at the recipe's weight_decay (0.1) exactly as they did under AdamW. Its momentum is
torch's default 0.95 with Nesterov (Keller Jordan's settings); beta1 stays AdamW's.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from blink.model.config import TrainConfig

HIDDEN_SCOPE = "trunk.blocks."  # every nn.Linear weight under here goes to Muon
MUON_PARTS = ("muon", "adamw")


def build_adamw(params: Sequence[nn.Parameter], cfg: TrainConfig, device_type: str) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (ndim >= 2); fused kernels on CUDA."""
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups, lr=cfg.peak_lr, betas=(cfg.beta1, cfg.beta2), fused=device_type == "cuda"
    )


@dataclass(frozen=True)
class ParameterSplit:
    """Parameter names by owner, in the model's own order."""

    muon: tuple[str, ...]
    adamw_decay: tuple[str, ...]
    adamw_no_decay: tuple[str, ...]


def hidden_matrix_names(model: nn.Module) -> frozenset[str]:
    """The weight of every nn.Linear inside trunk.blocks: the attention and FFN projections.

    Chosen by module type rather than by shape, so a 2D parameter that is not a projection (a bias
    table, say) never lands on Muon by accident."""
    return frozenset(
        f"{name}.weight"
        for name, module in model.named_modules()
        if name.startswith(HIDDEN_SCOPE) and isinstance(module, nn.Linear)
    )


def split_parameters(model: nn.Module) -> ParameterSplit:
    hidden = hidden_matrix_names(model)
    if not hidden:
        raise ValueError(f"optimizer 'muon' found no hidden matrices (nn.Linear under {HIDDEN_SCOPE!r})")
    named = list(model.named_parameters())
    return ParameterSplit(
        muon=tuple(name for name, _ in named if name in hidden),
        adamw_decay=tuple(name for name, p in named if name not in hidden and p.ndim >= 2),
        adamw_no_decay=tuple(name for name, p in named if name not in hidden and p.ndim < 2),
    )


@dataclass(frozen=True)
class MuonAdamW:
    """Muon and AdamW stepped as one optimizer, with the interface the trainer uses.

    `param_groups` lists both optimizers' own group dicts (the same objects), so the trainer's
    per-step `group["lr"] = lr` reaches both from the WSD schedule and any resume LR scale. The
    state dict nests each optimizer's own, so a checkpoint restores both and --resume, the NaN
    rollback and a preview branch continue exactly where the run left off.
    """

    muon: torch.optim.Muon
    adamw: torch.optim.AdamW

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return [*self.muon.param_groups, *self.adamw.param_groups]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.muon.step()
        self.adamw.step()

    def state_dict(self) -> dict[str, Any]:
        return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if sorted(state) != sorted(MUON_PARTS):
            raise ValueError(
                f"this optimizer state has parts {sorted(state)}, not Muon + AdamW {sorted(MUON_PARTS)}: "
                "the checkpoint was written with another train.optimizer; resume it with that one"
            )
        self.muon.load_state_dict(state["muon"])
        self.adamw.load_state_dict(state["adamw"])

    def summary(self) -> str:
        """One log line, so a run's log shows which optimizer owns what."""
        muon, adamw = _owned(self.muon), _owned(self.adamw)
        adjust = self.muon.param_groups[0]["adjust_lr_fn"]
        return (
            f"optimizer: Muon ({adjust}) on {len(muon)} hidden matrices ({_size(muon):,} parameters), "
            f"AdamW on {len(adamw)} others ({_size(adamw):,})"
        )


def _owned(optimizer: torch.optim.Optimizer) -> list[nn.Parameter]:
    return [p for group in optimizer.param_groups for p in group["params"]]


def _size(params: Sequence[nn.Parameter]) -> int:
    return sum(p.numel() for p in params)


def build_muon_adamw(model: nn.Module, cfg: TrainConfig, device_type: str) -> MuonAdamW:
    split = split_parameters(model)
    owned = set(split.muon)
    named = list(model.named_parameters())
    muon = torch.optim.Muon(
        [p for name, p in named if name in owned],
        lr=cfg.peak_lr,
        weight_decay=cfg.weight_decay,
        adjust_lr_fn=cfg.muon_adjust_lr_fn,
    )
    adamw = build_adamw([p for name, p in named if name not in owned], cfg, device_type)
    return MuonAdamW(muon, adamw)


def build_optimizer(model: nn.Module, cfg: TrainConfig, device_type: str) -> torch.optim.AdamW | MuonAdamW:
    """Recipe D's single AdamW over every parameter, or with optimizer "muon" the pair above."""
    if cfg.optimizer == "muon":
        return build_muon_adamw(model, cfg, device_type)
    return build_adamw(list(model.parameters()), cfg, device_type)
