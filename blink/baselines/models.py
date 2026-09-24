"""The learned baseline rungs: linear 768 -> 1 and MLP 768-512-256-1, a sigmoid win% for the mover.

Both read the 768 piece-square bits of blink.baselines.features, built here on the device from square
codes so training never materialises a [N, 768] float array. Weights files hold plain tensors and
values only and load with weights_only=True, like every Blink weights file.
"""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from blink.baselines import features
from blink.train.atomic import replace_with_retry

KINDS = ("linear", "mlp")
HIDDEN = (512, 256)
FORMAT = "blink-baseline-v1"


def unpack_torch(packed: torch.Tensor) -> torch.Tensor:
    """uint8 [N, 32] packed boards -> int64 [N, 64] codes (even square in the low nibble)."""
    wide = packed.long()
    return torch.stack((wide & 0x0F, wide >> 4), dim=-1).reshape(packed.shape[0], 64)


def features_torch(codes: torch.Tensor) -> torch.Tensor:
    """float32 [N, 768] bits from int64 [N, 64] codes: the same bits as features.features."""
    table = torch.as_tensor(features.PLANE_OF_CODE, device=codes.device)
    planes = table[codes] + 1  # 0 marks "no plane"
    one_hot = nn.functional.one_hot(planes, features.NUM_PLANES + 1)[..., 1:]
    return one_hot.permute(0, 2, 1).reshape(codes.shape[0], features.NUM_FEATURES).float()


class LinearBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(features.NUM_FEATURES, 1)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return self.linear(bits).squeeze(-1)


class MlpBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        widths = (features.NUM_FEATURES, *HIDDEN)
        layers: list[nn.Module] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:], strict=True):
            layers += [nn.Linear(fan_in, fan_out), nn.ReLU()]
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(HIDDEN[-1], 1)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(bits)).squeeze(-1)


def build(kind: str) -> nn.Module:
    if kind == "linear":
        return LinearBaseline()
    if kind == "mlp":
        return MlpBaseline()
    raise ValueError(f"baseline kind must be one of {KINDS}, got {kind!r}")


def logits_from_codes(model: nn.Module, codes: torch.Tensor) -> torch.Tensor:
    return model(features_torch(codes))


def save(path: Path, model: nn.Module, kind: str, meta: dict[str, Any]) -> None:
    """Write {"format", "kind", "model", "meta"} via a temp file, so a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format": FORMAT, "kind": kind, "model": model.state_dict(), "meta": dict(meta)}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    replace_with_retry(tmp, path)


def load(path: Path, device: str = "cpu") -> tuple[nn.Module, str, dict[str, Any]]:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if state.get("format") != FORMAT:
        raise ValueError(f"{path} is not a Blink baseline file (format {state.get('format')!r})")
    model = build(state["kind"])
    model.load_state_dict(state["model"])
    return model.to(torch.device(device)).eval(), state["kind"], state["meta"]
