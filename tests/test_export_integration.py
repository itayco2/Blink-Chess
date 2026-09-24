"""Export must load a trained Blink model through the train area's loader (found only when the areas met)."""

import dataclasses

import pytest

torch = pytest.importorskip("torch")

from blink.export import models  # noqa: E402
from blink.model.config import ModelConfig  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402


@pytest.mark.torch
def test_export_loads_a_trained_blink_model_from_a_weights_file(tmp_path):
    cfg = ModelConfig(d_model=64, n_layers=1, n_heads=2, head_dim=32, ffn_mult=2, gab=False)
    net = BlinkNet(cfg).eval()
    path = tmp_path / "blink.pt"
    torch.save({"config": dataclasses.asdict(cfg), "model": net.state_dict()}, path)

    module = models.load_module(str(path))

    tokens = torch.randint(0, 16, (3, 64))
    with torch.no_grad():
        want_policy, want_value = net(tokens)
        got_policy, got_value = module(tokens)
    assert isinstance(module, BlinkNet) and not module.training
    assert torch.equal(want_policy, got_policy) and torch.equal(want_value, got_value)
