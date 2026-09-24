import pytest

torch = pytest.importorskip("torch")

from train_helpers import tiny_model_config  # noqa: E402

from blink.model import gab  # noqa: E402
from blink.model.config import ModelConfig  # noqa: E402
from blink.model.transformer import BlinkNet, count_parameters  # noqa: E402

pytestmark = pytest.mark.torch

S = ModelConfig(d_model=256, n_layers=8, n_heads=8, head_dim=32, gab=True)
M = ModelConfig(d_model=512, n_layers=10, n_heads=16, head_dim=32, gab=True)


def _tokens(batch: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 16, (batch, 64), generator=torch.Generator().manual_seed(seed))


def _randomise(module: torch.nn.Module, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.05)


def test_gab_bias_shape_is_batch_heads_64_64():
    module = gab.GabLite(d_model=64, n_heads=2)
    bias = module(torch.randn(3, 64, 64))
    assert bias.shape == (3, 2, 64, 64)


def test_gab_lite_adds_at_most_15_percent_of_trunk_params_at_s_and_m():
    """PF34: 598,016 = 14.3% of the S trunk and 671,744 = 3.2% of the M trunk (8 d^2 per block)."""
    for cfg, expected, share in ((S, 598_016, 14.3), (M, 671_744, 3.2)):
        model = BlinkNet(cfg)
        added = count_parameters(model.trunk.gab)
        blocks = count_parameters(model.trunk.blocks)
        matrices = 8 * cfg.d_model**2 * cfg.n_layers
        assert added == expected == gab.parameter_count(cfg.d_model, cfg.n_heads)
        assert added / blocks <= 0.15
        assert round(100 * added / matrices, 1) == share


def test_gab_is_one_module_computed_once_and_shared_by_every_layer():
    cfg = tiny_model_config(n_layers=3, gab=True)
    model = BlinkNet(cfg)
    _randomise(model)
    calls, seen = [], []
    model.trunk.gab.register_forward_hook(lambda mod, args, out: calls.append(out))
    for block in model.trunk.blocks:
        block.attn.register_forward_pre_hook(lambda mod, args: seen.append(args[1]))
    model(_tokens(2))
    assert len(calls) == 1 and len(seen) == 3
    assert all(bias is calls[0] for bias in seen)


def test_a_fresh_gab_adds_a_zero_bias_so_the_net_starts_like_the_plain_trunk():
    torch.manual_seed(0)
    plain = BlinkNet(tiny_model_config(gab=False))
    with_gab = BlinkNet(tiny_model_config(gab=True))
    with_gab.load_state_dict(plain.state_dict(), strict=False)
    _randomise(plain.policy)
    with_gab.policy.load_state_dict(plain.policy.state_dict())
    tokens = _tokens(4)
    assert torch.count_nonzero(with_gab.trunk.gab(with_gab.trunk.embed(tokens))) == 0
    torch.testing.assert_close(with_gab(tokens)[0], plain(tokens)[0])


def test_the_bias_changes_attention_once_the_generator_has_learned():
    model = BlinkNet(tiny_model_config(gab=True))
    _randomise(model)
    tokens = _tokens(2)
    before = model(tokens)[0]
    with torch.no_grad():
        model.trunk.gab.generator.weight.mul_(3.0)
    assert not torch.allclose(before, model(tokens)[0])


def test_every_gab_weight_gets_a_gradient_after_one_update():
    model = BlinkNet(tiny_model_config(gab=True))
    _randomise(model.policy)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(2):
        optimizer.zero_grad()
        policy, value = model(_tokens(8))
        (policy.logsumexp(-1).mean() - policy[:, 0].mean() + value.pow(2).mean()).backward()
        optimizer.step()
    for name, p in model.trunk.gab.named_parameters():
        assert p.grad is not None and torch.count_nonzero(p.grad) > 0, name


@pytest.mark.cuda
def test_the_gab_bias_runs_through_bf16_sdpa_on_cuda_with_gradients():
    model = BlinkNet(tiny_model_config(gab=True)).cuda()
    _randomise(model)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        policy, value = model(_tokens(16).cuda())
    (policy.float().logsumexp(-1).mean() + value.float().mean()).backward()
    assert torch.count_nonzero(model.trunk.gab.generator.weight.grad) > 0
