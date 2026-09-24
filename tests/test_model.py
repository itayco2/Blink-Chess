import math

import pytest

torch = pytest.importorskip("torch")

from train_helpers import tiny_model_config  # noqa: E402

from blink.board import moves  # noqa: E402
from blink.model.config import ModelConfig, load_config  # noqa: E402
from blink.model.transformer import BlinkNet, count_parameters  # noqa: E402

pytestmark = pytest.mark.torch


def _random_tokens(batch: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 16, (batch, 64), generator=generator)


def _randomise_heads(model: BlinkNet, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.1)


def test_attention_policy_head_outputs_1880_logits():
    model = BlinkNet(tiny_model_config())
    policy, value = model(_random_tokens(3))
    assert policy.shape == (3, moves.NUM_MOVES) == (3, 1880)
    assert value.shape == (3, 128)


def test_zero_initialised_heads_start_with_uniform_outputs():
    model = BlinkNet(tiny_model_config())
    policy, value = model(_random_tokens(4))
    assert torch.count_nonzero(policy) == 0
    assert torch.count_nonzero(value) == 0


def test_a_from_to_logit_is_the_from_query_dotted_with_the_to_key():
    model = BlinkNet(tiny_model_config())
    _randomise_heads(model)
    tokens = _random_tokens(2)
    policy, _ = model(tokens)
    hidden = model.trunk(tokens)
    q, k = model.policy.query(hidden), model.policy.key(hidden)
    scale = 1.0 / math.sqrt(q.shape[-1])
    for index in (0, 500, moves.NUM_FROM_TO - 1):
        frm, to = moves.FROM_TO[index]
        expected = (q[:, frm] * k[:, to]).sum(-1) * scale
        torch.testing.assert_close(policy[:, index], expected, rtol=1e-5, atol=1e-5)


def test_a_promotion_logit_is_its_pair_logit_plus_a_per_piece_bias_from_the_to_key():
    model = BlinkNet(tiny_model_config())
    _randomise_heads(model)
    tokens = _random_tokens(2)
    policy, _ = model(tokens)
    hidden = model.trunk(tokens)
    q, k = model.policy.query(hidden), model.policy.key(hidden)
    scale = 1.0 / math.sqrt(q.shape[-1])
    bias = model.policy.promotion(k)
    for pair_index, (frm, to) in enumerate(moves.PROMO_PAIRS):
        pair_logit = (q[:, frm] * k[:, to]).sum(-1) * scale
        for piece in range(len(moves.PROMO_PIECES)):
            index = moves.NUM_FROM_TO + pair_index * len(moves.PROMO_PIECES) + piece
            expected = pair_logit + bias[:, to, piece]
            torch.testing.assert_close(policy[:, index], expected, rtol=1e-5, atol=1e-5)


def test_rows_do_not_leak_into_each_other_across_the_batch():
    model = BlinkNet(tiny_model_config())
    _randomise_heads(model)
    tokens = _random_tokens(4)
    policy, value = model(tokens)
    single_policy, single_value = model(tokens[2:3])
    torch.testing.assert_close(policy[2:3], single_policy, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(value[2:3], single_value, rtol=1e-5, atol=1e-5)


def test_the_head_count_must_tile_the_model_width():
    with pytest.raises(ValueError):
        ModelConfig(d_model=100, n_heads=4, head_dim=32)


def test_the_skeleton_config_is_about_0_4m_parameters(repo_root):
    cfg = load_config(repo_root / "configs" / "t.toml")
    assert (cfg.model.d_model, cfg.model.n_layers, cfg.model.n_heads, cfg.model.gab) == (128, 2, 4, False)
    assert 300_000 <= count_parameters(BlinkNet(cfg.model)) <= 500_000
