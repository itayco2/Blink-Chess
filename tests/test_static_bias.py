"""Arm a06: a learned static 64 x 64 attention bias per head, in GAB-lite's place (plan P5)."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_mixed_training import children_from  # noqa: E402
from train_helpers import fixture_records, tiny_model_config, tiny_train_config  # noqa: E402

from blink.model import static_bias  # noqa: E402
from blink.model.config import ModelConfig, TrainConfig, config_from_dict, read_tables  # noqa: E402
from blink.model.transformer import BlinkNet, count_parameters, parameter_report  # noqa: E402
from blink.train import loop, sweep  # noqa: E402
from blink.train.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from blink.train.source import InMemorySource, mixed_source  # noqa: E402

pytestmark = pytest.mark.torch
REPO = Path(__file__).resolve().parents[1]
S = ModelConfig(d_model=256, n_layers=8, n_heads=8, head_dim=32, gab=False, static_bias=True)
WORLD = "0123456789ab"


def _tokens(batch: int, seed: int = 0) -> torch.Tensor:
    return torch.randint(0, 16, (batch, 64), generator=torch.Generator().manual_seed(seed))


def _randomise(module: torch.nn.Module, seed: int = 1) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * 0.05)


def _bf16_step(forward, tokens: torch.Tensor) -> None:
    """One bf16 forward and backward, as the training step runs it on CUDA."""
    with torch.autocast(tokens.device.type, dtype=torch.bfloat16):
        policy, value = forward(tokens)
    (policy.float().logsumexp(-1).mean() + value.float().mean()).backward()


def _bias_grad(forward, model: BlinkNet, tokens: torch.Tensor) -> torch.Tensor:
    model.zero_grad()
    _bf16_step(forward, tokens)
    return model.trunk.static_bias.bias.grad.clone()


def _a06_train(run_dir: Path, device: str = "cpu", **overrides) -> loop.TrainResult:
    """A tiny a06 run on mixed roots and children, micro-batched as the P5 arms train."""
    settings = {
        "steps": 12,
        "warmup_steps": 3,
        "batch_size": 16,
        "child_frac": 0.25,
        "micro_batch": 8,
        "metrics_every": 4,
        "eval_every": 1000,
        "ckpt_every_steps": 1000,
        **overrides,
    }
    cfg = tiny_train_config(model=tiny_model_config(gab=False, static_bias=True), **settings)
    records = fixture_records()
    roots = InMemorySource(records[:48], 12, seed=3).batches
    children = InMemorySource(children_from(records[48:]), 4, seed=4).batches
    spec = loop.RunSpec(run_dir=run_dir, world=WORLD, device=device)
    return loop.train(cfg, spec, mixed_source(roots, children), val=records, log=lambda _: None)


def test_static_bias_is_off_by_default():
    assert ModelConfig().static_bias is False
    assert BlinkNet(tiny_model_config()).trunk.static_bias is None


def test_gab_and_the_static_bias_together_are_refused():
    with pytest.raises(ValueError, match="gab.*static_bias"):
        ModelConfig(gab=True, static_bias=True)
    with pytest.raises(ValueError, match="gab.*static_bias"):
        config_from_dict({"model": {"gab": True, "static_bias": True}})


def test_the_static_bias_is_one_64_by_64_matrix_per_head_broadcast_over_the_batch():
    model = BlinkNet(tiny_model_config(n_heads=2, static_bias=True))
    assert model.trunk.static_bias.bias.shape == (2, 64, 64)
    bias = model.trunk.static_bias()
    assert bias.shape == (1, 2, 64, 64)
    assert torch.broadcast_shapes(bias.shape, (5, 2, 64, 64)) == (5, 2, 64, 64)


def test_the_static_bias_holds_heads_times_4096_parameters_under_its_own_report_key():
    model = BlinkNet(S)
    added = count_parameters(model.trunk.static_bias)
    assert added == static_bias.parameter_count(S.n_heads) == 8 * 64 * 64 == 32_768
    report = parameter_report(model)
    assert report["static_bias"] == added and report["gab"] == 0
    assert report["total"] == count_parameters(model)
    plain = BlinkNet(dataclasses.replace(S, static_bias=False))
    assert report["total"] - added == count_parameters(plain)
    assert parameter_report(plain)["static_bias"] == 0


def test_a_fresh_static_bias_is_zero_so_the_net_starts_like_the_plain_trunk():
    torch.manual_seed(0)
    plain = BlinkNet(tiny_model_config())
    torch.manual_seed(0)
    biased = BlinkNet(tiny_model_config(static_bias=True))
    assert torch.count_nonzero(biased.trunk.static_bias.bias) == 0
    for name, tensor in plain.state_dict().items():
        assert torch.equal(tensor, biased.state_dict()[name]), name
    _randomise(plain.policy)
    biased.policy.load_state_dict(plain.policy.state_dict())
    tokens = _tokens(4)
    torch.testing.assert_close(biased(tokens)[0], plain(tokens)[0])


def test_without_either_bias_the_model_is_todays_model():
    """gab = false and static_bias = false: the same parameters, keys and outputs as before a06 existed."""
    torch.manual_seed(0)
    model = BlinkNet(tiny_model_config(gab=False, static_bias=False))
    assert model.trunk.gab is None and model.trunk.static_bias is None
    assert not any("static_bias" in name or "gab" in name for name in model.state_dict())
    assert model.trunk.attention_bias(model.trunk.embed(_tokens(2))) is None


def test_the_static_bias_is_computed_once_and_shared_by_every_layer():
    model = BlinkNet(tiny_model_config(n_layers=3, static_bias=True))
    _randomise(model)
    calls, seen = [], []
    model.trunk.static_bias.register_forward_hook(lambda mod, args, out: calls.append(out))
    for block in model.trunk.blocks:
        block.attn.register_forward_pre_hook(lambda mod, args: seen.append(args[1]))
    model(_tokens(2))
    assert len(calls) == 1 and len(seen) == 3
    assert all(bias is calls[0] for bias in seen)


def test_the_static_bias_is_added_to_the_attention_logits():
    model = BlinkNet(tiny_model_config(static_bias=True))
    _randomise(model)
    tokens = _tokens(2)
    x = model.trunk.embed(tokens)
    block = model.trunk.blocks[0]
    attn, bias = block.attn, model.trunk.static_bias()
    b, t, _ = x.shape
    q, k, v = attn.qkv(block.attn_norm(x)).view(b, t, 3, attn.n_heads, attn.head_dim).unbind(2)
    q, k, v = attn.q_norm(q).transpose(1, 2), attn.k_norm(k).transpose(1, 2), v.transpose(1, 2)
    logits = q @ k.transpose(-1, -2) / attn.head_dim**0.5 + bias
    expected = attn.out((logits.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1))
    torch.testing.assert_close(attn(block.attn_norm(x), bias), expected, rtol=1e-5, atol=1e-5)


def test_a_learned_static_bias_changes_the_outputs():
    model = BlinkNet(tiny_model_config(static_bias=True))
    _randomise(model)
    tokens = _tokens(2)
    before = model(tokens)[0]
    with torch.no_grad():
        model.trunk.static_bias.bias.mul_(3.0)
    assert not torch.allclose(before, model(tokens)[0])


def test_the_static_bias_gets_a_gradient_and_learns():
    model = BlinkNet(tiny_model_config(static_bias=True))
    _randomise(model.policy)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(2):
        optimizer.zero_grad()
        policy, value = model(_tokens(8))
        (policy.logsumexp(-1).mean() - policy[:, 0].mean() + value.pow(2).mean()).backward()
        optimizer.step()
    grad = model.trunk.static_bias.bias.grad
    assert grad is not None and torch.count_nonzero(grad) > 0
    assert torch.count_nonzero(model.trunk.static_bias.bias) > 0


def test_the_a06_arm_loads_over_the_recipe_at_s():
    arm = sweep.load_arm(REPO / "configs" / "ablations" / "a06.toml")
    merged = sweep.merged_config(read_tables(REPO / "configs" / "s.toml"), arm, steps=10_000)
    sweep.validate(merged)
    cfg = config_from_dict({**merged["train"], "model": merged["model"]})
    assert (cfg.model.gab, cfg.model.static_bias) == (False, True)
    assert isinstance(cfg, TrainConfig)


def test_a_tiny_a06_run_trains_on_cpu_and_reports_its_bias(tmp_path):
    result = _a06_train(tmp_path / "a06")
    assert result.step == 12
    assert np.isfinite(result.last_metrics["loss_policy"]) and np.isfinite(result.last_metrics["loss_value"])
    saved = json.loads((tmp_path / "a06" / "config.json").read_text(encoding="utf-8"))
    assert saved["parameter_report"]["static_bias"] == static_bias.parameter_count(2) > 0
    assert saved["config"]["model"]["static_bias"] is True


def test_the_broadcast_static_bias_compiles_as_one_graph_with_eagers_gradient_under_bf16_autocast():
    """P5 trains every arm compiled (train.compile = "inductor") in bf16, and the static bias is the only
    attention bias that reaches SDPA as [1, H, 64, 64] with a stride-0 batch dimension. Inductor needs a
    C++ or CUDA toolchain, so on CPU this traces the same graph through Dynamo and AOTAutograd (backend
    aot_eager): no graph break, a gradient summed over the batch, and eager's value. The inductor
    kernels themselves are the cuda tests' job."""
    model = BlinkNet(tiny_model_config(static_bias=True))
    _randomise(model)
    tokens = _tokens(8)
    eager = _bias_grad(model, model, tokens)
    compiled = _bias_grad(torch.compile(model, backend="aot_eager", fullgraph=True), model, tokens)
    assert compiled.shape == (2, 64, 64) and torch.count_nonzero(compiled) > 0
    assert (compiled - eager).abs().max() <= 0.02 * eager.abs().max()


@pytest.mark.cuda
def test_the_static_bias_runs_through_bf16_sdpa_on_cuda_with_gradients():
    model = BlinkNet(tiny_model_config(static_bias=True)).cuda()
    _randomise(model)
    _bf16_step(model, _tokens(16).cuda())
    assert torch.count_nonzero(model.trunk.static_bias.bias.grad) > 0


def _sdpa_kernels(model: BlinkNet) -> set[str]:
    """The aten SDPA kernels one bf16 training step dispatches to: a fused one or the math fallback."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities) as profile:
        _bf16_step(model, _tokens(16).cuda())
    return {e.key for e in profile.key_averages() if e.key.startswith("aten::_scaled_dot_product_")}


@pytest.mark.cuda
def test_the_static_bias_keeps_attention_on_gab_lites_sdpa_kernels_on_cuda():
    """supervise stops an arm that trains 15% under its bench rate, which Recipe D measured with GAB-lite's
    dense [B, H, 64, 64] bias. A broadcast bias that sent SDPA to its math fallback would trip that rule
    and waste the arm's retries, so a06 must dispatch to the same kernels as GAB-lite."""
    static = _sdpa_kernels(BlinkNet(tiny_model_config(static_bias=True)).cuda())
    gab = _sdpa_kernels(BlinkNet(tiny_model_config(gab=True)).cuda())
    assert static and static == gab, (static, gab)


@pytest.mark.cuda
def test_an_a06_run_trains_compiled_by_inductor_in_bf16_on_cuda_as_p5_runs_it(tmp_path):
    """The path P5 takes (compile = "inductor", bf16, micro-batches), which must pass before a06 leaves
    configs/ablations/plan.toml's held list: the compiled run trains to finite losses within 2% of eager
    and learns the bias. The bias starts at zero and decaying zero leaves it zero, so a non-zero bias in
    the checkpoint means it got non-zero gradients."""
    losses = {}
    for mode in ("off", "inductor"):
        run_dir = tmp_path / mode
        result = _a06_train(run_dir, "cuda", steps=30, warmup_steps=5, compile=mode)
        metrics_path = run_dir / "metrics.jsonl"
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        assert result.step == 30 and rows
        assert all(np.isfinite(row["loss_policy"]) and np.isfinite(row["loss_value"]) for row in rows)
        state = load_checkpoint(latest_checkpoint(run_dir))
        assert torch.count_nonzero(state["model"]["trunk.static_bias.bias"]) > 0
        assert not any("_orig_mod" in name for name in state["model"])
        losses[mode] = rows[-1]["loss_policy"] + rows[-1]["loss_value"]
    assert abs(losses["inductor"] - losses["off"]) <= 0.02 * losses["off"], losses


def test_the_static_bias_is_decayed_like_the_gab_generator():
    model = BlinkNet(tiny_model_config(static_bias=True))
    decay, no_decay = loop.build_optimizer(model, tiny_train_config(), "cpu").param_groups
    assert any(p is model.trunk.static_bias.bias for p in decay["params"])
    assert decay["weight_decay"] > 0 and not any(
        p is model.trunk.static_bias.bias for p in no_decay["params"]
    )
