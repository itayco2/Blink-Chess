"""Arm a10 (P5): torch.optim.Muon on the trunk's hidden matrices, AdamW on everything else."""

import inspect
import json
import re

import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_model_config, tiny_train_config  # noqa: E402

from blink.model.config import MUON_ADJUST_LR_FNS  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train import bench, loop, optimizers  # noqa: E402
from blink.train.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from blink.train.schedule import wsd_lr  # noqa: E402
from blink.train.source import InMemorySource  # noqa: E402

pytestmark = pytest.mark.torch
WORLD = "0123456789ab"
HIDDEN = ("attn.qkv.weight", "attn.out.weight", "ffn_in.weight", "ffn_out.weight")


def _model(n_layers: int = 2, gab: bool = True) -> BlinkNet:
    return BlinkNet(tiny_model_config(n_layers=n_layers, gab=gab))


def _muon_config(**overrides):
    return tiny_train_config(optimizer="muon", **overrides)


def _spec(run_dir, **overrides) -> loop.RunSpec:
    return loop.RunSpec(**{"run_dir": run_dir, "world": WORLD, "device": "cpu", **overrides})


def _repeat(batch):
    def source(start_step: int):
        while True:
            yield batch

    return source


def test_the_adjust_lr_fns_blink_accepts_are_exactly_the_ones_the_installed_torch_muon_accepts():
    """The config is torch-free, so it keeps its own list; this pins it to torch.optim.Muon's check."""
    source = inspect.getsource(torch.optim.Muon.__init__)
    listed = re.search(r"adjust_lr_fn not in \[(.*?)\]", source, re.DOTALL)
    assert listed, "torch.optim.Muon no longer validates adjust_lr_fn as a list; re-read its source"
    assert set(re.findall(r'"(\w+)"', listed.group(1))) == set(MUON_ADJUST_LR_FNS)
    weight = torch.nn.Parameter(torch.zeros(4, 4))
    for name in MUON_ADJUST_LR_FNS:
        assert torch.optim.Muon([weight], adjust_lr_fn=name).param_groups[0]["adjust_lr_fn"] == name
    with pytest.raises(ValueError):
        torch.optim.Muon([weight], adjust_lr_fn="match_rms")


def test_muon_owns_exactly_the_attention_and_ffn_projections_of_every_block():
    model = _model(n_layers=2)
    split = optimizers.split_parameters(model)
    assert set(split.muon) == {f"trunk.blocks.{i}.{name}" for i in range(2) for name in HIDDEN}


def test_embeddings_gab_lite_norms_biases_and_both_heads_stay_on_adamw():
    split = optimizers.split_parameters(_model(n_layers=2))
    adamw = set(split.adamw_decay) | set(split.adamw_no_decay)
    for name in (
        "trunk.token_embedding.weight",
        "trunk.square_embedding",
        "trunk.gab.compress.weight",
        "trunk.gab.hidden.weight",
        "trunk.gab.heads.weight",
        "trunk.gab.generator.weight",
        "trunk.blocks.0.attn_norm.weight",
        "trunk.blocks.1.attn.q_norm.weight",
        "trunk.final_norm.weight",
        "policy.query.weight",
        "policy.key.weight",
        "policy.promotion.weight",
        "value.hidden.weight",
        "value.hidden.bias",
        "value.out.weight",
        "value.out.bias",
    ):
        assert name in adamw, f"{name} should be on AdamW"


def test_every_parameter_is_in_exactly_one_optimizer():
    model = _model(n_layers=2)
    optimizer = loop.build_optimizer(model, _muon_config(), "cpu")
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert sorted(owned) == sorted(id(p) for p in model.parameters())
    assert len(owned) == len(set(owned))
    muon_owned = {id(p) for group in optimizer.muon.param_groups for p in group["params"]}
    adamw_owned = {id(p) for group in optimizer.adamw.param_groups for p in group["params"]}
    assert muon_owned.isdisjoint(adamw_owned) and len(muon_owned) == 8


def test_muon_decays_its_matrices_like_adamw_and_adamw_keeps_decay_on_matrices_only():
    model = _model(n_layers=2)
    cfg = _muon_config(weight_decay=0.1, muon_adjust_lr_fn="match_rms_adamw")
    optimizer = loop.build_optimizer(model, cfg, "cpu")
    assert isinstance(optimizer.muon, torch.optim.Muon) and isinstance(optimizer.adamw, torch.optim.AdamW)
    for muon_group in optimizer.muon.param_groups:
        assert (muon_group["weight_decay"], muon_group["adjust_lr_fn"]) == (0.1, "match_rms_adamw")
    decay, no_decay = optimizer.adamw.param_groups
    assert decay["weight_decay"] == 0.1 and all(p.ndim >= 2 for p in decay["params"])
    assert no_decay["weight_decay"] == 0.0 and all(p.ndim < 2 for p in no_decay["params"])
    assert (decay["betas"], decay["lr"]) == ((cfg.beta1, cfg.beta2), cfg.peak_lr)


def test_the_configured_adjust_lr_fn_reaches_muon():
    cfg = _muon_config(muon_adjust_lr_fn="spectral_unclamped")
    optimizer = loop.build_optimizer(_model(), cfg, "cpu")
    assert {group["adjust_lr_fn"] for group in optimizer.muon.param_groups} == {"spectral_unclamped"}


def test_muon_takes_each_qkv_weight_as_q_k_and_v_and_every_other_matrix_whole():
    model = _model(n_layers=2)
    optimizer = loop.build_optimizer(model, _muon_config(), "cpu")
    names = {id(p): name for name, p in model.named_parameters()}
    blocks = {
        names[id(p)]: group[optimizers.ROW_BLOCKS]
        for group in optimizer.muon.param_groups
        for p in group["params"]
    }
    expected = {
        f"trunk.blocks.{i}.{name}": 3 if name == "attn.qkv.weight" else 1 for i in range(2) for name in HIDDEN
    }
    assert blocks == expected
    assert "Q, K and V separately" in optimizer.summary()


def test_the_three_row_blocks_of_a_qkv_weight_are_whole_query_key_and_value_projections():
    """The split relies on Attention reading qkv(x) as [3, heads, head_dim]: rows [0, d) are Q,
    [d, 2d) K and [2d, 3d) V. A per-head interleaved layout would fail every line below."""
    attn = _model(n_layers=1).trunk.blocks[0].attn
    x = torch.randn(2, 64, 64, generator=torch.Generator().manual_seed(0))

    def output_without(block: int) -> torch.Tensor:
        weight = attn.qkv.weight
        saved = weight.detach().clone()
        with torch.no_grad():
            weight.chunk(3)[block].zero_()
            y = attn(x)
            weight.copy_(saved)
        return y

    for block in (0, 1):  # no query or no key: every position attends uniformly, so all outputs agree
        y = output_without(block)
        assert torch.allclose(y, y[:, :1].expand_as(y), atol=1e-6)
    assert torch.equal(output_without(2), torch.zeros_like(x))  # no value: nothing to attend to


def _orthogonal(d: int, seed: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(seed)))
    return q


def test_q_k_and_v_each_get_a_full_orthogonal_update_when_one_gradient_dominates():
    """Keller Jordan's Muon post: Muon works better applied to Q, K and V separately than together.

    Orthogonalised as one [3d, d] matrix, a query gradient 100x the others takes almost the whole
    update and K and V barely move (singular values near 0.01). Split, each [d, d] block gets an
    orthogonal update of its own (Newton-Schulz puts singular values in roughly 0.5-1.5)."""
    model = _model(n_layers=1)
    cfg = _muon_config(weight_decay=0.0, peak_lr=1.0, muon_adjust_lr_fn="original")  # a [d, d] step of 1
    optimizer = loop.build_optimizer(model, cfg, "cpu")
    qkv = model.trunk.blocks[0].attn.qkv.weight
    d = qkv.shape[1]
    qkv.grad = torch.cat([100 * _orthogonal(d, 1), _orthogonal(d, 2), _orthogonal(d, 3)])
    before = qkv.detach().clone()
    optimizer.step()
    for name, block in zip("QKV", (before - qkv.detach()).chunk(3), strict=True):
        singular = torch.linalg.svdvals(block)
        assert singular.min() > 0.5 and singular.max() < 1.5, (
            f"{name}: {singular.min():.4f}-{singular.max():.4f}"
        )


def _muon_pair(params, reference_params, **settings):
    return optimizers.BlockwiseMuon(params, **settings), torch.optim.Muon(reference_params, **settings)


SETTINGS = {"lr": 0.02, "weight_decay": 0.1, "adjust_lr_fn": "match_rms_adamw"}


def test_a_qkv_weight_steps_bitwise_like_torch_muon_on_separate_q_k_and_v():
    d = 32
    generator = torch.Generator().manual_seed(0)
    start = torch.randn(3 * d, d, generator=generator)
    fused = torch.nn.Parameter(start.clone())
    separate = [torch.nn.Parameter(block.clone()) for block in start.chunk(3)]
    ours, reference = _muon_pair([{"params": [fused], optimizers.ROW_BLOCKS: 3}], separate, **SETTINGS)
    for _ in range(3):  # momentum and weight decay both carry across steps
        grad = torch.randn(3 * d, d, generator=generator)
        fused.grad = grad.clone()
        for param, block in zip(separate, grad.chunk(3), strict=True):
            param.grad = block.clone()
        ours.step()
        reference.step()
    assert torch.equal(fused.detach(), torch.cat([p.detach() for p in separate]))
    buffers = [reference.state[p]["momentum_buffer"] for p in separate]
    assert torch.equal(ours.state[fused]["momentum_buffer"], torch.cat(buffers))


@pytest.mark.parametrize("shape", [(64, 32), (32, 64), (32, 32)])
def test_a_matrix_taken_whole_steps_bitwise_like_torch_muon(shape):
    generator = torch.Generator().manual_seed(1)
    start = torch.randn(*shape, generator=generator)
    ours_param, reference_param = torch.nn.Parameter(start.clone()), torch.nn.Parameter(start.clone())
    ours, reference = _muon_pair([ours_param], [reference_param], **SETTINGS)  # no row_blocks: 1
    for _ in range(3):
        grad = torch.randn(*shape, generator=generator)
        ours_param.grad, reference_param.grad = grad.clone(), grad.clone()
        ours.step()
        reference.step()
    assert torch.equal(ours_param.detach(), reference_param.detach())


@pytest.mark.parametrize("blocks", [3, 0])
def test_a_row_block_count_that_does_not_split_a_matrix_evenly_is_refused(blocks):
    weight = torch.nn.Parameter(torch.zeros(10, 4))
    with pytest.raises(ValueError, match="row blocks"):
        optimizers.BlockwiseMuon([{"params": [weight], optimizers.ROW_BLOCKS: blocks}])


def test_the_default_optimizer_is_still_one_adamw_over_the_model_in_order():
    """Recipe D's optimizer, group for group: matrices with decay, then the rest without."""
    model = _model()
    optimizer = loop.build_optimizer(model, tiny_train_config(), "cpu")
    assert type(optimizer) is torch.optim.AdamW
    params = list(model.parameters())
    decay, no_decay = optimizer.param_groups
    assert [id(p) for p in decay["params"]] == [id(p) for p in params if p.ndim >= 2]
    assert [id(p) for p in no_decay["params"]] == [id(p) for p in params if p.ndim < 2]
    assert (decay["weight_decay"], no_decay["weight_decay"]) == (0.1, 0.0)


def test_a_model_without_hidden_matrices_is_refused_for_muon():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError, match="trunk.blocks"):
        loop.build_optimizer(model, _muon_config(), "cpu")


def test_the_pair_zeroes_and_steps_both_optimizers():
    model = _model()
    optimizer = loop.build_optimizer(model, _muon_config(weight_decay=0.0), "cpu")
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    model.trunk(torch.randint(0, 16, (4, 64))).square().mean().backward()  # the heads get no gradient
    optimizer.step()
    moved = {name for name, p in model.named_parameters() if not torch.equal(p, before[name])}
    assert {"trunk.blocks.0.attn.qkv.weight", "trunk.blocks.1.ffn_out.weight"} <= moved  # Muon
    assert {"trunk.square_embedding", "trunk.final_norm.weight"} <= moved  # AdamW
    assert not any(name.startswith(("policy.", "value.")) for name in moved)
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())


def test_a_state_dict_from_a_single_adamw_is_refused_by_the_pair():
    model = _model()
    adamw_state = loop.build_optimizer(model, tiny_train_config(), "cpu").state_dict()
    pair = loop.build_optimizer(model, _muon_config(), "cpu")
    with pytest.raises(ValueError, match="optimizer"):
        pair.load_state_dict(adamw_state)


def _checkpoint_lrs(state) -> set[float]:
    return {group["lr"] for part in ("muon", "adamw") for group in state["optimizer"][part]["param_groups"]}


def test_the_wsd_schedule_and_a_resume_lr_scale_drive_both_optimizers(tmp_path):
    records = fixture_records()
    cfg = _muon_config(steps=40, warmup_steps=20, batch_size=16, ckpt_every_steps=10)
    run_dir = tmp_path / "run"
    quiet = {"val": records, "log": lambda _: None}
    loop.train(cfg, _spec(run_dir, max_steps=10), InMemorySource(records, 16, seed=3).batches, **quiet)
    warm = wsd_lr(9, cfg.peak_lr, cfg.warmup_steps, cfg.steps, cfg.cooldown_frac)
    assert warm < cfg.peak_lr  # still in the warmup, so the schedule, not the default, set it
    assert _checkpoint_lrs(load_checkpoint(latest_checkpoint(run_dir))) == {warm}

    resumed = _spec(run_dir, resume=True, lr_scale=0.5, max_steps=20)
    loop.train(cfg, resumed, InMemorySource(records, 16, seed=3).batches, **quiet)
    scaled = 0.5 * wsd_lr(19, cfg.peak_lr, cfg.warmup_steps, cfg.steps, cfg.cooldown_frac)
    state = load_checkpoint(latest_checkpoint(run_dir))
    assert state["step"] == 20 and state["lr_scale"] == 0.5
    assert _checkpoint_lrs(state) == {scaled}


def _metrics(run_dir) -> list[dict]:
    return [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]


def test_a_tiny_muon_run_trains_and_its_losses_fall(tmp_path):
    records = fixture_records()[:16]
    cfg = _muon_config(steps=60, warmup_steps=5, batch_size=16, metrics_every=10, ckpt_every_steps=60)
    run_dir = tmp_path / "run"

    loop.train(cfg, _spec(run_dir), _repeat(records), val=records, log=lambda _: None)
    first, last = _metrics(run_dir)[0], _metrics(run_dir)[-1]
    assert last["step"] == 60
    assert last["loss_policy"] < first["loss_policy"] - 2.0
    assert last["loss_value"] < first["loss_value"] - 0.5
    state = load_checkpoint(latest_checkpoint(run_dir))
    assert set(state["optimizer"]) == {"muon", "adamw"}
    assert all("momentum_buffer" in slots for slots in state["optimizer"]["muon"]["state"].values())


def _state_after(cfg, run_dir, source, **spec_overrides):
    loop.train(cfg, _spec(run_dir, **spec_overrides), source, val=fixture_records(), log=lambda _: None)
    return load_checkpoint(latest_checkpoint(run_dir))


def _assert_same_optimizer_state(straight, resumed) -> None:
    for part in ("muon", "adamw"):
        slots_straight = straight["optimizer"][part]["state"]
        slots_resumed = resumed["optimizer"][part]["state"]
        assert slots_straight.keys() == slots_resumed.keys()
        for index, slots in slots_straight.items():
            for slot, tensor in slots.items():
                assert torch.equal(tensor, slots_resumed[index][slot]), f"{part} {index}.{slot} differs"


def test_a_muon_resume_after_30_steps_equals_60_straight_steps_bitwise_on_cpu(tmp_path):
    records = fixture_records()
    cfg = _muon_config(steps=60, ckpt_every_steps=30, warmup_steps=10, batch_size=16)
    straight = _state_after(cfg, tmp_path / "straight", InMemorySource(records, 16, seed=3).batches)

    split_dir = tmp_path / "split"
    first = _state_after(cfg, split_dir, InMemorySource(records, 16, seed=3).batches, max_steps=30)
    assert first["step"] == 30
    resumed = _state_after(cfg, split_dir, InMemorySource(records, 16, seed=3).batches, resume=True)

    assert resumed["step"] == straight["step"] == 60
    for key in ("model", "ema"):
        assert straight[key].keys() == resumed[key].keys()
        for name, tensor in straight[key].items():
            assert torch.equal(tensor, resumed[key][name]), f"{key}.{name} differs after resume"
    _assert_same_optimizer_state(straight, resumed)


def test_resuming_an_adamw_checkpoint_as_muon_is_refused(tmp_path):
    records = fixture_records()[:16]
    run_dir = tmp_path / "run"

    adamw = tiny_train_config(steps=20, warmup_steps=2, batch_size=16, ckpt_every_steps=10)
    loop.train(adamw, _spec(run_dir, max_steps=10), _repeat(records), val=None, log=lambda _: None)
    muon = _muon_config(steps=20, warmup_steps=2, batch_size=16, ckpt_every_steps=10)
    with pytest.raises(ValueError, match="optimizer"):
        loop.train(muon, _spec(run_dir, resume=True), _repeat(records), val=None, log=lambda _: None)


def test_a_checkpoint_from_before_the_optimizer_keys_resumes_without_a_config_warning(tmp_path):
    """A run started on older code (P5's frozen worktree) resumes on this code as the AdamW it was."""
    records = fixture_records()[:16]
    run_dir = tmp_path / "run"

    cfg = tiny_train_config(steps=20, warmup_steps=2, batch_size=16, ckpt_every_steps=10)
    loop.train(cfg, _spec(run_dir, max_steps=10), _repeat(records), val=None, log=lambda _: None)
    path = latest_checkpoint(run_dir)
    state = load_checkpoint(path)
    del state["config"]["optimizer"], state["config"]["muon_adjust_lr_fn"]
    torch.save(state, path)
    lines: list[str] = []
    result = loop.train(cfg, _spec(run_dir, resume=True), _repeat(records), val=None, log=lines.append)
    assert result.step == 20
    assert not [line for line in lines if "config differs" in line], lines

    changed = tiny_train_config(steps=20, warmup_steps=2, batch_size=16, ckpt_every_steps=10, seed=5)
    lines.clear()
    loop.train(changed, _spec(run_dir, resume=True), _repeat(records), val=None, log=lines.append)
    assert [line for line in lines if "config differs" in line and "['seed']" in line]


def test_the_throughput_bench_steps_a_muon_config(tmp_path):
    """The sweep turns an arm's hours into steps from a bench rate; a10 can be measured at its own."""
    config = tmp_path / "muon.toml"
    config.write_text(
        "[model]\nd_model = 64\nn_layers = 1\nn_heads = 2\nhead_dim = 32\n"
        '[train]\nbatch_size = 16\nsteps = 10\nwarmup_steps = 1\noptimizer = "muon"\n',
        encoding="utf-8",
    )
    spec = bench.ThroughputSpec("muon", config, micro=8, compile="off", steps=1, warmup=1, device="cpu")
    row = bench.measure_throughput(spec)
    assert row["error"] is None and row["samples_per_s"] > 0


@pytest.mark.cuda
def test_a_short_cuda_muon_run_trains_in_bf16_and_resumes(tmp_path):
    records = fixture_records()
    cfg = _muon_config(steps=40, warmup_steps=4, batch_size=32, ckpt_every_steps=20, eval_every=20)
    run_dir = tmp_path / "gpu"
    source = InMemorySource(records, 32, seed=1).batches
    quiet = {"val": records, "log": lambda _: None}
    first = loop.train(cfg, _spec(run_dir, device="cuda", max_steps=20), source, **quiet)
    assert first.step == 20
    done = loop.train(cfg, _spec(run_dir, device="cuda", resume=True), source, **quiet)
    assert done.step == 40 and done.last_eval["policy_ce"] < 7.54
    state = load_checkpoint(latest_checkpoint(run_dir))
    assert set(state["optimizer"]) == {"muon", "adamw"} and state["step"] == 40
