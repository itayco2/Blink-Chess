"""ONNX export: one self-contained opset-20 file that reproduces torch (P1 UCI-and-export, P10)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from torch import nn  # noqa: E402

from blink.board import moves, value  # noqa: E402

pytestmark = pytest.mark.torch


class TinyNet(nn.Module):
    """A stand-in with BlinkNet's signature: tokens [B, 64] -> ([B, 1880], [B, 128])."""

    def __init__(self, d: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.embed = nn.Embedding(16, d)
        self.pos = nn.Parameter(torch.randn(64, d) * 0.1)
        self.qkv = nn.Linear(d, 3 * d)
        self.norm = nn.LayerNorm(d)
        self.policy = nn.Linear(d, moves.NUM_MOVES)
        self.value = nn.Linear(d, value.NUM_BINS)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(tokens) + self.pos
        b, s, d = x.shape
        q, k, v = self.qkv(x).view(b, s, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = nn.functional.scaled_dot_product_attention(q, k, v)
        x = self.norm(x + attn.transpose(1, 2).reshape(b, s, d))
        pooled = x.mean(dim=1)
        return self.policy(pooled), self.value(pooled)


@pytest.fixture(scope="module")
def tiny() -> nn.Module:
    torch.manual_seed(0)
    return TinyNet().eval()


@pytest.fixture(scope="module")
def exported(tiny, tmp_path_factory):
    from blink.export import onnx as export_onnx

    out = tmp_path_factory.mktemp("onnx") / "model.onnx"
    export_onnx.export(tiny, out)
    return out


def test_export_is_one_self_contained_file(exported):
    model = onnx.load(str(exported), load_external_data=False)
    opsets = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
    assert opsets["ai.onnx"] == 20
    assert [p.name for p in exported.parent.iterdir()] == ["model.onnx"]
    assert all(t.data_location != onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
    assert all(len(t.external_data) == 0 for t in model.graph.initializer)


def test_an_export_that_dies_mid_write_leaves_the_previous_model_in_place(tmp_path):
    from blink.export import onnx as export_onnx

    out = tmp_path / "model.onnx"
    out.write_bytes(b"previous model")

    def killed_mid_write(path):
        path.write_bytes(b"half a model")
        raise RuntimeError("the exporter died")

    with pytest.raises(RuntimeError, match="died"):
        export_onnx.write_checked(out, killed_mid_write)
    assert out.read_bytes() == b"previous model"
    assert [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_a_written_file_that_fails_the_check_never_replaces_the_destination(tmp_path):
    from blink.export import onnx as export_onnx

    out = tmp_path / "model.onnx"
    out.write_bytes(b"previous model")
    with pytest.raises(export_onnx.ExportError, match="not a readable ONNX file"):
        export_onnx.write_checked(out, lambda path: path.write_bytes(b"not an onnx file"))
    assert out.read_bytes() == b"previous model"
    assert [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_a_checked_export_replaces_the_destination_in_one_step(tiny, exported, tmp_path):
    from blink.export import onnx as export_onnx

    out = tmp_path / "model.onnx"
    out.write_bytes(b"previous model")
    export_onnx.write_checked(out, lambda path: path.write_bytes(exported.read_bytes()))
    assert out.read_bytes() == exported.read_bytes()
    assert [p.name for p in tmp_path.iterdir()] == ["model.onnx"]


def test_the_exported_graph_takes_int64_tokens_with_a_dynamic_batch(exported):
    model = onnx.load(str(exported), load_external_data=False)
    (tokens,) = model.graph.input
    assert tokens.name == "tokens"
    assert tokens.type.tensor_type.elem_type == onnx.TensorProto.INT64
    batch, squares = tokens.type.tensor_type.shape.dim
    assert batch.dim_param and not batch.HasField("dim_value")
    assert squares.dim_value == 64
    shapes = {
        o.name: [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim] for o in model.graph.output
    }
    assert shapes == {
        "policy_logits": [batch.dim_param, moves.NUM_MOVES],
        "value_logits": [batch.dim_param, value.NUM_BINS],
    }


def test_onnx_matches_torch_within_1e_4(tiny, exported):
    from blink.export import onnx as export_onnx
    from blink.export import positions

    tokens = positions.encode_fens(positions.random_fens(1_000, seed=7))
    assert tokens.shape == (1_000, 64)
    report = export_onnx.compare(tiny, exported, tokens, batch_size=333)
    assert report.positions == 1_000
    assert report.max_abs_policy <= 1e-4
    assert report.max_abs_value <= 1e-4


def test_the_onnx_evaluator_returns_value_probabilities_that_sum_to_one(exported):
    from blink.export import evaluators, positions

    codes = positions.encode_fens(positions.random_fens(5, seed=1)).astype(np.uint8)
    evaluation = evaluators.OnnxEvaluator(exported).evaluate(codes)
    assert evaluation.policy_logits.shape == (5, moves.NUM_MOVES)
    np.testing.assert_allclose(evaluation.value_probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.all((evaluation.win_probability() > 0) & (evaluation.win_probability() < 1))


def test_the_packaged_stand_in_has_the_blinknet_signature():
    from blink.export import standin

    net = standin.build(seed=0)
    policy, values = net(torch.zeros(3, 64, dtype=torch.long))
    assert policy.shape == (3, moves.NUM_MOVES)
    assert values.shape == (3, value.NUM_BINS)


def test_the_stand_in_is_identical_for_the_same_seed():
    from blink.export import standin

    a, b = standin.build(seed=3), standin.build(seed=3)
    for (name, pa), (_, pb) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert torch.equal(pa, pb), name


def test_an_unknown_selector_names_the_interface_it_needs(monkeypatch):
    from blink.export import models

    def fake_import(name):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setattr(models.importlib, "import_module", fake_import)
    with pytest.raises(models.ModelUnavailable, match="blink.model.loading"):
        models.load_module("run:skeleton")
