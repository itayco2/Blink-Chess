"""`blink export quantize --int8`: dynamic per-channel int8 for the single-thread WASM path (P10, PF30)."""

import hashlib
import json

import pytest

from blink import cli


def _opsets(model) -> dict[str, int]:
    return {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}


@pytest.fixture(scope="module")
def fp32_export(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    out = tmp_path_factory.mktemp("export") / "stand-in" / "model.onnx"
    assert cli.main(["export", "onnx", "--model", "stand-in", "--out", str(out)]) == 0
    return out


@pytest.fixture(scope="module")
def int8_export(fp32_export):
    assert cli.main(["export", "quantize", "--int8", "--model", str(fp32_export.parent)]) == 0
    return fp32_export.parent / "int8" / "model.onnx"


@pytest.mark.torch
def test_the_int8_file_is_one_self_contained_opset_20_file_under_a_third_of_the_fp32_size(
    fp32_export, int8_export
):
    import onnx

    model = onnx.load(str(int8_export), load_external_data=False)
    assert _opsets(model)["ai.onnx"] == 20
    assert all(t.data_location != onnx.TensorProto.EXTERNAL for t in model.graph.initializer)
    assert [o.name for o in model.graph.output] == ["policy_logits", "value_logits"]
    assert int8_export.stat().st_size < fp32_export.stat().st_size / 3
    assert sorted(p.name for p in int8_export.parent.iterdir()) == ["model.json", "model.onnx"]


@pytest.mark.torch
def test_quantization_is_dynamic_with_one_weight_scale_per_output_channel(int8_export):
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(int8_export), load_external_data=False)
    ops = {node.op_type for node in model.graph.node}
    assert {"DynamicQuantizeLinear", "MatMulInteger"} <= ops, "activations are quantized on the fly"
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    weights = [node.input[1] for node in model.graph.node if node.op_type == "MatMulInteger"]
    quantized = [name for name in weights if name in inits]
    assert quantized and all(inits[name].dtype.name == "int8" for name in quantized)
    scales = [inits[name.replace("_quantized", "_scale")] for name in quantized]
    assert all(scale.size == inits[name].shape[1] > 1 for scale, name in zip(scales, quantized, strict=True))


@pytest.mark.torch
def test_the_int8_card_names_its_file_hash_precision_and_the_fp32_it_came_from(fp32_export, int8_export):
    card = json.loads(int8_export.with_name("model.json").read_text(encoding="utf-8"))
    fp32_card = json.loads(fp32_export.with_name("model.json").read_text(encoding="utf-8"))
    assert card["file"] == "model.onnx"
    assert card["precision"] == "int8"
    assert card["bytes"] == int8_export.stat().st_size
    assert card["sha256"] == hashlib.sha256(int8_export.read_bytes()).hexdigest()
    assert card["quantization"]["fp32_sha256"] == fp32_card["sha256"]
    assert card["quantization"]["fp32_bytes"] == fp32_card["bytes"]
    assert card["parameters"] == fp32_card["parameters"]
    assert card["label"] == "one look, int8 WASM"


def test_quant_pre_process_runs_first_on_the_fp32_file_and_quantize_reads_its_output(monkeypatch, tmp_path):
    from onnxruntime import quantization
    from onnxruntime.quantization import shape_inference

    from blink.export import onnx as export_onnx
    from blink.export import quantize

    calls = []
    monkeypatch.setattr(
        shape_inference, "quant_pre_process", lambda src, dst, **kw: calls.append(("pre", src, dst))
    )

    def fake_quantize(src, dst, **kwargs):
        calls.append(("quantize", src, dst, kwargs["per_channel"], kwargs["weight_type"]))
        open(dst, "wb").close()

    monkeypatch.setattr(quantization, "quantize_dynamic", fake_quantize)
    monkeypatch.setattr(export_onnx, "check_self_contained", lambda path: 20)
    fp32 = tmp_path / "model.onnx"
    fp32.write_bytes(b"fp32")
    out = quantize.quantize_int8(fp32, tmp_path / "int8" / "model.onnx")
    assert out.is_file()
    assert [c[0] for c in calls] == ["pre", "quantize"]
    assert calls[0][1] == str(fp32) and calls[1][1] == calls[0][2], "quantize reads the pre-processed file"
    assert calls[1][3] is True and calls[1][4] == quantization.QuantType.QInt8
    assert sorted(p.name for p in out.parent.iterdir()) == ["model.onnx"], "no temporary file is left"


def test_a_failed_quantization_leaves_no_int8_file(monkeypatch, tmp_path):
    from onnxruntime import quantization
    from onnxruntime.quantization import shape_inference

    from blink.export import quantize

    monkeypatch.setattr(shape_inference, "quant_pre_process", lambda src, dst, **kw: open(dst, "wb").close())

    def broken(src, dst, **kwargs):
        open(dst, "wb").close()
        raise RuntimeError("unsupported op")

    monkeypatch.setattr(quantization, "quantize_dynamic", broken)
    fp32 = tmp_path / "model.onnx"
    fp32.write_bytes(b"fp32")
    with pytest.raises(RuntimeError, match="unsupported op"):
        quantize.quantize_int8(fp32, tmp_path / "int8" / "model.onnx")
    assert list((tmp_path / "int8").iterdir()) == []


def test_an_export_directory_comes_from_a_path_or_a_selector(tmp_path, monkeypatch):
    from blink.export import quantize

    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "model.onnx").write_bytes(b"")
    assert quantize.resolve_export_dir(str(tmp_path / "dir")) == tmp_path / "dir"
    assert quantize.resolve_export_dir(str(tmp_path / "dir" / "model.onnx")) == tmp_path / "dir"
    assert quantize.resolve_export_dir("run:skeleton:ema") == tmp_path / "home" / "export" / "skeleton-ema"
    assert quantize.resolve_export_dir("ship") == tmp_path / "home" / "export" / "ship"
    assert quantize.int8_path(tmp_path / "dir") == tmp_path / "dir" / "int8" / "model.onnx"


def test_quantize_names_the_scheme_explicitly_and_refuses_a_missing_fp32_file(tmp_path, capsys):
    assert cli.main(["export", "quantize", "--model", str(tmp_path)]) == 2
    assert "--int8" in capsys.readouterr().err
    assert cli.main(["export", "quantize", "--int8", "--model", str(tmp_path)]) == 2
    assert "blink export onnx" in capsys.readouterr().err
