"""`blink export onnx | vocab | golden` end to end, on the packaged stand-in."""

import hashlib
import json

import pytest

from blink import cli


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_export_vocab_writes_the_file_the_page_reads(tmp_path, capsys):
    from blink.export import vocab

    out = tmp_path / "vocab.json"
    assert cli.main(["export", "vocab", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == vocab.render(vocab.build())
    assert "ok:" in capsys.readouterr().out


def test_model_selectors_become_directory_names():
    from blink.export import models

    assert models.slug("stand-in") == "stand-in"
    assert models.slug("run:skeleton") == "skeleton"
    assert models.slug("run:skeleton:ema") == "skeleton-ema"
    assert models.slug("release:model-v1") == "model-v1"
    assert models.slug("D:/blink/runs/s/ckpt_000003000.pt") == "ckpt_000003000"
    assert models.slug("ship") == "ship"


def test_an_onnx_selector_is_a_file_or_a_directory_holding_model_onnx(tmp_path):
    from blink.export import models

    assert models.onnx_path(str(tmp_path / "x.onnx")) == tmp_path / "x.onnx"
    assert models.onnx_path(str(tmp_path)) is None
    (tmp_path / "model.onnx").write_bytes(b"")
    assert models.onnx_path(str(tmp_path)) == tmp_path / "model.onnx"
    assert models.onnx_path("run:skeleton") is None


@pytest.fixture(scope="module")
def stand_in_export(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    out = tmp_path_factory.mktemp("export") / "stand-in" / "model.onnx"
    assert cli.main(["export", "onnx", "--model", "stand-in", "--out", str(out)]) == 0
    return out


@pytest.mark.torch
def test_export_onnx_writes_the_model_and_a_card_that_records_parity(stand_in_export):
    card = json.loads(stand_in_export.with_name("model.json").read_text(encoding="utf-8"))
    assert card["selector"] == "stand-in"
    assert card["opset"] == 20
    assert card["parity_positions"] == 1_000
    assert max(card["max_abs_policy"], card["max_abs_value"]) <= 1e-4
    assert card["bytes"] == stand_in_export.stat().st_size
    assert card["sha256"] == _sha(stand_in_export)
    assert sorted(p.name for p in stand_in_export.parent.iterdir()) == ["model.json", "model.onnx"]


@pytest.mark.torch
def test_export_golden_from_an_onnx_file_records_that_file(stand_in_export, tmp_path):
    out = tmp_path / "golden.json"
    assert cli.main(["export", "golden", "--model", str(stand_in_export), "--out", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["model"] == {"selector": str(stand_in_export), "onnx_sha256": _sha(stand_in_export)}
    assert len(data["positions"]) == 50


@pytest.mark.torch
def test_golden_from_the_stand_in_and_from_its_onnx_agree(stand_in_export, tmp_path):
    torch_out, onnx_out = tmp_path / "torch.json", tmp_path / "onnx.json"
    onnx_flag = ["--onnx", str(stand_in_export)]
    assert cli.main(["export", "golden", "--model", "stand-in", *onnx_flag, "--out", str(torch_out)]) == 0
    assert cli.main(["export", "golden", "--model", str(stand_in_export), "--out", str(onnx_out)]) == 0
    a = json.loads(torch_out.read_text(encoding="utf-8"))["positions"]
    b = json.loads(onnx_out.read_text(encoding="utf-8"))["positions"]
    for x, y in zip(a, b, strict=True):
        assert x["top5"][0]["index"] == y["top5"][0]["index"], x["fen"]
        assert abs(x["win"] - y["win"]) < 1e-5
        for p, q in zip(x["top5"], y["top5"], strict=True):
            assert abs(p["prob"] - q["prob"]) < 1e-4


def test_golden_refuses_a_missing_onnx_file(tmp_path):
    from blink.export import models

    with pytest.raises(models.ModelUnavailable, match="no ONNX file"):
        models.load_evaluator(str(tmp_path / "missing.onnx"))
