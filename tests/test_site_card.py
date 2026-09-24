"""The browser model's card, models/model.json: what the page's model card shows (P10)."""

import hashlib
import json

import pytest


def _fp32_card() -> dict:
    return {
        "selector": "run:skeleton:ema",
        "file": "model.onnx",
        "bytes": 1_576_222,
        "sha256": "0e" * 32,
        "opset": 20,
        "parameters": 339_456,
    }


def test_the_browser_card_names_the_int8_file_its_hash_and_the_fp32_it_came_from(tmp_path):
    from blink.site import card

    model = tmp_path / "model.onnx"
    model.write_bytes(b"int8 weights")
    made = card.for_file(_fp32_card(), model, precision="int8", method="dynamic")
    assert made["file"] == "model.onnx"
    assert made["bytes"] == len(b"int8 weights")
    assert made["sha256"] == hashlib.sha256(b"int8 weights").hexdigest()
    assert made["precision"] == "int8" and made["label"] == "one look, int8 WASM"
    assert made["mode"] == "policy" and made["backend"] == "wasm" and made["threads"] == 1
    assert made["name"] == "Blink skeleton-ema"
    assert made["parameters"] == 339_456 and made["selector"] == "run:skeleton:ema"
    assert made["quantization"] == {
        "method": "dynamic",
        "fp32_bytes": 1_576_222,
        "fp32_sha256": "0e" * 32,
        "gate": None,
    }


def test_an_fp32_browser_card_has_no_quantization_block(tmp_path):
    from blink.site import card

    model = tmp_path / "model.onnx"
    model.write_bytes(b"fp32")
    made = card.for_file(_fp32_card(), model, precision="fp32")
    assert made["label"] == "one look, fp32 WASM"
    assert made["quantization"] is None


def test_an_unknown_precision_is_refused(tmp_path):
    from blink.site import card

    with pytest.raises(ValueError, match="fp16"):
        card.browser_card(_fp32_card(), 1, "ab", precision="fp16")


def test_the_gate_numbers_land_under_quantization_gate_and_nothing_else_changes():
    from blink.export import qgate
    from blink.site import card

    base = card.browser_card(_fp32_card(), 400_000, "ab" * 32, precision="int8", method="dynamic")
    gated = card.with_gate(base, qgate.example().to_dict())
    assert gated["quantization"]["gate"] == qgate.example().to_dict()
    assert {k: v for k, v in gated.items() if k != "quantization"} == {
        k: v for k, v in base.items() if k != "quantization"
    }
    assert base["quantization"]["gate"] is None, "with_gate returns a new card"


def test_a_card_without_a_quantization_block_cannot_take_gate_numbers():
    from blink.export import qgate
    from blink.site import card

    fp32 = card.browser_card(_fp32_card(), 1, "ab", precision="fp32")
    with pytest.raises(ValueError, match="int8"):
        card.with_gate(fp32, qgate.example().to_dict())


def test_cards_round_trip_through_their_file(tmp_path):
    from blink.site import card

    path = card.write(card.example(), tmp_path / "int8" / "model.json")
    assert card.read(path) == card.example()
    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_the_tracked_card_fixture_is_current(repo_root):
    from blink.site import card

    tracked = repo_root / "site" / "tests" / "card.json"
    assert tracked.read_text(encoding="utf-8") == card.render(card.example())
    assert json.loads(tracked.read_text(encoding="utf-8"))["quantization"]["gate"]["passed"] is True
