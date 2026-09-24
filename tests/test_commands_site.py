"""`blink site pieces | vendor | smoke` and the train-area model selectors, without a browser."""

import types
import xml.etree.ElementTree as ET

from blink import cli

SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="45" height="45"><path id="x" d="M0 0h45"/></svg>'


def test_site_pieces_builds_the_sprite_from_the_twelve_commons_files(tmp_path, capsys):
    from blink.site import pieces

    src = tmp_path / "commons"
    src.mkdir()
    for name in pieces.source_names().values():
        (src / name).write_text(SVG, encoding="utf-8")
    out = tmp_path / "cburnett.svg"
    assert cli.main(["site", "pieces", "--src", str(src), "--out", str(out)]) == 0
    ids = {el.get("id") for el in ET.parse(out).getroot().iter() if el.get("id")}
    assert ids == set(pieces.source_names())
    assert "12 pieces" in capsys.readouterr().out


def test_vendoring_copies_the_nine_core_modules_and_the_licence_with_ascii_punctuation(tmp_path):
    from blink.site import vendor

    package = tmp_path / "cm-chessboard"
    for name in vendor.CM_CHESSBOARD_FILES:
        (package / "src" / name).parent.mkdir(parents=True, exist_ok=True)
        (package / "src" / name).write_text(f"// {name} " + chr(0x2014) + " core\n", encoding="utf-8")
    (package / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    out = tmp_path / "vendor" / "cm-chessboard"
    written = vendor.vendor_cm_chessboard(package, out)
    assert len(written) == len(vendor.CM_CHESSBOARD_FILES) + 1
    view = (out / "view" / "ChessboardView.js").read_text(encoding="utf-8")
    assert view == "// view/ChessboardView.js - core\n"
    assert (out / "LICENSE").read_text(encoding="utf-8") == "MIT License\n"


def _report(**overrides):
    from blink.site import smoke

    fields = dict(
        url="u",
        loaded=True,
        user_moves=10,
        legal_replies=10,
        illegal=(),
        arrows=3,
        min_arrows=3,
        console_errors=(),
        timings_ms=(5.0,),
        games=1,
        load_seconds=1.0,
    )
    return smoke.SmokeReport(**{**fields, **overrides})


def test_site_smoke_exits_zero_on_a_clean_report_and_one_on_a_missed_criterion(monkeypatch, capsys):
    from blink.site import smoke

    monkeypatch.setattr(smoke, "run", lambda url, moves, seed, timeout_s: _report())
    assert cli.main(["site", "smoke", "--url", "http://127.0.0.1:8000/"]) == 0
    assert "ok: 10 legal replies, 3 arrows, 0 console errors" in capsys.readouterr().out
    monkeypatch.setattr(smoke, "run", lambda url, moves, seed, timeout_s: _report(console_errors=("boom",)))
    assert cli.main(["site", "smoke"]) == 1
    assert "FAIL: 1 console error: boom" in capsys.readouterr().out


def _fake_loading(monkeypatch, loading):
    from blink.export import models

    monkeypatch.setattr(models.importlib, "import_module", lambda name: loading)
    return models


class _Net:
    def __init__(self):
        self.evaluated = False

    def eval(self):
        self.evaluated = True
        return self


def test_load_module_asks_the_train_areas_load_model_for_a_cpu_module(monkeypatch):
    """The real interface, known once the areas met: blink.model.loading.load_model(selector, device)."""
    net = _Net()
    calls = []

    def load_model(selector, device):
        calls.append((selector, device))
        return net

    models = _fake_loading(monkeypatch, types.SimpleNamespace(load_model=load_model))
    assert models.load_module("run:skeleton") is net
    assert calls == [("run:skeleton", "cpu")]
    assert net.evaluated


def test_golden_uses_the_train_areas_evaluator_for_a_train_selector(monkeypatch):
    sentinel = object()
    loading = types.SimpleNamespace(load_evaluator=lambda selector, device: (selector, device, sentinel))
    models = _fake_loading(monkeypatch, loading)
    assert models.load_evaluator("run:skeleton:ema") == ("run:skeleton:ema", "cpu", sentinel)
