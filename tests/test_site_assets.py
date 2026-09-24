"""What the browser page ships: vendored code, the piece sprite, and how it loads onnxruntime-web (P10)."""

import json
import re
import xml.etree.ElementTree as ET

import pytest

SVG_NS = "{http://www.w3.org/2000/svg}"
PIECE_IDS = {f"{color}{piece}" for color in "wb" for piece in "kqrbnp"}
PERMISSIVE = {
    "MIT": "Permission is hereby granted, free of charge",
    "BSD": "Redistribution and use in source and binary forms",
    "Apache-2.0": "Apache License",
    "OFL-1.1": "SIL OPEN FONT LICENSE",
}
# Built from pieces so that this file itself never trips a licence scanner.
BANNED_TEXT = [
    *("CC " + "BY-SA", "CC " + "BY-NC", "Share" + "Alike", "Non" + "Commercial", "GNU " + "General Public"),
    *("GNU " + "Affero", "A" + "GPL", "GPL" + "-3", "GPL" + "v", "Creative Commons " + "Attribution-Share"),
]
BANNED_FILES = {"standard.svg", "staunty.svg"}
OTHER_ORT_ENTRIES = (
    *("ort.min.mjs", "ort.bundle.min.mjs", "ort.all", "ort.webgpu", "ort.webgl", "ort.jspi", "webgpu"),
)


def _page_sources(repo_root):
    site = repo_root / "site"
    return {p: p.read_text(encoding="utf-8") for p in [*site.glob("*.js"), *site.glob("*.html")]}


def test_vendored_files_carry_permissive_licences(repo_root):
    vendor = repo_root / "site" / "vendor"
    packages = [p for p in vendor.iterdir() if p.is_dir()]
    assert packages, "site/vendor holds no vendored package"
    for package in packages:
        licence = (package / "LICENSE").read_text(encoding="utf-8")
        assert any(marker in licence for marker in PERMISSIVE.values()), package.name
    shipped = [p for p in (repo_root / "site").rglob("*") if p.is_file() and "node_modules" not in p.parts]
    assert not [p.name for p in shipped if p.name in BANNED_FILES]
    for path in (p for p in vendor.rglob("*") if p.is_file()):
        assert path.stat().st_size < 512 * 1024, path.name
        text = path.read_text(encoding="utf-8")
        hits = [banned for banned in BANNED_TEXT if banned.lower() in text.lower()]
        assert hits == [], f"{path.relative_to(vendor)}: {hits}"


def test_the_piece_set_elects_the_bsd_3_clause_option_with_the_cburnett_attribution(repo_root):
    pieces = repo_root / "site" / "assets" / "pieces"
    licence = (pieces / "LICENSE-pieces.txt").read_text(encoding="utf-8")
    assert "BSD 3-Clause" in licence
    assert "Cburnett" in licence
    assert PERMISSIVE["BSD"] in licence
    for path in pieces.iterdir():
        text = path.read_text(encoding="utf-8")
        assert [b for b in BANNED_TEXT if b.lower() in text.lower()] == [], path.name


def test_the_sprite_holds_the_twelve_cburnett_pieces_under_cm_chessboard_ids(repo_root):
    root = ET.parse(repo_root / "site" / "assets" / "pieces" / "cburnett.svg").getroot()
    ids = [el.get("id") for el in root.iter() if el.get("id")]
    assert sorted(ids) == sorted(PIECE_IDS)
    assert root.get("viewBox") == "0 0 45 45"


def test_build_sprite_wraps_each_piece_in_a_group_and_drops_inner_ids():
    from blink.site import pieces

    svg = '<svg xmlns="http://www.w3.org/2000/svg"><g id="inner"><path d="M0 0"/></g></svg>'
    sprite = pieces.build_sprite({piece_id: svg for piece_id in PIECE_IDS})
    root = ET.fromstring(sprite)
    groups = root.findall(f"{SVG_NS}g")
    assert sorted(g.get("id") for g in groups) == sorted(PIECE_IDS)
    assert [el.get("id") for g in groups for el in g.iter() if el is not g and el.get("id")] == []
    assert "Cburnett" in sprite


def test_build_sprite_refuses_a_missing_piece():
    from blink.site import pieces

    svg = '<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
    with pytest.raises(ValueError, match="bq"):
        pieces.build_sprite({piece_id: svg for piece_id in PIECE_IDS - {"bq"}})


def test_the_page_uses_the_onnxruntime_web_wasm_entry_with_one_thread(repo_root):
    from blink.site import serve

    worker = (repo_root / "site" / "worker.js").read_text(encoding="utf-8")
    entry = f"vendor/ort/{serve.ORT_WASM_ENTRY}"
    assert re.search(rf'import \* as ort from "\./{re.escape(entry)}"', worker)
    assert re.search(r"ort\.env\.wasm\.numThreads\s*=\s*1\b", worker)
    assert 'executionProviders: ["wasm"]' in worker
    assert re.search(r'new Worker\([^)]*type:\s*"module"', (repo_root / "site" / "app.js").read_text("utf-8"))
    for path, text in _page_sources(repo_root).items():
        assert [e for e in OTHER_ORT_ENTRIES if e in text] == [], path.name


def test_the_wasm_entry_is_what_the_npm_package_exports_as_onnxruntime_web_wasm(repo_root):
    from blink.site import serve

    package = repo_root / "site" / "node_modules" / "onnxruntime-web" / "package.json"
    if not package.is_file():
        pytest.skip("site/node_modules is absent (run npm ci --prefix site)")
    exports = json.loads(package.read_text(encoding="utf-8"))["exports"]["./wasm"]["import"]["default"]
    assert exports == f"./dist/{serve.ORT_WASM_ENTRY}"


def test_the_page_sets_the_pieces_file_to_the_cburnett_sprite(repo_root):
    app = (repo_root / "site" / "app.js").read_text(encoding="utf-8")
    assert re.search(r'file:\s*"assets/pieces/cburnett\.svg"', app)
    assert re.search(r"tileSize:\s*45\b", app)
    for text in _page_sources(repo_root).values():
        assert "standard.svg" not in text and "staunty.svg" not in text


def test_the_page_loads_the_model_from_models_model_onnx(repo_root):
    sources = "\n".join(_page_sources(repo_root).values())
    assert "models/model.onnx" in sources


def test_vendor_replaces_machine_punctuation_with_ascii():
    from blink.site import vendor

    text = "see `https://" + chr(0x2026) + "` " + chr(0x2014) + " done " + chr(0x201C) + "x" + chr(0x201D)
    assert vendor.to_ascii_punctuation(text) == 'see `https://...` - done "x"'


def test_vendored_cm_chessboard_matches_npm_except_ascii_punctuation(repo_root):
    from blink.site import vendor

    source = repo_root / "site" / "node_modules" / "cm-chessboard"
    if not source.is_dir():
        pytest.skip("site/node_modules is absent (run npm ci --prefix site)")
    target = repo_root / "site" / "vendor" / "cm-chessboard"
    for name in vendor.CM_CHESSBOARD_FILES:
        expected = vendor.to_ascii_punctuation((source / "src" / name).read_text(encoding="utf-8"))
        assert (target / name).read_text(encoding="utf-8") == expected, name
    licence = (source / "LICENSE").read_text(encoding="utf-8")
    assert (target / "LICENSE").read_text(encoding="utf-8") == licence
