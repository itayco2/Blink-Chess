"""`blink site stage --out DIR`: the tree GitHub Pages serves, from the same file table as `site serve`.

chess.js and onnxruntime-web come from npm (pinned in site/package-lock.json) and are never tracked;
the deploy copies exactly the files layout.NPM_FILES lists, so pages.yml cannot forget one (P1, P10).
"""

import re
from pathlib import Path

import pytest
from test_site_serve import _fake_tree

from blink import cli

RELATIVE_IMPORT = re.compile(r"""(?:\bfrom\s+|\bimport\s*\(\s*|\bimport\s+)["'](\.{1,2}/[^"']+)["']""")
WORKER = re.compile(r"""new Worker\(\s*["']([^"']+)["']""")
HTML_ASSET = re.compile(r"""<(?:script|link)\b[^>]*\b(?:src|href)="([^":#]+)\"""")


def _tree(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_stage_copies_the_page_and_only_the_listed_npm_files(tmp_path):
    from blink.site import stage

    site, _, _ = _fake_tree(tmp_path)
    out = tmp_path / "_site"
    stage.stage(site, out)
    assert _tree(out) == {
        "index.html",
        "app.js",
        "vocab.json",
        "vendor/ort/ort.wasm.bundle.min.mjs",
        "vendor/ort/ort-wasm-simd-threaded.wasm",
        "vendor/ort/ort-wasm-simd-threaded.mjs",
        "vendor/chess.js/chess.js",
        "vendor/chess.js/LICENSE",
    }
    assert (out / "vendor" / "chess.js" / "chess.js").read_text(encoding="utf-8") == "export class Chess {}"


def test_stage_puts_the_model_and_its_card_under_models(tmp_path):
    from blink.site import stage

    site, model_dir, _ = _fake_tree(tmp_path)
    (model_dir / "model.json").write_text('{"selector": "stand-in"}', encoding="utf-8")
    out = tmp_path / "_site"
    stage.stage(site, out, model=model_dir)
    assert (out / "models" / "model.onnx").read_bytes() == b"onnx-bytes"
    assert (out / "models" / "model.json").read_text(encoding="utf-8") == '{"selector": "stand-in"}'


def test_stage_refuses_a_missing_npm_file_and_says_to_run_npm_ci(tmp_path):
    from blink.site import stage

    site, _, _ = _fake_tree(tmp_path)
    (site / "node_modules" / "chess.js" / "dist" / "esm" / "chess.js").unlink()
    with pytest.raises(stage.StageError, match=r"chess\.js.*npm ci --prefix site"):
        stage.stage(site, tmp_path / "_site")
    assert not (tmp_path / "_site").exists()


def test_stage_refuses_an_output_directory_that_is_not_empty_or_inside_site(tmp_path):
    from blink.site import stage

    site, _, _ = _fake_tree(tmp_path)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(stage.StageError, match="not empty"):
        stage.stage(site, busy)
    with pytest.raises(stage.StageError, match="inside"):
        stage.stage(site, site / "_site")
    assert (busy / "keep.txt").read_text(encoding="utf-8") == "mine"


def test_serve_and_stage_publish_the_same_npm_files(tmp_path):
    from blink.site import layout, serve, stage

    site, model_dir, _ = _fake_tree(tmp_path)
    out = tmp_path / "_site"
    stage.stage(site, out)
    staged = {path for path in _tree(out) if path.startswith("vendor/")}
    assert staged == set(layout.npm_sources(site))
    cfg = serve.SiteConfig(site_dir=site, model=serve.resolve_model(model_dir))
    for published, source in layout.npm_sources(site).items():
        assert serve.resolve_request(cfg, "/" + published) == source


def test_site_stage_writes_the_tree_and_reports_it(tmp_path, capsys, monkeypatch):
    from blink.commands import site as site_command

    site, model_dir, _ = _fake_tree(tmp_path)
    monkeypatch.setattr(site_command, "SITE_DIR", site)
    out = tmp_path / "_site"
    assert cli.main(["site", "stage", "--out", str(out), "--model", str(model_dir)]) == 0
    assert "10 files" in capsys.readouterr().out
    assert cli.main(["site", "stage", "--out", str(out)]) == 2
    assert "not empty" in capsys.readouterr().err


def _missing_targets(root: Path) -> list[str]:
    """Every relative module, worker or HTML asset a staged file points at that the tree lacks."""
    missing = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".js", ".mjs", ".html"} or path.parent.name == "ort":
            continue  # ORT's minified bundle finds its wasm through wasmPaths, which worker.js sets
        text = path.read_text(encoding="utf-8")
        targets = RELATIVE_IMPORT.findall(text) + WORKER.findall(text) + HTML_ASSET.findall(text)
        missing += [f"{path.name} -> {t}" for t in targets if not (path.parent / t).resolve().is_file()]
    return missing


def test_every_module_the_real_page_imports_is_in_the_staged_tree(repo_root, tmp_path):
    from blink.site import stage

    if not (repo_root / "site" / "node_modules" / "chess.js").is_dir():
        pytest.skip("site/node_modules is absent (run npm ci --prefix site)")
    out = tmp_path / "_site"
    stage.stage(repo_root / "site", out)
    assert _missing_targets(out) == []
    assert {"index.html", "app.js", "rules.js", "worker.js", "vendor/chess.js/chess.js"} <= _tree(out)
    assert not {path for path in _tree(out) if path.startswith(("tests/", "node_modules/", "package"))}


def test_the_import_scan_notices_a_module_the_tree_lacks(tmp_path):
    source = 'import { Chess } from "./vendor/chess.js/chess.js";\n'
    (tmp_path / "app.js").write_text(source, encoding="utf-8")
    assert _missing_targets(tmp_path) == ["app.js -> ./vendor/chess.js/chess.js"]
