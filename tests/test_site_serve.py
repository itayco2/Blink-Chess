"""`blink site serve`: loopback only, the page plus its npm files and the model mapped in (P1, P10)."""

import argparse
import inspect
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from blink import cli


def _fake_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    site = tmp_path / "site"
    (site / "tests").mkdir(parents=True)
    (site / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (site / "app.js").write_text("export {}", encoding="utf-8")
    (site / "vocab.json").write_text("{}", encoding="utf-8")
    ort = site / "node_modules" / "onnxruntime-web" / "dist"
    ort.mkdir(parents=True)
    (ort / "ort.wasm.bundle.min.mjs").write_text("export const ort = 1", encoding="utf-8")
    (ort / "ort-wasm-simd-threaded.wasm").write_bytes(b"\0asm")
    (ort / "ort-wasm-simd-threaded.mjs").write_text("export default 1", encoding="utf-8")
    (ort / "ort.all.min.mjs").write_text("export const everything = 1", encoding="utf-8")
    esm = site / "node_modules" / "chess.js" / "dist" / "esm"
    esm.mkdir(parents=True)
    (esm / "chess.js").write_text("export class Chess {}", encoding="utf-8")
    (esm / "chess.js.map").write_text("{}", encoding="utf-8")
    (site / "node_modules" / "chess.js" / "LICENSE").write_text("BSD 2-Clause", encoding="utf-8")
    (site / "package.json").write_text("{}", encoding="utf-8")
    (site / "tests" / "golden.json").write_text("{}", encoding="utf-8")
    model_dir = tmp_path / "export" / "stand-in"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"onnx-bytes")
    (tmp_path / "secret.txt").write_text("outside the site", encoding="utf-8")
    return site, model_dir, tmp_path / "secret.txt"


@pytest.fixture
def served(tmp_path):
    from blink.site import serve

    site, model_dir, _ = _fake_tree(tmp_path)
    server = serve.make_server(serve.SiteConfig(site_dir=site, model=serve.resolve_model(model_dir)), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _get(url: str) -> tuple[int, str, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, "", b""


def test_site_serve_binds_loopback_only(tmp_path):
    from blink.site import serve

    site, model_dir, _ = _fake_tree(tmp_path)
    server = serve.make_server(serve.SiteConfig(site_dir=site, model=serve.resolve_model(model_dir)), port=0)
    try:
        assert server.server_address[0] == "127.0.0.1"
    finally:
        server.server_close()
    assert serve.HOST == "127.0.0.1"
    assert not {"host", "bind", "address"} & set(inspect.signature(serve.make_server).parameters)
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["site", "serve", "--model", str(model_dir), "--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["site", "serve", "--model", str(model_dir), "--bind", "0.0.0.0"])


def test_site_serve_parses_a_model_and_a_port():
    args = cli.build_parser().parse_args(["site", "serve", "--model", "D:/blink/export/x", "--port", "8123"])
    assert isinstance(args, argparse.Namespace)
    assert (args.model, args.port) == ("D:/blink/export/x", 8123)


def test_the_page_vendor_ort_and_the_model_are_served_with_their_types(served):
    status, kind, body = _get(f"{served}/")
    assert (status, kind.split(";")[0]) == (200, "text/html")
    assert _get(f"{served}/app.js")[1].startswith("text/javascript")
    status, kind, _ = _get(f"{served}/vendor/ort/ort.wasm.bundle.min.mjs")
    assert (status, kind.split(";")[0]) == (200, "text/javascript")
    assert _get(f"{served}/vendor/ort/ort-wasm-simd-threaded.wasm")[:2] == (200, "application/wasm")
    assert _get(f"{served}/vendor/chess.js/chess.js")[2] == b"export class Chess {}"
    status, kind, body = _get(f"{served}/models/model.onnx")
    assert (status, kind, body) == (200, "application/octet-stream", b"onnx-bytes")
    assert _get(f"{served}/vocab.json")[1].startswith("application/json")


def test_only_the_npm_files_the_page_deploys_are_served(served):
    assert _get(f"{served}/vendor/chess.js/LICENSE")[2] == b"BSD 2-Clause"
    assert _get(f"{served}/vendor/ort/ort-wasm-simd-threaded.mjs")[0] == 200
    for path in ("/vendor/ort/ort.all.min.mjs", "/vendor/chess.js/chess.js.map", "/vendor/ort/"):
        assert _get(served + path)[0] == 404, path


def test_files_that_are_never_deployed_are_never_served(served):
    for path in ("/package.json", "/tests/golden.json", "/node_modules", "/models/"):
        assert _get(served + path)[0] == 404, path


def test_a_missing_model_card_is_served_as_a_minimal_one_so_the_page_logs_no_404(served):
    status, kind, body = _get(f"{served}/models/model.json")
    assert (status, kind.split(";")[0]) == (200, "application/json")
    assert b'"file": "model.onnx"' in body


def test_paths_outside_the_mapped_roots_are_refused(served):
    for path in (
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/vendor/ort/../../../../secret.txt",
        "/vendor/ort/..%2f..%2fpackage.json",
        "/models/../secret.txt",
        "/models/other.onnx",
        "/node_modules/onnxruntime-web/dist/ort.wasm.bundle.min.mjs",
        "/index.html%00.txt",
        "/C:/Windows/win.ini",
    ):
        assert _get(served + path)[0] == 404, path


def test_serve_with_a_missing_model_exits_with_a_message_not_a_traceback(tmp_path, capsys):
    assert cli.main(["site", "serve", "--model", str(tmp_path / "nowhere")]) == 2
    assert "no ONNX model" in capsys.readouterr().err


def test_resolve_model_accepts_a_file_or_a_directory_and_refuses_anything_else(tmp_path):
    from blink.site import serve

    _, model_dir, _ = _fake_tree(tmp_path)
    assert serve.resolve_model(model_dir) == model_dir / "model.onnx"
    assert serve.resolve_model(model_dir / "model.onnx") == model_dir / "model.onnx"
    with pytest.raises(FileNotFoundError):
        serve.resolve_model(tmp_path / "nowhere")
