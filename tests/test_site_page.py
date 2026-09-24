"""The page's P10 parts: model cache, value histogram, backend, model card, rating widget, selftest."""

import json
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "site"
PROPERTY = re.compile(r"\b(?:card|quant|gate|overall|band)\.([a-z0-9_]+)\b")


def _text(name: str) -> str:
    return (SITE / name).read_text(encoding="utf-8")


def _keys(data) -> set[str]:
    out = set()
    if isinstance(data, dict):
        for key, value in data.items():
            out.add(key)
            out |= _keys(value)
    return out


def test_the_rating_widget_uses_the_results_schema_publish_rule():
    from blink.report import results_schema

    text = _text("rating.js")
    assert re.search(rf"export const MIN_GAMES = {results_schema.PUBLISH_MIN_GAMES};", text)
    assert re.search(rf"export const MAX_RD = {results_schema.PUBLISH_MAX_RD};", text)
    assert 'export const ACCRUING = "rating accruing";' in text
    assert "stats.games >= MIN_GAMES && stats.rd < MAX_RD" in text


def test_the_page_reads_the_bot_name_from_config_json_and_hard_codes_none():
    config = json.loads(_text("config.json"))
    assert set(config) == {"lichess_bot", "lichess_perf"}
    assert config["lichess_perf"] == "blitz"
    assert 'const CONFIG_URL = "config.json";' in _text("app.js")
    assert "https://lichess.org/api/user/${bot}" in _text("rating.js")
    for name in ("app.js", "rating.js", "panel.js", "index.html"):
        assert not re.search(r"(Blink_?BOT|BlinkBot|OneLookBot)", _text(name)), name


def test_the_model_is_cached_under_the_sha256_its_card_records():
    worker, app, cache = _text("worker.js"), _text("app.js"), _text("modelcache.js")
    assert 'import { loadModel } from "./modelcache.js";' in worker
    assert "await loadModel({ url, sha256 })" in worker
    assert "app.engine.load(MODEL_URL, app.card.sha256)" in app
    assert "cachesImpl.open(CACHE_NAME)" in cache
    assert 'key.searchParams.set("sha256", sha256)' in cache
    assert "does not match its card" in cache


def test_the_value_histogram_sits_beside_the_win_bar_with_128_columns():
    page = _text("index.html")
    column = page[page.index('<section class="board-column"') : page.index("</section>")]
    assert column.index('id="winbar"') < column.index('id="value-hist"') < column.index('id="status"')
    assert re.search(r'<svg id="value-hist"[^>]*viewBox="0 0 128 32"', column)
    assert 'panel.renderHistogram($("value-hist"), look && look.bins' in _text("app.js")
    assert "bins: tok.valueProbabilities(out.value)" in _text("rules.js")


def test_the_page_shows_ms_per_move_and_the_backend_it_runs_on():
    page, app = _text("index.html"), _text("app.js")
    assert 'id="ms-last"' in page and 'id="backend"' in page
    assert "panel.backendLabel(app.backend, app.card)" in app
    assert 'backend: "wasm"' in _text("worker.js")


def test_every_card_key_the_panel_reads_is_one_the_card_writer_writes():
    from blink.site import card

    lines = _text("panel.js").splitlines()
    code = "\n".join(line for line in lines if not line.lstrip().startswith("//"))
    read = set(PROPERTY.findall(code))
    assert {"top1_agreement", "mean_abs_dwin_pt", "drop_pt", "fp32_bytes", "passed", "failures"} <= read
    assert read - _keys(card.example()) == set()


def test_the_page_links_the_training_replay():
    assert '<a href="replay/index.html">' in _text("index.html"), "blink site serve maps no directory index"


# ------------------------------------------------------------------------------ in Edge (local)

EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


@pytest.fixture(scope="module")
def carded_model(tmp_path_factory):
    if sys.platform != "win32" or not EDGE.is_file():
        pytest.skip("Microsoft Edge is not installed (Playwright uses channel msedge and never downloads)")
    if not (SITE / "node_modules" / "onnxruntime-web").is_dir():
        pytest.skip("site/node_modules is absent (run npm ci --prefix site)")
    pytest.importorskip("torch")
    from blink.export import onnx as export_onnx
    from blink.export import standin
    from blink.site import card

    model = export_onnx.export(standin.build(seed=0), tmp_path_factory.mktemp("carded") / "model.onnx")
    fp32 = {"selector": "stand-in", "bytes": model.stat().st_size, "sha256": card.sha256_of(model)}
    card.write(card.for_file(fp32, model, precision="fp32"), model.with_name(card.CARD_FILE))
    return model


@contextmanager
def _served(model: Path):
    from blink.site import serve

    class Server(serve.SiteServer):
        # Edge fetches a dozen modules at once; the default listen backlog of 5 can refuse some
        request_queue_size = 64

    server = Server(serve.SiteConfig(site_dir=SITE, model=model), 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def _edge():
    from playwright.sync_api import sync_playwright

    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_context().new_page()
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            yield page, errors
        finally:
            browser.close()


@pytest.mark.local
@pytest.mark.torch
def test_the_page_shows_the_card_histogram_and_backend_and_reloads_the_model_from_cache(carded_model):
    from blink.site import smoke

    with _served(carded_model) as url, _edge() as (page, errors):
        page.goto(url)
        page.wait_for_function(smoke.WAIT_READY, timeout=120_000)
        first = page.evaluate("window.__blink.state()")
        smoke._click_square(page, "e2")
        smoke._click_square(page, "e4")
        page.wait_for_function("() => window.__blink.state().replies === 1", timeout=120_000)
        after = page.evaluate("window.__blink.state()")
        card_text = page.text_content("#model-card")
        backend = page.text_content("#backend")
        note = page.text_content("#hist-note")
        page.reload()
        page.wait_for_function(smoke.WAIT_READY, timeout=120_000)
        second = page.evaluate("window.__blink.state()")
    assert {k: first["backend"][k] for k in ("backend", "threads", "source")} == {
        "backend": "wasm",
        "threads": 1,
        "source": "network",
    }
    assert first["backend"]["loadMs"] > 0
    assert second["backend"]["source"] == "cache", "the reload reads the model from the Cache API"
    assert after["histogramBins"] == 128
    assert "one look, fp32 WASM" in card_text and "sha256" in card_text
    assert backend == "Backend: WASM, 1 thread, fp32 weights (downloaded)"
    assert "128 bins" in note and "mean" in note
    assert after["rating"] == "rating accruing"
    assert errors == []


@pytest.mark.local
@pytest.mark.torch
def test_the_selftest_checks_a_legal_move_one_call_and_the_histogram(carded_model):
    with _served(carded_model) as url, _edge() as (page, errors):
        page.goto(url + "?selftest=1")
        page.wait_for_function("() => window.__blinkSelfTest !== undefined", timeout=120_000)
        result = page.evaluate("window.__blinkSelfTest")
    assert result["ok"] is True and result["histogram"] is True
    assert result["backend"] == "wasm"
    assert errors == []


@pytest.mark.local
@pytest.mark.torch
def test_the_replay_page_loads_the_frozen_run_with_no_console_errors(carded_model):
    run = json.loads((SITE / "replay" / "run.json").read_text(encoding="utf-8"))
    with _served(carded_model) as url, _edge() as (page, errors):
        page.goto(url + "replay/index.html")
        page.wait_for_function("() => window.__blinkReplay && window.__blinkReplay.ready", timeout=60_000)
        replay = page.evaluate("window.__blinkReplay")
        about = page.text_content("#about")
        page.click("#play")
        page.wait_for_function("() => document.getElementById('play').textContent === 'Play'", timeout=30_000)
        values = page.eval_on_selector_all(".card .value", "cards => cards.map(c => c.textContent)")
    assert replay == {
        "ready": True,
        "run": run["run"],
        "metrics": run["metrics_rows"],
        "evals": run["evals_rows"],
    }
    assert f"run {run['run']}" in about and "one per 2,000 steps" in about
    assert len(values) == 6 and "-" not in values
    assert errors == []
