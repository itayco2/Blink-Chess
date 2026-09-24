"""`blink site smoke`: Playwright on the installed Edge plays the page and checks what it shows (P1, P10)."""

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import chess
import pytest

EDGE_EXE = "Microsoft/Edge/Application/msedge.exe"
EDGE_PATHS = (
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / EDGE_EXE,
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / EDGE_EXE,
)


def _report(**overrides):
    from blink.site import smoke

    fields = {
        "url": "http://127.0.0.1:8000/",
        "loaded": True,
        "user_moves": 10,
        "legal_replies": 10,
        "illegal": (),
        "arrows": 3,
        "min_arrows": 3,
        "console_errors": (),
        "timings_ms": (12.0, 10.0, 11.0),
        "games": 1,
        "load_seconds": 1.5,
        "problems": (),
    }
    return smoke.SmokeReport(**{**fields, **overrides})


def test_replaying_a_game_counts_only_blinks_legal_replies():
    from blink.site import smoke

    history = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"]
    assert smoke.replay(chess.STARTING_FEN, history, user_color="w") == (2, [])
    assert smoke.replay(chess.STARTING_FEN, history, user_color="b") == (3, [])


def test_replaying_stops_at_the_first_illegal_move_and_names_it():
    from blink.site import smoke

    replies, illegal = smoke.replay(chess.STARTING_FEN, ["e2e4", "e7e4", "g1f3"], user_color="w")
    assert replies == 0
    assert len(illegal) == 1 and "e7e4" in illegal[0]


def test_replaying_accepts_castling_written_as_the_kings_two_square_move():
    from blink.site import smoke

    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    assert smoke.replay(fen, ["e1g1", "e8c8"], user_color="w") == (1, [])


def test_a_reply_should_show_three_arrows_or_one_per_legal_move_when_fewer():
    from blink.site import smoke

    assert smoke.expected_arrows(chess.STARTING_FEN, ["e2e4", "e7e5"], last_rule=None) == 3
    in_check = "4k3/8/8/8/8/8/3q4/4K3 w - - 0 1"  # the white king has two legal moves, one reply from Blink
    assert smoke.expected_arrows(in_check, ["e1f1"], last_rule=None) == 2
    assert smoke.expected_arrows(chess.STARTING_FEN, [], last_rule=None) == 0


def test_a_mate_now_reply_is_played_without_a_look_so_it_shows_no_arrows():
    from blink.site import smoke

    history = ["f2f3", "e7e5", "g2g4", "d8h4"]
    assert smoke.expected_arrows(chess.STARTING_FEN, history, last_rule="R2") == 0
    assert smoke.failures(_report(arrows=0, expected_arrows=0, last_rule="R2"), moves=10) == []
    assert "0 arrows" in " ".join(smoke.failures(_report(arrows=0, expected_arrows=3), moves=10))


def test_a_clean_report_has_no_failures():
    from blink.site import smoke

    assert smoke.failures(_report(), moves=10) == []


def test_failures_name_every_missed_criterion():
    from blink.site import smoke

    report = _report(
        loaded=False, legal_replies=7, illegal=("ply 3: e7e4",), arrows=2, console_errors=("boom",)
    )
    text = " | ".join(smoke.failures(report, moves=10))
    for needle in ("never became ready", "7 legal replies", "e7e4", "2 arrows", "1 console error"):
        assert needle in text


def test_smoke_side_problems_such_as_a_timeout_fail_the_run():
    from blink.site import smoke

    report = _report(problems=("timed out waiting for reply 4",))
    assert smoke.failures(report, moves=10) == ["timed out waiting for reply 4"]


def test_the_report_summarises_ms_per_move():
    summary = _report(timings_ms=(10.0, 30.0, 20.0, 40.0)).to_dict()
    assert summary["ms_per_move_median"] == 25.0
    assert summary["ms_per_move_max"] == 40.0
    assert summary["moves_timed"] == 4


def _edge_installed() -> bool:
    return sys.platform == "win32" and any(path.is_file() for path in EDGE_PATHS)


@pytest.fixture(scope="module")
def stand_in_onnx(repo_root, tmp_path_factory):
    if not _edge_installed():
        pytest.skip("Microsoft Edge is not installed (Playwright uses channel msedge and never downloads)")
    if not (repo_root / "site" / "node_modules" / "onnxruntime-web").is_dir():
        pytest.skip("site/node_modules is absent (run npm ci --prefix site)")
    from blink.export import onnx as export_onnx
    from blink.export import standin

    return export_onnx.export(standin.build(seed=0), tmp_path_factory.mktemp("page") / "model.onnx")


@contextmanager
def _serving(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="module")
def page_url(repo_root, stand_in_onnx):
    from blink.site import serve

    config = serve.SiteConfig(site_dir=repo_root / "site", model=stand_in_onnx)
    with _serving(serve.make_server(config, port=0)) as url:
        yield url


@pytest.mark.local
@pytest.mark.torch
def test_the_page_plays_ten_legal_replies_with_three_arrows_and_no_console_errors(page_url):
    from blink.site import smoke

    report = smoke.run(page_url, moves=10, seed=1)
    assert smoke.failures(report, moves=10) == []
    assert report.legal_replies >= 10
    assert report.arrows == 3
    assert report.console_errors == ()


@pytest.mark.local
@pytest.mark.torch
def test_the_page_plays_a_pasted_mate_in_one_without_a_network_call(page_url):
    from playwright.sync_api import sync_playwright

    from blink.site import smoke

    errors: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page()
            page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(page_url)
            page.wait_for_function(smoke.WAIT_READY, timeout=60_000)
            page.fill("#fen-input", "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1")  # Blink is Black: Ra1 mates
            page.click("#fen-form button[type=submit]")
            page.wait_for_function("() => window.__blink.state().replies === 1", timeout=60_000)
            state = page.evaluate("window.__blink.state()")
            note = page.text_content("#rule-note")
        finally:
            browser.close()
    assert state["history"] == ["a8a1"]
    assert state["lastRule"] == "R2"
    assert state["arrows"] == 0
    assert state["timings"] == [], "the mate was played without a network call"
    assert "R2" in note
    assert errors == []
