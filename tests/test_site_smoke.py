"""`blink site smoke`: Playwright on the installed Edge plays the page and checks what it shows (P1, P10)."""

import os
import sys
import threading
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


def test_the_report_summarises_ms_per_move():
    summary = _report(timings_ms=(10.0, 30.0, 20.0, 40.0)).to_dict()
    assert summary["ms_per_move_median"] == 25.0
    assert summary["ms_per_move_max"] == 40.0
    assert summary["moves_timed"] == 4


def _edge_installed() -> bool:
    return sys.platform == "win32" and any(path.is_file() for path in EDGE_PATHS)


@pytest.mark.local
@pytest.mark.torch
def test_the_page_plays_ten_legal_replies_with_three_arrows_and_no_console_errors(repo_root, tmp_path):
    if not _edge_installed():
        pytest.skip("Microsoft Edge is not installed (Playwright uses channel msedge and never downloads)")
    if not (repo_root / "site" / "node_modules" / "onnxruntime-web").is_dir():
        pytest.skip("site/node_modules is absent (run npm ci --prefix site)")
    from blink.export import onnx as export_onnx
    from blink.export import standin
    from blink.site import serve, smoke

    model = export_onnx.export(standin.build(seed=0), tmp_path / "model.onnx")
    server = serve.make_server(serve.SiteConfig(site_dir=repo_root / "site", model=model), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        report = smoke.run(f"http://127.0.0.1:{server.server_address[1]}/", moves=10, seed=1)
    finally:
        server.shutdown()
        server.server_close()
    assert smoke.failures(report, moves=10) == []
    assert report.legal_replies >= 10
    assert report.arrows == 3
    assert report.console_errors == ()
