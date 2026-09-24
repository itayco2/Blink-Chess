"""`blink film render`: one 4:5 1080x1350 master per language, from film.json, with no encoder tag."""

import json
import shutil
import struct
import sys
import zlib
from dataclasses import asdict
from pathlib import Path

import chess
import numpy as np
import pytest
from test_report_fixtures import lichess, write_bundle

from blink.film import extract, render
from blink.report import claims

PAGE = Path(render.__file__).resolve().parent / "page"
EDGE_PATHS = (
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
)
ROW = {
    "PuzzleId": "mX46Y",
    "FEN": "2kr3r/ppqnbpp1/4p3/1P1p2Np/N2P2n1/P6P/2PB1PP1/R2Q1RK1 w - - 2 16",
    "Moves": "g5f7 c7h2",
    "Rating": "581",
    "Themes": "mate mateIn1",
}


PROOF = {
    "position_blocked": True, "line_blocked": True, "blocklist_sha256": "ab" * 32,
    "pack_blocklist_sha256": "ab" * 32, "pack_world": "w",
}  # fmt: skip


def _film(n_frames: int = 3) -> dict:
    """A small film.json made by hand: the policy drifts towards the solution, the value sharpens."""
    position = extract.position_from_row(ROW)
    legal_moves = sorted(m.uci() for m in chess.Board(position.fen).legal_moves)
    frames = []
    for i in range(n_frames):
        weights = np.ones(len(legal_moves))
        weights[legal_moves.index(position.solution)] += 30.0 * i
        legal = dict(zip(legal_moves, (weights / weights.sum()).tolist(), strict=True))
        bins = np.exp(-0.5 * ((np.arange(128) - 64 - 20 * i) / (40 - 12 * i)) ** 2)
        bins = (bins / bins.sum()).tolist()
        frames.append(
            {"index": i + 1, "step": 250 * i, "kind": "init" if i == 0 else "ema", "origin": "film",
             "interpolated": False, "positions_seen": 256 * 250 * i, "gpu_hours": 0.01 * i,
             "top5": extract.top_moves(legal), "legal": legal, "value_bins": bins,
             "win": float(np.dot(bins, np.linspace(0.5 / 128, 1 - 0.5 / 128, 128)))}
        )  # fmt: skip
    return {
        "format": 1, "run": "demo", "world": "w", "position": asdict(position),
        "never_in_training": PROOF, "frames": frames, "measured_frames": n_frames,
        "note": "",
    }  # fmt: skip


def _png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 1))
        + chunk(b"IEND", b"")
    )


def _have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def _have_edge() -> bool:
    return sys.platform == "win32" and any(p.is_file() for p in EDGE_PATHS)


def test_the_hook_is_the_claims_hook_for_each_language_and_mode():
    for mode in claims.MODES:
        for lang in claims.LANGS:
            assert render.hook_for(lang, mode) == claims.hook(lang, mode)


def test_the_timeline_is_a_3_s_hook_21_frames_of_0_3_plus_0_5_s_and_a_3_s_end_card():
    assert render.duration_s(21) == pytest.approx(22.8)
    assert render.frame_count(21, 30) == 684
    assert render.MIN_S <= render.duration_s(21) <= render.MAX_S


def test_the_payload_carries_the_hook_the_frames_and_an_end_card_with_the_repo_as_plain_text():
    payload = render.build_payload(_film(), "en", "policy", lichess())
    assert payload["hook"] == claims.hook("en", "policy") and payload["dir"] == "ltr"
    assert payload["timing"] == {"hook": 3.0, "morph": 0.3, "hold": 0.5, "end": 3.0}
    assert len(payload["frames"]) == 3 and "legal" not in payload["frames"][0]
    end = payload["end"]
    assert end["repo"] == "github.com/itayco2/Blink-Chess" and "http" not in json.dumps(payload)
    assert "1950" in end["rating"] and "2026-10-11" in end["rating"]
    assert end["never_seen"] == "This position was never in its training data."


def test_the_end_card_says_accruing_until_the_rating_is_publishable():
    assert "accruing" in render.build_payload(_film(), "en", "value", lichess(n=40, rd=120))["end"]["rating"]
    assert "accruing" in render.build_payload(_film(), "en", "value", None)["end"]["rating"]
    he = render.build_payload(_film(), "he", "value", None)
    assert (
        he["dir"] == "rtl" and he["hook"] == claims.hook("he", "value") and "Lichess" in he["end"]["rating"]
    )


def test_an_interpolated_film_says_so_on_screen_in_both_languages():
    film = {**_film(), "measured_frames": 2}
    film["frames"][1] = {**film["frames"][1], "interpolated": True, "kind": "interpolated"}
    for lang in ("en", "he"):
        payload = render.build_payload(film, lang, "policy", None)
        assert "1" in payload["note"] and "3" in payload["note"] and "2" in payload["note"]
    assert "interpolated" in render.build_payload(film, "en", "policy", None)["note"]


def test_the_mode_defaults_to_the_shipped_mode_and_is_refused_without_one(tmp_path):
    assert render.resolve_mode(None, write_bundle(tmp_path / "results")) == "value"
    assert render.resolve_mode("policy", tmp_path / "empty") == "policy"
    with pytest.raises(extract.FilmError, match="--mode"):
        render.resolve_mode(None, tmp_path / "empty")


def test_once_a_mode_ships_the_film_cannot_burn_in_the_other_modes_hook(tmp_path):
    """results.json ships value: --mode policy would put 'never searches' on a film of a value model."""
    results = write_bundle(tmp_path / "results")
    with pytest.raises(extract.FilmError, match="ships 'value'"):
        render.resolve_mode("policy", results)
    assert render.resolve_mode("value", results) == "value"


def test_a_render_before_any_mode_ships_is_marked_a_preview(tmp_path):
    film_path = extract.write_film(_film(), tmp_path / "film.json")
    _, payload, meta = render.prepare(film_path, "en", "policy", tmp_path / "empty")
    assert meta == {"mode": "policy", "preview": True} and payload["hook"] == claims.hook("en", "policy")
    _, _, meta = render.prepare(film_path, "en", None, write_bundle(tmp_path / "results"))
    assert meta == {"mode": "value", "preview": False}


@pytest.mark.parametrize(
    "proof",
    [
        {},
        {**PROOF, "position_blocked": False},
        {**PROOF, "pack_blocklist_sha256": "cd" * 32},
        {k: v for k, v in PROOF.items() if k != "pack_blocklist_sha256"},
    ],
)
def test_a_film_without_a_full_never_in_training_proof_gets_no_end_card(proof):
    film = {**_film(), "never_in_training": proof}
    with pytest.raises(extract.FilmError, match="never in"):
        render.build_payload(film, "en", "value", None)


def test_ffmpeg_is_told_to_write_no_encoder_tag_and_no_x264_version_sei(tmp_path):
    command = " ".join(render.ffmpeg_command(tmp_path / "out.mp4", 30))
    for part in (
        "-c:v libx264", "-pix_fmt yuv420p", "-crf 18", "-movflags +faststart", "-map_metadata -1",
        "-fflags +bitexact", "-flags:v +bitexact", "-bsf:v filter_units=remove_types=6", "-framerate 30",
        "-metadata:s:v:0 encoder=",
    ):  # fmt: skip
        assert part in command


def test_the_probe_check_flags_every_published_limit():
    good = {"width": 1080, "height": 1350, "fps": 30.0, "duration": 22.8, "bytes": 3_000_000,
            "pix_fmt": "yuv420p", "encoder_tags": [], "has_x264": False}  # fmt: skip
    assert render.check_probe(good, 30) == []
    bad = {**good, "width": 1920, "fps": 25.0, "duration": 30.0, "bytes": 11_000_000,
           "encoder_tags": ["Lavf62"], "has_x264": True}  # fmt: skip
    problems = " ".join(render.check_probe(bad, 30))
    for word in ("1080x1350", "fps", "25 s", "10 MB", "encoder", "x264"):
        assert word in problems


def test_the_rendered_mp4_has_no_encoder_tag(tmp_path):
    """ffprobe shows no encoder tag in the container or the stream, and no byte of the file says x264."""
    if not _have_ffmpeg():
        pytest.skip("ffmpeg and ffprobe are not on PATH")
    frames = [_png(render.WIDTH, render.HEIGHT, (243, 239, 230 - 8 * i)) for i in range(10)]
    out = tmp_path / "tiny.mp4"
    assert render.encode(iter(frames), out, 30) == 10
    info = render.probe(out)
    assert info["encoder_tags"] == [] and not info["has_x264"]
    assert b"x264" not in out.read_bytes() and b"Lavf" not in out.read_bytes()
    assert (info["width"], info["height"], info["fps"], info["pix_fmt"]) == (1080, 1350, 30.0, "yuv420p")
    assert render.check_probe(info, 30, full=False) == []


def test_the_hebrew_render_is_rtl_with_the_vendored_ofl_font():
    css = (PAGE / "film.css").read_text(encoding="utf-8")
    assert 'src: url("fonts/Heebo-wght.ttf")' in css and 'font-family: "Heebo"' in css
    font = PAGE / "fonts" / "Heebo-wght.ttf"
    licence = (PAGE / "fonts" / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License, Version 1.1" in licence and "Heebo" in licence
    assert "Heebo".encode("utf-16-be") in font.read_bytes()  # the font's own name table
    payload = render.build_payload(_film(), "he", "policy", None)
    assert payload["dir"] == "rtl" and payload["lang"] == "he"
    if not _have_edge():
        pytest.skip("Microsoft Edge is not installed (Playwright uses channel msedge and never downloads)")
    with render.open_page(payload) as session:
        page = session.page
        facts = page.evaluate(
            """() => {
                const hook = document.getElementById('hook-text');
                const style = getComputedStyle(hook);
                return {dir: document.documentElement.dir, lang: document.documentElement.lang,
                        hook: hook.textContent, direction: style.direction,
                        family: style.fontFamily.replaceAll('"', ''),
                        loaded: document.fonts.check('700 40px Heebo', 'בלינק')};
            }"""
        )
        assert session.errors == []
    assert facts == {
        "dir": "rtl", "lang": "he", "hook": claims.hook("he", "policy"), "direction": "rtl",
        "family": "Heebo, sans-serif", "loaded": True,
    }  # fmt: skip
    assert "fonts/Heebo-wght.ttf" in session.served


@pytest.mark.local
def test_a_short_render_in_edge_seeks_frames_and_encodes_them_without_page_errors(tmp_path):
    if not (_have_edge() and _have_ffmpeg()):
        pytest.skip("needs Microsoft Edge and ffmpeg")
    film_path = extract.write_film(_film(), tmp_path / "film.json")
    results = write_bundle(tmp_path / "results")
    report = render.render(
        film_path, tmp_path / "film-en.mp4", "en", fps=5, results_dir=results, max_seconds=4.0
    )
    assert report["page_errors"] == [] and report["frames"] == 20
    assert report["probe"]["encoder_tags"] == [] and not report["probe"]["has_x264"]
    sidecar = json.loads((tmp_path / "film-en.json").read_text(encoding="utf-8"))
    assert sidecar["hook"] == claims.hook("en", "value") and sidecar["mode"] == "value"
    assert sidecar["preview"] is False


def test_a_failed_capture_leaves_no_partial_film_behind(tmp_path):
    if not _have_ffmpeg():
        pytest.skip("ffmpeg is not on PATH")

    def frames():
        yield _png(64, 80, (1, 2, 3))
        raise RuntimeError("the browser crashed")

    with pytest.raises(RuntimeError, match="crashed"):
        render.encode(frames(), tmp_path / "film.mp4", 30)
    assert list(tmp_path.iterdir()) == []


def test_an_ffmpeg_failure_is_a_film_error_with_its_message(tmp_path):
    if not _have_ffmpeg():
        pytest.skip("ffmpeg is not on PATH")
    with pytest.raises(extract.FilmError, match="ffmpeg exited"):
        render.encode(iter([b"not a png"] * 3), tmp_path / "film.mp4", 30)
    assert list(tmp_path.iterdir()) == []


def test_blink_film_render_reports_a_film_error_as_exit_1(tmp_path, capsys):
    from blink import cli

    film_path = extract.write_film(_film(), tmp_path / "film.json")
    args = ["film", "render", "--lang", "en", "--film", str(film_path), "--results", str(tmp_path / "none")]
    assert cli.main(args) == 1
    assert "--mode" in capsys.readouterr().err


def test_film_commands_refuse_a_run_name_that_is_a_path(capsys):
    from blink import cli

    assert cli.main(["film", "render", "--lang", "en", "--run", "../outside", "--mode", "policy"]) == 1
    assert "bad run name" in capsys.readouterr().err


def test_milestones_on_screen_say_what_they_were_measured_on():
    milestone = {"label": "passed the MLP", "step": 250, "metric": "vaa", "threshold": 0.2}
    film = {**_film(), "milestones": [milestone]}
    assert render.build_payload(film, "en", "value", None)["milestones"] == [
        {"label": "passed the MLP (on the val probe)", "step": 250}
    ]
    he = render.build_payload(film, "he", "value", None)["milestones"][0]
    assert he["step"] == 250 and he["label"].startswith("passed the MLP (") and "val probe" not in he["label"]
