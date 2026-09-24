"""`blink film render --lang en|he`: film.json -> one 4:5 1080x1350 master per language.

The page (blink/film/page) is pure rendering: it exposes seek(t), duration() and window.__ready,
and draws every frame from t alone (no CSS transition, no timer), so a capture is exact. Playwright
drives the installed Edge (channel msedge, nothing downloaded) and serves the page, the vendored
Heebo font (OFL-1.1), the cburnett piece sprite and the payload from memory under a fake origin:
every other request is aborted, so the render touches no network. Each seek is one screenshot, piped
as PNG into ffmpeg: libx264, yuv420p, crf 18, +faststart, with -map_metadata -1, -fflags +bitexact,
-flags:v +bitexact and the SEI filter, so neither the container's encoder tag nor x264's version
string is written. ffmpeg 8.1.1 still tags the video stream "Lavc libx264" under +bitexact (it drops
only the version), so the stream's encoder tag is also set empty, which removes it.

Timeline: a 3 s hook (claims.HOOK_EN or HOOK_HE for the shipped mode), then each of the 21 frames
as a 0.3 s morph and a 0.5 s hold, then a 3 s end card (the Lichess rating with its date, "this
position was never in its training data", the repo URL as plain text): 22.8 s at 30 fps.
"""

import contextlib
import json
import os
import shutil
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from blink.film import extract
from blink.report import claims
from blink.report import results_schema as rs
from blink.train.atomic import write_text_atomic

PAGE_DIR = Path(__file__).resolve().parent / "page"
REPO_ROOT = Path(__file__).resolve().parents[2]
PIECES = REPO_ROOT / "site" / "assets" / "pieces" / "cburnett.svg"
ORIGIN = "http://blink-film.invalid/"
WIDTH, HEIGHT = 1080, 1350
HOOK_S, MORPH_S, HOLD_S, END_S = 3.0, 0.3, 0.5, 3.0
MIN_S, MAX_S = 15.0, 25.0
MAX_BYTES = 10_000_000  # GitHub's 10 MB README video limit
REPO_URL = "github.com/itayco2/Blink-Chess"
READY_TIMEOUT_MS = 60_000
FRAME_KEYS = (
    "index",
    "step",
    "kind",
    "interpolated",
    "positions_seen",
    "gpu_hours",
    "top5",
    "value_bins",
    "win",
)
CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".ttf": "font/ttf",
    ".svg": "image/svg+xml",
    ".txt": "text/plain; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


TEXTS = {
    "en": {
        "sub": "Watch it learn one position.",
        "puzzle": "Puzzle {id}, rated {rating}",
        "white": "White",
        "black": "Black",
        "to_play": "{side} to play",
        "frame": "frame {i} of {n}",
        "kind_init": "at birth: random weights",
        "kind_final": "the final weights",
        "kind_interpolated": "interpolated between measured frames",
        "step": "training step {step}",
        "seen": "positions seen {n}",
        "gpu": "GPU-hours {h}",
        "win": "win chance for {side}: {pct}%",
        "found": "top move = the puzzle's solution",
        "never_seen": "This position was never in its training data.",
        "rating": "Lichess blitz rating {rating} ({date})",
        "accruing": "Lichess blitz rating: still accruing",
        "run": "run {run}",
        "padded": "{padded} of {total} frames are interpolated between {measured} measured ones",
    },
    "he": {
        "sub": "צפו בה לומדת עמדה אחת.",
        "puzzle": "חידה {id}, דירוג {rating}",
        "white": "הלבן",
        "black": "השחור",
        "to_play": "תור {side}",
        "frame": "שלב {i} מתוך {n}",
        "kind_init": "בלידה: משקלים אקראיים",
        "kind_final": "המשקלים הסופיים",
        "kind_interpolated": "אינטרפולציה בין שלבים מדודים",
        "step": "צעד אימון {step}",
        "seen": "עמדות שנראו {n}",
        "gpu": "שעות GPU {h}",
        "win": "סיכויי הניצחון של {side}: {pct}%",
        "found": "המהלך המוביל = פתרון החידה",
        "never_seen": "העמדה הזאת מעולם לא הופיעה בנתוני האימון שלה.",
        "rating": "דירוג בליץ ב-Lichess: {rating} ({date})",
        "accruing": "דירוג הבליץ ב-Lichess עוד נצבר",
        "run": "ריצה {run}",
        "padded": "{padded} מתוך {total} השלבים הם אינטרפולציה בין {measured} שלבים מדודים",
    },
}


def hook_for(lang: str, mode: str) -> str:
    return claims.hook(lang, mode)


def duration_s(n_frames: int) -> float:
    return HOOK_S + n_frames * (MORPH_S + HOLD_S) + END_S


def frame_count(n_frames: int, fps: int) -> int:
    return round(duration_s(n_frames) * fps)


def resolve_mode(mode: str | None, results_dir: Path) -> str:
    if mode:
        return mode
    shipped = claims.shipped_mode(results_dir)
    if shipped is None:
        raise extract.FilmError(f"{results_dir} names no shipped mode yet: pass --mode policy|value")
    return shipped


def read_lichess(results_dir: Path) -> rs.LichessSnapshot | None:
    path = Path(results_dir) / "lichess.json"
    return rs.lichess_from_json(path.read_text(encoding="utf-8")) if path.is_file() else None


def _rating_line(lang: str, snap: rs.LichessSnapshot | None) -> str:
    text = TEXTS[lang]
    if snap is None or not snap.publishable:
        return text["accruing"]
    return text["rating"].format(rating=snap.rating, date=snap.snapshot_date)


def _note(lang: str, film: dict) -> str:
    total, measured = len(film["frames"]), film.get("measured_frames", len(film["frames"]))
    run = TEXTS[lang]["run"].format(run=film["run"])
    if measured >= total:
        return run
    padded = TEXTS[lang]["padded"].format(padded=total - measured, total=total, measured=measured)
    return f"{run}. {padded}"


def build_payload(film: dict, lang: str, mode: str, lichess: rs.LichessSnapshot | None) -> dict:
    text = TEXTS[lang]
    return {
        "lang": lang,
        "dir": "rtl" if lang == "he" else "ltr",
        "hook": hook_for(lang, mode),
        "text": {k: v for k, v in text.items() if k not in ("rating", "accruing")},
        "timing": {"hook": HOOK_S, "morph": MORPH_S, "hold": HOLD_S, "end": END_S},
        "position": film["position"],
        "frames": [{k: frame[k] for k in FRAME_KEYS} for frame in film["frames"]],
        "milestones": film.get("milestones", []),
        "note": _note(lang, film),
        "end": {"rating": _rating_line(lang, lichess), "never_seen": text["never_seen"], "repo": REPO_URL},
    }


# ------------------------------------------------------------------------------------------ the page


@dataclass
class PageSession:
    page: object
    served: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _files(payload: dict) -> dict[str, bytes]:
    files = {p.relative_to(PAGE_DIR).as_posix(): p for p in PAGE_DIR.rglob("*") if p.is_file()}
    blobs = {name: path.read_bytes() for name, path in files.items()}
    blobs["pieces/cburnett.svg"] = PIECES.read_bytes()
    blobs["data.json"] = json.dumps(payload).encode("utf-8")
    return blobs


def _router(blobs: dict[str, bytes], session: PageSession):
    def handle(route) -> None:
        url = route.request.url
        name = url[len(ORIGIN) :].split("?", 1)[0] if url.startswith(ORIGIN) else None
        if name not in blobs:
            session.errors.append(f"blocked request: {url}")
            route.abort()
            return
        session.served.append(name)
        kind = CONTENT_TYPES.get(Path(name).suffix, "application/octet-stream")
        route.fulfill(status=200, body=blobs[name], headers={"Content-Type": kind})

    return handle


@contextmanager
def open_page(payload: dict) -> Iterator[PageSession]:
    """The film page loaded in headless Edge at 1080x1350, fonts and data ready."""
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="msedge", headless=True)
        try:
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)
            session = PageSession(page)
            page.on("pageerror", lambda exc: session.errors.append(f"page error: {exc}"))
            page.on("console", lambda msg: session.errors.append(msg.text) if msg.type == "error" else None)
            page.route("**/*", _router(_files(payload), session))
            try:
                page.goto(ORIGIN + "film.html")
                page.wait_for_function(
                    "window.__ready === true || !!window.__filmError", timeout=READY_TIMEOUT_MS
                )
            except PlaywrightError as exc:
                raise extract.FilmError(f"the film page never became ready: {exc}; {session.errors}") from exc
            failure = page.evaluate("window.__filmError || null")
            if failure:
                raise extract.FilmError(f"the film page failed to start: {failure}")
            yield session
        finally:
            browser.close()


def capture(session: PageSession, fps: int, frames: int) -> Iterator[bytes]:
    for i in range(frames):
        session.page.evaluate("t => window.seek(t)", i / fps)
        yield session.page.screenshot(type="png", animations="disabled")


# ------------------------------------------------------------------------------------------ ffmpeg


def ffmpeg_command(out: Path, fps: int) -> list[str]:
    return [
        shutil.which("ffmpeg") or "ffmpeg",
        "-y", "-hide_banner", "-loglevel", "error",
        "-f", "image2pipe", "-framerate", str(fps), "-c:v", "png", "-i", "-",
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-movflags", "+faststart", "-map_metadata", "-1",
        "-fflags", "+bitexact", "-flags:v", "+bitexact", "-metadata:s:v:0", "encoder=",
        "-bsf:v", "filter_units=remove_types=6",
        str(out),
    ]  # fmt: skip


def _feed(proc: subprocess.Popen, frames: Iterable[bytes]) -> int:
    count = 0
    for png in frames:
        try:
            proc.stdin.write(png)
        except OSError:
            break  # ffmpeg exited early; its stderr says why
        count += 1
    return count


def encode(frames: Iterable[bytes], out: Path, fps: int) -> int:
    """Pipe PNG frames into ffmpeg; `out` appears only when every frame is in and ffmpeg succeeds."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.mp4")
    proc = subprocess.Popen(ffmpeg_command(tmp, fps), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        count = _feed(proc, frames)
    except BaseException:
        proc.kill()  # the capture failed: no half film is left behind
        proc.wait()
        tmp.unlink(missing_ok=True)
        raise
    with contextlib.suppress(OSError):
        proc.stdin.close()
    errors = proc.stderr.read().decode("utf-8", errors="replace")
    code = proc.wait()
    if code != 0:
        tmp.unlink(missing_ok=True)
        raise extract.FilmError(f"ffmpeg exited {code}: {errors.strip()[-800:]}")
    os.replace(tmp, out)
    return count


def probe(path: Path) -> dict:
    command = [shutil.which("ffprobe") or "ffprobe", "-v", "error", "-print_format", "json"]
    command += ["-show_format", "-show_streams", str(path)]
    data = json.loads(subprocess.run(command, capture_output=True, check=True).stdout)
    stream = next(s for s in data["streams"] if s.get("codec_type") == "video")
    tag_sets = [data["format"].get("tags", {}), *(s.get("tags", {}) for s in data["streams"])]
    return {
        "width": stream["width"],
        "height": stream["height"],
        "fps": float(Fraction(stream.get("avg_frame_rate") or stream["r_frame_rate"])),
        "duration": float(data["format"]["duration"]),
        "bytes": Path(path).stat().st_size,
        "pix_fmt": stream.get("pix_fmt"),
        "encoder_tags": [v for tags in tag_sets for k, v in tags.items() if k.lower() == "encoder"],
        "has_x264": b"x264" in Path(path).read_bytes(),
    }


def check_probe(info: dict, fps: int, full: bool = True) -> list[str]:
    problems = []
    if (info["width"], info["height"]) != (WIDTH, HEIGHT):
        problems.append(f"{info['width']}x{info['height']}, not {WIDTH}x{HEIGHT}")
    if abs(info["fps"] - fps) > 1e-6:
        problems.append(f"{info['fps']:g} fps, not {fps} fps")
    if full and not MIN_S <= info["duration"] <= MAX_S:
        problems.append(f"{info['duration']:.2f} s is outside {MIN_S:g} to {MAX_S:g} s")
    if info["bytes"] > MAX_BYTES:
        problems.append(f"{info['bytes']:,} B is over the 10 MB README video limit")
    if info["encoder_tags"]:
        problems.append(f"encoder tag written: {info['encoder_tags']}")
    if info["has_x264"]:
        problems.append("the file contains the string x264")
    return problems


# ------------------------------------------------------------------------------------------ render


def render(
    film_path: Path,
    out: Path,
    lang: str,
    fps: int = 30,
    mode: str | None = None,
    results_dir: Path = claims.RESULTS_DIR,
    max_seconds: float | None = None,
) -> dict:
    film = extract.read_film(film_path)
    mode = resolve_mode(mode, results_dir)
    payload = build_payload(film, lang, mode, read_lichess(results_dir))
    total = frame_count(len(film["frames"]), fps)
    frames = min(total, round(max_seconds * fps)) if max_seconds else total
    with open_page(payload) as session:
        written = encode(capture(session, fps, frames), out, fps)
        page_errors = list(session.errors)
    info = probe(out)
    problems = check_probe(info, fps, full=max_seconds is None) + page_errors
    report = {
        "out": str(out), "lang": lang, "mode": mode, "hook": payload["hook"], "run": film["run"],
        "fps": fps, "frames": written, "film_frames": len(film["frames"]),
        "measured_frames": film.get("measured_frames"), "probe": info, "page_errors": page_errors,
        "problems": problems,
    }  # fmt: skip
    write_sidecar(report, out)
    return report


def write_sidecar(report: dict, video: Path) -> Path:
    """The render report next to its video (film_en.mp4 -> film_en.json), via a .tmp and a replace."""
    sidecar = Path(video).with_suffix(".json")
    write_text_atomic(sidecar, json.dumps(report, indent=2) + "\n")
    return sidecar


def format_report(report: dict) -> str:
    info = report["probe"]
    lines = [
        f"{report['out']}: {info['width']}x{info['height']}, {info['fps']:g} fps, {info['duration']:.2f} s, "
        f"{info['bytes']:,} B, {info['pix_fmt']}, encoder tags {info['encoder_tags'] or 'none'}, "
        f"x264 string {'present' if info['has_x264'] else 'absent'}",
        f"{report['frames']} video frames from {report['film_frames']} film frames "
        f"({report['measured_frames']} measured); hook: {report['hook']}",
    ]
    lines += [f"PROBLEM: {p}" for p in report["problems"]]
    return "\n".join(lines)
