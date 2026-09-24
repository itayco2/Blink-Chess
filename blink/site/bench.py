"""`blink site bench`: one-look latency per backend, measured in the installed Edge (P10).

The staged page (the same tree Pages serves) is served on 127.0.0.1 with cross-origin isolation, so
multi-threaded WASM can run, plus the fp32 model and onnxruntime-web's WebGPU runtime files, which the
deployed page never ships. Playwright (channel msedge, no browser download) first loads the page itself
in a fresh profile (empty caches) through a paced server, then opens bench.html on an unpaced one, which
times `runs` looks at batch 1 per (model, backend) pair. int8 is never paired with WebGPU (PF30).

Loopback alone would hide the download, the largest part of a first visit (the 14.2 MB ORT wasm plus the
model), so the cold load is paced over a stated link, COLD_LOAD_NETWORK unless --network names another:
every response waits one round trip, then its bytes queue on one downlink shared by all requests. The
bytes are uncompressed, so where Pages compresses a file the real transfer is smaller; TCP and TLS setup
are not modelled. It is an estimate for that link, not what every visitor sees; the profile and the bytes
served are recorded beside the time.

The report (BLINK_HOME/eval/site_bench.json) holds nearest-rank percentiles and the browser-model gate
from the plan: the int8 file at most 30 MB, one look p50 at most 250 ms on int8 wasm-1t (the path the page
runs), and a cold load of at most 5 s over that link. The machine's CPU load at the start is recorded
beside them, since a busy desktop slows every number. `selftest()` is the post-deploy check of pages.yml:
it loads the page with ?selftest=1 and reads window.__blinkSelfTest.
"""

import functools
import math
import os
import platform
import shutil
import statistics
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from blink.site import card, stage

SITE_DIR = Path(__file__).resolve().parents[2] / "site"
HOST = "127.0.0.1"
BACKENDS = ("wasm-1t", "wasm-mt", "webgpu")
GPU_BACKENDS = frozenset({"webgpu"})
PRECISIONS = ("int8", "fp32")
SHIPPED = ("int8", "wasm-1t")
FP32_FILE = "fp32.onnx"
GPU_ORT_FILES = (
    "ort.webgpu.bundle.min.mjs",
    "ort-wasm-simd-threaded.jsep.wasm",
    "ort-wasm-simd-threaded.jsep.mjs",
    "ort-wasm-simd-threaded.asyncify.wasm",
    "ort-wasm-simd-threaded.asyncify.mjs",
)
MAX_INT8_BYTES = 30_000_000
MAX_P50_MS = 250.0
MAX_COLD_LOAD_S = 5.0
PF30 = "WebGPU never gets int8 (PF30): integer kernels there are slow or missing"
WAIT_BENCH = "() => window.__blinkBench && window.__blinkBench.done"
WAIT_PAGE = "() => window.__blink && window.__blink.state().ready"
WAIT_SELFTEST = "() => window.__blinkSelfTest !== undefined"
CHUNK = 64 * 1024  # bytes per paced write
LINK_MODEL = (
    "one downlink shared by every request, paced on uncompressed bytes (an upper bound where Pages "
    "compresses), one round trip before each response; TCP and TLS setup not modelled"
)


class BenchError(RuntimeError):
    pass


@dataclass(frozen=True)
class Pair:
    precision: str
    backend: str


@dataclass(frozen=True)
class Network:
    """The link a cold load is paced over: downlink bandwidth and one round trip per response."""

    down_mbit_s: float
    rtt_ms: float

    @property
    def bytes_per_s(self) -> float:
        return self.down_mbit_s * 1e6 / 8

    def describe(self) -> str:
        return f"{self.down_mbit_s:g} Mbit/s down, {self.rtt_ms:g} ms RTT"

    def to_dict(self) -> dict:
        return {
            "down_mbit_s": self.down_mbit_s,
            "rtt_ms": self.rtt_ms,
            "profile": self.describe(),
            "model": LINK_MODEL,
        }


# The plan's own limits only fit together on a fast link: a 30 MB int8 file plus the 14.2 MB ORT wasm is
# 44 MB, which needs 71 Mbit/s just to arrive within the 5 s cold load, before any startup work. So the
# gate assumes fast home broadband by default; --network measures any other link.
COLD_LOAD_NETWORK = Network(down_mbit_s=100.0, rtt_ms=40.0)


def parse_network(text: str) -> Network:
    """'MBIT:RTT_MS', for example '20:40' for 20 Mbit/s down and a 40 ms round trip."""
    try:
        down, rtt = (float(part) for part in text.split(":"))
    except ValueError as exc:
        raise ValueError(f"a network is MBIT:RTT_MS such as 20:40, got {text!r}") from exc
    if not (math.isfinite(down) and math.isfinite(rtt) and down > 0 and rtt >= 0):
        raise ValueError(f"a network is MBIT:RTT_MS with MBIT > 0 and RTT_MS >= 0, got {text!r}")
    return Network(down_mbit_s=down, rtt_ms=rtt)


class Link:
    """One downlink for every response of a server: bytes queue for the same bandwidth, in order."""

    def __init__(self, network: Network) -> None:
        self.network = network
        self._lock = threading.Lock()
        self._free_at = 0.0
        self._sent = 0

    @property
    def bytes_sent(self) -> int:
        with self._lock:
            return self._sent

    def round_trip(self) -> None:
        time.sleep(self.network.rtt_ms / 1000)

    def send(self, data: bytes, out) -> None:
        """Write `data` when the link has carried it, after every byte queued before it."""
        with self._lock:
            start = max(time.perf_counter(), self._free_at)
            self._free_at = start + len(data) / self.network.bytes_per_s
            self._sent += len(data)
            arrives = self._free_at
        delay = arrives - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        out.write(data)


@dataclass(frozen=True)
class SelfTest:
    ok: bool
    result: dict
    console_errors: tuple[str, ...]


def plan(precisions: Sequence[str], backends: Sequence[str]) -> tuple[list[Pair], list[dict]]:
    """The pairs to measure and the skipped ones with their reason; int8 never goes to a GPU backend."""
    unknown = [b for b in backends if b not in BACKENDS] + [p for p in precisions if p not in PRECISIONS]
    if unknown:
        raise BenchError(f"unknown backend or model {unknown[0]!r}: backends {BACKENDS}, models {PRECISIONS}")
    pairs, skipped = [], []
    for precision in precisions:
        for backend in backends:
            if precision == "int8" and backend in GPU_BACKENDS:
                skipped.append({"precision": precision, "backend": backend, "reason": PF30})
            else:
                pairs.append(Pair(precision, backend))
    return pairs, skipped


def query(pairs: Sequence[Pair], runs: int, warmup: int) -> str:
    listed = ",".join(f"{p.precision}:{p.backend}" for p in pairs)
    return f"?pairs={listed}&runs={runs}&warmup={warmup}"


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def _stats(values: Sequence[float]) -> dict:
    return {
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p99": percentile(values, 99),
        "mean": statistics.fmean(values),
        "min": min(values),
        "n": len(values),
    }


def summarize(results: Sequence[dict]) -> list[dict]:
    """Per pair: status, threads, cold load, first run, and run/look percentiles (ok rows only)."""
    rows = []
    for result in results:
        if result["precision"] == "int8" and result["backend"] in GPU_BACKENDS:
            raise BenchError(f"refused a result of int8 on {result['backend']}: {PF30}")
        row = {k: result[k] for k in ("precision", "backend", "status") if k in result}
        if result["status"] != "ok":
            rows.append({**row, "reason": result.get("reason", "")})
            continue
        extra = {k: result[k] for k in ("threads", "cold_ms", "first_run_ms", "bytes") if k in result}
        rows.append(
            {**row, **extra, "run_ms": _stats(result["run_ms"]), "look_ms": _stats(result["look_ms"])}
        )
    return rows


def gate(
    int8_bytes: int, rows: Sequence[dict], cold_load_s: float | None, network: Network = COLD_LOAD_NETWORK
) -> dict:
    """The plan's browser-model gate on the shipped path (int8, WASM, 1 thread); cold load over `network`."""
    failures = []
    if int8_bytes > MAX_INT8_BYTES:
        failures.append(f"int8 file {int8_bytes:,} B is over {MAX_INT8_BYTES:,} B")
    shipped = next(
        (r for r in rows if (r["precision"], r["backend"]) == SHIPPED and r["status"] == "ok"), None
    )
    p50 = shipped["look_ms"]["p50"] if shipped else None
    if shipped is None:
        failures.append("no int8 wasm-1t measurement: the shipped path was not measured")
    elif p50 > MAX_P50_MS:
        failures.append(f"one look p50 {p50:.1f} ms on int8 wasm-1t is over {MAX_P50_MS:g} ms")
    if cold_load_s is None or cold_load_s > MAX_COLD_LOAD_S:
        shown = "not measured" if cold_load_s is None else f"{cold_load_s:.2f} s"
        failures.append(f"cold load {shown} over {network.describe()} is over {MAX_COLD_LOAD_S:g} s")
    return {
        "int8_bytes": int8_bytes,
        "int8_bytes_max": MAX_INT8_BYTES,
        "p50_ms": p50,
        "p50_ms_max": MAX_P50_MS,
        "cold_load_s": cold_load_s,
        "cold_load_s_max": MAX_COLD_LOAD_S,
        "cold_load_network": network.to_dict(),
        "failures": failures,
        "passed": not failures,
    }


# ----------------------------------------------------------------------------- the local tree and server


def stage_tree(site_dir: Path, out: Path, int8: Path, fp32: Path) -> Path:
    """The staged page with int8 as models/model.onnx, plus models/fp32.onnx and the GPU runtime files."""
    stage.stage(site_dir, out, model=int8)
    shutil.copyfile(fp32, out / "models" / FP32_FILE)
    dist = site_dir / "node_modules" / "onnxruntime-web" / "dist"
    for name in GPU_ORT_FILES:
        if not (dist / name).is_file():
            raise BenchError(f"missing {dist / name}: {stage.NPM_CI_HINT}")
        shutil.copyfile(dist / name, out / "vendor" / "ort" / name)
    return out


class _IsolatedHandler(SimpleHTTPRequestHandler):
    """Static files with cross-origin isolation, so SharedArrayBuffer (and WASM threads) are available."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".onnx": "application/octet-stream",
    }

    def send_head(self):
        if self.server.link is not None:
            self.server.link.round_trip()
        return super().send_head()

    def copyfile(self, source, outputfile) -> None:
        if self.server.link is None:
            super().copyfile(source, outputfile)
            return
        while chunk := source.read(CHUNK):
            self.server.link.send(chunk, outputfile)

    def end_headers(self) -> None:
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - the base class names it format
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 64  # a page fetches a dozen modules at once; a backlog of 5 refuses some

    def __init__(self, address, handler, link: Link | None) -> None:
        self.link = link
        super().__init__(address, handler)


@contextmanager
def serving(root: Path, link: Link | None = None) -> Iterator[str]:
    """Serve `root` on 127.0.0.1 (never another interface) at a free port, paced over `link` if given."""
    server = _Server((HOST, 0), functools.partial(_IsolatedHandler, directory=str(root)), link)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{HOST}:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()


# ----------------------------------------------------------------------------- Edge


@contextmanager
def _edge() -> Iterator:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        try:
            yield browser
        finally:
            browser.close()


def _open(browser, errors: list[str]):
    page = browser.new_context().new_page()
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(f"page error: {exc}"))
    return page


def _page_cold_load(browser, url: str, timeout_ms: float) -> dict:
    """The page in a fresh profile (empty cache): seconds from navigation until a move can be made."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    errors: list[str] = []
    page = _open(browser, errors)
    started = time.perf_counter()
    page.goto(url)
    try:
        page.wait_for_function(WAIT_PAGE, timeout=timeout_ms)
    except PlaywrightTimeout:
        waited = f"not ready after {timeout_ms / 1000:g} s"
        return {"ready": False, "cold_load_s": None, "console_errors": [*errors, waited]}
    seconds = time.perf_counter() - started
    state = page.evaluate("window.__blink.state()")
    page.context.close()
    return {
        "ready": True,
        "cold_load_s": seconds,
        "backend": state["backend"],
        "card": state["card"],
        "console_errors": errors,
    }


def _bench_page(browser, url: str, timeout_ms: float) -> tuple[dict, list[str]]:
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    errors: list[str] = []
    page = _open(browser, errors)
    page.goto(url)
    try:
        page.wait_for_function(WAIT_BENCH, timeout=timeout_ms)
    except PlaywrightTimeout as exc:
        raise BenchError(f"bench.html did not finish within {timeout_ms / 1000:g} s: {errors[:3]}") from exc
    result = page.evaluate("window.__blinkBench")
    page.context.close()
    if result.get("error"):
        raise BenchError(f"bench.html failed: {result['error']}")
    return result, errors


def file_info(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": card.sha256_of(path)}


def machine_info(browser_version: str) -> dict:
    import psutil

    return {
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "cpu_percent_before": psutil.cpu_percent(interval=1.0),
        "os": platform.platform(),
        "browser": f"Microsoft Edge {browser_version} (headless, Playwright channel msedge)",
    }


def run_bench(
    int8: Path,
    fp32: Path,
    runs: int,
    warmup: int,
    backends: Sequence[str],
    timeout_s: float,
    network: Network = COLD_LOAD_NETWORK,
) -> dict:
    """Stage, serve and measure (the cold load paced over `network`); returns the site_bench.json report."""
    pairs, skipped = plan(PRECISIONS, backends)
    timeout_ms = timeout_s * 1000
    link = Link(network)
    with tempfile.TemporaryDirectory(prefix="blink-bench-") as tmp:
        root = stage_tree(SITE_DIR, Path(tmp) / "site", int8, fp32)
        with _edge() as browser:
            machine = machine_info(browser.version)
            with serving(root, link) as url:
                cold = _page_cold_load(browser, url, timeout_ms)
            with serving(root) as url:
                bench_url = url + "bench.html" + query(pairs, runs, warmup)
                result, errors = _bench_page(browser, bench_url, timeout_ms)
    page = {**cold, "network": network.to_dict(), "bytes_served": link.bytes_sent}
    rows = summarize(result["results"])
    return {
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "machine": machine,
        "method": (
            f"one look = one session run at batch 1 on seeded game positions; {warmup} untimed then "
            f"{runs} timed per pair in a fresh worker; look_ms is the page's round trip (message, run, "
            "reply), run_ms the run alone; files served on loopback; the cold load paced over "
            f"{network.describe()}"
        ),
        "models": {"int8": file_info(int8), "fp32": file_info(fp32)},
        "page": page,
        "env": result.get("env", {}),
        "backends": rows,
        "skipped": skipped + [s for s in result.get("skipped", []) if s not in skipped],
        "bench_console_errors": errors,
        "gate": gate(int8.stat().st_size, rows, page.get("cold_load_s"), network),
    }


def write(report: dict, out: Path) -> Path:
    import json

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return out


# ----------------------------------------------------------------------------- the post-deploy selftest


def selftest_url(url: str) -> str:
    parts = urlsplit(url)
    if "selftest" in {item.split("=")[0] for item in parts.query.split("&") if item}:
        return url
    joined = f"{parts.query}&selftest=1" if parts.query else "selftest=1"
    return urlunsplit(parts._replace(query=joined))


def selftest(url: str, timeout_s: float = 60.0) -> SelfTest:
    """Load the page with ?selftest=1 and read window.__blinkSelfTest (ok, move, ms, histogram, backend)."""
    from playwright.sync_api import TimeoutError as PlaywrightTimeout

    errors: list[str] = []
    with _edge() as browser:
        page = _open(browser, errors)
        page.goto(selftest_url(url))
        try:
            page.wait_for_function(WAIT_SELFTEST, timeout=timeout_s * 1000)
        except PlaywrightTimeout:
            result = {"ok": False, "error": f"no __blinkSelfTest within {timeout_s:g} s"}
            return SelfTest(ok=False, result=result, console_errors=tuple(errors))
        result = page.evaluate("window.__blinkSelfTest")
    return SelfTest(ok=result.get("ok") is True, result=result, console_errors=tuple(errors))
