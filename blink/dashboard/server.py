"""Live dashboard server (a port of driving-rl's live.py).

Binds 127.0.0.1 only, hard-coded with no override: no firewall prompt and no LAN exposure. Requests
whose Host header is not the loopback address are refused, which also blocks DNS rebinding.

Routes:
  /                     live.html
  /api/runs             every run under the runs root with its LIVE badge (heartbeat age <= 30 s)
  /api/tail?run=&file=&offset=
                        complete lines of metrics.jsonl or evals.jsonl from a byte offset, plus the
                        offset to ask for next; a torn last line is left for the next poll
"""

import json
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from blink.train.status import exit_code, list_runs, valid_run_name

HOST = "127.0.0.1"
DEFAULT_PORT = 8767
TAIL_FILES = frozenset({"metrics.jsonl", "evals.jsonl"})
MAX_TAIL_BYTES = 1 << 20
LIVE_HTML = Path(__file__).with_name("live.html")
LOOPBACK_NAMES = ("127.0.0.1", "localhost")
RUN_FIELDS = ("name", "live", "state", "step", "steps", "heartbeat_age_s")


def resolve_tail_path(runs_root: Path, run: str, name: str) -> Path:
    """The file to tail, or ValueError for any run or file name that could leave the runs root."""
    if not valid_run_name(run) or name not in TAIL_FILES:
        raise ValueError(f"refused: run={run!r} file={name!r}")
    root = runs_root.resolve()
    path = (root / run / name).resolve()
    if path.parent.parent != root:
        raise ValueError(f"refused: {path} is outside {root}")
    return path


def tail_lines(path: Path, offset: int, max_bytes: int = MAX_TAIL_BYTES) -> dict[str, Any]:
    """Complete lines from a byte offset. An offset past the end means the file was rewritten: restart."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return {"lines": [], "offset": 0, "reset": False}
    reset = offset > size
    start = 0 if reset else offset
    with open(path, "rb") as handle:
        handle.seek(start)
        chunk = handle.read(max_bytes)
    end = chunk.rfind(b"\n") + 1
    lines = chunk[:end].decode("utf-8", errors="replace").splitlines()
    return {"lines": lines, "offset": start + end, "reset": reset}


def runs_payload(runs_root: Path) -> dict[str, Any]:
    """Only the badge fields: charts come from /api/tail, and a NaN loss must not break strict JSON."""
    runs = []
    for report in list_runs(runs_root):
        fields = asdict(report)
        runs.append({**{k: fields[k] for k in RUN_FIELDS}, "healthy": exit_code(report) == 0})
    return {"runs": runs}


class DashboardHandler(BaseHTTPRequestHandler):
    runs_root: Path = Path(".")
    server_version = "BlinkDashboard/1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - the stdlib's signature
        return  # quiet: the trainer's log is the one that matters

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in LOOPBACK_NAMES

    def do_GET(self) -> None:  # noqa: N802 - the stdlib's name
        if not self._host_ok():
            self._json(HTTPStatus.FORBIDDEN, {"error": "loopback host only"})
            return
        url = urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            self._send(HTTPStatus.OK, LIVE_HTML.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/api/runs":
            self._json(HTTPStatus.OK, runs_payload(self.runs_root))
        elif url.path == "/api/tail":
            self._tail(parse_qs(url.query))
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _tail(self, query: dict[str, list[str]]) -> None:
        try:
            run, name = query["run"][0], query["file"][0]
            offset = int(query.get("offset", ["0"])[0])
            if offset < 0:
                raise ValueError("negative offset")
            path = resolve_tail_path(self.runs_root, run, name)
        except (KeyError, IndexError, ValueError):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "bad run, file or offset"})
            return
        self._json(HTTPStatus.OK, tail_lines(path, offset))


def make_server(runs_root: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """A server bound to 127.0.0.1 only. There is deliberately no host parameter."""
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"runs_root": Path(runs_root)})
    httpd = ThreadingHTTPServer((HOST, port), handler)
    httpd.daemon_threads = True
    return httpd


def serve(runs_root: Path, port: int = DEFAULT_PORT) -> None:
    httpd = make_server(runs_root, port)
    print(f"dashboard: http://{HOST}:{httpd.server_address[1]}/ (runs under {runs_root}); Ctrl+C stops it")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
