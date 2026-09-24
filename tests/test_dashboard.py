import http.client
import inspect
import json
import threading

import pytest

from blink import cli, heartbeat
from blink.commands import dashboard as dashboard_command
from blink.dashboard import server


@pytest.fixture
def runs_root(tmp_path):
    root = tmp_path / "runs"
    (root / "alpha").mkdir(parents=True)
    (tmp_path / "secret.txt").write_text("TOP SECRET", encoding="utf-8")
    return root


@pytest.fixture
def live_server(runs_root):
    httpd = server.make_server(runs_root, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()


def _get(httpd, path: str, host: str | None = None) -> tuple[int, bytes, dict]:
    port = httpd.server_address[1]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": host} if host else {}
    conn.request("GET", path, headers=headers)
    response = conn.getresponse()
    body = response.read()
    conn.close()
    return response.status, body, dict(response.getheaders())


def test_dashboard_tail_returns_only_complete_lines_from_an_offset(runs_root, live_server):
    metrics = runs_root / "alpha" / "metrics.jsonl"
    metrics.write_bytes(b'{"step": 1}\n{"step": 50}\n{"step": 10')
    first = server.tail_lines(metrics, 0)
    assert first["lines"] == ['{"step": 1}', '{"step": 50}']
    assert first["offset"] == len(b'{"step": 1}\n{"step": 50}\n')

    status, body, _ = _get(live_server, f"/api/tail?run=alpha&file=metrics.jsonl&offset={first['offset']}")
    assert status == 200 and json.loads(body)["lines"] == []

    with open(metrics, "ab") as handle:
        handle.write(b'0}\n{"step": 150}\n')
    status, body, _ = _get(live_server, f"/api/tail?run=alpha&file=metrics.jsonl&offset={first['offset']}")
    payload = json.loads(body)
    assert payload["lines"] == ['{"step": 100}', '{"step": 150}']
    assert payload["offset"] == metrics.stat().st_size and payload["reset"] is False


def test_a_file_rewritten_shorter_is_tailed_again_from_zero(runs_root):
    metrics = runs_root / "alpha" / "metrics.jsonl"
    metrics.write_bytes(b'{"step": 1}\n')
    payload = server.tail_lines(metrics, 500)
    assert payload["reset"] is True and payload["lines"] == ['{"step": 1}']


def test_a_missing_run_file_tails_as_empty(runs_root):
    payload = server.tail_lines(runs_root / "alpha" / "evals.jsonl", 0)
    assert payload == {"lines": [], "offset": 0, "reset": False}


@pytest.mark.parametrize(
    "query",
    [
        "run=..&file=metrics.jsonl",
        "run=../alpha&file=metrics.jsonl",
        "run=%2e%2e&file=secret.txt",
        "run=alpha&file=../../secret.txt",
        "run=alpha&file=..%2F..%2Fsecret.txt",
        "run=alpha&file=config.json",
        "run=alpha&file=heartbeat.json",
        "run=C:&file=metrics.jsonl",
        "run=alpha&file=metrics.jsonl&offset=-5",
        "run=alpha&file=metrics.jsonl&offset=abc",
        "file=metrics.jsonl",
    ],
)
def test_dashboard_rejects_path_traversal(live_server, query):
    status, body, _ = _get(live_server, f"/api/tail?{query}")
    assert status == 400
    assert b"TOP SECRET" not in body


def test_the_path_resolver_refuses_anything_outside_the_runs_root(runs_root):
    assert (
        server.resolve_tail_path(runs_root, "alpha", "metrics.jsonl")
        == (runs_root / "alpha" / "metrics.jsonl").resolve()
    )
    for run, name in (("..", "metrics.jsonl"), ("alpha", "../x"), ("alpha/..", "evals.jsonl")):
        with pytest.raises(ValueError):
            server.resolve_tail_path(runs_root, run, name)


def test_dashboard_binds_loopback_only(runs_root):
    httpd = server.make_server(runs_root, port=0)
    try:
        assert httpd.server_address[0] == "127.0.0.1" == server.HOST
    finally:
        httpd.server_close()
    assert "host" not in inspect.signature(server.make_server).parameters
    assert "host" not in inspect.signature(server.serve).parameters
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["dashboard", "--host", "0.0.0.0"])
    assert server.DEFAULT_PORT == 8767


def test_a_foreign_host_header_is_refused(live_server):
    status, _, _ = _get(live_server, "/api/runs", host="evil.example:8767")
    assert status == 403


def test_the_runs_api_marks_a_fresh_heartbeat_live(runs_root, live_server):
    heartbeat.write(runs_root / "alpha" / "heartbeat.json", {"state": "running", "step": 7, "steps": 10})
    (runs_root / "beta").mkdir()
    status, body, headers = _get(live_server, "/api/runs")
    runs = {r["name"]: r for r in json.loads(body)["runs"]}
    assert status == 200 and headers["Content-Type"].startswith("application/json")
    assert runs["alpha"]["live"] is True and runs["alpha"]["step"] == 7
    assert runs["beta"]["live"] is False


def test_the_page_is_served_with_the_dark_palette_and_six_charts(live_server):
    status, body, headers = _get(live_server, "/")
    page = body.decode("utf-8")
    assert status == 200 and headers["Content-Type"].startswith("text/html")
    for colour in ("#0b0e13", "#ffd54a", "#3ddc84"):
        assert colour in page
    for chart in ("policy", "value", "top1", "speed", "lr", "grad"):
        assert f'data-chart="{chart}"' in page
    assert "no-store" in headers["Cache-Control"]


def test_unknown_paths_are_not_found(live_server):
    assert _get(live_server, "/etc/passwd")[0] == 404
    assert _get(live_server, "/api/nope")[0] == 404


def test_the_dashboard_command_serves_the_runs_under_blink_home(tmp_path, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(tmp_path))
    seen = {}
    monkeypatch.setattr(
        dashboard_command.server, "serve", lambda root, port: seen.update(root=root, port=port)
    )
    assert cli.main(["dashboard"]) == 0
    assert seen == {"root": tmp_path / "runs", "port": 8767}
    assert cli.main(["dashboard", "--port", "9001"]) == 0
    assert seen["port"] == 9001
