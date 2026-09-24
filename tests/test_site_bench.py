"""`blink site bench`: one-look latency per backend in the local Edge, written to BLINK_HOME/eval (P10)."""

import json
import sys
import urllib.request
from pathlib import Path

import pytest

from blink import cli

REPO = Path(__file__).resolve().parent.parent
SITE = REPO / "site"


def _row(precision="int8", backend="wasm-1t", p50=12.0, status="ok", **extra):
    return {"precision": precision, "backend": backend, "status": status, "look_ms": {"p50": p50}, **extra}


def test_int8_is_never_sent_to_webgpu():
    from blink.site import bench

    pairs, skipped = bench.plan(("int8", "fp32"), bench.BACKENDS)
    assert bench.Pair("int8", "webgpu") not in pairs
    assert bench.Pair("fp32", "webgpu") in pairs and bench.Pair("int8", "wasm-1t") in pairs
    assert [(s["precision"], s["backend"]) for s in skipped] == [("int8", "webgpu")]
    assert "PF30" in skipped[0]["reason"]
    with pytest.raises(bench.BenchError, match="int8 on webgpu"):
        bench.summarize(
            [{"precision": "int8", "backend": "webgpu", "status": "ok", "run_ms": [1.0], "look_ms": [1.0]}]
        )
    plan_js = (SITE / "bench" / "plan.js").read_text(encoding="utf-8")
    assert 'precision === "int8" && BACKENDS[backend].gpu' in plan_js
    assert "never gets int8 (PF30)" in plan_js


def test_the_plan_refuses_an_unknown_backend_or_precision():
    from blink.site import bench

    with pytest.raises(bench.BenchError, match="webgl"):
        bench.plan(("int8",), ("webgl",))
    with pytest.raises(bench.BenchError, match="fp16"):
        bench.plan(("fp16",), ("wasm-1t",))


def test_the_bench_query_names_each_pair_and_the_run_counts():
    from blink.site import bench

    pairs, _ = bench.plan(("int8", "fp32"), ("wasm-1t", "webgpu"))
    query = bench.query(pairs, runs=200, warmup=10)
    assert query == "?pairs=int8:wasm-1t,fp32:wasm-1t,fp32:webgpu&runs=200&warmup=10"


def test_latency_summaries_are_nearest_rank_percentiles():
    from blink.site import bench

    rows = bench.summarize(
        [
            {
                "precision": "int8",
                "backend": "wasm-1t",
                "status": "ok",
                "threads": 1,
                "cold_ms": 850.0,
                "first_run_ms": 9.0,
                "run_ms": [float(x) for x in range(1, 101)],
                "look_ms": [float(x) + 0.5 for x in range(1, 101)],
            },
            {
                "precision": "fp32",
                "backend": "webgpu",
                "status": "unavailable",
                "reason": "no WebGPU adapter",
            },
        ]
    )
    ok, gpu = rows
    assert ok["run_ms"] == {"p50": 50.0, "p90": 90.0, "p99": 99.0, "mean": 50.5, "min": 1.0, "n": 100}
    assert ok["look_ms"]["p50"] == 50.5
    assert ok["cold_ms"] == 850.0 and ok["threads"] == 1
    assert gpu == {
        "precision": "fp32",
        "backend": "webgpu",
        "status": "unavailable",
        "reason": "no WebGPU adapter",
    }


def test_the_browser_gate_needs_int8_within_30_mb_p50_within_250_ms_and_cold_load_within_5_s():
    from blink.site import bench

    fast = bench.gate(int8_bytes=419_787, rows=[_row(p50=12.0)], cold_load_s=1.3)
    assert fast["passed"] is True and fast["failures"] == []
    assert fast["p50_ms"] == 12.0 and fast["int8_bytes"] == 419_787
    slow = bench.gate(int8_bytes=31_000_000, rows=[_row(p50=260.0)], cold_load_s=5.5)
    assert slow["failures"] == [
        "int8 file 31,000,000 B is over 30,000,000 B",
        "one look p50 260.0 ms on int8 wasm-1t is over 250 ms",
        "cold load 5.50 s is over 5 s",
    ]
    missing = bench.gate(int8_bytes=1, rows=[_row(backend="wasm-mt")], cold_load_s=1.0)
    assert missing["failures"] == ["no int8 wasm-1t measurement: the shipped path was not measured"]


def _bench_site(tmp_path: Path) -> Path:
    from test_site_serve import _fake_tree

    site, _, _ = _fake_tree(tmp_path)
    ort = site / "node_modules" / "onnxruntime-web" / "dist"
    for name in (
        "ort.webgpu.bundle.min.mjs",
        "ort-wasm-simd-threaded.jsep.wasm",
        "ort-wasm-simd-threaded.jsep.mjs",
        "ort-wasm-simd-threaded.asyncify.wasm",
        "ort-wasm-simd-threaded.asyncify.mjs",
    ):
        (ort / name).write_bytes(b"x")
    return site


def test_the_bench_tree_is_the_staged_page_plus_fp32_and_the_gpu_runtime_files(tmp_path):
    from blink.site import bench

    site = _bench_site(tmp_path)
    int8 = tmp_path / "int8" / "model.onnx"
    int8.parent.mkdir()
    int8.write_bytes(b"int8")
    (int8.parent / "model.json").write_text('{"precision": "int8"}', encoding="utf-8")
    fp32 = tmp_path / "fp32.onnx"
    fp32.write_bytes(b"fp32")
    out = bench.stage_tree(site, tmp_path / "bench", int8, fp32)
    assert (out / "models" / "model.onnx").read_bytes() == b"int8"
    assert (out / "models" / "fp32.onnx").read_bytes() == b"fp32"
    assert (out / "models" / "model.json").is_file()
    assert (out / "vendor" / "ort" / "ort.webgpu.bundle.min.mjs").is_file()
    assert (out / "vendor" / "ort" / "ort.wasm.bundle.min.mjs").is_file()
    assert not (out / "vendor" / "ort" / "ort.all.min.mjs").exists()


def test_the_bench_server_is_loopback_only_and_cross_origin_isolated(tmp_path):
    from blink.site import bench

    (tmp_path / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (tmp_path / "a.mjs").write_text("export {}", encoding="utf-8")
    with bench.serving(tmp_path) as url:
        assert url.startswith("http://127.0.0.1:")
        with urllib.request.urlopen(url + "a.mjs", timeout=5) as response:
            headers = response.headers
    assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
    assert headers["Cross-Origin-Embedder-Policy"] == "require-corp"
    assert headers["Content-Type"].startswith("text/javascript")


def _fake_report(passed: bool) -> dict:
    return {
        "backends": [_row()],
        "skipped": [],
        "gate": {"passed": passed, "failures": [] if passed else ["x"]},
    }


@pytest.mark.parametrize("passed", [True, False])
def test_site_bench_writes_its_report_under_blink_home_eval_and_exits_on_the_gate(
    tmp_path, monkeypatch, capsys, passed
):
    from blink.site import bench

    monkeypatch.setenv("BLINK_HOME", str(tmp_path / "home"))
    export = tmp_path / "export"
    (export / "int8").mkdir(parents=True)
    (export / "model.onnx").write_bytes(b"fp32")
    (export / "int8" / "model.onnx").write_bytes(b"int8")
    seen = {}

    def fake_run(int8, fp32, runs, warmup, backends, timeout_s):
        seen.update(int8=int8, fp32=fp32, runs=runs, backends=backends)
        return _fake_report(passed)

    monkeypatch.setattr(bench, "run_bench", fake_run)
    code = cli.main(["site", "bench", "--model", str(export), "--runs", "50", "--backends", "wasm-1t,webgpu"])
    assert code == (0 if passed else 1)
    written = json.loads((tmp_path / "home" / "eval" / "site_bench.json").read_text(encoding="utf-8"))
    assert written["gate"]["passed"] is passed
    assert seen == {
        "int8": export / "int8" / "model.onnx",
        "fp32": export / "model.onnx",
        "runs": 50,
        "backends": ("wasm-1t", "webgpu"),
    }
    assert "site_bench.json" in capsys.readouterr().out


def test_site_bench_names_the_missing_model_and_the_command_that_makes_it(tmp_path, capsys):
    assert cli.main(["site", "bench", "--model", str(tmp_path)]) == 2
    assert "blink export onnx" in capsys.readouterr().err


def test_the_top_level_bench_page_only_loads_its_module_from_bench():
    page = (SITE / "bench.html").read_text(encoding="utf-8")
    assert '<script type="module" src="bench/bench.js"></script>' in page
    assert "ort." not in page


@pytest.mark.parametrize("ok", [True, False])
def test_smoke_with_selftest_also_requires_blink_self_test_ok(monkeypatch, capsys, ok):
    from test_site_smoke import _report

    from blink.site import bench, smoke

    monkeypatch.setattr(smoke, "run", lambda url, moves, seed, timeout_s: _report())
    calls = []

    def fake_selftest(url, timeout_s):
        calls.append(url)
        return bench.SelfTest(ok=ok, result={"ok": ok, "move": "e2e4"}, console_errors=())

    monkeypatch.setattr(bench, "selftest", fake_selftest)
    code = cli.main(["site", "smoke", "--url", "https://example.test/Blink-Chess/?selftest=1", "--selftest"])
    assert calls == ["https://example.test/Blink-Chess/?selftest=1"]
    assert code == (0 if ok else 1)
    assert ("FAIL: __blinkSelfTest.ok is not true" in capsys.readouterr().out) is not ok


def test_the_selftest_url_gains_the_query_only_when_it_lacks_it():
    from blink.site import bench

    assert bench.selftest_url("https://x.test/Blink-Chess/") == "https://x.test/Blink-Chess/?selftest=1"
    assert bench.selftest_url("https://x.test/?selftest=1") == "https://x.test/?selftest=1"
    assert bench.selftest_url("http://127.0.0.1:8000/?a=b") == "http://127.0.0.1:8000/?a=b&selftest=1"


EDGE = Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")


@pytest.mark.local
@pytest.mark.torch
def test_the_bench_measures_int8_and_fp32_on_wasm_in_edge(tmp_path):
    if sys.platform != "win32" or not EDGE.is_file():
        pytest.skip("Microsoft Edge is not installed")
    if not (SITE / "node_modules" / "onnxruntime-web").is_dir():
        pytest.skip("site/node_modules is absent (run npm ci --prefix site)")
    from blink.export import onnx as export_onnx
    from blink.export import quantize, standin
    from blink.site import bench, card

    fp32 = export_onnx.export(standin.build(seed=0), tmp_path / "model.onnx")
    int8 = quantize.quantize_int8(fp32, tmp_path / "int8" / "model.onnx")
    fp32_card = {"selector": "stand-in", "bytes": fp32.stat().st_size, "sha256": card.sha256_of(fp32)}
    card.write(card.for_file(fp32_card, int8, "int8"), int8.with_name("model.json"))
    report = bench.run_bench(int8, fp32, runs=20, warmup=3, backends=("wasm-1t", "wasm-mt"), timeout_s=180)
    rows = {(r["precision"], r["backend"]): r for r in report["backends"]}
    assert set(rows) == {("int8", "wasm-1t"), ("int8", "wasm-mt"), ("fp32", "wasm-1t"), ("fp32", "wasm-mt")}
    assert rows[("int8", "wasm-1t")]["status"] == "ok" and rows[("int8", "wasm-1t")]["run_ms"]["n"] == 20
    assert rows[("int8", "wasm-mt")]["threads"] > 1, "the bench server is cross-origin isolated"
    assert report["page"]["ready"] is True and report["page"]["console_errors"] == []
    assert report["gate"]["int8_bytes"] == int8.stat().st_size
