"""`blink site serve | stage | smoke | bench | replay | pieces | vendor`: the browser page and its checks."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = REPO_ROOT / "site"


def _cmd_serve(args: argparse.Namespace) -> int:
    from blink.site import serve

    try:
        model = serve.resolve_model(Path(args.model))
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    serve.run(serve.SiteConfig(site_dir=SITE_DIR, model=model), port=args.port)
    return 0


def _cmd_stage(args: argparse.Namespace) -> int:
    from blink.site import stage

    out = Path(args.out)
    try:
        written = stage.stage(SITE_DIR, out, model=Path(args.model) if args.model else None)
    except (stage.StageError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    total = sum(path.stat().st_size for path in written)
    print(f"ok: {len(written)} files, {total:,} B in {out}")
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    from blink.site import smoke

    report = smoke.run(args.url, moves=args.moves, seed=args.seed, timeout_s=args.timeout)
    print(json.dumps(report.to_dict(), indent=2))
    failures = smoke.failures(report, moves=args.moves)
    if args.selftest:
        failures += _selftest_failures(args.url, args.timeout)
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print(f"ok: {report.legal_replies} legal replies, {report.arrows} arrows, 0 console errors")
    return 1 if failures else 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from blink import paths
    from blink.site import replay

    try:
        run = replay.run_dir(paths.home() / "runs", args.run)
    except replay.ReplayError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = replay.build(run, Path(args.out), every=args.every)
    print(json.dumps(summary, indent=2))
    rows = f"{summary['metrics_rows']} metrics rows and {summary['evals_rows']} eval rows"
    print(f"ok: {rows} (one per {args.every:,} steps) in {args.out}")
    return 0


def _selftest_failures(url: str, timeout: float) -> list[str]:
    from blink.site import bench

    check = bench.selftest(url, timeout_s=timeout)
    print(json.dumps({"selftest": check.result}, indent=2))
    failures = [] if check.ok else ["__blinkSelfTest.ok is not true"]
    return failures + [f"console error during the selftest: {error}" for error in check.console_errors]


def _print_bench(report: dict) -> None:
    for row in report["backends"]:
        look = row.get("look_ms", {})
        timing = f"look p50 {look['p50']:.2f} ms" if "p50" in look else row.get("reason", "")
        cold = f", cold {row['cold_ms']:.0f} ms" if "cold_ms" in row else ""
        print(f"  {row['precision']:>4} {row['backend']:<8} {row['status']:<11} {timing}{cold}")
    for row in report["skipped"]:
        print(f"  {row['precision']:>4} {row['backend']:<8} skipped     {row['reason']}")
    for failure in report["gate"]["failures"]:
        print(f"FAIL: {failure}")


def _cmd_bench(args: argparse.Namespace) -> int:
    from blink import paths
    from blink.export import quantize
    from blink.site import bench

    export_dir = quantize.resolve_export_dir(args.model)
    fp32, int8 = quantize.fp32_path(export_dir), quantize.int8_path(export_dir)
    for path, hint in ((fp32, "blink export onnx"), (int8, "blink export quantize --int8")):
        if not path.is_file():
            print(f"error: no model at {path} (run: {hint} --model {args.model})", file=sys.stderr)
            return 2
    backends = tuple(name.strip() for name in args.backends.split(",") if name.strip())
    try:
        report = bench.run_bench(int8, fp32, args.runs, args.warmup, backends, args.timeout)
    except bench.BenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = bench.write(report, Path(args.out) if args.out else paths.home() / "eval" / "site_bench.json")
    _print_bench(report)
    verdict = "passes" if report["gate"]["passed"] else "fails"
    print(f"ok: wrote {out}; the browser-model gate {verdict}")
    return 0 if report["gate"]["passed"] else 1


def _cmd_pieces(args: argparse.Namespace) -> int:
    from blink.site import pieces

    out = pieces.write_sprite(Path(args.src), Path(args.out))
    print(f"ok: {out} ({out.stat().st_size:,} B, 12 pieces)")
    return 0


def _cmd_vendor(args: argparse.Namespace) -> int:
    from blink.site import vendor

    target = SITE_DIR / "vendor" / "cm-chessboard"
    written = vendor.vendor_cm_chessboard(SITE_DIR / "node_modules" / "cm-chessboard", target)
    print(f"ok: {len(written)} files into {target}")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    from blink.site import serve

    site = sub.add_parser("site", help="the local browser page: serve it, smoke-test it, build its assets")
    tasks = site.add_subparsers(dest="site_command", required=True)

    serve_cmd = tasks.add_parser("serve", help=f"serve site/ on http://{serve.HOST}:<port> (loopback only)")
    serve_cmd.add_argument("--model", required=True, help="model.onnx, or the directory holding it")
    serve_cmd.add_argument("--port", type=int, default=serve.DEFAULT_PORT)
    serve_cmd.set_defaults(func=_cmd_serve)

    stage_cmd = tasks.add_parser("stage", help="copy the deployable page (and its npm files) into a new dir")
    stage_cmd.add_argument("--out", required=True, help="a new or empty directory outside site/")
    stage_cmd.add_argument("--model", help="model.onnx, or the directory holding it (with its model.json)")
    stage_cmd.set_defaults(func=_cmd_stage)

    smoke_cmd = tasks.add_parser("smoke", help="Playwright on Edge: legal replies, 3 arrows, no errors")
    smoke_cmd.add_argument("--url", default=f"http://{serve.HOST}:{serve.DEFAULT_PORT}/")
    smoke_cmd.add_argument("--moves", type=int, default=10, help="user moves to play (each needs a reply)")
    smoke_cmd.add_argument("--seed", type=int, default=0)
    smoke_cmd.add_argument("--timeout", type=float, default=60.0, help="seconds to wait for the model")
    smoke_cmd.add_argument(
        "--selftest", action="store_true", help="also load ?selftest=1: __blinkSelfTest.ok"
    )
    smoke_cmd.set_defaults(func=_cmd_smoke)

    bench_cmd = tasks.add_parser("bench", help="one-look latency per backend in Edge -> BLINK_HOME/eval")
    bench_cmd.add_argument(
        "--model", default="ship", help="export dir holding model.onnx and int8/model.onnx"
    )
    bench_cmd.add_argument("--runs", type=int, default=200, help="timed looks per model and backend")
    bench_cmd.add_argument("--warmup", type=int, default=10)
    bench_cmd.add_argument(
        "--backends", default="wasm-1t,wasm-mt,webgpu", help="comma list; int8 skips webgpu"
    )
    bench_cmd.add_argument("--timeout", type=float, default=600.0, help="seconds for the whole bench page")
    bench_cmd.add_argument("--out", help="default: BLINK_HOME/eval/site_bench.json")
    bench_cmd.set_defaults(func=_cmd_bench)

    replay_cmd = tasks.add_parser(
        "replay", help="freeze a run's curves into site/replay (1 row per 2,000 steps)"
    )
    replay_cmd.add_argument("--run", required=True, help="a run under BLINK_HOME/runs")
    replay_cmd.add_argument("--every", type=int, default=2000, help="steps per kept row")
    replay_cmd.add_argument("--out", default=str(SITE_DIR / "replay"))
    replay_cmd.set_defaults(func=_cmd_replay)

    pieces_cmd = tasks.add_parser("pieces", help="build the cburnett sprite from the 12 Commons SVGs")
    pieces_cmd.add_argument("--src", required=True, help="directory holding Chess_{k,q,r,b,n,p}{l,d}t45.svg")
    pieces_cmd.add_argument("--out", default=str(SITE_DIR / "assets" / "pieces" / "cburnett.svg"))
    pieces_cmd.set_defaults(func=_cmd_pieces)

    vendor_cmd = tasks.add_parser("vendor", help="copy cm-chessboard's core src from site/node_modules")
    vendor_cmd.set_defaults(func=_cmd_vendor)
