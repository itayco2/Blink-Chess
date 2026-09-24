"""`blink site serve | stage | smoke | pieces | vendor`: the browser page, its deploy tree and its checks."""

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
    for failure in failures:
        print(f"FAIL: {failure}")
    if not failures:
        print(f"ok: {report.legal_replies} legal replies, {report.arrows} arrows, 0 console errors")
    return 1 if failures else 0


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
    smoke_cmd.set_defaults(func=_cmd_smoke)

    pieces_cmd = tasks.add_parser("pieces", help="build the cburnett sprite from the 12 Commons SVGs")
    pieces_cmd.add_argument("--src", required=True, help="directory holding Chess_{k,q,r,b,n,p}{l,d}t45.svg")
    pieces_cmd.add_argument("--out", default=str(SITE_DIR / "assets" / "pieces" / "cburnett.svg"))
    pieces_cmd.set_defaults(func=_cmd_pieces)

    vendor_cmd = tasks.add_parser("vendor", help="copy cm-chessboard's core src from site/node_modules")
    vendor_cmd.set_defaults(func=_cmd_vendor)
