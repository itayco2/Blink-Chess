"""`blink export onnx | vocab | golden | rules`: the files the browser page runs on and is checked against."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from blink import paths

REPO_ROOT = Path(__file__).resolve().parents[2]
SITE_DIR = REPO_ROOT / "site"
PARITY_POSITIONS = 1_000
PARITY_SEED = 7
MODEL_HELP = "stand-in | run:<name>[:ema] | ship | release:<tag> | <.pt path>"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def default_onnx_path(selector: str) -> Path:
    from blink.export import models

    return paths.home() / "export" / models.slug(selector) / models.MODEL_FILE


def _cmd_onnx(args: argparse.Namespace) -> int:
    from blink.export import models, positions
    from blink.export import onnx as export_onnx

    out = Path(args.out) if args.out else default_onnx_path(args.model)
    module = models.load_module(args.model)
    started = time.perf_counter()
    export_onnx.export(module, out)
    seconds = time.perf_counter() - started
    tokens = positions.encode_fens(positions.random_fens(PARITY_POSITIONS, seed=PARITY_SEED))
    report = export_onnx.compare(module, out, tokens)
    card = {
        "selector": args.model,
        "file": out.name,
        "bytes": out.stat().st_size,
        "sha256": sha256_of(out),
        "opset": export_onnx.OPSET,
        "parameters": sum(p.numel() for p in module.parameters()),
        "parity_positions": report.positions,
        "max_abs_policy": report.max_abs_policy,
        "max_abs_value": report.max_abs_value,
        "export_seconds": round(seconds, 2),
    }
    card_path = out.with_name("model.json")
    card_path.write_text(json.dumps(card, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(card, indent=2))
    if not report.passed:
        print(f"FAIL: max |torch - onnx| {report.max_abs:.3g} > {export_onnx.PARITY_TOLERANCE}")
        return 1
    print(f"ok: {out} ({card['bytes']:,} B, opset {card['opset']}, max |torch - onnx| {report.max_abs:.3g})")
    return 0


def _cmd_vocab(args: argparse.Namespace) -> int:
    from blink.export import vocab

    out = vocab.write(Path(args.out))
    print(f"ok: {out} ({out.stat().st_size:,} B)")
    return 0


def _cmd_rules(args: argparse.Namespace) -> int:
    from blink.export import rulecases

    out = rulecases.write(Path(args.out))
    print(f"ok: {out} ({len(rulecases.RULE_CASES)} cases, {out.stat().st_size:,} B)")
    return 0


def _golden_model_info(selector: str, onnx: Path | None) -> dict:
    from blink.export import models

    info: dict = {"selector": selector}
    candidate = onnx or models.onnx_path(selector) or default_onnx_path(selector)
    if candidate.is_file():
        info["onnx_sha256"] = sha256_of(candidate)
    return info


def _cmd_golden(args: argparse.Namespace) -> int:
    from blink.export import golden, models

    evaluator = models.load_evaluator(args.model)
    info = _golden_model_info(args.model, Path(args.onnx) if args.onnx else None)
    data = golden.build(evaluator, info)
    out = golden.write(data, Path(args.out))
    tagged = {tag for entry in data["positions"] for tag in entry["tags"]}
    print(f"ok: {out} ({len(data['positions'])} positions; tags {sorted(tagged)}; model {info})")
    if "onnx_sha256" not in info:
        print("note: no ONNX file found for this selector, so the Node parity test will skip onnxruntime-web")
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    export = sub.add_parser("export", help="ONNX export, move vocabulary, golden positions, rule cases")
    tasks = export.add_subparsers(dest="export_command", required=True)

    onnx = tasks.add_parser("onnx", help="write one self-contained opset-20 ONNX file and check it")
    onnx.add_argument("--model", required=True, help=MODEL_HELP)
    onnx.add_argument("--out", help="default: BLINK_HOME/export/<model>/model.onnx")
    onnx.set_defaults(func=_cmd_onnx)

    vocab = tasks.add_parser("vocab", help="write site/vocab.json from the frozen contract")
    vocab.add_argument("--out", default=str(SITE_DIR / "vocab.json"))
    vocab.set_defaults(func=_cmd_vocab)

    gold = tasks.add_parser("golden", help="write site/tests/golden.json (50 positions, one batch)")
    gold.add_argument("--model", required=True, help="stand-in | <.onnx path or dir> | a train selector")
    gold.add_argument("--onnx", help="the ONNX file the Node parity test runs (its sha256 is recorded)")
    gold.add_argument("--out", default=str(SITE_DIR / "tests" / "golden.json"))
    gold.set_defaults(func=_cmd_golden)

    rules = tasks.add_parser("rules", help="write site/tests/rules.json (R2 and R3 per child)")
    rules.add_argument("--out", default=str(SITE_DIR / "tests" / "rules.json"))
    rules.set_defaults(func=_cmd_rules)
