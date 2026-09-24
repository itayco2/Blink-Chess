"""`blink export onnx | quantize | qgate | vocab | golden | rules`: the files the browser page runs on."""

import argparse
import hashlib
import json
import sys
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


def _cmd_quantize(args: argparse.Namespace) -> int:
    from blink.export import quantize
    from blink.site import card

    if not args.int8:
        print(
            "error: name the scheme: blink export quantize --int8 (the only one the page runs)",
            file=sys.stderr,
        )
        return 2
    export_dir = quantize.resolve_export_dir(args.model)
    fp32 = quantize.fp32_path(export_dir)
    if not fp32.is_file():
        print(f"error: no fp32 model at {fp32} (run: blink export onnx --model <selector>)", file=sys.stderr)
        return 2
    out = Path(args.out) if args.out else quantize.int8_path(export_dir)
    started = time.perf_counter()
    quantize.quantize_int8(fp32, out)
    fp32_card_path = fp32.with_name(card.CARD_FILE)
    fp32_card = card.read(fp32_card_path) if fp32_card_path.is_file() else {"selector": args.model}
    fp32_card = {**fp32_card, "bytes": fp32.stat().st_size, "sha256": sha256_of(fp32)}
    made = card.for_file(fp32_card, out, precision=quantize.PRECISION, method=quantize.METHOD)
    card.write(made, out.with_name(card.CARD_FILE))
    ratio = made["bytes"] / fp32_card["bytes"]
    seconds = time.perf_counter() - started
    print(
        f"ok: {out} ({made['bytes']:,} B, {ratio:.1%} of the fp32 {fp32_card['bytes']:,} B, {seconds:.1f} s)"
    )
    print("next: blink export qgate --model " + str(export_dir))
    return 0


def _print_gate(report) -> None:
    a = report.agreement
    print(f"runtime: {report.runtime}")
    share = f"{100 * a.top1_agreement:.2f}%"
    where = f"{report.positions_source}, {a.batch_size} per call"
    print(f"top-1 agreement: {share} of {a.positions:,} positions ({where})")
    print(f"win% change: mean {a.mean_abs_dwin_pt:.3f} pt, max {a.max_abs_dwin_pt:.2f} pt")
    o = report.overall
    print(f"puzzles ({report.puzzle_set}, n={o.n:,}): fp32 {o.fp32_pct:.2f}%, int8 {o.int8_pct:.2f}%")
    for name, band in report.bands:
        scores = f"fp32 {band.fp32_pct:6.2f}%  int8 {band.int8_pct:6.2f}%"
        print(f"  {name:>9}: n={band.n:5,}  {scores}  drop {band.drop_pt:+.2f} pt")


def _cmd_qgate(args: argparse.Namespace) -> int:
    from blink.export import qgate, quantize
    from blink.site import card

    export_dir = quantize.resolve_export_dir(args.model)
    fp32, int8 = quantize.fp32_path(export_dir), quantize.int8_path(export_dir)
    missing = [path for path in (fp32, int8) if not path.is_file()]
    if missing:
        hint = "blink export quantize --int8" if missing[0] == int8 else "blink export onnx"
        print(f"error: no model at {missing[0]} (run: {hint} --model {args.model})", file=sys.stderr)
        return 2
    try:
        report = qgate.run(
            fp32, int8, args.runtime, args.positions, args.puzzles, args.puzzle_limit, args.position_limit
        )
    except qgate.GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    gate = report.to_dict()
    int8.with_name("qgate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8", newline="\n")
    card_path = int8.with_name(card.CARD_FILE)
    if card_path.is_file() and not report.exploratory:
        card.write(card.with_gate(card.read(card_path), gate), card_path)
    _print_gate(report)
    for failure in report.failures():
        print(f"FAIL: {failure}")
    for deviation in report.deviations():
        print(f"EXPLORATORY: {deviation}")
    if report.exploratory:
        print("card not stamped: only the pre-registered sample (the defaults, no limits) can pass the gate")
    if report.passed:
        print(f"ok: int8 passes the quantization gate ({card_path})")
    return 0 if report.passed else 1


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
    export = sub.add_parser("export", help="ONNX export and int8, move vocabulary, golden positions, rules")
    tasks = export.add_subparsers(dest="export_command", required=True)

    onnx = tasks.add_parser("onnx", help="write one self-contained opset-20 ONNX file and check it")
    onnx.add_argument("--model", required=True, help=MODEL_HELP)
    onnx.add_argument("--out", help="default: BLINK_HOME/export/<model>/model.onnx")
    onnx.set_defaults(func=_cmd_onnx)

    quant = tasks.add_parser("quantize", help="int8 dynamic per-channel copy for single-thread WASM")
    quant.add_argument("--int8", action="store_true", help="the scheme: int8 weights, dynamic activations")
    quant.add_argument(
        "--model", default="ship", help="export dir, its model.onnx, or a selector (default ship)"
    )
    quant.add_argument("--out", help="default: <export dir>/int8/model.onnx")
    quant.set_defaults(func=_cmd_quantize)

    gate = tasks.add_parser(
        "qgate", help="int8 vs fp32: 99%% top-1, puzzles within 0.5 pt, |dwin%%| within 1"
    )
    tryout = "; anything but the default makes the run exploratory (never a pass, card untouched)"
    gate.add_argument("--model", default="ship", help="export dir holding model.onnx and int8/model.onnx")
    gate.add_argument(
        "--runtime", choices=("web", "python"), default="web", help="web: onnxruntime-web in Node" + tryout
    )
    gate.add_argument(
        "--positions", default="games10k", help="games10k | random | <root records .npy>" + tryout
    )
    gate.add_argument("--position-limit", type=int, help="default 10,000" + tryout)
    gate.add_argument("--puzzles", default="bands", help="bands (lichess_bands.csv) | dm10k | <csv>" + tryout)
    gate.add_argument(
        "--puzzle-limit",
        type=int,
        help="the first N rows (the band set is sorted by rating, so low bands come first)" + tryout,
    )
    gate.set_defaults(func=_cmd_qgate)

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
