"""The browser model's card: models/model.json beside models/model.onnx, read by the page (P10).

`blink export quantize --int8` writes it from the fp32 export's card; `blink export qgate` adds the
gate's numbers under quantization.gate. The page's model card (site/panel.js) shows the label, the size,
the hash and those numbers, and site/app.js keys the model's Cache API entry by the sha256 recorded here.
site/tests/card.json is example(), the fixture that the Python and Node tests both read, so the two
sides cannot drift apart on a key name.
"""

import hashlib
import json
import os
from pathlib import Path

CARD_FILE = "model.json"
MODEL_FILE = "model.onnx"
LABELS = {"int8": "one look, int8 WASM", "fp32": "one look, fp32 WASM"}
MODE = "policy"  # the browser runs one look only; value mode in the browser is out of scope


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_name(selector: str) -> str:
    from blink.export import models

    return f"Blink {models.slug(selector)}"


def browser_card(fp32_card: dict, size: int, sha256: str, precision: str, method: str | None = None) -> dict:
    """The card of a browser model of `size` bytes, made from its fp32 export's card."""
    if precision not in LABELS:
        raise ValueError(f"precision must be one of {sorted(LABELS)}, got {precision!r}")
    selector = fp32_card.get("selector", "")
    quantization = None
    if precision == "int8":
        quantization = {
            "method": method,
            "fp32_bytes": fp32_card.get("bytes"),
            "fp32_sha256": fp32_card.get("sha256"),
            "gate": None,
        }
    return {
        "name": fp32_card.get("name") or display_name(selector),
        "selector": selector,
        "mode": MODE,
        "label": LABELS[precision],
        "backend": "wasm",
        "threads": 1,
        "file": MODEL_FILE,
        "bytes": size,
        "sha256": sha256,
        "precision": precision,
        "parameters": fp32_card.get("parameters"),
        "opset": fp32_card.get("opset"),
        "quantization": quantization,
    }


def for_file(fp32_card: dict, model: Path, precision: str, method: str | None = None) -> dict:
    return browser_card(fp32_card, model.stat().st_size, sha256_of(model), precision, method)


def with_gate(card: dict, gate: dict) -> dict:
    """A new card holding the quantization gate's report (never an exploratory run's)."""
    if not card.get("quantization"):
        raise ValueError("only an int8 card has quantization numbers to hold")
    if gate.get("exploratory", True):
        raise ValueError(f"an exploratory gate run cannot stamp the card: {gate.get('deviations')}")
    return {**card, "quantization": {**card["quantization"], "gate": gate}}


def render(card: dict) -> str:
    return json.dumps(card, indent=2) + "\n"


def write(card: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(render(card), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def example() -> dict:
    """A gated int8 card with plausible numbers: the tracked fixture site/tests/card.json."""
    from blink.export import qgate

    fp32 = {
        "selector": "run:example:ema",
        "bytes": 1_576_222,
        "sha256": "0eb4a4240c68e2049e4408a151a6769571d2ced20124d8828e3264f896016d50",
        "opset": 20,
        "parameters": 339_456,
    }
    sha = "5a3c" * 16
    card = browser_card(fp32, 419_787, sha, "int8", "dynamic, int8 weights per output channel")
    return with_gate(card, qgate.example().to_dict())
