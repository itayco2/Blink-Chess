"""torch -> ONNX: one self-contained opset-20 file with a dynamic batch (P10, PF29).

The dynamo exporter writes weights to a side file by default (external_data=True, PF29), which a
static site cannot serve as one fetch, so external_data=False is passed and then checked on the file.
The file is written as <name>.tmp, checked, and only then moved onto its name with os.replace, so a
killed or failed export never leaves a partial model where `blink site serve` would find it.
"""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blink.board import moves, value

OPSET = 20
INPUT_NAME = "tokens"
OUTPUT_NAMES = ("policy_logits", "value_logits")
EXAMPLE_BATCH = 2  # batch 1 would let torch.export specialise the batch dimension to a constant
MAX_BATCH = 4096
PARITY_TOLERANCE = 1e-4


class ExportError(RuntimeError):
    pass


@dataclass(frozen=True)
class ParityReport:
    positions: int
    max_abs_policy: float
    max_abs_value: float

    @property
    def max_abs(self) -> float:
        return max(self.max_abs_policy, self.max_abs_value)

    @property
    def passed(self) -> bool:
        return self.max_abs <= PARITY_TOLERANCE


def export(module, out: Path) -> Path:
    """Write `module` (tokens [B, 64] -> policy [B, 1880], value [B, 128]) to `out` and check the file."""
    import torch

    module = module.eval()
    example = torch.zeros((EXAMPLE_BATCH, 64), dtype=torch.long)
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)

    def write(path: Path) -> None:
        torch.onnx.export(
            module,
            (example,),
            str(path),
            dynamo=True,
            opset_version=OPSET,
            external_data=False,
            input_names=[INPUT_NAME],
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes=({0: batch},),
            optimize=True,
            verbose=False,
        )

    return write_checked(out, write)


def write_checked(out: Path, write: Callable[[Path], None]) -> Path:
    """`write` fills <out>.tmp; the file replaces `out` only once check_self_contained passes on it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:
        write(tmp)
        check_self_contained(tmp)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return out


def check_self_contained(path: Path) -> int:
    """The ai.onnx opset of a file that holds every weight inline; raises ExportError otherwise."""
    import onnx

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:  # onnx raises protobuf's DecodeError (and others) on a truncated file
        raise ExportError(f"{path.name} is not a readable ONNX file: {exc}") from exc
    opsets = {entry.domain or "ai.onnx": entry.version for entry in model.opset_import}
    if opsets.get("ai.onnx") != OPSET:
        raise ExportError(f"{path.name}: ai.onnx opset {opsets.get('ai.onnx')}, expected {OPSET}")
    external = [t.name for t in model.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL]
    if external:
        raise ExportError(f"{path.name}: {len(external)} initializers live outside the file ({external[0]})")
    outputs = [o.name for o in model.graph.output]
    if tuple(outputs) != OUTPUT_NAMES:
        raise ExportError(f"{path.name}: outputs {outputs}, expected {list(OUTPUT_NAMES)}")
    return OPSET


def _torch_outputs(module, tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    device = next(module.parameters()).device
    with torch.no_grad():
        policy, values = module(torch.from_numpy(tokens.astype(np.int64)).to(device))
    return policy.float().cpu().numpy(), values.float().cpu().numpy()


def compare(module, path: Path, tokens: np.ndarray, batch_size: int = 256) -> ParityReport:
    """Max |torch - onnxruntime CPU| over every logit of every position, in batches of batch_size."""
    from blink.export.evaluators import OnnxEvaluator

    session = OnnxEvaluator(path)
    max_policy = max_value = 0.0
    for start in range(0, len(tokens), batch_size):
        chunk = tokens[start : start + batch_size]
        torch_policy, torch_value = _torch_outputs(module, chunk)
        onnx_policy, onnx_value = session.run(chunk)
        expected = ((len(chunk), moves.NUM_MOVES), (len(chunk), value.NUM_BINS))
        if (onnx_policy.shape, onnx_value.shape) != expected:
            raise ExportError(f"unexpected output shapes {onnx_policy.shape}, {onnx_value.shape}")
        max_policy = max(max_policy, float(np.abs(torch_policy - onnx_policy).max()))
        max_value = max(max_value, float(np.abs(torch_value - onnx_value).max()))
    if math.isnan(max_policy) or math.isnan(max_value):
        raise ExportError("NaN in the parity comparison")
    return ParityReport(positions=len(tokens), max_abs_policy=max_policy, max_abs_value=max_value)
