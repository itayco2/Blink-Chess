"""fp32 ONNX -> int8 for the browser's single-thread WASM path (P10, PF30).

Dynamic quantization, per channel: every MatMul and Gather weight is stored as int8 with one scale per
output channel, and activations are quantized on the fly (DynamicQuantizeLinear feeding MatMulInteger),
so no calibration set is needed. onnxruntime's own advice is followed: quant_pre_process (symbolic
shape inference plus basic graph optimisation) runs first, and quantize_dynamic reads its output.

The int8 file is for onnxruntime-web's WASM backend only. Integer kernels are slow or missing on
WebGPU (PF30), so the page runs WASM alone and `blink site bench` never pairs int8 with WebGPU.
Layout: <export dir>/model.onnx (fp32) -> <export dir>/int8/model.onnx plus its card, model.json,
which is exactly the models/ directory the page loads. The file is written as .tmp, checked by
blink.export.onnx.check_self_contained, and only then moved onto its name.
"""

from pathlib import Path

from blink import paths
from blink.export import models
from blink.export import onnx as export_onnx

QUANT_DIR = "int8"
PRECISION = "int8"
METHOD = "dynamic, int8 weights per output channel, uint8 activations (onnxruntime.quantization)"


def resolve_export_dir(model: str) -> Path:
    """The export directory a --model names: a directory, an .onnx file's directory, or a selector's."""
    path = Path(model)
    if path.suffix.lower() == ".onnx":
        return path.parent
    if path.is_dir():
        return path
    return paths.home() / "export" / models.slug(model)


def fp32_path(export_dir: Path) -> Path:
    return export_dir / models.MODEL_FILE


def int8_path(export_dir: Path) -> Path:
    return export_dir / QUANT_DIR / models.MODEL_FILE


def quantize_int8(fp32: Path, out: Path) -> Path:
    """Write the int8 model of `fp32` to `out` (checked, then replaced in one step); returns `out`."""
    from onnxruntime import quantization
    from onnxruntime.quantization import shape_inference

    def write(tmp: Path) -> None:
        pre = tmp.with_name(tmp.name + ".pre")
        try:
            shape_inference.quant_pre_process(str(fp32), str(pre))
            quantization.quantize_dynamic(
                str(pre),
                str(tmp),
                per_channel=True,
                weight_type=quantization.QuantType.QInt8,
            )
        finally:
            pre.unlink(missing_ok=True)

    return export_onnx.write_checked(out, write)
