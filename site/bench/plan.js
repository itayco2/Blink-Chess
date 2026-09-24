// Which (model, backend) pairs the bench runs, from ?pairs=int8:wasm-1t,fp32:webgpu. The browser model
// (models/model.onnx) is the int8 file; `blink site bench` also stages the fp32 source as
// models/fp32.onnx. int8 is never handed to a GPU backend: onnxruntime-web's WebGPU has slow or missing
// integer kernels (PF30), which is why the page itself runs WASM only.

export const BACKENDS = Object.freeze({
  "wasm-1t": Object.freeze({ entry: "ort.wasm.bundle.min.mjs", ep: "wasm", threads: 1, gpu: false }),
  "wasm-mt": Object.freeze({ entry: "ort.wasm.bundle.min.mjs", ep: "wasm", threads: 4, gpu: false }),
  webgpu: Object.freeze({ entry: "ort.webgpu.bundle.min.mjs", ep: "webgpu", threads: 1, gpu: true }),
});
export const MODELS = Object.freeze({ int8: "models/model.onnx", fp32: "models/fp32.onnx" });
export const DEFAULT_PAIRS = "int8:wasm-1t";

export function parsePairs(text) {
  return String(text || DEFAULT_PAIRS)
    .split(",")
    .map((pair) => pair.trim())
    .filter(Boolean)
    .map((pair) => {
      const [precision, backend] = pair.split(":");
      return { precision, backend };
    });
}

// { runs, skipped }: the pairs to measure, and the ones refused with their reason.
export function plan(pairs) {
  const runs = [];
  const skipped = [];
  for (const { precision, backend } of pairs) {
    if (!(backend in BACKENDS) || !(precision in MODELS)) {
      skipped.push({ precision, backend, reason: `unknown pair ${precision}:${backend}` });
    } else if (precision === "int8" && BACKENDS[backend].gpu) {
      skipped.push({ precision, backend, reason: "WebGPU never gets int8 (PF30): integer kernels there are slow or missing" });
    } else {
      runs.push({ precision, backend });
    }
  }
  return { runs, skipped };
}
