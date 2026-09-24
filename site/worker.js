// The network runs here, off the page's main thread: onnxruntime-web's CPU (WASM) backend, one thread.
// ort.wasm.bundle.min.mjs is the file `import "onnxruntime-web/wasm"` resolves to; it loads only the
// CPU wasm (ort-wasm-simd-threaded.wasm), never WebGPU or WebGL. One "evaluate" is one session run.

import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;

let session = null;

async function load(url) {
  const started = performance.now();
  session = await ort.InferenceSession.create(url, { executionProviders: ["wasm"] });
  return { loadMs: performance.now() - started, inputs: session.inputNames, outputs: session.outputNames };
}

async function evaluate(codes) {
  if (!session) {
    throw new Error("the model is not loaded yet");
  }
  const tokens = new ort.Tensor("int64", BigInt64Array.from(codes, (code) => BigInt(code)), [1, 64]);
  const started = performance.now();
  const out = await session.run({ tokens });
  const runMs = performance.now() - started;
  return { policy: out.policy_logits.data, value: out.value_logits.data, runMs };
}

self.onmessage = async (event) => {
  const { id, type } = event.data;
  try {
    const result = type === "load" ? await load(event.data.url) : await evaluate(event.data.codes);
    self.postMessage({ id, type: "ok", ...result });
  } catch (error) {
    self.postMessage({ id, type: "error", message: String(error && error.message ? error.message : error) });
  }
};
