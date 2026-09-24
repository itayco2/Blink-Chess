// The network runs here, off the page's main thread: onnxruntime-web's CPU (WASM) backend, one thread.
// ort.wasm.bundle.min.mjs is the file `import "onnxruntime-web/wasm"` resolves to; it loads only the
// CPU wasm (ort-wasm-simd-threaded.wasm), never WebGPU or WebGL. The int8 model's integer kernels are
// WASM kernels (PF30). One "evaluate" is one session run. The model's bytes come from the Cache API
// when this browser already holds the file with the sha256 the model card names (modelcache.js).

import * as ort from "./vendor/ort/ort.wasm.bundle.min.mjs";
import { loadModel } from "./modelcache.js";

ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths = new URL("./vendor/ort/", import.meta.url).href;

let session = null;

async function load(url, sha256) {
  const started = performance.now();
  const model = await loadModel({ url, sha256 });
  const fetchMs = performance.now() - started;
  session = await ort.InferenceSession.create(model.bytes, { executionProviders: ["wasm"] });
  return {
    loadMs: performance.now() - started,
    fetchMs,
    source: model.source,
    verified: model.verified,
    backend: "wasm",
    threads: ort.env.wasm.numThreads,
    inputs: session.inputNames,
    outputs: session.outputNames,
  };
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
    const result = type === "load" ? await load(event.data.url, event.data.sha256) : await evaluate(event.data.codes);
    self.postMessage({ id, type: "ok", ...result });
  } catch (error) {
    self.postMessage({ id, type: "error", message: String(error && error.message ? error.message : error) });
  }
};
