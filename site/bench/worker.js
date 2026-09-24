// One (model, backend) pair per worker: the bench loads the backend's onnxruntime-web entry, creates a
// session for one model, and answers "evaluate" with the time of one session run (one look, batch 1).
// The page's own worker (../worker.js) is the shipped path; this one exists only to compare backends.

import { BACKENDS, MODELS } from "./plan.js";

let ort = null;
let session = null;

async function gpuAdapter() {
  return self.navigator && self.navigator.gpu ? self.navigator.gpu.requestAdapter() : null;
}

async function load({ backend, precision, threads }) {
  const spec = BACKENDS[backend];
  if (precision === "int8" && spec.gpu) {
    throw new Error("refused: int8 on a GPU backend (PF30)");
  }
  if (spec.gpu && !(await gpuAdapter())) {
    return { unavailable: "no WebGPU adapter in this browser" };
  }
  const started = performance.now();
  ort = await import(new URL(`../vendor/ort/${spec.entry}`, import.meta.url).href);
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.wasmPaths = new URL("../vendor/ort/", import.meta.url).href;
  const response = await fetch(new URL(`../${MODELS[precision]}`, import.meta.url), { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${MODELS[precision]}: HTTP ${response.status}`);
  }
  const bytes = new Uint8Array(await response.arrayBuffer());
  session = await ort.InferenceSession.create(bytes, { executionProviders: [spec.ep] });
  return { coldMs: performance.now() - started, threads: ort.env.wasm.numThreads, bytes: bytes.length };
}

async function evaluate(codes) {
  const tokens = new ort.Tensor("int64", BigInt64Array.from(codes, (code) => BigInt(code)), [1, 64]);
  const started = performance.now();
  const out = await session.run({ tokens });
  const runMs = performance.now() - started;
  if (out.policy_logits.dims[1] !== 1880) {
    throw new Error(`policy has ${out.policy_logits.dims[1]} moves, expected 1880`);
  }
  return { runMs };
}

self.onmessage = async (event) => {
  const { id, type } = event.data;
  try {
    const result = type === "load" ? await load(event.data) : await evaluate(event.data.codes);
    self.postMessage({ id, type: "ok", ...result });
  } catch (error) {
    self.postMessage({ id, type: "error", message: String(error && error.message ? error.message : error) });
  }
};
