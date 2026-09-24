// The bench's pair plan: int8 never reaches a GPU backend (PF30), unknown pairs are refused.

import { test } from "node:test";
import assert from "node:assert/strict";
import { BACKENDS, DEFAULT_PAIRS, MODELS, parsePairs, plan } from "../bench/plan.js";

test("int8 is never sent to WebGPU; fp32 may be", () => {
  const { runs, skipped } = plan(parsePairs("int8:wasm-1t,int8:webgpu,fp32:webgpu,int8:wasm-mt"));
  assert.deepEqual(runs, [
    { precision: "int8", backend: "wasm-1t" },
    { precision: "fp32", backend: "webgpu" },
    { precision: "int8", backend: "wasm-mt" },
  ]);
  assert.equal(skipped.length, 1);
  assert.deepEqual([skipped[0].precision, skipped[0].backend], ["int8", "webgpu"]);
  assert.match(skipped[0].reason, /PF30/);
});

test("every GPU backend is refused for int8, whatever its name", () => {
  for (const [backend, spec] of Object.entries(BACKENDS)) {
    const { runs } = plan([{ precision: "int8", backend }]);
    assert.equal(runs.length, spec.gpu ? 0 : 1, backend);
  }
});

test("unknown models or backends are skipped with a reason, never run", () => {
  const { runs, skipped } = plan(parsePairs("fp16:wasm-1t,int8:webgl"));
  assert.equal(runs.length, 0);
  assert.deepEqual(skipped.map((s) => s.reason), ["unknown pair fp16:wasm-1t", "unknown pair int8:webgl"]);
});

test("without pairs the bench measures the shipped path: int8 on one WASM thread", () => {
  assert.equal(DEFAULT_PAIRS, "int8:wasm-1t");
  assert.deepEqual(plan(parsePairs("")).runs, [{ precision: "int8", backend: "wasm-1t" }]);
  assert.equal(MODELS.int8, "models/model.onnx");
  assert.equal(BACKENDS["wasm-1t"].entry, "ort.wasm.bundle.min.mjs");
  assert.equal(BACKENDS["wasm-1t"].threads, 1);
});
