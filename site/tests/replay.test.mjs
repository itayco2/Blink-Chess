// The training replay's pure helpers, on the tracked frozen run (site/replay/*.jsonl).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import * as replay from "../replay/replay.js";

const HERE = new URL(".", import.meta.url);
const read = (name) => readFileSync(new URL(`../replay/${name}`, HERE), "utf-8");

test("the frozen metrics and evals parse as one JSON row per line, in step order", () => {
  const run = JSON.parse(read("run.json"));
  const metrics = replay.parseJsonl(read("metrics.jsonl"));
  const evals = replay.parseJsonl(read("evals.jsonl"));
  assert.equal(metrics.length, run.metrics_rows);
  assert.equal(evals.length, run.evals_rows);
  for (const rows of [metrics, evals]) {
    assert.deepEqual(rows.map((r) => r.step), [...rows.map((r) => r.step)].sort((a, b) => a - b));
  }
});

test("upTo keeps the rows at or before the cursor and series skips nulls", () => {
  const rows = [{ step: 0, x: 1 }, { step: 2000, x: null }, { step: 4000, x: 3 }];
  assert.deepEqual(replay.upTo(rows, 2000).map((r) => r.step), [0, 2000]);
  assert.deepEqual(replay.series(rows, "x"), [[0, 1], [4000, 3]]);
});

test("bounds span every line and the reference, and never collapse to zero width", () => {
  assert.deepEqual(replay.bounds([[[0, 2], [10, 4]], [[5, 3]]], 7.5), { x0: 0, x1: 10, y0: 2, y1: 7.5 });
  const flat = replay.bounds([[[3, 1]]]);
  assert.ok(flat.x1 > flat.x0 && flat.y1 > flat.y0);
  assert.equal(replay.bounds([[]]), null);
});

test("fmt writes big numbers with separators and small ones in exponent form", () => {
  assert.equal(replay.fmt(19125.3, 0), "19,125");
  assert.equal(replay.fmt(0.0005, 5), "5.00e-4");
  assert.equal(replay.fmt(2.9426, 3), "2.943");
  assert.equal(replay.fmt(null, 3), "-");
});
