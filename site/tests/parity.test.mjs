// JS/Python parity (PF31): the browser tokenizer and onnxruntime-web must reproduce golden.json,
// which `blink export golden` wrote from python-chess and the Python model.
// Run from the repo root with `node --test "site/tests/**/*.test.mjs"` (after `npm ci --prefix site`).
// BLINK_ONNX=<path> points the onnxruntime-web check at an exported model; the default is
// site/models/model.onnx. Without a model file that check is skipped, never faked.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { Chess } from "chess.js";
import * as ort from "onnxruntime-web/wasm";
import * as tok from "../tokenizer.js";

const HERE = new URL(".", import.meta.url);
const golden = JSON.parse(readFileSync(new URL("golden.json", HERE), "utf-8"));
const vocab = tok.createVocab(JSON.parse(readFileSync(new URL("../vocab.json", HERE), "utf-8")));
const PROB_TOLERANCE = 1e-4;
const WIN_TOLERANCE = 1e-4;

function modelPath() {
  return process.env.BLINK_ONNX || fileURLToPath(new URL("../models/model.onnx", HERE));
}

test("the vocabulary has 1880 distinct moves that round-trip through their index", () => {
  assert.equal(vocab.numMoves, 1880);
  const seen = new Set();
  for (let index = 0; index < vocab.numMoves; index++) {
    const move = tok.decodeMove(vocab, "w", index);
    assert.equal(tok.encodeMove(vocab, "w", move), index);
    assert.equal(tok.encodeMove(vocab, "b", tok.decodeMove(vocab, "b", index)), index);
    seen.add(move.uci);
  }
  assert.equal(seen.size, 1880);
});

test("golden.json holds 50 positions covering castling, promotion, en passant and black to move", () => {
  assert.equal(golden.positions.length, 50);
  const tags = golden.positions.flatMap((entry) => entry.tags);
  for (const tag of ["castling", "promotion", "en_passant", "black_to_move"]) {
    assert.ok(tags.includes(tag), `no golden position is tagged ${tag}`);
  }
});

test("sanitizeFen drops castling rights without their king or rook, as python-chess does", () => {
  assert.equal(tok.sanitizeFen("4k3/8/8/8/8/8/8/4K2R w KQ - 0 1"), "4k3/8/8/8/8/8/8/4K2R w K - 0 1");
  assert.equal(tok.sanitizeFen("r3k2r/8/8/8/8/8/8/R4K1R w KQkq - 0 1"), "r3k2r/8/8/8/8/8/8/R4K1R w kq - 0 1");
  assert.equal(tok.sanitizeFen("4k3/8/8/8/8/8/8/4K3 w KQkq - 0 1"), "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
});

test("sanitizeFen keeps an en-passant square only when a double-pushed pawn stands in front of it", () => {
  const start = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
  assert.equal(tok.sanitizeFen(start), start);
  assert.equal(tok.sanitizeFen("4k3/8/8/8/8/8/8/4K3 b - e3 0 1"), "4k3/8/8/8/8/8/8/4K3 b - - 0 1");
});

test("tokenizer codes equal golden on all 50 FENs", () => {
  for (const entry of golden.positions) {
    const chess = tok.loadPosition(Chess, entry.fen);
    assert.deepEqual(Array.from(tok.encodeBoard(vocab, chess)), entry.codes, entry.fen);
  }
});

test("legal move indices and their uci equal golden on all 50 FENs", () => {
  for (const entry of golden.positions) {
    const chess = tok.loadPosition(Chess, entry.fen);
    const legal = tok.legalMoves(vocab, chess).map((move) => [move.index, move.uci]);
    assert.deepEqual(legal, entry.legal, entry.fen);
    for (const [index, uci] of entry.legal) {
      assert.equal(tok.decodeMove(vocab, chess.turn(), index).uci, uci, entry.fen);
    }
  }
});

function checkPosition(entry, policyRow, valueRow) {
  const chess = tok.loadPosition(Chess, entry.fen);
  const legal = tok.legalMoves(vocab, chess);
  const top = tok.policyTopK(legal, policyRow, golden.top_k);
  const probOf = new Map(tok.policyTopK(legal, policyRow, legal.length).map((m) => [m.index, m.prob]));
  let probDelta = 0;
  entry.top5.forEach((want, k) => {
    const got = probOf.get(want.index);
    probDelta = Math.max(probDelta, Math.abs(got - want.prob), Math.abs(top[k].prob - want.prob));
    assert.ok(Math.abs(got - want.prob) <= PROB_TOLERANCE, `${entry.fen} ${want.uci}: ${got} vs ${want.prob}`);
    assert.ok(Math.abs(top[k].prob - want.prob) <= PROB_TOLERANCE, `${entry.fen} rank ${k + 1}`);
  });
  const winDelta = Math.abs(tok.winProbability(valueRow) - entry.win);
  assert.ok(winDelta <= WIN_TOLERANCE, `${entry.fen} win off by ${winDelta}`);
  return { probDelta, winDelta, sameTop1: top[0].index === entry.top5[0].index };
}

test("onnxruntime-web (wasm, one thread) reproduces golden top-5 and win% on all 50 FENs", async (t) => {
  const path = modelPath();
  if (!existsSync(path)) {
    t.skip(`no model at ${path}: export one and set BLINK_ONNX to run this check`);
    return;
  }
  const bytes = readFileSync(path);
  const sha = createHash("sha256").update(bytes).digest("hex");
  assert.equal(sha, golden.model.onnx_sha256, "golden.json was made for another model: rerun blink export golden");
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
  const tokens = new BigInt64Array(golden.positions.length * 64);
  golden.positions.forEach((entry, row) => entry.codes.forEach((code, i) => (tokens[row * 64 + i] = BigInt(code))));
  const feeds = { tokens: new ort.Tensor("int64", tokens, [golden.positions.length, 64]) };
  const out = await session.run(feeds);
  const policy = out.policy_logits.data;
  const values = out.value_logits.data;
  const results = golden.positions.map((entry, row) => {
    const policyRow = policy.subarray(row * vocab.numMoves, (row + 1) * vocab.numMoves);
    const valueRow = values.subarray(row * vocab.numBins, (row + 1) * vocab.numBins);
    return checkPosition(entry, policyRow, valueRow);
  });
  const maxProb = Math.max(...results.map((r) => r.probDelta));
  const maxWin = Math.max(...results.map((r) => r.winDelta));
  const top1 = results.filter((r) => r.sameTop1).length;
  t.diagnostic(`top-1 agrees on ${top1}/50; max |dprob| ${maxProb.toExponential(2)}; max |dwin| ${maxWin.toExponential(2)}`);
});
