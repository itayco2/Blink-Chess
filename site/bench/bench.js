// The latency bench behind bench.html: for each (model, backend) pair in ?pairs=, a fresh worker loads
// the model (cold load), then plays one look per position: `runs` timed session runs of batch 1 after
// `warmup` untimed ones. run_ms is the session run inside the worker; look_ms is the round trip the page
// pays per move (message to the worker, run, reply). Results land in window.__blinkBench for
// `blink site bench`, and in the table for anyone who opens the page.

import { Chess } from "../vendor/chess.js/chess.js";
import * as tok from "../tokenizer.js";
import { BACKENDS, DEFAULT_PAIRS, parsePairs, plan } from "./plan.js";

const POSITIONS = 64;
const SEED = 20260924;
const params = new URLSearchParams(location.search);

function count(name, fallback, max) {
  const value = Number.parseInt(params.get(name) || "", 10);
  return Number.isFinite(value) && value > 0 ? Math.min(value, max) : fallback;
}

const RUNS = count("runs", 200, 5000);
const WARMUP = count("warmup", 10, 500);

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Seeded random-game positions, encoded as the network sees them.
export function positions(vocab, n, seed = SEED) {
  const random = mulberry32(seed);
  const out = [];
  let game = new Chess();
  while (out.length < n) {
    const moves = game.moves();
    if (moves.length === 0 || game.isGameOver()) {
      game = new Chess();
      continue;
    }
    out.push(Array.from(tok.encodeBoard(vocab, game)));
    game.move(moves[Math.floor(random() * moves.length)]);
  }
  return out;
}

function connect() {
  const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  let nextId = 1;
  const pending = new Map();
  worker.onmessage = ({ data }) => {
    const job = pending.get(data.id);
    pending.delete(data.id);
    if (job) {
      data.type === "error" ? job.reject(new Error(data.message)) : job.resolve(data);
    }
  };
  worker.onerror = (event) => pending.forEach((job) => job.reject(new Error(event.message || "worker failed")));
  const call = (message) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ ...message, id });
    });
  return { call, close: () => worker.terminate() };
}

async function timedLooks(call, codes) {
  const first = await call({ type: "evaluate", codes: codes[0] });
  for (let i = 0; i < WARMUP; i++) {
    await call({ type: "evaluate", codes: codes[(i + 1) % codes.length] });
  }
  const runMs = [];
  const lookMs = [];
  for (let i = 0; i < RUNS; i++) {
    const started = performance.now();
    const reply = await call({ type: "evaluate", codes: codes[i % codes.length] });
    lookMs.push(performance.now() - started);
    runMs.push(reply.runMs);
  }
  return { first_run_ms: first.runMs, run_ms: runMs, look_ms: lookMs };
}

async function measure(pair, codes) {
  const threads = BACKENDS[pair.backend].threads > 1 ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  const { call, close } = connect();
  try {
    const loaded = await call({ type: "load", ...pair, threads });
    if (loaded.unavailable) {
      return { ...pair, status: "unavailable", reason: loaded.unavailable };
    }
    const looks = await timedLooks(call, codes);
    return { ...pair, status: "ok", threads: loaded.threads, cold_ms: loaded.coldMs, bytes: loaded.bytes, ...looks };
  } catch (error) {
    return { ...pair, status: "error", reason: error.message };
  } finally {
    close();
  }
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : NaN;
}

function addRow(result) {
  const cells = [
    result.precision,
    result.backend,
    result.threads ?? "",
    result.cold_ms !== undefined ? result.cold_ms.toFixed(0) : "",
    result.look_ms ? median(result.look_ms).toFixed(2) : "",
    result.run_ms ? median(result.run_ms).toFixed(2) : "",
    result.status === "ok" ? "ok" : `${result.status}: ${result.reason}`,
  ];
  const row = document.createElement("tr");
  row.replaceChildren(...cells.map((text) => Object.assign(document.createElement("td"), { textContent: String(text) })));
  document.querySelector("#bench tbody").appendChild(row);
}

async function main() {
  const vocab = tok.createVocab(await (await fetch(new URL("../vocab.json", import.meta.url))).json());
  const codes = positions(vocab, POSITIONS);
  const { runs, skipped } = plan(parsePairs(params.get("pairs") || DEFAULT_PAIRS));
  skipped.forEach((pair) => addRow({ ...pair, status: "skipped" }));
  const results = [];
  for (const pair of runs) {
    document.getElementById("bench-status").textContent = `Measuring ${pair.precision} on ${pair.backend}`;
    const result = await measure(pair, codes);
    results.push(result);
    addRow(result);
  }
  const env = {
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    crossOriginIsolated: self.crossOriginIsolated === true,
    gpu: Boolean(navigator.gpu),
    runs: RUNS,
    warmup: WARMUP,
    positions: codes.length,
  };
  document.getElementById("bench-status").textContent = `Done: ${results.length} pairs, ${RUNS} looks each`;
  window.__blinkBench = Object.freeze({ done: true, results, skipped, env });
}

if (typeof document !== "undefined") {
  main().catch((error) => {
    document.getElementById("bench-status").textContent = `The bench failed: ${error.message}`;
    window.__blinkBench = Object.freeze({ done: true, error: error.message, results: [], skipped: [] });
  });
}
