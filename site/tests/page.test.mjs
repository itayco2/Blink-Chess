// The page's pure pieces: the Cache API keyed by the model's sha256, the Lichess rating widget,
// the model card text, and the 128-bin value histogram (P10).

import { test } from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import * as modelcache from "../modelcache.js";
import * as panel from "../panel.js";
import * as rating from "../rating.js";
import * as tok from "../tokenizer.js";

const HERE = new URL(".", import.meta.url);
const card = JSON.parse(readFileSync(new URL("card.json", HERE), "utf-8"));
const MODEL_URL = "https://example.test/models/model.onnx";

const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");

function fakeCaches() {
  const stores = new Map();
  return {
    stores,
    async open(name) {
      if (!stores.has(name)) {
        stores.set(name, new Map());
      }
      const store = stores.get(name);
      return {
        async match(key) {
          return store.has(key) ? new Response(store.get(key)) : undefined;
        },
        async put(key, response) {
          store.set(key, new Uint8Array(await response.arrayBuffer()));
        },
        async keys() {
          return [...store.keys()].map((url) => ({ url }));
        },
        async delete(key) {
          return store.delete(typeof key === "string" ? key : key.url);
        },
      };
    },
  };
}

function fakeFetch(bytes) {
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url, init });
    return new Response(bytes);
  };
  return { calls, fetchImpl };
}

test("the first load fetches the model, checks its sha256 and caches it under that hash", async () => {
  const bytes = new Uint8Array([1, 2, 3, 4]);
  const caches = fakeCaches();
  const { calls, fetchImpl } = fakeFetch(bytes);
  const model = await modelcache.loadModel({ url: MODEL_URL, sha256: sha(bytes), fetchImpl, cachesImpl: caches });
  assert.equal(model.source, "network");
  assert.equal(model.verified, true);
  assert.deepEqual(Array.from(model.bytes), [1, 2, 3, 4]);
  assert.equal(calls.length, 1);
  const store = caches.stores.get(modelcache.CACHE_NAME);
  assert.deepEqual([...store.keys()], [modelcache.cacheKey(MODEL_URL, sha(bytes))]);
  assert.ok(modelcache.cacheKey(MODEL_URL, sha(bytes)).endsWith(`sha256=${sha(bytes)}`));
});

test("a second load with the same hash comes from the cache without a network request", async () => {
  const bytes = new Uint8Array([9, 8, 7]);
  const caches = fakeCaches();
  await modelcache.loadModel({ url: MODEL_URL, sha256: sha(bytes), fetchImpl: fakeFetch(bytes).fetchImpl, cachesImpl: caches });
  const again = fakeFetch(bytes);
  const model = await modelcache.loadModel({ url: MODEL_URL, sha256: sha(bytes), fetchImpl: again.fetchImpl, cachesImpl: caches });
  assert.equal(model.source, "cache");
  assert.equal(again.calls.length, 0);
});

test("a new model hash replaces the old cache entry, so stale weights never pile up", async () => {
  const caches = fakeCaches();
  const first = new Uint8Array([1]);
  const second = new Uint8Array([2]);
  await modelcache.loadModel({ url: MODEL_URL, sha256: sha(first), fetchImpl: fakeFetch(first).fetchImpl, cachesImpl: caches });
  await modelcache.loadModel({ url: MODEL_URL, sha256: sha(second), fetchImpl: fakeFetch(second).fetchImpl, cachesImpl: caches });
  assert.deepEqual([...caches.stores.get(modelcache.CACHE_NAME).keys()], [modelcache.cacheKey(MODEL_URL, sha(second))]);
});

test("bytes that do not match the card's sha256 are refused and never cached", async () => {
  const caches = fakeCaches();
  const { fetchImpl } = fakeFetch(new Uint8Array([5, 5]));
  await assert.rejects(
    modelcache.loadModel({ url: MODEL_URL, sha256: "ab".repeat(32), fetchImpl, cachesImpl: caches }),
    /does not match its card/,
  );
  const store = caches.stores.get(modelcache.CACHE_NAME);
  assert.equal(store ? store.size : 0, 0);
});

test("a corrupted cache entry is dropped and the model is fetched again", async () => {
  const bytes = new Uint8Array([3, 1, 4]);
  const caches = fakeCaches();
  const store = new Map([[modelcache.cacheKey(MODEL_URL, sha(bytes)), new Uint8Array([0])]]);
  caches.stores.set(modelcache.CACHE_NAME, store);
  const { calls, fetchImpl } = fakeFetch(bytes);
  const model = await modelcache.loadModel({ url: MODEL_URL, sha256: sha(bytes), fetchImpl, cachesImpl: caches });
  assert.equal(model.source, "network");
  assert.equal(calls.length, 1);
  assert.deepEqual(Array.from(store.get(modelcache.cacheKey(MODEL_URL, sha(bytes)))), [3, 1, 4]);
});

test("without a hash or a Cache API the model is fetched and nothing is cached", async () => {
  const bytes = new Uint8Array([7]);
  const caches = fakeCaches();
  const noHash = await modelcache.loadModel({ url: MODEL_URL, sha256: undefined, fetchImpl: fakeFetch(bytes).fetchImpl, cachesImpl: caches });
  assert.equal(noHash.source, "network");
  assert.equal(noHash.verified, false);
  assert.equal(caches.stores.size, 0);
  const noCaches = await modelcache.loadModel({ url: MODEL_URL, sha256: sha(bytes), fetchImpl: fakeFetch(bytes).fetchImpl, cachesImpl: undefined });
  assert.equal(noCaches.source, "network");
  assert.equal(noCaches.verified, true);
});

// --- The Lichess rating widget -----------------------------------------------------------------

const user = (games, rd, rating = 1843) => ({ username: "BlinkBot", perfs: { blitz: { games, rating, rd, prov: rd >= 110 } } });

test("the rating shows only at 200 or more rated games and an RD under 75", () => {
  assert.equal(rating.MIN_GAMES, 200);
  assert.equal(rating.MAX_RD, 75);
  assert.deepEqual(rating.ratingView(user(200, 74), "blitz"), {
    publishable: true,
    text: "1843",
    note: "RD 74, 200 rated blitz games, live from Lichess",
  });
  for (const [games, rd] of [[199, 60], [200, 75], [500, 90], [0, 500]]) {
    const view = rating.ratingView(user(games, rd), "blitz");
    assert.equal(view.publishable, false, `${games} games, RD ${rd}`);
    assert.equal(view.text, "rating accruing");
    assert.equal(view.note, `${games} of 200 rated blitz games`);
  }
});

test("a missing perf, a missing user or no bot name all read rating accruing", async () => {
  assert.equal(rating.ratingView({ perfs: {} }, "blitz").text, "rating accruing");
  assert.equal(rating.ratingView(null, "blitz").text, "rating accruing");
  let fetched = 0;
  const fetchImpl = async () => {
    fetched += 1;
    return new Response("{}");
  };
  const none = await rating.fetchRating({ lichess_bot: "", lichess_perf: "blitz" }, fetchImpl);
  assert.equal(none.text, "rating accruing");
  assert.equal(none.note, "");
  assert.equal(fetched, 0, "no bot name, no request");
});

test("the rating is read from the public user API for the configured bot", async () => {
  const urls = [];
  const fetchImpl = async (url) => {
    urls.push(url);
    return new Response(JSON.stringify(user(250, 60)));
  };
  const view = await rating.fetchRating({ lichess_bot: "BlinkBot", lichess_perf: "blitz" }, fetchImpl);
  assert.deepEqual(urls, ["https://lichess.org/api/user/BlinkBot"]);
  assert.equal(view.text, "1843");
  assert.equal(view.url, "https://lichess.org/@/BlinkBot");
});

test("a bot name that is not a Lichess username is never put in a URL", async () => {
  let fetched = 0;
  const fetchImpl = async () => {
    fetched += 1;
    return new Response("{}");
  };
  const view = await rating.fetchRating({ lichess_bot: "../api/x?y=1", lichess_perf: "blitz" }, fetchImpl);
  assert.equal(view.text, "rating accruing");
  assert.equal(fetched, 0);
});

test("a failed or refused request reads rating accruing, never an error", async () => {
  const refused = async () => new Response("{}", { status: 429 });
  assert.equal((await rating.fetchRating({ lichess_bot: "BlinkBot", lichess_perf: "blitz" }, refused)).text, "rating accruing");
  const offline = async () => {
    throw new TypeError("fetch failed");
  };
  assert.equal((await rating.fetchRating({ lichess_bot: "BlinkBot", lichess_perf: "blitz" }, offline)).text, "rating accruing");
});

// --- The model card and the value histogram --------------------------------------------------------

test("the model card names the label, the size and every quantization delta", () => {
  const lines = panel.cardLines(card);
  const text = lines.join("\n");
  assert.equal(lines[0], "Blink example-ema, one look, int8 WASM");
  assert.match(text, /339,456 parameters/);
  assert.match(text, /0\.42 MB int8 \(fp32 1\.58 MB\)/);
  assert.match(text, /top-1 agreement with fp32 99\.31% of 10,000 positions/);
  assert.match(text, /puzzles 31\.62% fp32, 31\.57% int8 \(-0\.06 pt, 5,281 puzzles; worst band -0\.10 pt\)/);
  assert.match(text, /mean \|dwin%\| 0\.21 pt/);
  assert.match(text, /quantization gate: passed/);
  assert.match(text, /sha256 5a3c5a3c5a3c/);
});

test("a failed gate is named on the card, with its first failure", () => {
  const gate = { ...card.quantization.gate, passed: false, failures: ["top-1 agreement 96.42% on 10,000 positions is below 99%"] };
  const failed = { ...card, quantization: { ...card.quantization, gate } };
  assert.match(panel.cardLines(failed).join("\n"), /quantization gate: failed \(top-1 agreement 96\.42% on 10,000 positions is below 99%\)/);
});

test("a card without quantization numbers still names the model", () => {
  const lines = panel.cardLines({ selector: "stand-in", bytes: 2_219_744 });
  assert.equal(lines[0], "stand-in");
  assert.match(lines.join("\n"), /2\.22 MB/);
  assert.match(lines.join("\n"), /Random, untrained weights/);
});

test("value bins are a softmax whose mean over bin centres is the win probability", () => {
  const logits = Float32Array.from({ length: 128 }, (_, i) => Math.sin(i / 7) * 3);
  const bins = tok.valueProbabilities(logits);
  assert.equal(bins.length, 128);
  assert.ok(Math.abs(bins.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  const mean = bins.reduce((sum, p, i) => sum + p * ((i + 0.5) / 128), 0);
  assert.ok(Math.abs(mean - tok.winProbability(logits)) < 1e-9);
});

test("the histogram is drawn from White's side, so Black's bins are reversed", () => {
  const bins = Array.from({ length: 128 }, (_, i) => (i === 100 ? 1 : 0));
  assert.equal(panel.whiteView(bins, "w").indexOf(1), 100);
  assert.equal(panel.whiteView(bins, "b").indexOf(1), 27);
  const bars = panel.histogramBars(bins, "w", 32);
  assert.equal(bars.length, 128);
  assert.deepEqual(bars[100], { x: 100, height: 32 });
  assert.equal(bars[0].height, 0);
});

test("the backend label says WASM, one thread, the precision and where the weights came from", () => {
  assert.equal(panel.backendLabel({ backend: "wasm", threads: 1, source: "cache" }, card), "WASM, 1 thread, int8 weights (from this browser's cache)");
  assert.equal(panel.backendLabel({ backend: "wasm", threads: 1, source: "network" }, {}), "WASM, 1 thread, fp32 weights (downloaded)");
});
