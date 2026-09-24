// The model's bytes, kept in the browser's Cache API under the sha256 its card (models/model.json)
// records. A visitor downloads a given model once; a new model has a new hash, so it is fetched again
// and the old entry is deleted. Every byte string is hashed before use, from the network or the cache,
// and a mismatch is an error: the page never runs weights its card does not describe.
// Pure except for its injected fetch, caches and crypto, so site/tests/page.test.mjs runs it in Node.

export const CACHE_NAME = "blink-models";

export function toHex(buffer) {
  return Array.from(new Uint8Array(buffer), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function sha256Hex(bytes, subtle = globalThis.crypto && globalThis.crypto.subtle) {
  return toHex(await subtle.digest("SHA-256", bytes));
}

// One entry per model file and content hash.
export function cacheKey(url, sha256) {
  const key = new URL(url);
  key.searchParams.set("sha256", sha256);
  return key.href;
}

async function download(url, fetchImpl) {
  const response = await fetchImpl(url, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

async function openCache(cachesImpl) {
  try {
    return cachesImpl ? await cachesImpl.open(CACHE_NAME) : null;
  } catch {
    return null; // storage blocked (private mode, quota): run without the cache
  }
}

async function fromCache(cache, key, sha256, subtle) {
  const hit = await cache.match(key);
  if (!hit) {
    return null;
  }
  const bytes = new Uint8Array(await hit.arrayBuffer());
  if ((await sha256Hex(bytes, subtle)) === sha256) {
    return bytes;
  }
  await cache.delete(key); // a damaged entry: drop it and download again
  return null;
}

async function store(cache, key, bytes) {
  try {
    await cache.put(key, new Response(bytes, { headers: { "Content-Type": "application/octet-stream" } }));
    for (const request of await cache.keys()) {
      if (request.url !== key) {
        await cache.delete(request);
      }
    }
  } catch {
    // a full or blocked cache only costs the next visit a download
  }
}

// { bytes, source: "cache" | "network", verified }. `sha256` comes from the model card; without it
// (or without a Cache API) the model is simply downloaded.
export async function loadModel({ url, sha256, fetchImpl = globalThis.fetch, cachesImpl = globalThis.caches, subtle }) {
  const crypto = subtle || (globalThis.crypto && globalThis.crypto.subtle);
  if (!sha256 || !crypto) {
    return { bytes: await download(url, fetchImpl), source: "network", verified: false };
  }
  const cache = await openCache(cachesImpl);
  const key = cacheKey(url, sha256);
  const cached = cache ? await fromCache(cache, key, sha256, crypto) : null;
  if (cached) {
    return { bytes: cached, source: "cache", verified: true };
  }
  const bytes = await download(url, fetchImpl);
  const digest = await sha256Hex(bytes, crypto);
  if (digest !== sha256) {
    throw new Error(`the model does not match its card: sha256 ${digest.slice(0, 12)} is not ${sha256.slice(0, 12)}`);
  }
  if (cache) {
    await store(cache, key, bytes);
  }
  return { bytes, source: "network", verified: true };
}
