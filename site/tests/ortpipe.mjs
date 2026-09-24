// The web runtime of `blink export qgate`: onnxruntime-web's WASM backend on one thread, the browser's
// own kernels, driven by Python over stdin/stdout. Not a test itself (no .test.mjs suffix).
// Usage: node site/tests/ortpipe.mjs <model.onnx> [<model.onnx> ...]
// On start it writes "BLNK", a uint32 length and a JSON {ort, node}. Then, per request frame
// (uint32 model index, uint32 n, n x 64 uint8 square codes), it replies with uint32 n, float32 policy
// logits [n, 1880] and float32 value logits [n, 128]. An error replies n = 0xFFFFFFFF, a uint32 length and
// the message. All integers are little-endian. It exits when stdin closes.

import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web/wasm";

const HEADER = 8;
const SQUARES = 64;
const ERROR = 0xffffffff;

ort.env.wasm.numThreads = 1;

function u32(value) {
  const out = Buffer.alloc(4);
  out.writeUInt32LE(value >>> 0, 0);
  return out;
}

function write(buffer) {
  return new Promise((resolve) => (process.stdout.write(buffer) ? resolve() : process.stdout.once("drain", resolve)));
}

async function answer(sessions, frame) {
  const index = frame.readUInt32LE(0);
  const n = frame.readUInt32LE(4);
  try {
    if (!sessions[index]) {
      throw new Error(`no model ${index} (${sessions.length} loaded)`);
    }
    const tokens = new BigInt64Array(n * SQUARES);
    for (let i = 0; i < tokens.length; i++) {
      tokens[i] = BigInt(frame[HEADER + i]);
    }
    const out = await sessions[index].run({ tokens: new ort.Tensor("int64", tokens, [n, SQUARES]) });
    const policy = out.policy_logits.data;
    const value = out.value_logits.data;
    await write(Buffer.concat([u32(n), Buffer.from(policy.buffer, policy.byteOffset, policy.byteLength), Buffer.from(value.buffer, value.byteOffset, value.byteLength)]));
  } catch (error) {
    const message = Buffer.from(String(error && error.message ? error.message : error), "utf-8");
    await write(Buffer.concat([u32(ERROR), u32(message.length), message]));
  }
}

async function main() {
  const paths = process.argv.slice(2);
  const sessions = [];
  for (const path of paths) {
    sessions.push(await ort.InferenceSession.create(readFileSync(path), { executionProviders: ["wasm"] }));
  }
  const info = Buffer.from(JSON.stringify({ ort: ort.env.versions.web, node: process.version }), "utf-8");
  await write(Buffer.concat([Buffer.from("BLNK", "ascii"), u32(info.length), info]));

  let pending = Buffer.alloc(0);
  let queue = Promise.resolve();
  process.stdin.on("data", (chunk) => {
    pending = Buffer.concat([pending, chunk]);
    while (pending.length >= HEADER) {
      const size = HEADER + pending.readUInt32LE(4) * SQUARES;
      if (pending.length < size) {
        break;
      }
      const frame = pending.subarray(0, size);
      pending = pending.subarray(size);
      queue = queue.then(() => answer(sessions, frame));
    }
  });
  process.stdin.on("end", () => queue.then(() => process.exit(0)));
}

main().catch((error) => {
  process.stderr.write(`${error && error.stack ? error.stack : error}\n`);
  process.exit(1);
});
