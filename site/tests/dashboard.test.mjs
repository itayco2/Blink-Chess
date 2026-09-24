// The live dashboard's speed WARN (blink/dashboard/live.html) against the cases blink.train.status.speed_check
// is held to (tests/fixtures/dashboard_speed_cases.json), so the page and `blink status` warn alike.
// The page's inline script runs in a bare VM context: its fetch fails, so it only defines its functions.
// Run from the repo root with `node --test "site/tests/**/*.test.mjs"`.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const REPO = new URL("../../", import.meta.url);
const html = readFileSync(new URL("blink/dashboard/live.html", REPO), "utf-8");
const { cases } = JSON.parse(readFileSync(new URL("tests/fixtures/dashboard_speed_cases.json", REPO), "utf-8"));

function loadPage() {
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  const element = () => ({ textContent: "", className: "", replaceChildren() {}, querySelector: element });
  const context = vm.createContext({
    console,
    document: { getElementById: element, querySelector: element },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    fetch: () => Promise.reject(new Error("no server in this test")),
    setInterval() {},
  });
  vm.runInContext(script, context);
  return context;
}

const page = loadPage();
const verdicts = vm.runInContext("verdicts", page); // a top-level const: not a property of the global
const near = (got, want) => (want === null ? got === null : Math.abs(got - want) <= 1e-9 * Math.max(1, Math.abs(want)));

for (const { name, rows, expect } of cases) {
  test(`speedCheck: ${name}`, () => {
    const check = page.speedCheck(rows);
    assert.equal(check.warn, expect.warn);
    for (const field of ["reference", "rate", "slow_s"]) {
      assert.ok(near(check[field], expect[field]), `${field}: got ${check[field]}, want ${expect[field]}`);
    }
  });
}

test("the speed caption warns only on a lasting drop and names the loader wait otherwise", () => {
  const warn = cases.find((c) => c.name === "slow for five minutes of wall time warns").rows;
  const [kind, text] = verdicts.speed(warn, []);
  assert.equal(kind, "sick");
  assert.match(text, /^WARN: 600 samples\/s/);
  const single = cases.find((c) => c.name === "one slow row does not warn").rows;
  assert.equal(verdicts.speed(single, [])[0], "watch");
  const steady = cases.find((c) => c.name === "steady rows never warn").rows.map((r) => ({ ...r, data_wait_frac: 0.031 }));
  const [healthy, caption] = verdicts.speed(steady, []);
  assert.equal(healthy, "healthy");
  assert.match(caption, /loader wait 3\.1%/);
});

test("parseLine keeps text fields and reads bare NaN and Infinity as numbers", () => {
  const row = page.parseLine('{"step": 50, "phase": "eval", "loss_policy": NaN, "grad_norm": -Infinity}');
  assert.equal(row.phase, "eval");
  assert.ok(Number.isNaN(row.loss_policy));
  assert.equal(row.grad_norm, -Infinity);
  assert.equal(row.step, 50);
});
