// The training replay: the live dashboard's six cards, drawn from three static files that
// `blink site replay` froze (run.json, metrics.jsonl, evals.jsonl; one row per 2,000 steps).
// The slider picks a step and every card shows the run up to it; Play sweeps it from start to end.

const COLORS = { accent: "#ffd54a", val: "#8ab4f8", grid: "#1f2633", muted: "#7d8795" };
const LN_MOVES = Math.log(1880);
const LN_BINS = Math.log(128);
const RANDOM_LEGAL_TOP1 = 0.03;
const PLAY_MS = 8000;

// --- Pure helpers (tested in Node) ------------------------------------------------------------------

export function parseJsonl(text) {
  return text
    .split("\n")
    .filter((line) => line.trim())
    .map((line) => JSON.parse(line));
}

export function upTo(rows, step) {
  return rows.filter((row) => row.step <= step);
}

export function series(rows, key) {
  return rows.filter((row) => Number.isFinite(row[key])).map((row) => [row.step, row[key]]);
}

// Axis bounds over the whole run, so the curves grow into fixed axes while the replay plays.
export function bounds(lines, ref) {
  const points = lines.flat();
  const ys = points.map((p) => p[1]).concat(ref === undefined ? [] : [ref]);
  const xs = points.map((p) => p[0]);
  if (!xs.length) {
    return null;
  }
  let [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  if (x1 === x0) {
    x1 = x0 + 1;
  }
  if (y1 === y0) {
    [y0, y1] = [y0 - Math.abs(y0) * 0.05 - 1e-9, y1 + Math.abs(y1) * 0.05 + 1e-9];
  }
  return { x0, x1, y0, y1 };
}

export function fmt(x, digits) {
  if (x === undefined || x === null || !Number.isFinite(x)) {
    return "-";
  }
  if (x === 0) {
    return "0";
  }
  if (Math.abs(x) >= 1000) {
    return Math.round(x).toLocaleString("en-US");
  }
  return Math.abs(x) < 0.01 ? x.toExponential(2) : x.toFixed(digits);
}

// --- Drawing --------------------------------------------------------------------------------------

function drawChart(canvas, lines, box, ref) {
  const ratio = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  canvas.width = Math.round(w * ratio);
  canvas.height = Math.round(h * ratio);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!box) {
    return;
  }
  const pad = { l: 44, r: 8, t: 6, b: 18 };
  const sx = (x) => pad.l + ((x - box.x0) / (box.x1 - box.x0)) * (w - pad.l - pad.r);
  const sy = (y) => h - pad.b - ((y - box.y0) / (box.y1 - box.y0)) * (h - pad.t - pad.b);
  ctx.strokeStyle = COLORS.grid;
  ctx.fillStyle = COLORS.muted;
  ctx.lineWidth = 1;
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  for (const y of [box.y0, (box.y0 + box.y1) / 2, box.y1]) {
    ctx.beginPath();
    ctx.moveTo(pad.l, sy(y));
    ctx.lineTo(w - pad.r, sy(y));
    ctx.stroke();
    ctx.fillText(fmt(y, 3), 2, sy(y) + 4);
  }
  ctx.fillText(fmt(box.x0, 0), pad.l, h - 4);
  ctx.fillText(fmt(box.x1, 0), w - pad.r - 40, h - 4);
  if (ref !== undefined) {
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = COLORS.muted;
    ctx.beginPath();
    ctx.moveTo(pad.l, sy(ref));
    ctx.lineTo(w - pad.r, sy(ref));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  for (const { points, color } of lines) {
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach(([x, y], i) => (i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y))));
    ctx.stroke();
    for (const [x, y] of points) {
      ctx.beginPath();
      ctx.arc(sx(x), sy(y), 2.5, 0, 7);
      ctx.fill();
    }
  }
}

// name -> [big number, lines as [rows, key, color], reference line]
const CARDS = {
  policy: (m, e) => [fmt(m.at(-1)?.loss_policy, 3), [[m, "loss_policy", COLORS.accent], [e, "policy_ce", COLORS.val]], LN_MOVES],
  value: (m, e) => [fmt(m.at(-1)?.loss_value, 3), [[m, "loss_value", COLORS.accent], [e, "value_ce", COLORS.val]], LN_BINS],
  top1: (m, e) => [e.length ? `${fmt(e.at(-1).top1 * 100, 1)}%` : "-", [[e, "top1", COLORS.val], [e, "ema_top1", COLORS.accent]], RANDOM_LEGAL_TOP1],
  speed: (m) => [fmt(m.at(-1)?.samples_per_s, 0), [[m.slice(1), "samples_per_s", COLORS.accent]], undefined],
  lr: (m) => [fmt(m.at(-1)?.lr, 5), [[m, "lr", COLORS.accent]], undefined],
  grad: (m) => [fmt(m.at(-1)?.grad_norm, 3), [[m, "grad_norm", COLORS.accent]], 1.0],
};

const state = { run: null, metrics: [], evals: [], cursor: 0, last: 0, playing: false };

function render() {
  const m = upTo(state.metrics, state.cursor);
  const e = upTo(state.evals, state.cursor);
  for (const [name, card] of Object.entries(CARDS)) {
    const element = document.querySelector(`[data-chart="${name}"]`);
    const [big, specs, ref] = card(m, e);
    const [, fullSpecs] = card(state.metrics, state.evals);
    const box = bounds(fullSpecs.map(([rows, key]) => series(rows, key)), ref);
    element.querySelector(".value").textContent = big;
    drawChart(element.querySelector("canvas"), specs.map(([rows, key, color]) => ({ points: series(rows, key), color })), box, ref);
  }
  document.getElementById("cursor-label").textContent = `step ${state.cursor.toLocaleString("en-US")}`;
  document.getElementById("cursor").value = String(state.cursor);
}

function describe(run) {
  const parts = [`run ${run.run}`, `WORLD ${run.world}`];
  if (run.parameters) {
    parts.push(`${run.parameters.toLocaleString("en-US")} parameters`);
  }
  if (run.steps) {
    parts.push(`${run.steps.toLocaleString("en-US")} steps`);
  }
  const kept = `${run.metrics_rows} of ${run.source_metrics_rows} metric rows kept (one per ${run.every.toLocaleString("en-US")} steps)`;
  return `${parts.join(", ")}. ${kept}.`;
}

function play() {
  if (state.playing) {
    state.playing = false;
    return;
  }
  state.playing = true;
  document.getElementById("play").textContent = "Pause";
  const started = performance.now() - (state.cursor >= state.last ? 0 : (state.cursor / state.last) * PLAY_MS);
  const frame = (now) => {
    const share = Math.min(1, (now - started) / PLAY_MS);
    state.cursor = Math.round(share * state.last);
    render();
    if (state.playing && share < 1) {
      requestAnimationFrame(frame);
    } else {
      state.playing = false;
      document.getElementById("play").textContent = "Play";
    }
  };
  requestAnimationFrame(frame);
}

async function text(name) {
  const response = await fetch(name, { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`${name}: HTTP ${response.status}`);
  }
  return response.text();
}

async function main() {
  const [run, metrics, evals] = await Promise.all([text("run.json"), text("metrics.jsonl"), text("evals.jsonl")]);
  state.run = JSON.parse(run);
  state.metrics = parseJsonl(metrics);
  state.evals = parseJsonl(evals);
  state.last = Math.max(0, ...state.metrics.map((r) => r.step), ...state.evals.map((r) => r.step));
  state.cursor = state.last;
  const slider = document.getElementById("cursor");
  slider.max = String(state.last);
  slider.addEventListener("input", () => {
    state.playing = false;
    state.cursor = Number(slider.value);
    render();
  });
  document.getElementById("play").addEventListener("click", play);
  window.addEventListener("resize", render);
  document.getElementById("about").textContent = describe(state.run);
  render();
  window.__blinkReplay = Object.freeze({ ready: true, run: state.run.run, metrics: state.metrics.length, evals: state.evals.length });
}

if (typeof document !== "undefined") {
  main().catch((error) => {
    document.getElementById("about").textContent = `The replay did not load: ${error.message}`;
    console.error(error);
  });
}
