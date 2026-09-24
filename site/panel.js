// What the side panel says about the model: the model card (from models/model.json, which
// `blink export quantize` and `blink export qgate` write), the backend line, and the 128-bin value
// histogram drawn beside the win bar. The text builders are pure; site/tests/page.test.mjs runs them
// on site/tests/card.json, the fixture blink/site/card.py writes, so the key names cannot drift.

const SVG_NS = "http://www.w3.org/2000/svg";

const mb = (bytes) => `${(bytes / 1e6).toFixed(2)} MB`;
const pct = (share) => `${(share * 100).toFixed(2)}%`;
const signed = (x) => `${x > 0 ? "+" : x < 0 ? "-" : ""}${Math.abs(x).toFixed(2)}`;

function sizeLine(card) {
  const quant = card.quantization;
  if (card.precision === "int8" && quant && quant.fp32_bytes) {
    return `${mb(card.bytes)} int8 (fp32 ${mb(quant.fp32_bytes)})`;
  }
  return card.bytes ? mb(card.bytes) : null;
}

function worstBandDrop(bands) {
  const drops = Object.values(bands || {}).filter((band) => band.n > 0).map((band) => band.drop_pt);
  return drops.length ? Math.max(...drops) : 0;
}

function gateLines(gate) {
  const lines = [`top-1 agreement with fp32 ${pct(gate.top1_agreement)} of ${gate.positions.toLocaleString("en-US")} positions`];
  const overall = gate.puzzles && gate.puzzles.overall;
  if (overall) {
    const change = `${signed(-overall.drop_pt)} pt, ${overall.n.toLocaleString("en-US")} puzzles; worst band ${signed(-worstBandDrop(gate.puzzles.bands))} pt`;
    lines.push(`puzzles ${overall.fp32_pct.toFixed(2)}% fp32, ${overall.int8_pct.toFixed(2)}% int8 (${change})`);
  }
  lines.push(`mean |dwin%| ${gate.mean_abs_dwin_pt.toFixed(2)} pt`);
  const verdict = gate.passed ? "passed" : `failed (${(gate.failures || [])[0] || "see qgate.json"})`;
  lines.push(`quantization gate: ${verdict}`);
  return lines;
}

// The card as lines of text, first line the model's name and label.
export function cardLines(card) {
  const title = [card.name || card.selector || "model.onnx", card.label].filter(Boolean).join(", ");
  const facts = [
    card.parameters ? `${card.parameters.toLocaleString("en-US")} parameters` : null,
    sizeLine(card),
    card.sha256 ? `sha256 ${card.sha256.slice(0, 12)}` : null,
  ].filter(Boolean);
  const lines = [title, facts.join(", ")];
  const gate = card.quantization && card.quantization.gate;
  if (gate) {
    lines.push(...gateLines(gate));
  } else if (card.precision === "int8") {
    lines.push("quantization gate: not run yet (blink export qgate)");
  }
  if (card.selector === "stand-in") {
    lines.push("Random, untrained weights: its moves test the pipeline, not chess.");
  }
  return lines;
}

export function backendLabel(info, card) {
  const where = info.source === "cache" ? "from this browser's cache" : "downloaded";
  const threads = `${info.threads} thread${info.threads === 1 ? "" : "s"}`;
  return `${String(info.backend).toUpperCase()}, ${threads}, ${card.precision || "fp32"} weights (${where})`;
}

// The value head's bins from White's side: bin i is White's win chance i/128, whoever is to move.
export function whiteView(bins, turn) {
  return turn === "w" ? Array.from(bins) : Array.from(bins).reverse();
}

export function histogramBars(bins, turn, height) {
  const white = whiteView(bins, turn);
  const top = Math.max(...white) || 1;
  return white.map((p, x) => ({ x, height: (p / top) * height }));
}

// --- DOM ----------------------------------------------------------------------------------------------

export function renderCard(element, card) {
  const [title, ...rest] = cardLines(card);
  const heading = document.createElement("strong");
  heading.textContent = title;
  const items = rest.filter(Boolean).map((line) => {
    const item = document.createElement("li");
    item.textContent = line;
    return item;
  });
  const list = document.createElement("ul");
  list.className = "card-facts";
  list.replaceChildren(...items);
  element.replaceChildren(heading, list);
}

export function renderHistogram(svg, bins, turn) {
  const height = Number(svg.viewBox.baseVal.height) || 32;
  const bars = bins ? histogramBars(bins, turn, height) : [];
  const rects = bars.map(({ x, height: h }) => {
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", String(x + 0.1));
    rect.setAttribute("width", "0.8");
    rect.setAttribute("y", (height - h).toFixed(2));
    rect.setAttribute("height", h.toFixed(2));
    return rect;
  });
  svg.replaceChildren(...rects);
  svg.dataset.bins = String(rects.length);
}
