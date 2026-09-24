/*
 * The Blink learning film, drawn from time alone: window.seek(t) paints the exact frame at t seconds,
 * window.duration() is the film's length, window.__ready turns true once data, pieces and the font are in.
 * No CSS transitions and no timers, so every captured frame is reproducible (blink/film/render.py).
 */
"use strict";

(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const SQUARE = 100;
  const PIECE_BOX = 45;
  const HIST_W = 860;
  const HIST_H = 100;
  const MIN_ARROW_P = 0.004;
  const HOOK_FADE_IN = 0.35;
  const HOOK_FADE_OUT = 0.3;
  const END_FADE_IN = 0.4;
  let data = null;
  let text = null;
  let histMax = 1;

  const byId = (id) => document.getElementById(id);
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (x) => (x < 0.5 ? 2 * x * x : 1 - Math.pow(-2 * x + 2, 2) / 2);
  const fill = (template, values) => template.replace(/\{(\w+)\}/g, (_, key) => String(values[key]));
  const whole = (n) => Math.round(n).toLocaleString("en-US");

  function hours(h) {
    if (h === null || h === undefined) return "-";
    if (h < 0.1) return h.toFixed(3);
    return h < 10 ? h.toFixed(2) : h.toFixed(1);
  }

  function percent(p) {
    return p >= 0.095 ? `${Math.round(p * 100)}%` : `${(p * 100).toFixed(1)}%`;
  }

  function svg(tag, attrs, parent) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, val] of Object.entries(attrs)) node.setAttribute(key, String(val));
    if (parent) parent.appendChild(node);
    return node;
  }

  // ------------------------------------------------------------------------------------------ board

  const flipped = () => data.position.side_to_move === "black";

  function squareXY(square) {
    const file = square.charCodeAt(0) - 97;
    const rank = square.charCodeAt(1) - 49;
    return flipped() ? [(7 - file) * SQUARE, rank * SQUARE] : [file * SQUARE, (7 - rank) * SQUARE];
  }

  function centre(square) {
    const [x, y] = squareXY(square);
    return [x + SQUARE / 2, y + SQUARE / 2];
  }

  function installSprite(source) {
    const doc = new DOMParser().parseFromString(source, "image/svg+xml");
    const defs = byId("sprite");
    for (const group of doc.documentElement.querySelectorAll(":scope > g[id]")) {
      defs.appendChild(document.importNode(group, true));
    }
  }

  function buildSquares() {
    const layer = byId("squares");
    for (let rank = 0; rank < 8; rank++) {
      for (let file = 0; file < 8; file++) {
        const [x, y] = squareXY(String.fromCharCode(97 + file) + String(rank + 1));
        const shade = (file + rank) % 2 === 0 ? "dark" : "light";
        svg("rect", { x, y, width: SQUARE, height: SQUARE, class: shade }, layer);
      }
    }
  }

  function buildCoords() {
    const layer = byId("coords");
    const bottomRank = flipped() ? "8" : "1";
    const leftFile = flipped() ? "h" : "a";
    for (let i = 0; i < 8; i++) {
      const file = String.fromCharCode(97 + i);
      const [fx, fy] = squareXY(file + bottomRank);
      svg("text", { x: fx + SQUARE - 7, y: fy + SQUARE - 7, "text-anchor": "end", class: "coord" }, layer)
        .textContent = file;
      const rank = String(i + 1);
      const [rx, ry] = squareXY(leftFile + rank);
      svg("text", { x: rx + 6, y: ry + 20, class: "coord" }, layer).textContent = rank;
    }
  }

  function buildPieces() {
    const layer = byId("pieces");
    const rows = data.position.fen.split(" ")[0].split("/");
    rows.forEach((row, i) => {
      let file = 0;
      for (const ch of row) {
        if (/\d/.test(ch)) {
          file += Number(ch);
          continue;
        }
        const [x, y] = squareXY(String.fromCharCode(97 + file) + String(8 - i));
        const colour = ch === ch.toUpperCase() ? "w" : "b";
        const transform = `translate(${x} ${y}) scale(${SQUARE / PIECE_BOX})`;
        svg("use", { href: `#${colour}${ch.toLowerCase()}`, transform }, layer);
        file += 1;
      }
    });
  }

  // ------------------------------------------------------------------------------------------ arrows

  function pOf(frame, move) {
    if (!frame) return 0;
    const hit = frame.top5.find((m) => m.move === move);
    return hit ? hit.p : 0;
  }

  function drawArrow(layer, from, to, width, opacity, found) {
    const [x1, y1] = centre(from);
    const [x2, y2] = centre(to);
    const len = Math.hypot(x2 - x1, y2 - y1);
    const [ux, uy] = [(x2 - x1) / len, (y2 - y1) / len];
    const head = Math.min(len * 0.45, width * 1.4 + 16);
    const [bx, by] = [x2 - ux * head, y2 - uy * head];
    const half = width * 0.9 + 10;
    const group = svg("g", { class: found ? "arrow found" : "arrow", opacity: opacity.toFixed(3) }, layer);
    svg("line", { x1: x1 + ux * 16, y1: y1 + uy * 16, x2: bx, y2: by, "stroke-width": width.toFixed(2) }, group);
    const points = [
      [x2, y2],
      [bx - uy * half, by + ux * half],
      [bx + uy * half, by - ux * half],
    ];
    svg("polygon", { points: points.map((p) => p.map((v) => v.toFixed(1)).join(",")).join(" ") }, group);
    return [x1 + (x2 - x1) * 0.58, y1 + (y2 - y1) * 0.58];
  }

  function drawLabel(layer, x, y, label, opacity) {
    const group = svg("g", { class: "label", opacity: opacity.toFixed(3) }, layer);
    const width = 18 + 15 * label.length;
    svg("rect", { x: x - width / 2, y: y - 20, width, height: 40, rx: 20 }, group);
    svg("text", { x, y: y + 9, "text-anchor": "middle" }, group).textContent = label;
  }

  function drawArrows(prev, cur, a) {
    const layer = byId("arrows");
    const labels = byId("labels");
    layer.replaceChildren();
    labels.replaceChildren();
    const moves = new Set([...(prev ? prev.top5 : []), ...cur.top5].map((m) => m.move));
    const items = [...moves]
      .map((move) => ({ move, p: lerp(pOf(prev, move), pOf(cur, move), a) }))
      .filter((item) => item.p >= MIN_ARROW_P)
      .sort((x, y) => x.p - y.p || (x.move < y.move ? -1 : 1));
    const curTop = cur.top5.slice(0, 3).map((m) => m.move);
    const prevTop = prev ? prev.top5.slice(0, 3).map((m) => m.move) : [];
    const best = cur.top5[0].move;
    for (const item of items) {
      const width = 7 + 36 * Math.sqrt(item.p);
      const opacity = 0.2 + 0.7 * Math.sqrt(item.p);
      const found = item.move === data.position.solution && item.move === best;
      const [lx, ly] = drawArrow(layer, item.move.slice(0, 2), item.move.slice(2, 4), width, opacity, found);
      const shown = Math.max(curTop.includes(item.move) ? a : 0, prevTop.includes(item.move) ? 1 - a : 0);
      if (shown > 0.01) drawLabel(labels, lx, ly, percent(item.p), shown);
    }
  }

  // ------------------------------------------------------------------------------------------ value

  function buildHistogram() {
    histMax = Math.max(...data.frames.flatMap((f) => f.value_bins), 1e-9);
    const bars = byId("bars");
    const w = HIST_W / 128;
    for (let i = 0; i < 128; i++) {
      svg("rect", { x: (i * w).toFixed(2), y: HIST_H, width: (w - 1).toFixed(2), height: 0, class: "bar" }, bars);
    }
    const axis = byId("axis");
    [["0%", 0, "start"], ["50%", HIST_W / 2, "middle"], ["100%", HIST_W, "end"]].forEach(([label, x, anchor]) => {
      svg("text", { x, y: 128, "text-anchor": anchor }, axis).textContent = label;
    });
  }

  function drawValue(prev, cur, a) {
    const bars = byId("bars").children;
    for (let i = 0; i < 128; i++) {
      const v = lerp(prev ? prev.value_bins[i] : 0, cur.value_bins[i], a);
      const h = (HIST_H * v) / histMax;
      bars[i].setAttribute("y", (HIST_H - h).toFixed(2));
      bars[i].setAttribute("height", h.toFixed(2));
    }
    const win = prev ? lerp(prev.win, cur.win, a) : cur.win;
    const mark = byId("win-mark");
    mark.setAttribute("x1", (win * HIST_W).toFixed(1));
    mark.setAttribute("x2", (win * HIST_W).toFixed(1));
    mark.setAttribute("opacity", prev ? "1" : a.toFixed(3));
    const side = text[data.position.side_to_move];
    byId("win-text").textContent = fill(text.win, { side, pct: Math.round(win * 100) });
  }

  // ------------------------------------------------------------------------------------------ labels

  function kindLabel(frame) {
    if (frame.interpolated) return text.kind_interpolated;
    return { init: text.kind_init, final: text.kind_final }[frame.kind] || "";
  }

  function found(frame) {
    return !!frame && frame.top5[0].move === data.position.solution;
  }

  function drawLabels(prev, cur, a) {
    byId("frame-label").textContent = fill(text.frame, { i: cur.index, n: data.frames.length });
    byId("kind").textContent = kindLabel(cur);
    const shown = found(cur) ? (found(prev) ? 1 : a) : found(prev) ? 1 - a : 0;
    const badge = byId("found");
    badge.textContent = text.found;
    badge.style.opacity = shown.toFixed(3);
    const step = lerp(prev ? prev.step : 0, cur.step, a);
    const seen = lerp(prev ? prev.positions_seen : 0, cur.positions_seen, a);
    const gpu = cur.gpu_hours === null ? null : lerp(prev && prev.gpu_hours !== null ? prev.gpu_hours : 0, cur.gpu_hours, a);
    byId("step").textContent = fill(text.step, { step: whole(step) });
    byId("seen").textContent = fill(text.seen, { n: whole(seen) });
    byId("gpu").textContent = fill(text.gpu, { h: hours(gpu) });
    const lo = prev ? prev.step : -1;
    const passed = data.milestones.filter((m) => m.step > lo && m.step <= cur.step).map((m) => m.label);
    const milestone = byId("milestone");
    milestone.textContent = passed.join("  ");
    milestone.style.opacity = a.toFixed(3);
  }

  // ------------------------------------------------------------------------------------------ time

  function show(id, opacity) {
    const node = byId(id);
    node.style.opacity = opacity.toFixed(3);
    node.style.visibility = opacity > 0 ? "visible" : "hidden";
  }

  function draw(t) {
    const timing = data.timing;
    const n = data.frames.length;
    const seg = timing.morph + timing.hold;
    const filmT = t - timing.hook;
    const endT = filmT - n * seg;
    let k = 0;
    let a = 0;
    if (endT >= 0) {
      [k, a] = [n - 1, 1];
    } else if (filmT >= 0) {
      k = Math.floor(filmT / seg);
      const within = filmT - k * seg;
      a = within < timing.morph ? ease(within / timing.morph) : 1;
    }
    const prev = k > 0 ? data.frames[k - 1] : null;
    const cur = data.frames[k];
    drawArrows(prev, cur, a);
    drawValue(prev, cur, a);
    drawLabels(prev, cur, a);
    const hookIn = Math.min(t / HOOK_FADE_IN, (timing.hook - t) / HOOK_FADE_OUT);
    show("hook", filmT < 0 ? clamp(hookIn, 0, 1) : 0);
    show("film", filmT < 0 ? 0 : 1);
    show("end", endT >= 0 ? clamp(endT / END_FADE_IN, 0, 1) : 0);
  }

  function applyText() {
    const root = document.documentElement;
    root.lang = data.lang;
    root.dir = data.dir;
    const pos = data.position;
    byId("hook-text").textContent = data.hook;
    byId("hook-sub").textContent = text.sub;
    byId("puzzle").textContent = fill(text.puzzle, { id: pos.puzzle_id, rating: pos.rating });
    byId("to-play").textContent = fill(text.to_play, { side: text[pos.side_to_move] });
    byId("note").textContent = data.note;
    byId("end-rating").textContent = data.end.rating;
    byId("end-never").textContent = data.end.never_seen;
    byId("end-repo").textContent = data.end.repo;
  }

  async function fetchOk(url, as) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return as === "json" ? response.json() : response.text();
  }

  async function boot() {
    const [film, sprite] = await Promise.all([fetchOk("data.json", "json"), fetchOk("pieces/cburnett.svg", "text")]);
    data = film;
    text = film.text;
    installSprite(sprite);
    applyText();
    buildSquares();
    buildCoords();
    buildPieces();
    buildHistogram();
    await document.fonts.load('800 88px "Heebo"', data.hook);
    await document.fonts.load('400 30px "Heebo"', "0123456789%");
    await document.fonts.ready;
    draw(0);
    window.__ready = true;
  }

  window.__ready = false;
  window.duration = () => {
    const timing = data.timing;
    return timing.hook + data.frames.length * (timing.morph + timing.hold) + timing.end;
  };
  window.seek = (t) => {
    draw(t);
    return t;
  };
  boot().catch((err) => {
    window.__filmError = String((err && err.stack) || err);
  });
})();
