// Blink in the browser: one look (policy mode) per move. The network runs in worker.js
// (onnxruntime-web, WASM, one thread); this module owns the game, the board and what is drawn.
// Blink's move is the legal move with the highest policy probability, one forward pass and no search,
// with the bot's rule checks from rules.js: a mate in one is played without looking (R2), and draws
// by rule are avoided when clearly winning and taken when clearly losing (R3). panel.js draws the model
// card and the value histogram; rating.js reads the bot's live Lichess rating (config.json names it).

import { Chessboard, COLOR, FEN, INPUT_EVENT_TYPE } from "./vendor/cm-chessboard/Chessboard.js";
import { Chess, validateFen } from "./vendor/chess.js/chess.js";
import * as panel from "./panel.js";
import * as rating from "./rating.js";
import * as rules from "./rules.js";
import * as tok from "./tokenizer.js";

const MODEL_URL = "models/model.onnx";
const CARD_URL = "models/model.json";
const VOCAB_URL = "vocab.json";
const CONFIG_URL = "config.json";
const ARROWS = 3;
const SQUARE = 100; // arrow overlay units per square: the overlay's viewBox is 0 0 800 800
const SVG_NS = "http://www.w3.org/2000/svg";

const $ = (id) => document.getElementById(id);

const app = {
  vocab: null,
  engine: null,
  board: null,
  game: new Chess(),
  startFen: FEN.start,
  userColor: "w",
  generation: 0,
  thinking: false,
  ready: false,
  replies: 0,
  timings: [],
  lastLook: null,
  card: {},
  backend: null,
  rating: null,
};

// --- The worker: one queued request at a time --------------------------------------------------------

function createEngine() {
  const worker = new Worker("worker.js", { type: "module" });
  const pending = new Map();
  let nextId = 1;
  let queue = Promise.resolve();
  worker.onmessage = ({ data }) => {
    const job = pending.get(data.id);
    pending.delete(data.id);
    if (job) {
      data.type === "error" ? job.reject(new Error(data.message)) : job.resolve(data);
    }
  };
  worker.onerror = (event) => {
    const error = new Error(event.message || "the model worker failed to start");
    pending.forEach((job) => job.reject(error));
    pending.clear();
  };
  const send = (message) =>
    new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ ...message, id });
    });
  const call = (message) => {
    const result = queue.then(() => send(message));
    queue = result.catch(() => undefined);
    return result;
  };
  return {
    load: (url, sha256) => call({ type: "load", url: new URL(url, document.baseURI).href, sha256 }),
    evaluate: (codes) => call({ type: "evaluate", codes: Array.from(codes) }),
  };
}

// --- One look ------------------------------------------------------------------------------------------

function lookOnce(game) {
  return rules.decide(app.vocab, game, Chess, app.engine.evaluate, ARROWS);
}

async function blinkMoves() {
  if (!app.ready || gameOverStatus()) {
    return;
  }
  if (app.game.turn() === app.userColor) {
    setStatus("Your move");
    return;
  }
  const generation = app.generation;
  app.thinking = true;
  setStatus("Blink is looking");
  try {
    const look = await lookOnce(app.game);
    if (generation !== app.generation) {
      return; // a new game or a pasted FEN replaced this position while the network ran
    }
    const { from, to, promotion } = look.move;
    app.game.move({ from, to, promotion });
    app.timings = look.calls ? [...app.timings, look.ms] : app.timings; // R2 made no network call
    app.lastLook = look;
    await app.board.setPosition(app.game.fen(), true);
    if (generation !== app.generation) {
      return;
    }
    renderLook(look);
    renderMoves();
    app.replies += 1; // counted once the reply is fully shown: moved, arrows drawn, win bar set
    if (!gameOverStatus()) {
      setStatus("Your move");
    }
  } catch (error) {
    fail("Blink could not move", error);
  } finally {
    if (generation === app.generation) {
      app.thinking = false;
    }
  }
}

// --- The board and the user's moves --------------------------------------------------------------------

function createBoard() {
  return new Chessboard($("board"), {
    position: FEN.start,
    orientation: COLOR.white,
    assetsUrl: "./",
    style: {
      cssClass: "blink",
      borderType: "none",
      showCoordinates: true,
      animationDuration: 150,
      pieces: { file: "assets/pieces/cburnett.svg", tileSize: 45 },
    },
  });
}

function userMayMove(square) {
  return app.ready && !app.thinking && app.game.turn() === app.userColor && app.game.moves({ square }).length > 0;
}

function applyUserMove(from, to) {
  const candidates = app.game.moves({ square: from, verbose: true }).filter((move) => move.to === to);
  if (candidates.length === 0) {
    return false;
  }
  const promotion = candidates[0].promotion ? $("promotion").value : undefined;
  app.game.move({ from, to, promotion });
  return true;
}

// The board animates the user's own move and then re-shows the moved piece. A no-op position change
// queues behind both, so the redraw below never races them. Only castling, en passant and promotion
// leave the board different from the game, and only then is it redrawn.
async function afterUserMove() {
  clearArrows();
  await app.board.setPosition(app.board.getPosition(), false);
  if (app.board.getPosition() !== app.game.fen().split(" ")[0]) {
    await app.board.setPosition(app.game.fen(), true);
  }
  renderMoves();
  await blinkMoves();
}

function onMoveInput(event) {
  switch (event.type) {
    case INPUT_EVENT_TYPE.moveInputStarted:
      return userMayMove(event.squareFrom);
    case INPUT_EVENT_TYPE.validateMoveInput:
      return applyUserMove(event.squareFrom, event.squareTo);
    case INPUT_EVENT_TYPE.moveInputFinished:
      if (event.legalMove) {
        afterUserMove();
      }
      return true;
    default:
      return true;
  }
}

async function startGame(fen) {
  app.generation += 1;
  app.thinking = false;
  app.game = tok.loadPosition(Chess, fen);
  app.startFen = app.game.fen();
  app.lastLook = null;
  clearArrows();
  renderWin(null);
  renderValue(null);
  renderMoves();
  $("top-moves").replaceChildren();
  $("rule-note").textContent = "";
  await app.board.setPosition(app.game.fen(), false);
  await blinkMoves();
}

async function switchSide() {
  app.generation += 1; // a look already in flight was made for the old sides: drop it
  app.thinking = false;
  app.userColor = app.userColor === "w" ? "b" : "w";
  $("switch-side").textContent = app.userColor === "w" ? "Play Black" : "Play White";
  await app.board.setOrientation(app.userColor, false);
  if (app.lastLook) {
    drawArrows(app.lastLook.top);
  }
  await blinkMoves();
}

// --- Drawing ---------------------------------------------------------------------------------------

function squareCenter(square) {
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  const white = app.board.getOrientation() === COLOR.white;
  const x = (white ? file : 7 - file) * SQUARE + SQUARE / 2;
  const y = (white ? 7 - rank : rank) * SQUARE + SQUARE / 2;
  return [x, y];
}

function svg(name, attributes, parent) {
  const element = document.createElementNS(SVG_NS, name);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  parent.appendChild(element);
  return element;
}

const points = (list) => list.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

function drawArrow(layer, move, rank) {
  const [x0, y0] = squareCenter(move.from);
  const [x2, y2] = squareCenter(move.to);
  const length = Math.hypot(x2 - x0, y2 - y0);
  const [ux, uy] = [(x2 - x0) / length, (y2 - y0) / length];
  const [nx, ny] = [-uy, ux];
  const [x1, y1] = [x0 + ux * 16, y0 + uy * 16];
  const half = 4 + 11 * move.prob;
  const head = Math.min(length * 0.45, 22 + 20 * move.prob);
  const [sx, sy] = [x2 - ux * head, y2 - uy * head];
  const wing = half + 10;
  const group = svg("g", { class: `blink-arrow rank-${rank}`, "data-uci": move.uci, opacity: 0.55 + 0.4 * move.prob }, layer);
  const shaft = [[x1 + nx * half, y1 + ny * half], [sx + nx * half, sy + ny * half], [sx - nx * half, sy - ny * half], [x1 - nx * half, y1 - ny * half]];
  svg("polygon", { class: "shaft", points: points(shaft) }, group);
  svg("polygon", { class: "head", points: points([[sx + nx * wing, sy + ny * wing], [x2, y2], [sx - nx * wing, sy - ny * wing]]) }, group);
  const [lx, ly] = [(x1 + sx) / 2, (y1 + sy) / 2];
  svg("rect", { class: "label-bg", x: lx - 30, y: ly - 15, width: 60, height: 30, rx: 12 }, group);
  svg("text", { class: "label", x: lx, y: ly }, group).textContent = `${Math.round(move.prob * 100)}%`;
}

function clearArrows() {
  $("arrows").replaceChildren();
}

function drawArrows(top) {
  clearArrows();
  [...top].reverse().forEach((move, i) => drawArrow($("arrows"), move, top.length - i));
}

function renderWin(winWhite) {
  const share = winWhite === null ? 50 : winWhite * 100;
  $("win-white").style.width = `${share.toFixed(1)}%`;
  $("winbar").setAttribute("aria-valuenow", share.toFixed(0));
  $("win-label-white").textContent = winWhite === null ? "White" : `White ${share.toFixed(0)}%`;
  $("win-label-black").textContent = winWhite === null ? "Black" : `Black ${(100 - share).toFixed(0)}%`;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) / 2)];
}

function renderValue(look) {
  panel.renderHistogram($("value-hist"), look && look.bins, look ? look.turn : "w");
  if (!look) {
    $("hist-note").textContent = "The value head's 128 bins of White's win chance appear after Blink's next look.";
  } else if (!look.bins) {
    $("hist-note").textContent = "No look this move: a mate in one is played without the network (R2).";
  } else {
    const white = look.turn === "w" ? look.win : 1 - look.win;
    $("hist-note").textContent = `Value head, 128 bins of White's win chance (White's side on the left): mean ${(white * 100).toFixed(1)}%, from the same forward pass as the move.`;
  }
}

function renderLook(look) {
  drawArrows(look.top);
  renderWin(look.turn === "w" ? look.win : 1 - look.win);
  renderValue(look);
  $("ms-last").textContent = look.calls ? look.ms.toFixed(0) : "0";
  const timed = app.timings.length ? `median ${median(app.timings).toFixed(0)} ms over ${app.timings.length}` : "";
  $("ms-median").textContent = timed ? `(${timed})` : "";
  const items = look.top.map((move) => {
    const item = document.createElement("li");
    item.textContent = `${move.san}  ${(move.prob * 100).toFixed(1)}%`;
    return item;
  });
  $("top-moves").replaceChildren(...items);
  $("rule-note").textContent = rules.describe(look);
}

function renderMoves() {
  const history = app.game.history();
  const blackFirst = app.startFen.split(" ")[1] === "b";
  const plies = blackFirst ? ["...", ...history] : history;
  const items = [];
  for (let i = 0; i < plies.length; i += 2) {
    const item = document.createElement("li");
    item.textContent = plies.slice(i, i + 2).join(" ");
    items.push(item);
  }
  $("move-list").replaceChildren(...items);
  $("move-list").setAttribute("start", app.startFen.split(" ")[5] || "1");
}

function setStatus(text) {
  $("status").textContent = text;
}

function gameOverStatus() {
  const game = app.game;
  if (!game.isGameOver()) {
    return null;
  }
  const winner = game.turn() === "w" ? "Black" : "White";
  const text = game.isCheckmate()
    ? `Checkmate: ${winner} wins`
    : game.isStalemate()
      ? "Draw by stalemate"
      : game.isThreefoldRepetition()
        ? "Draw by threefold repetition"
        : game.isInsufficientMaterial()
          ? "Draw: insufficient material"
          : "Draw by the fifty-move rule";
  setStatus(text);
  return text;
}

function fail(what, error) {
  setStatus(`${what}: ${error.message}`);
  console.error(what, error);
}

// --- Controls, model card, self-test and read-only hooks for the smoke test -----------------------------

function fenProblem(text) {
  if (!text) {
    return "Paste a FEN first";
  }
  const check = validateFen(tok.sanitizeFen(text));
  return check.ok ? null : check.error;
}

function onFenSubmit(event) {
  event.preventDefault();
  const text = $("fen-input").value.trim();
  const problem = fenProblem(text);
  $("fen-error").textContent = problem || "";
  if (!problem) {
    startGame(text);
  }
}

async function getJson(url) {
  const response = await fetch(url, { cache: "no-cache" });
  return response.ok ? response.json() : {};
}

async function showRating() {
  const config = await getJson(CONFIG_URL);
  app.rating = await rating.fetchRating(config);
  $("rating").replaceChildren();
  if (app.rating.publishable && app.rating.url) {
    const link = document.createElement("a");
    link.href = app.rating.url;
    link.textContent = app.rating.text;
    $("rating").appendChild(link);
  } else {
    $("rating").textContent = app.rating.text;
  }
  $("rating-note").textContent = app.rating.note ? `(${app.rating.note})` : "";
}

function showModel(card) {
  panel.renderCard($("model-card"), card);
  if (card.label) {
    const label = card.label.charAt(0).toUpperCase() + card.label.slice(1);
    $("mode-label").textContent = `${label}: the legal move with the highest probability, from one forward pass. Like the bot, Blink plays a mate in one without looking, and avoids draws by rule when clearly winning but takes them when clearly losing. Value mode is not run in the browser.`;
  }
}

async function runSelfTest() {
  const game = new Chess();
  const look = await lookOnce(game);
  const legal = game.moves({ verbose: true }).some((move) => move.lan === look.move.uci);
  const histogram = Array.isArray(look.bins) && look.bins.length === app.vocab.numBins;
  const ok = legal && look.calls === 1 && Number.isFinite(look.win) && histogram;
  window.__blinkSelfTest = Object.freeze({ ok, move: look.move.uci, ms: look.ms, histogram, backend: app.backend.backend, source: app.backend.source });
}

function exposeHooks() {
  window.__blink = Object.freeze({
    state: () => ({
      ready: app.ready,
      thinking: app.thinking,
      fen: app.game.fen(),
      startFen: app.startFen,
      turn: app.game.turn(),
      userColor: app.userColor,
      history: app.game.history({ verbose: true }).map((move) => move.lan),
      replies: app.replies,
      arrows: document.querySelectorAll("#arrows .blink-arrow").length,
      lastRule: app.lastLook ? app.lastLook.rule : null,
      timings: [...app.timings],
      gameOver: app.game.isGameOver(),
      histogramBins: Number($("value-hist").dataset.bins || 0),
      backend: app.backend ? { backend: app.backend.backend, threads: app.backend.threads, source: app.backend.source, loadMs: app.backend.loadMs } : null,
      card: app.card.label || null,
      rating: app.rating ? app.rating.text : null,
    }),
    legalMoves: () => app.game.moves({ verbose: true }).map((move) => ({ from: move.from, to: move.to, uci: move.lan })),
  });
}

async function main() {
  app.board = createBoard();
  app.board.enableMoveInput(onMoveInput);
  $("new-game").addEventListener("click", () => startGame(FEN.start));
  $("switch-side").addEventListener("click", () => switchSide());
  $("fen-form").addEventListener("submit", onFenSubmit);
  exposeHooks();
  try {
    app.vocab = tok.createVocab(await (await fetch(VOCAB_URL)).json());
    app.card = await getJson(CARD_URL);
    showModel(app.card);
    showRating().catch((error) => console.warn("the Lichess rating did not load", error));
    app.engine = createEngine();
    app.backend = await app.engine.load(MODEL_URL, app.card.sha256);
    $("backend").textContent = `Backend: ${panel.backendLabel(app.backend, app.card)}`;
    app.ready = true;
    const where = app.backend.source === "cache" ? "from this browser's cache" : "downloaded";
    setStatus(`Model loaded in ${Math.round(app.backend.loadMs)} ms (${where}). Your move`);
    if (new URLSearchParams(location.search).has("selftest")) {
      await runSelfTest();
    }
    await blinkMoves();
  } catch (error) {
    fail("The model did not load", error);
  }
}

main();
