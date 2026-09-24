// NSC-1's rule checks for one look (policy mode), ported from blink/play/rules.py and agents.py so the
// page plays the moves the bot would. Nothing here calls the network except decide(), exactly once.
// R1: only legal moves, the policy softmax masked to them (tokenizer.js).
// R2: a checkmating child is played before any network call (the lowest vocabulary index if several).
// R3: rule-draw children (stalemate, insufficient material, the fifty-move rule, a third occurrence)
//     are demoted when the root value says Blink is clearly winning, and the one the policy likes best
//     is played when Blink is clearly losing; "clearly" is |win - 0.5| > 0.10, pre-registered.
// R4 (value-mode tie-break) and R5 (clock guard) belong to value mode and a clock; the page has neither.
// site/tests/rules.test.mjs checks the children against python-chess (rules.json).

import * as tok from "./tokenizer.js";

export const DRAW_DELTA = 0.1;
export const HALFMOVE_DRAW = 100;

// Placement, side to move, castling rights and the en-passant square: python-chess's repetition key.
// chess.js writes the ep square into a FEN only when an ep capture is legal, and castling rights stay
// clean (sanitizeFen on load, chess.js on every move), so the first four FEN fields are that key.
export function repetitionKey(fen) {
  return fen.split(" ").slice(0, 4).join(" ");
}

// How often each position occurred in this game, the current one included. Counting every position
// since the start equals Python's walk back to the last capture or pawn move: nothing before an
// irreversible move can occur again.
export function historyCounts(game) {
  const played = game.history({ verbose: true });
  const fens = played.length ? [played[0].before, ...played.map((move) => move.after)] : [game.fen()];
  const counts = new Map();
  for (const fen of fens) {
    const key = repetitionKey(fen);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return counts;
}

// R1: every legal move with its child position, in vocabulary order.
export function children(vocab, game, ChessClass) {
  return tok.legalMoves(vocab, game).map((move) => ({ move, board: new ChessClass(move.after) }));
}

// R2: the checkmating move with the lowest vocabulary index, or null.
export function mateNow(kids) {
  const mate = kids.find((kid) => kid.board.isCheckmate());
  return mate ? mate.move : null;
}

// R3: why the child is a draw by rule, or null. A mate is never a draw.
export function ruleDraw(child, counts) {
  if (child.isCheckmate()) {
    return null;
  }
  if (child.isStalemate()) {
    return "stalemate";
  }
  if (child.isInsufficientMaterial()) {
    return "insufficient material";
  }
  if (Number(child.fen().split(" ")[4]) >= HALFMOVE_DRAW) {
    return "fifty-move rule";
  }
  return (counts.get(repetitionKey(child.fen())) || 0) + 1 >= 3 ? "threefold repetition" : null;
}

// Vocabulary index -> reason, for every rule-draw child.
export function ruleDraws(kids, counts) {
  const draws = new Map();
  for (const kid of kids) {
    const reason = ruleDraw(kid.board, counts);
    if (reason) {
      draws.set(kid.move.index, reason);
    }
  }
  return draws;
}

// R3 in policy mode. `ranked` is the legal moves by descending policy; returns { move, fired }.
export function policyDrawChoice(ranked, draws, rootWin) {
  if (draws.size === 0 || Math.abs(rootWin - 0.5) <= DRAW_DELTA) {
    return { move: ranked[0], fired: false };
  }
  if (rootWin > 0.5) {
    return { move: ranked.find((move) => !draws.has(move.index)) || ranked[0], fired: true };
  }
  return { move: ranked.find((move) => draws.has(move.index)), fired: true };
}

// One look with R1-R3. `evaluate(codes)` is the network; it is called once, on the root, or never (R2).
// `top` is the network's k best legal moves (the arrows); `move` is what Blink plays.
export async function decide(vocab, game, ChessClass, evaluate, k) {
  const turn = game.turn();
  const kids = children(vocab, game, ChessClass);
  const mate = mateNow(kids);
  if (mate) {
    return Object.freeze({ turn, move: mate, rule: "R2", draw: null, calls: 0, top: [], win: 1, ms: 0, legalCount: kids.length });
  }
  const started = performance.now();
  const out = await evaluate(tok.encodeBoard(vocab, game));
  const ms = performance.now() - started;
  const ranked = tok.policyTopK(kids.map((kid) => kid.move), out.policy, kids.length);
  const win = tok.winProbability(out.value);
  const draws = ruleDraws(kids, historyCounts(game));
  const { move, fired } = policyDrawChoice(ranked, draws, win);
  return Object.freeze({
    turn,
    move,
    rule: fired ? "R3" : null,
    draw: fired ? draws.get(move.index) || null : null,
    avoided: fired && !draws.has(move.index) ? draws.size : 0,
    calls: 1,
    top: ranked.slice(0, k),
    win,
    ms,
    runMs: out.runMs,
    legalCount: kids.length,
  });
}
