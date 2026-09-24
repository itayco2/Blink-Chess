// NSC-1's rule checks in the browser (site/rules.js) against python-chess (rules.json, written by
// `blink export rules`), plus the one-look decision with a fake network so every call is counted.
// Run from the repo root with `node --test "site/tests/**/*.test.mjs"` (after `npm ci --prefix site`).

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Chess } from "chess.js";
import * as tok from "../tokenizer.js";
import * as rules from "../rules.js";

const HERE = new URL(".", import.meta.url);
const golden = JSON.parse(readFileSync(new URL("rules.json", HERE), "utf-8"));
const vocab = tok.createVocab(JSON.parse(readFileSync(new URL("../vocab.json", HERE), "utf-8")));

function replay(fen, moves) {
  const game = tok.loadPosition(Chess, fen);
  for (const uci of moves) {
    game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
  }
  return game;
}

// A network stand-in: `prefer` gets the top policy logit, and the value bins peak at `win`.
function fakeNetwork(win, prefer = []) {
  const calls = [];
  const evaluate = async (codes) => {
    calls.push(Array.from(codes));
    const policy = new Float32Array(vocab.numMoves);
    prefer.forEach((index, rank) => (policy[index] = 10 - rank));
    const value = new Float32Array(vocab.numBins).fill(-50);
    value[Math.min(vocab.numBins - 1, Math.floor(win * vocab.numBins))] = 50;
    return { policy, value, runMs: 0 };
  };
  return { calls, evaluate };
}

const ranked = (...indices) => indices.map((index) => ({ index, uci: `m${index}` }));

test("rules.js uses the delta and halfmove limit python-chess's cases were written with", () => {
  assert.equal(rules.DRAW_DELTA, golden.draw_delta);
  assert.equal(rules.HALFMOVE_DRAW, golden.halfmove_draw);
});

test("rules.js finds python-chess's mates and rule draws, with their reasons, in every case", () => {
  assert.ok(golden.cases.length >= 10);
  for (const entry of golden.cases) {
    const game = replay(entry.fen, entry.moves);
    const kids = rules.children(vocab, game, Chess);
    assert.equal(kids.length, entry.legal, entry.name);
    const mates = kids.filter((kid) => kid.board.isCheckmate()).map((kid) => [kid.move.index, kid.move.uci]);
    assert.deepEqual(mates, entry.mates, entry.name);
    const draws = rules.ruleDraws(kids, rules.historyCounts(game));
    const found = kids.filter((kid) => draws.has(kid.move.index)).map((kid) => [kid.move.index, kid.move.uci, draws.get(kid.move.index)]);
    assert.deepEqual(found, entry.draws, entry.name);
    const mate = rules.mateNow(kids);
    assert.equal(mate ? mate.index : null, entry.mates.length ? entry.mates[0][0] : null, entry.name);
  }
});

test("a repetition key is placement, side to move, castling rights and a usable en-passant square", () => {
  const afterE4 = replay("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1", ["e2e4"]);
  assert.equal(rules.repetitionKey(afterE4.fen()), "4k3/8/8/8/4P3/8/8/4K3 b - -");
  const capturable = replay("4k3/8/8/8/5p2/8/4P3/4K3 w - - 0 1", ["e2e4"]);
  assert.equal(rules.repetitionKey(capturable.fen()), "4k3/8/8/8/4Pp2/8/8/4K3 b - e3");
});

test("history counts include the current position and every earlier one in the game", () => {
  const game = replay(tok.sanitizeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), ["g1f3", "g8f6", "f3g1", "f6g8"]);
  const counts = rules.historyCounts(game);
  assert.equal(counts.get(rules.repetitionKey(game.fen())), 2);
  assert.equal([...counts.values()].reduce((sum, n) => sum + n, 0), 5);
  assert.equal(rules.historyCounts(new Chess()).get(rules.repetitionKey(new Chess().fen())), 1);
});

test("within delta of an even game, R3 leaves the policy's first choice alone", () => {
  const draws = new Map([[1, "stalemate"]]);
  for (const win of [0.4, 0.5, 0.6]) {
    assert.deepEqual(rules.policyDrawChoice(ranked(1, 2, 3), draws, win), { move: ranked(1)[0], fired: false });
  }
  assert.deepEqual(rules.policyDrawChoice(ranked(1, 2), new Map(), 0.95), { move: ranked(1)[0], fired: false });
});

test("clearly winning, R3 plays the best move that is not a rule draw", () => {
  const draws = new Map([[1, "threefold repetition"], [2, "stalemate"]]);
  assert.deepEqual(rules.policyDrawChoice(ranked(1, 2, 3), draws, 0.61), { move: ranked(3)[0], fired: true });
});

test("clearly winning with only rule draws left, R3 keeps the policy's first choice", () => {
  const draws = new Map([[1, "stalemate"], [2, "stalemate"]]);
  assert.deepEqual(rules.policyDrawChoice(ranked(1, 2), draws, 0.9), { move: ranked(1)[0], fired: true });
});

test("clearly losing, R3 plays the rule draw the policy likes best", () => {
  const draws = new Map([[3, "threefold repetition"], [2, "insufficient material"]]);
  assert.deepEqual(rules.policyDrawChoice(ranked(1, 2, 3), draws, 0.39), { move: ranked(2)[0], fired: true });
});

test("one look plays a mate in one without calling the network (R2)", async () => {
  const net = fakeNetwork(0.5);
  const game = replay("7k/5P2/6K1/8/8/8/8/8 w - - 0 1", []);
  const look = await rules.decide(vocab, game, Chess, net.evaluate, 3);
  assert.equal(net.calls.length, 0);
  assert.equal(look.rule, "R2");
  assert.equal(look.calls, 0);
  assert.equal(look.move.uci, "f7f8q");
  assert.deepEqual(look.top, []);
});

test("one look makes exactly one network call, on the root position's codes", async () => {
  const game = new Chess();
  const legal = tok.legalMoves(vocab, game);
  const net = fakeNetwork(0.5, [legal[5].index]);
  const look = await rules.decide(vocab, game, Chess, net.evaluate, 3);
  assert.equal(net.calls.length, 1);
  assert.deepEqual(net.calls[0], Array.from(tok.encodeBoard(vocab, game)));
  assert.equal(look.calls, 1);
  assert.equal(look.rule, null);
  assert.equal(look.move.index, legal[5].index);
  assert.equal(look.top.length, 3);
  assert.equal(look.top[0].index, legal[5].index);
});

test("winning, one look steps around the threefold repetition the policy wanted (R3)", async () => {
  const game = replay(tok.sanitizeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]);
  const legal = tok.legalMoves(vocab, game);
  const repeat = legal.find((move) => move.uci === "f6g8");
  const other = legal.find((move) => move.uci === "b8c6");
  const net = fakeNetwork(0.9, [repeat.index, other.index]);
  const look = await rules.decide(vocab, game, Chess, net.evaluate, 3);
  assert.equal(look.rule, "R3");
  assert.equal(look.move.uci, "b8c6");
  assert.equal(look.top[0].uci, "f6g8", "the arrows still show what the network preferred");
  assert.equal(net.calls.length, 1);
});

test("losing, one look takes the threefold repetition (R3)", async () => {
  const game = replay(tok.sanitizeFen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"), ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1"]);
  const legal = tok.legalMoves(vocab, game);
  const other = legal.find((move) => move.uci === "b8c6");
  const net = fakeNetwork(0.1, [other.index]);
  const look = await rules.decide(vocab, game, Chess, net.evaluate, 3);
  assert.equal(look.rule, "R3");
  assert.equal(look.move.uci, "f6g8");
  assert.equal(look.draw, "threefold repetition");
});
