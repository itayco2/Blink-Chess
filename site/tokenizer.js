// A JS port of blink/board/encode.py and blink/board/moves.py for the browser (and Node parity tests).
// The board is 64 square codes seen from the side to move; moves are indices into the 1880-move
// vocabulary read from vocab.json (written by `blink export vocab`), so the page and Python share it.
// Squares are 0..63 = file + 8 * rank (a1 = 0); Black's view flips the rank (sq ^ 56).

const FILES = "abcdefgh";

export function squareIndex(name) {
  return FILES.indexOf(name[0]) + 8 * (Number(name[1]) - 1);
}

export function squareName(index) {
  return FILES[index % 8] + String(Math.floor(index / 8) + 1);
}

export function frame(square, turn) {
  return turn === "w" ? square : square ^ 56;
}

export function createVocab(json) {
  const ftIndex = new Int16Array(64 * 64).fill(-1);
  json.from_to.forEach(([from, to], index) => (ftIndex[from * 64 + to] = index));
  const promoIndex = new Map(json.promo_pairs.map(([from, to], index) => [from * 64 + to, index]));
  return Object.freeze({
    numMoves: json.num_moves,
    numFromTo: json.num_from_to,
    numBins: json.num_bins,
    fromTo: json.from_to,
    promoPairs: json.promo_pairs,
    promoPieces: json.promo_pieces,
    pieceOrder: json.piece_order,
    codes: json.codes,
    ftIndex,
    promoIndex,
  });
}

// --- FEN hygiene: chess.js keeps whatever a FEN claims; python-chess cleans it -------------------

function placement(fen) {
  const board = new Map();
  fen.split(" ")[0].split("/").forEach((row, r) => {
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) {
        file += Number(ch);
      } else {
        board.set(FILES[file] + String(8 - r), ch);
        file += 1;
      }
    }
  });
  return board;
}

const CASTLING_NEEDS = { K: ["e1", "K", "h1", "R"], Q: ["e1", "K", "a1", "R"], k: ["e8", "k", "h8", "r"], q: ["e8", "k", "a8", "r"] };

function cleanCastling(board, field) {
  const kept = [...field].filter((right) => {
    const need = CASTLING_NEEDS[right];
    return need && board.get(need[0]) === need[1] && board.get(need[2]) === need[3];
  });
  return kept.length ? kept.join("") : "-";
}

function cleanEnPassant(board, turn, field) {
  if (field === "-" || board.has(field)) {
    return "-";
  }
  const behind = squareName(squareIndex(field) + (turn === "w" ? -8 : 8));
  return board.get(behind) === (turn === "w" ? "p" : "P") ? field : "-";
}

// Drop castling rights whose king or rook is not home, and an en-passant square with no double-pushed
// pawn in front of it, exactly as python-chess's clean_castling_rights and ep generation do.
export function sanitizeFen(fen) {
  const fields = fen.trim().split(/\s+/);
  while (fields.length < 6) {
    fields.push(["w", "-", "-", "0", "1"][fields.length - 1]);
  }
  const board = placement(fen);
  fields[2] = cleanCastling(board, fields[2]);
  fields[3] = cleanEnPassant(board, fields[1], fields[3]);
  return fields.join(" ");
}

export function loadPosition(ChessClass, fen) {
  return new ChessClass(sanitizeFen(fen));
}

// --- Board -> 64 codes ---------------------------------------------------------------------------

const CASTLE_ROOKS = [
  ["w", "k", "e1", "h1"],
  ["w", "q", "e1", "a1"],
  ["b", "k", "e8", "h8"],
  ["b", "q", "e8", "a8"],
];

export function encodeBoard(vocab, chess) {
  const { codes, pieceOrder } = vocab;
  const turn = chess.turn();
  const out = new Uint8Array(64);
  for (const row of chess.board()) {
    for (const piece of row) {
      if (piece) {
        const base = piece.color === turn ? codes.own : codes.opp;
        out[frame(squareIndex(piece.square), turn)] = base + pieceOrder.indexOf(piece.type);
      }
    }
  }
  for (const [color, side, kingSquare, rookSquare] of CASTLE_ROOKS) {
    const king = chess.get(kingSquare);
    const rook = chess.get(rookSquare);
    const homed = king && king.type === "k" && king.color === color && rook && rook.type === "r" && rook.color === color;
    if (homed && chess.getCastlingRights(color)[side]) {
      out[frame(squareIndex(rookSquare), turn)] = color === turn ? codes.own_castling_rook : codes.opp_castling_rook;
    }
  }
  const ep = chess.moves({ verbose: true }).find((move) => move.flags.includes("e"));
  if (ep) {
    out[frame(squareIndex(ep.to), turn)] = codes.ep_square;
  }
  return out;
}

// --- Moves <-> vocabulary indices ----------------------------------------------------------------

export function encodeMove(vocab, turn, move) {
  const from = frame(squareIndex(move.from), turn);
  const to = frame(squareIndex(move.to), turn);
  if (move.promotion) {
    const pair = vocab.promoIndex.get(from * 64 + to);
    if (pair === undefined) {
      throw new Error(`${move.from}${move.to} is not a promotion from the 7th to the 8th rank`);
    }
    return vocab.numFromTo + pair * vocab.promoPieces.length + vocab.promoPieces.indexOf(move.promotion);
  }
  const index = vocab.ftIndex[from * 64 + to];
  if (index < 0) {
    throw new Error(`${move.from}${move.to} is not a queen line or a knight jump`);
  }
  return index;
}

export function decodeMove(vocab, turn, index) {
  if (!(index >= 0 && index < vocab.numMoves)) {
    throw new Error(`move index ${index} outside 0..${vocab.numMoves - 1}`);
  }
  let pair;
  let promotion;
  if (index >= vocab.numFromTo) {
    const offset = index - vocab.numFromTo;
    pair = vocab.promoPairs[Math.floor(offset / vocab.promoPieces.length)];
    promotion = vocab.promoPieces[offset % vocab.promoPieces.length];
  } else {
    pair = vocab.fromTo[index];
  }
  const from = squareName(frame(pair[0], turn));
  const to = squareName(frame(pair[1], turn));
  return { from, to, promotion, uci: from + to + (promotion || "") };
}

// Every legal move with its vocabulary index and the child's FEN, sorted by index (castling is the
// king's two-square move).
export function legalMoves(vocab, chess) {
  const turn = chess.turn();
  return chess
    .moves({ verbose: true })
    .map((move) => ({
      index: encodeMove(vocab, turn, move),
      uci: move.lan,
      san: move.san,
      from: move.from,
      to: move.to,
      promotion: move.promotion,
      after: move.after,
    }))
    .sort((a, b) => a.index - b.index);
}

// --- Network outputs -> what the page shows --------------------------------------------------------

// One look: a softmax over the legal moves' logits only, best first (ties: lower index first).
export function policyTopK(legal, policyLogits, k) {
  const logits = legal.map((move) => policyLogits[move.index]);
  const max = Math.max(...logits);
  const exps = logits.map((logit) => Math.exp(logit - max));
  const total = exps.reduce((sum, x) => sum + x, 0);
  return legal
    .map((move, i) => ({ ...move, prob: exps[i] / total }))
    .sort((a, b) => b.prob - a.prob || a.index - b.index)
    .slice(0, k);
}

// Expected win probability for the side to move: softmax over the value bins times the bin centres.
export function winProbability(valueLogits) {
  const n = valueLogits.length;
  let max = -Infinity;
  for (const logit of valueLogits) {
    max = Math.max(max, logit);
  }
  let total = 0;
  let weighted = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.exp(valueLogits[i] - max);
    total += p;
    weighted += p * ((i + 0.5) / n);
  }
  return weighted / total;
}
