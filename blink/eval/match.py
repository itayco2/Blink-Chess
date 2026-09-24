"""In-process matches between two agents, with python-chess as the arbiter.

Openings are played in book order, each once per colour (games 2k and 2k+1 share opening k).
A game ends by checkmate, or is drawn by rule (stalemate, insufficient material, halfmove clock 100,
a third repetition), or is adjudicated a draw after 600 engine plies (book plies excluded; this
mirrors fastchess `-maxmoves 300` and Lichess's forced draw). There is no resignation and no win
adjudication. An illegal move or a crash loses the game; a NoSearchViolation stops the match.

Every engine move carries a fastchess-style comment `{<score>/1 <seconds>s, n=<rows>}`, so
`blink audit no-search` reads these PGNs exactly as it reads fastchess's.

`EngineAgent` puts a UCI engine (Stockfish 19) into the same loop, with python-chess sending the whole
game (start position plus moves) and `go movetime` or `go nodes`. Blink needs no clock in process
because its compute never depends on time (N4), so an in-process block plays exactly the moves a
fastchess block would. `pair_player` feeds the SPRT one opening (both colours) at a time, and
`audit_players` audits every searchless player of a PGN by its own name (Blink-*, DM-9M*, PF60).
"""

import datetime
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from blink.eval import nosearch
from blink.eval.books import Opening
from blink.play.agents import Agent, Decision
from blink.play.budget import NoSearchViolation
from blink.uci import win_to_cp

MAX_ENGINE_PLIES = 600
HALFMOVE_DRAW = 100
NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
SF_MOVETIME = 0.1  # Stockfish st=0.1, as in fastchess (plan match rules)
DRAW = "1/2-1/2"


@dataclass(frozen=True)
class GameRecord:
    white: str
    black: str
    result: str
    termination: str  # the PGN Termination tag: normal | adjudication | illegal move | abandoned
    reason: str  # checkmate, stalemate, threefold repetition, ..., or what went wrong
    engine_plies: int
    opening: int
    illegal_by: str | None = None
    crashed_by: str | None = None


def rule_outcome(board: chess.Board) -> tuple[str, str] | None:
    """(result, reason) when the game is over by rule, else None. The arbiter, never an agent."""
    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), "checkmate"
    if board.is_stalemate():
        return DRAW, "stalemate"
    if board.is_insufficient_material():
        return DRAW, "insufficient material"
    if board.halfmove_clock >= HALFMOVE_DRAW:
        return DRAW, "fifty-move rule"
    if board.is_repetition(3):
        return DRAW, "threefold repetition"
    return None


def move_comment(decision: Decision, seconds: float) -> str:
    if decision.mate_now:
        score = "+M1"
    elif decision.win is not None:
        score = f"{win_to_cp(decision.win) / 100:+.2f}"
    else:
        score = "0.00"
    return f"{score}/1 {seconds:.3f}s, n={decision.n_rows}"


def _loss_for(color: chess.Color) -> str:
    return "0-1" if color == chess.WHITE else "1-0"


def _start(opening: Opening) -> tuple[chess.Board, chess.pgn.Game, chess.pgn.GameNode]:
    board = chess.Board(opening.fen)
    game = chess.pgn.Game()
    if opening.fen != chess.STARTING_FEN:
        game.setup(board)
    node: chess.pgn.GameNode = game
    for uci in opening.moves:
        move = chess.Move.from_uci(uci)
        node = node.add_variation(move, comment="book")
        board.push(move)
    return board, game, node


def play_game(
    white: Agent, black: Agent, opening: Opening, game_id: str, max_plies: int = MAX_ENGINE_PLIES
) -> tuple[chess.pgn.Game, GameRecord]:
    board, game, node = _start(opening)
    plies, extra = 0, {}
    while True:
        over = rule_outcome(board)
        if over is not None:
            result, termination, reason = over[0], "normal", over[1]
            break
        if plies >= max_plies:
            result, termination, reason = DRAW, "adjudication", f"{max_plies} engine plies"
            break
        player = white if board.turn == chess.WHITE else black
        started = time.perf_counter()
        try:
            decision = player.choose(board, game=game_id)
        except NoSearchViolation:
            raise
        except Exception as exc:  # a crash loses this game; the match goes on and counts it
            result, termination, reason = _loss_for(board.turn), "abandoned", f"crash: {exc!r}"
            extra = {"crashed_by": player.name}
            break
        if decision.move not in board.legal_moves:
            result, termination = _loss_for(board.turn), "illegal move"
            reason, extra = f"illegal move {decision.move.uci()}", {"illegal_by": player.name}
            break
        node = node.add_variation(
            decision.move, comment=move_comment(decision, time.perf_counter() - started)
        )
        board.push(decision.move)
        plies += 1
    record = GameRecord(white.name, black.name, result, termination, reason, plies, opening.number, **extra)
    _set_headers(game, record, len(opening.moves) + plies)
    return game, record


def _set_headers(game: chess.pgn.Game, record: GameRecord, ply_count: int) -> None:
    game.headers["Event"] = "Blink match"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["White"] = record.white
    game.headers["Black"] = record.black
    game.headers["Result"] = record.result
    game.headers["Termination"] = record.termination
    game.headers["EndReason"] = record.reason
    game.headers["PlyCount"] = str(ply_count)
    game.headers["BookIndex"] = str(record.opening)


def _score_for_a(record: GameRecord, a_is_white: bool) -> float:
    if record.result == DRAW:
        return 0.5
    white_won = record.result == "1-0"
    return 1.0 if white_won == a_is_white else 0.0


def summarize(records: Sequence[tuple[GameRecord, bool]], a_name: str, b_name: str) -> dict:
    scores = [_score_for_a(record, a_white) for record, a_white in records]
    return {
        "a": a_name,
        "b": b_name,
        "games": len(records),
        "a_wins": scores.count(1.0),
        "draws": scores.count(0.5),
        "a_losses": scores.count(0.0),
        "a_score": sum(scores) / len(scores) if scores else 0.0,
        "illegal_moves": sum(record.illegal_by is not None for record, _ in records),
        "crashes": sum(record.crashed_by is not None for record, _ in records),
        "adjudications": sum(record.termination == "adjudication" for record, _ in records),
        "reasons": dict(Counter(record.reason for record, _ in records)),
        "penta": _penta(records),
        "records": [asdict(record) for record, _ in records],
    }


def _penta(records: Sequence[tuple[GameRecord, bool]]) -> list[int] | None:
    """A's pentanomial over consecutive game pairs, or None for an odd number of games."""
    if not records or len(records) % 2:
        return None
    counts = [0] * 5
    for (first, white1), (second, white2) in zip(records[::2], records[1::2], strict=True):
        counts[round(2 * (_score_for_a(first, white1) + _score_for_a(second, white2)))] += 1
    return counts


def run_match(
    a: Agent,
    b: Agent,
    openings: Sequence[Opening],
    games: int,
    pgn_path: Path,
    max_plies: int = MAX_ENGINE_PLIES,
    event: str = "Blink match",
) -> dict:
    """Play `games` games, A as White in even games; append each game to pgn_path as it finishes."""
    if games > 2 * len(openings):
        raise ValueError(f"{games} games need {(games + 1) // 2} openings, got {len(openings)}")
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for index in range(games):
        a_white = index % 2 == 0
        seat = Seat(a, b, a_white, openings[index // 2], index + 1)
        records.append((_play_and_append(seat, pgn_path, max_plies, event), a_white))
    return summarize(records, a.name, b.name)


@dataclass(frozen=True)
class Seat:
    """One game to play: who is A, who is B, A's colour, the opening and the game number."""

    a: Agent
    b: Agent
    a_white: bool
    opening: Opening
    number: int


def _play_and_append(seat: Seat, pgn: Path, max_plies: int, event: str) -> GameRecord:
    white, black = (seat.a, seat.b) if seat.a_white else (seat.b, seat.a)
    game, record = play_game(white, black, seat.opening, f"g{seat.number}", max_plies)
    game.headers["Round"] = str(seat.number)
    game.headers["Event"] = event
    with open(pgn, "a", encoding="utf-8") as handle:
        print(game, file=handle, end="\n\n")
    return record


def pair_player(
    a: Agent,
    b: Agent,
    openings: Sequence[Opening],
    pgn_path: Path,
    max_plies: int = MAX_ENGINE_PLIES,
    event: str = "Blink SPRT",
) -> Callable[[int], tuple[float, float]]:
    """index -> (A's score as White, A's score as Black) on opening `index`: one SPRT game pair."""
    pgn_path.parent.mkdir(parents=True, exist_ok=True)

    def play(index: int) -> tuple[float, float]:
        if index >= len(openings):
            raise ValueError(f"pair {index + 1} needs more than the {len(openings)} openings supplied")
        scores = []
        for a_white in (True, False):
            seat = Seat(a, b, a_white, openings[index], 2 * index + (1 if a_white else 2))
            scores.append(_score_for_a(_play_and_append(seat, pgn_path, max_plies, event), a_white))
        return scores[0], scores[1]

    return play


def audit_players(pgns: Sequence[Path], players: Sequence[str]) -> dict[str, dict]:
    """The no-search audit of each searchless player by its own name (Blink-*, DM-9M*): one report each."""
    files = [Path(p) for p in pgns if Path(p).is_file()]
    return {name: nosearch.audit(files, engine=name) for name in players}


class EngineAgent:
    """A UCI engine (Stockfish 19) as an agent: the whole game goes to it and it answers with one move.

    `movetime` (seconds) or `nodes` bounds each move; `options` are UCI options (Threads, Hash,
    UCI_LimitStrength, UCI_Elo). The process starts on first use and stops on close()."""

    def __init__(
        self,
        name: str,
        command: str | Path,
        movetime: float | None = None,
        nodes: int | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        if (movetime is None) == (nodes is None):
            raise ValueError("an engine agent needs exactly one of movetime and nodes")
        self.name = name
        self.command = str(command)
        self.limit = chess.engine.Limit(time=movetime) if nodes is None else chess.engine.Limit(nodes=nodes)
        self.options = dict(options or {})
        self._engine: chess.engine.SimpleEngine | None = None

    def _process(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.command)
            self._engine.configure(self.options)
        return self._engine

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        result = self._process().play(board, self.limit, game=game or None)
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move at {board.fen()}")
        return Decision(result.move)

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> "EngineAgent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def stockfish_agent(
    exe: Path, elo: int | None = None, movetime: float = SF_MOVETIME, nodes: int | None = None
) -> EngineAgent:
    """SF19 in process: a UCI_Elo anchor SF<elo> at `movetime`, or full strength at a node budget."""
    options: dict[str, object] = {"Threads": 1, "Hash": 16}
    if nodes is not None:
        return EngineAgent(f"SF19-n{nodes}", exe, nodes=nodes, options=options)
    if elo is not None:
        options |= {"UCI_LimitStrength": True, "UCI_Elo": elo}
        return EngineAgent(f"SF{elo}", exe, movetime=movetime, options=options)
    return EngineAgent("SF19", exe, movetime=movetime, options=options)


def match_report(summary: dict, pgn: Path) -> dict:
    """An in-process summary in the shape fastchess.match_report gives: A's W/D/L, score, pentanomial."""
    return {
        "a": summary["a"],
        "b": summary["b"],
        "games": summary["games"],
        "wins": summary["a_wins"],
        "draws": summary["draws"],
        "losses": summary["a_losses"],
        "score": summary["a_score"] if summary["games"] else None,
        "penta": summary["penta"],
        "pgn": str(pgn),
        "illegal_moves": summary["illegal_moves"],
        "crashes": summary["crashes"],
        "adjudications": summary["adjudications"],
    }


def merge_reports(first: dict, second: dict) -> dict:
    """Two reports of the same pairing (more games at a bracketing rung) as one."""
    if (first["a"], first["b"]) != (second["a"], second["b"]):
        raise ValueError(f"cannot merge {first['a']} vs {first['b']} with {second['a']} vs {second['b']}")
    games = first["games"] + second["games"]
    wins, draws = first["wins"] + second["wins"], first["draws"] + second["draws"]
    penta = None
    if first.get("penta") and second.get("penta"):
        penta = [x + y for x, y in zip(first["penta"], second["penta"], strict=True)]
    pgns = [p for r in (first, second) for p in (r.get("pgns") or [r["pgn"]])]
    return {
        **first,
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": first["losses"] + second["losses"],
        "score": (wins + draws / 2) / games if games else None,
        "penta": penta,
        "pgn": pgns[-1],
        "pgns": pgns,
    }


# ------------------------------------------------------------------------------ in-process blocks


def read_epsilon(results_dir: Path = Path("results")) -> float:
    """The R4 tie window chosen in E2b (results/epsilon.json), or the play default before E2b has run."""
    from blink.play.rules import DEFAULT_EPSILON

    path = Path(results_dir) / "epsilon.json"
    if not path.is_file():
        return DEFAULT_EPSILON
    return float(json.loads(path.read_text(encoding="utf-8"))["epsilon"])


def blink_agents(selector: str, device: str, epsilon: float | None = None) -> dict[str, Agent]:
    """Both modes of one model on one evaluator (one load, one CUDA context), named as fastchess does."""
    from blink.eval.fastchess import engine_name
    from blink.play import factory

    evaluator = factory.load_evaluator(selector, device=device)
    eps = read_epsilon() if epsilon is None else epsilon
    return {
        mode: replace(factory.make_agent(mode, evaluator, epsilon=eps), name=engine_name(selector, mode))
        for mode in factory.MODES
    }


def unique_path(path: Path) -> Path:
    """`path`, or path-2, path-3, ... when it exists: two matches never append to one PGN."""
    candidate, number = Path(path), 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{number}{path.suffix}")
        number += 1
    return candidate


def play_inprocess(
    a: Agent,
    b: Agent,
    games: int,
    book: str,
    out_dir: Path,
    skip: int = 0,
    max_plies: int = MAX_ENGINE_PLIES,
) -> dict:
    """`games` games of A against B from a book slice (after `skip` openings): PGN, summary and report."""
    from blink.eval.books import openings_for

    openings = openings_for(book, (games + 1) // 2, skip)
    tag = f"{NAME_SAFE.sub('_', a.name)}_vs_{NAME_SAFE.sub('_', b.name)}"
    pgn = unique_path(Path(out_dir) / f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}_{skip}.pgn")
    summary = run_match(a, b, openings, games, pgn, max_plies=max_plies)
    pgn.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return match_report(summary, pgn)
