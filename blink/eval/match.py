"""In-process matches between two agents, with python-chess as the arbiter.

Openings are played in book order, each once per colour (games 2k and 2k+1 share opening k).
A game ends by checkmate, or is drawn by rule (stalemate, insufficient material, halfmove clock 100,
a third repetition), or is adjudicated a draw after 600 engine plies (book plies excluded; this
mirrors fastchess `-maxmoves 300` and Lichess's forced draw). There is no resignation and no win
adjudication. An illegal move or a crash loses the game; a NoSearchViolation stops the match.

The clocks are fastchess's (EVAL.md section 3): Blink and DM-9M `st=1 timemargin=500`, so a move over
1.5 s loses on time; Stockfish `st=0.1 timemargin=100` (over 0.2 s loses), or a tc it is told its time
left under; Stockfish at a node count and the baselines have no clock. The time is measured around the
move choice, and every clocked player first makes one untimed warm-up move, as fastchess's isready lets
an engine start before its first `go`.

Every engine move carries a fastchess-style comment `{<score>/1 <seconds>s, n=<rows>}`, so
`blink audit no-search` reads these PGNs exactly as it reads fastchess's.

`EngineAgent` puts a UCI engine (Stockfish 19) into the same loop, with python-chess sending the whole
game (start position plus moves) and `go movetime`, `go nodes` or `go wtime btime winc binc`. Blink's
compute never depends on time (N4), so an in-process block plays exactly the moves a fastchess block
would. `pair_player` feeds the SPRT one opening (both colours) at a time, and `audit_players` audits
every searchless player of a PGN by its exact name (Blink-*, DM-9M*, PF60).
"""

import datetime
import json
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path

import chess
import chess.engine
import chess.pgn

from blink.eval import nosearch
from blink.eval.books import Opening
from blink.eval.fastchess import BLINK_MARGIN_MS, BLINK_ST, SF_MARGIN_MS
from blink.play.agents import Agent, Decision
from blink.play.budget import NoSearchViolation
from blink.uci import win_to_cp

MAX_ENGINE_PLIES = 600
HALFMOVE_DRAW = 100
NAME_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
SF_MOVETIME = 0.1  # Stockfish st=0.1, as in fastchess (plan match rules)
SF_MARGIN_S = SF_MARGIN_MS / 1000
DRAW = "1/2-1/2"
WARM_UP_GAME = "warm-up"
move_timer = time.perf_counter  # what times each move (a test swaps in a fake clock)


@dataclass(frozen=True)
class Clock:
    """One player's time control, kept as fastchess keeps it.

    `move_s` is a fixed time per move (fastchess st): a move slower than move_s + margin_s loses on time.
    Otherwise the player starts with `base_s`, gains `inc_s` after each move (fastchess tc), and loses on
    time when a move leaves less than -margin_s on its clock."""

    move_s: float | None = None
    base_s: float = 0.0
    inc_s: float = 0.0
    margin_s: float = 0.0

    @classmethod
    def from_tc(cls, tc: str, margin_s: float) -> "Clock":
        base, plus, inc = tc.partition("+")
        try:
            return cls(None, float(base), float(inc) if plus else 0.0, margin_s)
        except ValueError as exc:
            raise ValueError(f"a time control here is <base>+<increment> in seconds, got {tc!r}") from exc

    def start(self) -> float | None:
        """The time on the clock before the first move; None under a fixed time per move."""
        return None if self.move_s is not None else self.base_s

    def _left(self, left: float | None) -> float:
        return self.base_s if left is None else left

    def overran(self, elapsed: float, left: float | None) -> bool:
        if self.move_s is not None:
            return elapsed > self.move_s + self.margin_s
        return self._left(left) - elapsed < -self.margin_s

    def after(self, elapsed: float, left: float | None) -> float | None:
        """The time left after a move (None under a fixed time per move)."""
        return None if self.move_s is not None else self._left(left) - elapsed + self.inc_s

    def describe(self) -> str:
        margin = f"timemargin={round(self.margin_s * 1000)}"
        if self.move_s is not None:
            return f"st={self.move_s:g} {margin}"
        return f"tc={self.base_s:g}+{self.inc_s:g} {margin}"


BLINK_CLOCK = Clock(move_s=BLINK_ST, margin_s=BLINK_MARGIN_MS / 1000)


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
    time_forfeit_by: str | None = None


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


def _timed_choice(
    player: Agent, board: chess.Board, left: float | None, game_id: str
) -> tuple[Decision | None, float, Exception | None]:
    """(decision, seconds it took, None), or (None, seconds, the exception) when the player crashed."""
    started = move_timer()
    try:
        decision = player.choose(board, remaining_s=left, game=game_id)
    except NoSearchViolation:
        raise
    except Exception as exc:  # a crash loses this game; the match goes on and counts it
        return None, move_timer() - started, exc
    return decision, move_timer() - started, None


def _move_ends_game(
    player: Agent,
    clock: Clock | None,
    board: chess.Board,
    decision: Decision,
    took: float,
    left: float | None,
) -> tuple[str, str, dict] | None:
    """(termination, reason, record fields) when this move loses the game on time or by being illegal."""
    if clock is not None and clock.overran(took, left):
        reason = f"{took:.3f} s on one move ({clock.describe()})"
        return "time forfeit", reason, {"time_forfeit_by": player.name}
    if decision.move not in board.legal_moves:
        return "illegal move", f"illegal move {decision.move.uci()}", {"illegal_by": player.name}
    return None


def play_game(
    white: Agent, black: Agent, opening: Opening, game_id: str, max_plies: int = MAX_ENGINE_PLIES
) -> tuple[chess.pgn.Game, GameRecord]:
    board, game, node = _start(opening)
    clocks = {chess.WHITE: clock_for(white), chess.BLACK: clock_for(black)}
    left = {color: clock.start() if clock else None for color, clock in clocks.items()}
    plies, extra = 0, {}
    while True:
        over = rule_outcome(board)
        if over is not None:
            result, termination, reason = over[0], "normal", over[1]
            break
        if plies >= max_plies:
            result, termination, reason = DRAW, "adjudication", f"{max_plies} engine plies"
            break
        color = board.turn
        player, clock = (white if color == chess.WHITE else black), clocks[color]
        decision, took, crash = _timed_choice(player, board, left[color], game_id)
        if decision is None:
            result, termination, reason = _loss_for(color), "abandoned", f"crash: {crash!r}"
            extra = {"crashed_by": player.name}
            break
        ended = _move_ends_game(player, clock, board, decision, took, left[color])
        if ended is not None:
            (termination, reason, extra), result = ended, _loss_for(color)
            break
        left[color] = clock.after(took, left[color]) if clock else None
        node = node.add_variation(decision.move, comment=move_comment(decision, took))
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
        "time_forfeits": sum(record.time_forfeit_by is not None for record, _ in records),
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
    warm_up(a, b)
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
    warm_up(a, b)

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
    """The no-search audit of each searchless player by its exact name (Blink-*, DM-9M*): one report each."""
    files = [Path(p) for p in pgns if Path(p).is_file()]
    return nosearch.audit_each(files, players)


class EngineAgent:
    """A UCI engine (Stockfish 19) as an agent: the whole game goes to it and it answers with one move.

    `movetime` (seconds), `nodes` or a `tc` such as "60+0.6" bounds each move; `options` are UCI options
    (Threads, Hash, UCI_LimitStrength, UCI_Elo). Under movetime or tc the engine is on fastchess's clock
    (`clock`, timemargin 100 ms); under a tc it is told its time left each move. The process starts on
    first use (or warm_up) and stops on close()."""

    def __init__(
        self,
        name: str,
        command: str | Path,
        movetime: float | None = None,
        nodes: int | None = None,
        options: dict[str, object] | None = None,
        tc: str | None = None,
    ) -> None:
        if sum(bound is not None for bound in (movetime, nodes, tc)) != 1:
            raise ValueError("an engine agent needs exactly one of movetime, nodes and tc")
        self.name = name
        self.command = str(command)
        self.limit: chess.engine.Limit | None = None
        self.clock: Clock | None = None
        if movetime is not None:
            self.limit, self.clock = chess.engine.Limit(time=movetime), Clock(movetime, margin_s=SF_MARGIN_S)
        elif nodes is not None:
            self.limit = chess.engine.Limit(nodes=nodes)
        else:
            self.clock = Clock.from_tc(tc, SF_MARGIN_S)
        self.options = dict(options or {})
        self._engine: chess.engine.SimpleEngine | None = None

    def _process(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.command)
            self._engine.configure(self.options)
        return self._engine

    def _limit_for(self, remaining_s: float | None) -> chess.engine.Limit:
        if self.limit is not None:
            return self.limit
        left = max(0.0, self.clock.base_s if remaining_s is None else remaining_s)
        inc = self.clock.inc_s
        return chess.engine.Limit(white_clock=left, black_clock=left, white_inc=inc, black_inc=inc)

    def choose(self, board: chess.Board, remaining_s: float | None = None, game: str = "") -> Decision:
        result = self._process().play(board, self._limit_for(remaining_s), game=game or None)
        if result.move is None:
            raise RuntimeError(f"{self.name} returned no move at {board.fen()}")
        return Decision(result.move)

    def warm_up(self) -> None:
        """Start the process and wait for readyok before the first timed move."""
        self._process().ping()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> "EngineAgent":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def stockfish_agent(
    exe: Path,
    elo: int | None = None,
    movetime: float = SF_MOVETIME,
    nodes: int | None = None,
    tc: str | None = None,
) -> EngineAgent:
    """SF19 in process: a UCI_Elo anchor SF<elo> at `movetime` (or on a `tc`), or full strength at a
    node budget."""
    options: dict[str, object] = {"Threads": 1, "Hash": 16}
    if nodes is not None:
        return EngineAgent(f"SF19-n{nodes}", exe, nodes=nodes, options=options)
    bound: dict[str, object] = {"tc": tc} if tc else {"movetime": movetime}
    if elo is not None:
        options |= {"UCI_LimitStrength": True, "UCI_Elo": elo}
        return EngineAgent(f"SF{elo}", exe, options=options, **bound)
    return EngineAgent("SF19", exe, options=options, **bound)


def clock_for(player: Agent) -> Clock | None:
    """A player's clock: an engine agent's own; Blink's and DeepMind's st=1 timemargin=500; else none."""
    if isinstance(player, EngineAgent):
        return player.clock
    return BLINK_CLOCK if nosearch.is_searchless(player.name) else None


def warm_up(*players: Agent) -> None:
    """Get each clocked player ready before its first timed move, as fastchess's isready does: an engine
    process starts; a network makes one untimed decision (CUDA context, kernels) and logs nothing."""
    for player in players:
        if clock_for(player) is None:
            continue
        if isinstance(player, EngineAgent):
            player.warm_up()
            continue
        quiet = replace(player, sink=None) if is_dataclass(player) and hasattr(player, "sink") else player
        quiet.choose(chess.Board(), game=WARM_UP_GAME)


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
        "time_forfeits": summary["time_forfeits"],
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


def blink_agents(
    selector: str,
    device: str,
    epsilon: float | None = None,
    results_dir: Path = Path("results"),
    precision: str = "fp32",
    compile: bool = False,
) -> dict[str, Agent]:
    """Both modes of one model on one evaluator (one load, one CUDA context), named as fastchess does:
    in the fast play mode given (blink.play.fastmode; fp32 uncompiled by default), whose tag the names
    carry, so an in-process game and a fastchess game of one configuration file under one name."""
    from blink.eval.fastchess import engine_name
    from blink.play import factory

    evaluator = factory.load_evaluator(selector, device=device, precision=precision, compile=compile)
    eps = read_epsilon(results_dir) if epsilon is None else epsilon
    return {
        mode: replace(
            factory.make_agent(mode, evaluator, epsilon=eps),
            name=engine_name(selector, mode, precision, compile),
        )
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
