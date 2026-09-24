"""`blink audit no-search`: prove NSC-1 from the games alone.

The UCI engine reports, for every move, the number of positions its network scored (`nodes`).
fastchess writes that number into each move comment as `n=<nodes>` (verified on fastchess 1.8.2:
`{+0.53/1 0.001s, n=5}`, book moves `{book}`, the last move `{..., n=0, White mates}`), and
`blink match` writes the same format. This audit replays every game with python-chess, and for each
move by a Blink player checks:
  - the comment carries a node count;
  - nodes <= L + 1, where L is the number of legal moves in the position (value mode uses exactly L+1);
  - nodes == 0 only when the move played gives checkmate (rule R2).
It also counts game terminations, per-engine forfeits (time, illegal move, crash) and adjudications,
and writes the rows-per-move histogram as JSON.

`audit` checks every player whose name contains a filter (the CLI's --engine); `audit_each` checks named
players by their exact names, in one pass, so Blink-value-ship never picks up Blink-value-ship-rules-off's
moves and DM-9M never picks up DM-9M-ema's. The searchless players are Blink's and DeepMind's.
"""

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import chess
import chess.pgn

NODES = re.compile(r"(?:^|[\s,])n=(\d+)")
DEFAULT_ENGINE = "blink"
BENIGN_TERMINATIONS = {"normal", "adjudication", ""}
SEARCHLESS_PREFIXES = ("Blink", "DM-")


def is_searchless(name: str) -> bool:
    """Blink's players and DeepMind's (DM-9M[-ema]): the players the no-search audit must cover."""
    return name.startswith(SEARCHLESS_PREFIXES)


def pgn_files(target: Path) -> list[Path]:
    """A PGN file, or every *.pgn directly inside a directory, in name order."""
    if target.is_dir():
        return sorted(target.glob("*.pgn"))
    return [target]


def _read_games(path: Path) -> Iterable[chess.pgn.Game]:
    with open(path, encoding="utf-8", errors="replace") as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            yield game


class _Tally:
    def __init__(self) -> None:
        self.games = 0
        self.decisions = 0
        self.full_batches = 0
        self.missing = 0
        self.max_rows = 0
        self.max_legal = 0
        self.histogram: Counter = Counter()
        self.players: Counter = Counter()
        self.terminations: Counter = Counter()
        self.forfeits: dict[str, Counter] = {}
        self.adjudications = 0
        self.violations: list[dict] = []

    def violation(self, where: dict, rule: str, rows: int | None, legal: int) -> None:
        self.violations.append({**where, "rows": rows, "legal": legal, "rule": rule})


def _check_move(tally: _Tally, board: chess.Board, node: chess.pgn.ChildNode, where: dict) -> None:
    legal = board.legal_moves.count()
    found = NODES.search(node.comment)
    tally.decisions += 1
    tally.players[where["player"]] += 1
    tally.max_legal = max(tally.max_legal, legal)
    if found is None:
        tally.missing += 1
        tally.violation(where, "no node count", None, legal)
        return
    rows = int(found.group(1))
    tally.histogram[rows] += 1
    tally.max_rows = max(tally.max_rows, rows)
    tally.full_batches += rows == legal + 1
    if rows > legal + 1:
        tally.violation(where, "rows > L+1", rows, legal)
    elif rows == 0 and not node.board().is_checkmate():
        tally.violation(where, "0 rows without a mate", rows, legal)


def _count_ending(tally: _Tally, game: chess.pgn.Game) -> None:
    termination = game.headers.get("Termination", "")
    tally.terminations[termination] += 1
    result = game.headers.get("Result", "*")
    if termination == "adjudication":
        tally.adjudications += 1
    if termination not in BENIGN_TERMINATIONS and result in ("1-0", "0-1"):
        loser = game.headers.get("Black" if result == "1-0" else "White", "?")
        tally.forfeits.setdefault(loser, Counter())[termination] += 1


def _audit_game(tally: _Tally, game: chess.pgn.Game, source: str, wanted: Callable[[str], bool]) -> None:
    tally.games += 1
    _count_ending(tally, game)
    board = game.board()
    for node in game.mainline():
        player = game.headers.get("White" if board.turn == chess.WHITE else "Black", "?")
        if node.comment.strip() != "book" and wanted(player):
            where = {"file": source, "game": tally.games, "ply": board.ply(), "player": player}
            _check_move(tally, board, node, where)
        board.push(node.move)


def audit(files: Sequence[Path], engine: str = DEFAULT_ENGINE) -> dict:
    """Audit every game in `files`; players whose name contains `engine` (case-insensitive) are checked."""
    needle = engine.lower()
    tally = _Tally()
    for path in files:
        for game in _read_games(path):
            _audit_game(tally, game, path.name, lambda player: needle in player.lower())
    return _report(tally, len(files))


def audit_each(files: Sequence[Path], players: Iterable[str]) -> dict[str, dict]:
    """One report per named player, over the games it played, matching names exactly; one pass."""
    tallies = {name: _Tally() for name in players}
    for path in files:
        for game in _read_games(path):
            seated = {game.headers.get("White", "?"), game.headers.get("Black", "?")}
            for name in sorted(seated & tallies.keys()):
                _audit_game(tallies[name], game, path.name, lambda player, own=name: player == own)
    return {name: _report(tally, len(files)) for name, tally in tallies.items()}


def _report(tally: _Tally, files: int) -> dict:
    return {
        "files": files,
        "games": tally.games,
        "decisions": tally.decisions,
        "compliant": not tally.violations and tally.decisions > 0,
        "violations": tally.violations,
        "missing_counts": tally.missing,
        "histogram": dict(sorted(tally.histogram.items())),
        "value_mode_full_batches": tally.full_batches,
        "max_rows": tally.max_rows,
        "max_legal": tally.max_legal,
        "players": dict(tally.players),
        "terminations": dict(tally.terminations),
        "forfeits": {player: dict(counts) for player, counts in tally.forfeits.items()},
        "adjudications": tally.adjudications,
    }


def write_report(report: dict, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {**report, "histogram": {str(k): v for k, v in report["histogram"].items()}}
    out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    return out
