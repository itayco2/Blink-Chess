"""Stockfish 19 labels at a fixed node budget, cached on disk so no position is ever searched twice.

A label is Stockfish's score from the side to move of `fen`, either for the position itself or, with
`move`, for the position with the search restricted to that one root move (UCI `searchmoves`): the win%
Blink's chosen move keeps, on the same scale as the position's best score. Threads=1, Hash=64, as for
games10k (P2). The cache is one JSON line per label in BLINK_HOME/eval/sfcache/sf19-n<nodes>.jsonl, keyed
by (fen, move); it is append-only, so an interrupted run resumes where it stopped. Used by the E2 win%
regret, the E8 endgame screen, the E9 failure classes and the mateset's mate-preserving rate.
`label_many` spreads the cache misses over `procs` Stockfish processes (one thread each, as for games10k);
only the calling process writes the cache.
"""

import json
import multiprocessing as mp
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine

from blink import paths
from blink.board import value

HASH_MB = 64
CHUNK = 16  # searches per task handed to one worker process
Request = tuple[str, str | None]  # (fen, move in UCI or None)
Analyse = Callable[[chess.Board, int, chess.Move | None], "SfLabel"]


@dataclass(frozen=True)
class SfLabel:
    cp: int | None  # side to move of the labelled position; None when the score is a mate
    mate: int | None  # > 0: the side to move mates in that many; < 0: it is mated
    depth: int
    best: str | None  # Stockfish's first PV move (UCI), when it has one

    @property
    def win(self) -> float:
        """The Lichess win probability of this score for the side to move."""
        return value.win_probability(cp=self.cp, mate=self.mate)

    @property
    def pawns(self) -> float:
        """The score in pawns for the side to move, a mate counted as +-100."""
        if self.mate is not None:
            return 100.0 if self.mate > 0 else -100.0
        return (self.cp or 0) / 100


def cache_dir() -> Path:
    return paths.home() / "eval" / "sfcache"


def cache_key(fen: str, move: chess.Move | str | None) -> str:
    return f"{fen}|{move if move is not None else '-'}"


class SfCache:
    """(fen, move) -> SfLabel for one node budget, read once, appended line by line.

    A kill (taskkill /F before a measurement) can cut the last append short: that line is skipped and
    counted in `dropped` (its label is searched again when asked for), and the next append starts on a
    new line, so every whole line still resumes."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._labels: dict[str, SfLabel] = {}
        self.dropped = 0  # lines that are not a whole label (cut short by a kill)
        self._open_line = False  # the file does not end with a newline
        if self.path.is_file():
            text = self.path.read_text(encoding="utf-8")
            self._open_line = bool(text) and not text.endswith("\n")
            for line in text.splitlines():
                if line.strip():
                    self._read(line)

    def _read(self, line: str) -> None:
        try:
            row = json.loads(line)
            key = row.pop("key")
            label = SfLabel(**row)
        except (json.JSONDecodeError, AttributeError, KeyError, TypeError):
            self.dropped += 1
            return
        self._labels[key] = label

    def __len__(self) -> int:
        return len(self._labels)

    def get(self, key: str) -> SfLabel | None:
        return self._labels.get(key)

    def put(self, key: str, label: SfLabel) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(("\n" if self._open_line else "") + json.dumps({"key": key, **asdict(label)}) + "\n")
        self._open_line = False
        self._labels[key] = label


def _score_label(info: chess.engine.InfoDict, board: chess.Board) -> SfLabel:
    score = info["score"].pov(board.turn)
    pv = info.get("pv") or []
    return SfLabel(score.score(), score.mate(), int(info.get("depth", 0)), pv[0].uci() if pv else None)


def _search(engine: chess.engine.SimpleEngine, fen: str, move: str | None, nodes: int) -> SfLabel:
    board = chess.Board(fen)
    root_moves = [chess.Move.from_uci(move)] if move is not None else None
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=root_moves)
    return _score_label(info, board)


def _worker(task: tuple[str, int, list[Request]]) -> list[tuple[str, dict]]:
    """One Stockfish process for one chunk of searches (a spawned worker: plain data in and out)."""
    exe, nodes, requests = task
    with chess.engine.SimpleEngine.popen_uci(exe) as engine:
        engine.configure({"Threads": 1, "Hash": HASH_MB})
        return [(cache_key(fen, move), asdict(_search(engine, fen, move, nodes))) for fen, move in requests]


def _checked(fen: str, move: chess.Move | str | None) -> Request:
    uci = move.uci() if isinstance(move, chess.Move) else move
    if uci is not None and chess.Move.from_uci(uci) not in chess.Board(fen).legal_moves:
        raise ValueError(f"{uci} is not legal in {fen}")
    return fen, uci


class SfLabeler:
    """Labels at `nodes` nodes, from the cache when it can; Stockfish starts only on the first miss."""

    def __init__(
        self,
        nodes: int,
        exe: Path | None = None,
        cache_path: Path | None = None,
        analyse: Analyse | None = None,
        procs: int = 1,
    ) -> None:
        self.nodes = nodes
        self.exe = exe
        self.procs = max(1, procs)
        self.cache = SfCache(cache_path or cache_dir() / f"sf19-n{nodes}.jsonl")
        self._analyse = analyse
        self._engine: chess.engine.SimpleEngine | None = None
        self.searched = 0  # labels computed in this process (the rest came from the cache)

    def _stockfish(self, board: chess.Board, nodes: int, move: chess.Move | None) -> SfLabel:
        if self._engine is None:
            if self.exe is None:
                raise ValueError("no Stockfish executable was given and the label is not cached")
            self._engine = chess.engine.SimpleEngine.popen_uci(str(self.exe))
            self._engine.configure({"Threads": 1, "Hash": HASH_MB})
        root_moves = [move] if move is not None else None
        info = self._engine.analyse(board, chess.engine.Limit(nodes=nodes), root_moves=root_moves)
        return _score_label(info, board)

    def label(self, fen: str, move: chess.Move | str | None = None) -> SfLabel:
        key = cache_key(fen, move)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        board = chess.Board(fen)
        parsed = chess.Move.from_uci(move) if isinstance(move, str) else move
        if parsed is not None and parsed not in board.legal_moves:
            raise ValueError(f"{parsed} is not legal in {fen}")
        label = (self._analyse or self._stockfish)(board, self.nodes, parsed)
        self.cache.put(key, label)
        self.searched += 1
        return label

    def label_many(self, requests: Sequence[tuple[str, chess.Move | str | None]]) -> list[SfLabel]:
        """Labels for many (fen, move) at once; the misses are searched on `procs` processes."""
        checked = [_checked(fen, move) for fen, move in requests]
        missing = list(dict.fromkeys(r for r in checked if self.cache.get(cache_key(*r)) is None))
        if self.procs == 1 or self._analyse is not None or len(missing) <= CHUNK:
            return [self.label(fen, move) for fen, move in checked]
        if self.exe is None:
            raise ValueError("no Stockfish executable was given and the labels are not cached")
        tasks = [(str(self.exe), self.nodes, missing[i : i + CHUNK]) for i in range(0, len(missing), CHUNK)]
        with mp.get_context("spawn").Pool(self.procs) as pool:
            for batch in pool.imap_unordered(_worker, tasks):
                for key, fields in batch:
                    self.cache.put(key, SfLabel(**fields))
                self.searched += len(batch)
        return [self.cache.get(cache_key(fen, move)) for fen, move in checked]

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> "SfLabeler":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
