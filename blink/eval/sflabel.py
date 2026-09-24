"""Stockfish 19 labels at a fixed node budget, cached on disk so no position is ever searched twice.

A label is Stockfish's score from the side to move of `fen`, either for the position itself or, with
`move`, for the position with the search restricted to that one root move (UCI `searchmoves`): the win%
Blink's chosen move keeps, on the same scale as the position's best score. Threads=1, Hash=64, as for
games10k (P2). The cache is one JSON line per label in BLINK_HOME/eval/sfcache/sf19-n<nodes>.jsonl, keyed
by (fen, move); it is append-only, so an interrupted run resumes where it stopped. Used by the E2 win%
regret, the E8 endgame screen, the E9 failure classes and the mateset's mate-preserving rate.
"""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import chess.engine

from blink import paths
from blink.board import value

HASH_MB = 64
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
    """(fen, move) -> SfLabel for one node budget, read once, appended line by line."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._labels: dict[str, SfLabel] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = row.pop("key")
                    self._labels[key] = SfLabel(**row)

    def __len__(self) -> int:
        return len(self._labels)

    def get(self, key: str) -> SfLabel | None:
        return self._labels.get(key)

    def put(self, key: str, label: SfLabel) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, **asdict(label)}) + "\n")
        self._labels[key] = label


def _score_label(info: chess.engine.InfoDict, board: chess.Board) -> SfLabel:
    score = info["score"].pov(board.turn)
    pv = info.get("pv") or []
    return SfLabel(score.score(), score.mate(), int(info.get("depth", 0)), pv[0].uci() if pv else None)


class SfLabeler:
    """Labels at `nodes` nodes, from the cache when it can; Stockfish starts only on the first miss."""

    def __init__(
        self,
        nodes: int,
        exe: Path | None = None,
        cache_path: Path | None = None,
        analyse: Analyse | None = None,
    ) -> None:
        self.nodes = nodes
        self.exe = exe
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

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> "SfLabeler":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
