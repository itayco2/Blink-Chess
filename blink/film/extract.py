"""`blink film extract`: every frame of a run, run once on one puzzle position, written to film.json.

The position is a Lichess band puzzle after its setup move (the solver to move, the solution is the
next move of the line). The band puzzles are in the leakage blocklist, so the pack never let the
position, its colour mirror or any later position of its line into training; extract re-checks that
against the blocklist file and refuses a position the blocklist does not hold. It then proves the file
is the run's own blocklist: the run's config.json names its pack (data.dir, or --pack), whose manifest
must record the same blocklist sha256 and hash to the frames' WORLD. A run trained on raw data, on a
pack built without a blocklist (the P1 skeleton) or on an older blocklist is refused.

Frames come from runs/NAME/film/frame_<step>.pt (plan P7: step 0, 19 geometric EMA steps, the
shipped weights), sorted numerically and required to share one WORLD. A run saved before film
frames existed (the skeleton) falls back to its checkpoints' EMA weights plus the step-0 network
reproduced from the run's seed, exactly as the trainer builds it (torch.manual_seed(seed), then
BlinkNet on the CPU). `--pad-to 21` fills the gaps between measured frames by linear interpolation
of their predictions; every such frame says "interpolated" in film.json and on screen.

Each frame keeps the probability of every legal move (the softmax over legal moves only), the top 5,
the 128-bin value distribution and its mean win%, positions seen (step x batch rows) and the
GPU-hours the run had used by then (cumulative metrics.jsonl windows, as in results/compute.json).
"""

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import chess
import numpy as np

from blink.board import encode, moves, value
from blink.report import compute
from blink.train.atomic import write_text_atomic

FILM_FRAMES = 21
FORMAT = 1
TOP = 5


class FilmError(ValueError):
    """The film cannot be made as asked (a missing puzzle, a leak, mixed worlds, a wrong frame count)."""


@dataclass(frozen=True)
class FilmPosition:
    puzzle_id: str
    rating: int
    themes: str
    setup_fen: str
    setup_move: str
    fen: str  # the solver's position, after the setup move
    solution: str
    line: tuple[str, ...]  # every move of the puzzle, the setup move first
    side_to_move: str


@dataclass(frozen=True)
class FrameSource:
    step: int
    kind: str  # "init" | "ema" | "final"
    origin: str  # "film" | "checkpoint" | "reproduced-init"
    path: Path
    weights: str  # the state key: "model" or "ema"


def position_from_row(row: dict) -> FilmPosition:
    line = tuple(row["Moves"].split())
    board = chess.Board(row["FEN"])
    board.push_uci(line[0])
    return FilmPosition(
        puzzle_id=row["PuzzleId"],
        rating=int(row["Rating"]),
        themes=row.get("Themes", ""),
        setup_fen=row["FEN"],
        setup_move=line[0],
        fen=board.fen(),
        solution=line[1],
        line=line,
        side_to_move="white" if board.turn == chess.WHITE else "black",
    )


def read_band_puzzles(bands_csv: Path) -> list[FilmPosition]:
    with open(bands_csv, encoding="utf-8", newline="") as handle:
        return [position_from_row(row) for row in csv.DictReader(handle)]


def film_position(bands_csv: Path, puzzle_id: str) -> FilmPosition:
    with open(bands_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["PuzzleId"] == puzzle_id:
                return position_from_row(row)
    raise FilmError(f"puzzle {puzzle_id!r} is not in {bands_csv}")


# ------------------------------------------------------------------------------------------ leakage


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def never_in_training(position: FilmPosition, blocklist_path: Path) -> dict:
    """Proof from the blocklist file that the position and its line were kept out of training."""
    from blink.data import blocklist

    hashes = np.load(blocklist_path)
    board = chess.Board(position.setup_fen)
    line = [encode.position_hash(board)]
    for uci in position.line:
        board.push_uci(uci)
        line.append(encode.position_hash(board))
    target = encode.position_hash(chess.Board(position.fen))
    line_blocked = bool(blocklist.contains(hashes, np.array(line, dtype=np.uint64)).all())
    position_blocked = bool(blocklist.contains(hashes, np.array([target], dtype=np.uint64))[0])
    if not position_blocked:
        raise FilmError(
            f"puzzle {position.puzzle_id} is not in the blocklist {blocklist_path}: it may be in training"
        )
    return {
        "blocklist": Path(blocklist_path).name,
        "blocklist_sha256": _sha256(blocklist_path),
        "position_blocked": position_blocked,
        "line_blocked": line_blocked,
    }


def _pack_manifest_path(run_dir: Path, pack_dir: Path | None) -> Path:
    config_path = Path(run_dir) / "config.json"
    if not config_path.is_file():
        raise FilmError(
            f"{run_dir} has no config.json: the pack it trained on is unknown, so its blocklist is too"
        )
    data = json.loads(config_path.read_text(encoding="utf-8")).get("data") or {}
    if data.get("source") != "shards":
        raise FilmError(
            f"{Path(run_dir).name} trained on {data.get('source')!r} data, not a pack built with a "
            "blocklist: nothing proves the film position was kept out of its training"
        )
    folder = pack_dir if pack_dir is not None else data.get("dir")
    if not folder:
        raise FilmError(f"{Path(run_dir).name}'s config.json names no pack directory; pass --pack DIR")
    manifest = Path(folder) / "manifest.json"
    if not manifest.is_file():
        raise FilmError(f"the run's pack manifest {manifest} is missing; pass --pack DIR if the pack moved")
    return manifest


def pack_provenance(run_dir: Path, world: str, blocklist_sha256: str, pack_dir: Path | None = None) -> dict:
    """Proof that the run's own pack was built with this blocklist: its manifest names the same sha256,
    and the manifest's WORLD (commands.train_data.pack_world, which hashes the blocklist in) is the
    frames' world, so the manifest read here is the one the frames were trained on."""
    from blink.commands.train_data import pack_world

    path = _pack_manifest_path(run_dir, pack_dir)
    raw = path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    entry = manifest.get("blocklist")
    pack_sha = entry.get("sha256") if isinstance(entry, dict) else None
    if not pack_sha:
        raise FilmError(
            f"the pack {path.parent} was built without a blocklist: the position may be in training"
        )
    if pack_sha != blocklist_sha256:
        raise FilmError(
            f"--blocklist (sha256 {blocklist_sha256[:12]}) is not the blocklist the pack {path.parent} was "
            f"built with (sha256 {pack_sha[:12]})"
        )
    pack = pack_world(raw, manifest)
    if pack != world:
        raise FilmError(f"the manifest {path} is world {pack}, but the frames were trained in world {world}")
    return {"pack_manifest": str(path), "pack_blocklist_sha256": pack_sha, "pack_world": pack}


# ------------------------------------------------------------------------------------------ frames


def frame_sources(run_dir: Path) -> list[FrameSource]:
    """The run's film frames in step order, else its checkpoints' EMA weights after a reproduced init."""
    from blink.train import checkpoint, film

    frames = film.list_frames(run_dir)
    if frames:
        return [FrameSource(film.step_of(p), "frame", "film", p, "model") for p in frames]
    ckpts = checkpoint.list_checkpoints(run_dir)
    if not ckpts:
        raise FilmError(f"{run_dir} has neither film frames nor checkpoints")
    init = FrameSource(0, "init", "reproduced-init", ckpts[0], "model")
    return [init, *(FrameSource(checkpoint.step_of(p), "ema", "checkpoint", p, "ema") for p in ckpts)]


def _train_config(config: dict):
    from blink.model.config import ModelConfig, TrainConfig, config_from_dict

    if isinstance(config.get("model"), dict):
        return config_from_dict(config)
    return TrainConfig(model=ModelConfig(**config))


def load_frame(source: FrameSource, device: str = "cpu"):
    """(TorchEvaluator, {"world", "kind", "samples"}) for one frame source."""
    import torch

    from blink.model.evaluator import TorchEvaluator
    from blink.model.transformer import BlinkNet

    state = torch.load(source.path, map_location="cpu", weights_only=True)
    cfg = _train_config(state["config"])
    if source.origin == "reproduced-init":
        torch.manual_seed(cfg.seed)  # the trainer's own order: seed, then build the network on the CPU
        model = BlinkNet(cfg.model)
        samples = 0
    else:
        model = BlinkNet(cfg.model)
        model.load_state_dict(state[source.weights])
        samples = state.get("samples", source.step * cfg.batch_size)
    kind = state.get("kind", source.kind) if source.origin == "film" else source.kind
    if source.origin == "checkpoint" and source.step == cfg.steps:
        kind = "final"  # the EMA weights at the run's planned last step: what the run would ship
    meta = {"world": state["world"], "kind": kind, "samples": int(samples)}
    return TorchEvaluator(model, device), meta


def predict(evaluator, boards: list[chess.Board]) -> list[dict]:
    """One network call over all boards: legal-move probabilities, the value bins and the mean win%."""
    codes = np.stack([encode.encode_board(board) for board in boards])
    result = evaluator.evaluate(codes)
    out = []
    for i, board in enumerate(boards):
        mask = moves.legal_mask(board)
        index = np.flatnonzero(mask)
        logits = result.policy_logits[i][index].astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        legal = {moves.decode_move(board, int(j)).uci(): float(p) for j, p in zip(index, probs, strict=True)}
        bins = result.value_probs[i].astype(np.float64)
        out.append({"legal": legal, "value_bins": bins.tolist(), "win": float(bins @ value.BIN_CENTERS)})
    return out


def top_moves(legal: dict[str, float], n: int = TOP) -> list[dict]:
    ranked = sorted(legal.items(), key=lambda item: (-item[1], item[0]))[:n]
    return [{"move": move, "p": p} for move, p in ranked]


def gpu_hours_by_step(run_dir: Path) -> list[tuple[int, float]]:
    """(step, cumulative GPU-hours) at every metrics row; empty when the run has no telemetry."""
    metrics, config = Path(run_dir) / "metrics.jsonl", Path(run_dir) / "config.json"
    if not metrics.is_file() or not config.is_file():
        return []
    record = json.loads(config.read_text(encoding="utf-8"))
    batch = record.get("config", {}).get("batch_size")
    if not batch:
        return []
    try:
        start = compute.start_step(record)  # a branch counts from its parent's checkpoint step
    except ValueError:
        return []  # the branch step is unknown: no GPU-hours rather than the parent's counted twice
    rows = [row for row in compute.read_metrics(metrics) if row["step"] > start]
    cumulative = np.cumsum([seconds for seconds, _ in compute.windows(rows, batch, start)]) / 3600
    return [(row["step"], float(hours)) for row, hours in zip(rows, cumulative, strict=True)]


def _hours_at(table: list[tuple[int, float]], step: int) -> float | None:
    if not table:
        return None
    done = [hours for s, hours in table if s <= step]
    return done[-1] if done else 0.0


def _measured_frames(run_dir: Path, board: chess.Board, device: str) -> tuple[list[dict], set[str]]:
    hours = gpu_hours_by_step(run_dir)
    frames, worlds = [], set()
    for source in frame_sources(run_dir):
        evaluator, meta = load_frame(source, device)
        (pred,) = predict(evaluator, [board])
        worlds.add(meta["world"])
        frames.append(
            {
                "step": source.step,
                "kind": meta["kind"],
                "origin": source.origin,
                "interpolated": False,
                "positions_seen": meta["samples"],
                "gpu_hours": _hours_at(hours, source.step),
                "top5": top_moves(pred["legal"]),
                **pred,
            }
        )
    return frames, worlds


# ------------------------------------------------------------------------------------------ milestones


# A rung's threshold is its own measurement in results.json, never a typed number. VAA (value-mode move
# agreement, the one metric every ladder rung has) is preferred: the run's evals.jsonl ema_vaa is
# measured on the valprobe too. A policy-only rung falls back to policy top-1 (evals ema_top1 is on the
# run's validation sample). film.json records both sources and the page says "on the val probe".
RUNG_METRICS = (("value", "vaa", "ema_vaa"), ("policy", "top1", "ema_top1"))


def parse_milestone(text: str) -> tuple[str, str]:
    """'passed the MLP=MLP' -> ('passed the MLP', 'MLP'): the on-screen label and a ladder agent."""
    label, sep, agent = text.rpartition("=")
    if not sep or not label.strip() or not agent.strip():
        raise FilmError(f"a milestone is LABEL=AGENT (for example 'passed the MLP=MLP'), got {text!r}")
    return label.strip(), agent.strip()


def _rung(results, label: str, agent: str) -> dict:
    # E6's rungs: the baselines are ladder rows, s10m is rated as a Blink row (Blink-<mode>-run_s10m)
    ladder = {row.agent for row in results.strength if row.kind in ("ladder", "blink")}
    if agent not in ladder:
        raise FilmError(
            f"milestone {label!r}: {agent!r} is not a ladder row (or a Blink rung) in results.json "
            f"({sorted(ladder)})"
        )
    diagnostics = {(row.agent, row.mode): row for row in results.diagnostics}
    for mode, metric, key in RUNG_METRICS:
        row = diagnostics.get((agent, mode))
        value = getattr(row, metric) if row is not None else None
        if value is not None:
            return {
                "label": label,
                "agent": agent,
                "metric": metric,
                "threshold": value,
                "key": key,
                "threshold_source": f"results.json diagnostics ({agent}, {mode}): {metric} = {value:g}",
                "compared_with": f"{key} in the run's evals.jsonl (its val probe)",
            }
    raise FilmError(
        f"milestone {label!r}: results.json diagnostics hold no VAA or policy top-1 for {agent!r}"
    )


def ladder_rungs(results_dir: Path, milestones: Sequence[tuple[str, str]]) -> list[dict]:
    """Each (label, agent) resolved to its measured threshold in results_dir/results.json."""
    from blink.report import results_schema as rs

    if not milestones:
        return []
    path = Path(results_dir) / "results.json"
    if not path.is_file():
        raise FilmError(f"{path} is missing: milestone thresholds come from the ladder's measured results")
    results = rs.from_json(path.read_text(encoding="utf-8"))
    return [_rung(results, label, agent) for label, agent in milestones]


def find_milestones(run_dir: Path, rungs: Sequence[dict]) -> list[dict]:
    """The first eval step whose EMA metric reaches each rung's measured threshold, in step order."""
    path = Path(run_dir) / "evals.jsonl"
    if not rungs or not path.is_file():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = sorted((r for r in rows if "step" in r), key=lambda r: r["step"])
    found = []
    for rung in rungs:
        key, threshold = rung["key"], rung["threshold"]
        step = next((r["step"] for r in rows if r.get(key) is not None and r[key] >= threshold), None)
        if step is not None:
            found.append({**{k: v for k, v in rung.items() if k != "key"}, "step": step})
    return sorted(found, key=lambda m: (m["step"], m["label"]))


# ------------------------------------------------------------------------------------------ padding


def _lerp(a, b, t: float):
    return None if a is None or b is None else a + (b - a) * t


def _between(a: dict, b: dict, t: float) -> dict:
    legal = {m: _lerp(p, b["legal"][m], t) for m, p in a["legal"].items()}
    bins = (np.array(a["value_bins"]) * (1 - t) + np.array(b["value_bins"]) * t).tolist()
    return {
        "step": int(round(_lerp(a["step"], b["step"], t))),
        "kind": "interpolated",
        "origin": "interpolated",
        "interpolated": True,
        "positions_seen": int(round(_lerp(a["positions_seen"], b["positions_seen"], t))),
        "gpu_hours": _lerp(a["gpu_hours"], b["gpu_hours"], t),
        "top5": top_moves(legal),
        "legal": legal,
        "value_bins": bins,
        "win": _lerp(a["win"], b["win"], t),
    }


def pad_frames(frames: list[dict], total: int) -> list[dict]:
    """Insert interpolated frames between measured ones (earlier gaps first) until there are `total`."""
    if len(frames) > total:
        raise FilmError(f"{len(frames)} measured frames is more than the {total} asked for")
    if len(frames) < 2:
        raise FilmError("interpolation needs at least 2 measured frames")
    gaps = len(frames) - 1
    extra, spare = divmod(total - len(frames), gaps)
    out = [frames[0]]
    for g, (a, b) in enumerate(zip(frames, frames[1:], strict=False)):
        k = extra + (1 if g < spare else 0)
        out += [_between(a, b, j / (k + 1)) for j in range(1, k + 1)]
        out.append(b)
    return out


def extract(
    run_dir: Path,
    position: FilmPosition,
    blocklist_path: Path,
    pad_to: int | None = None,
    expect: int | None = FILM_FRAMES,
    device: str = "cpu",
    milestones: Sequence[dict] = (),
    pack_dir: Path | None = None,
) -> dict:
    proof = never_in_training(position, blocklist_path)
    frames, worlds = _measured_frames(Path(run_dir), chess.Board(position.fen), device)
    if len(worlds) != 1:
        raise FilmError(f"the frames come from {len(worlds)} worlds: {sorted(worlds)}")
    world = next(iter(worlds))
    proof |= pack_provenance(Path(run_dir), world, proof["blocklist_sha256"], pack_dir)
    measured = len(frames)
    if pad_to:
        frames = pad_frames(frames, pad_to)
    elif expect and measured != expect:
        raise FilmError(f"{run_dir} has {measured} frames; the film needs {expect} (--pad-to interpolates)")
    frames = [{"index": i, **frame} for i, frame in enumerate(frames, start=1)]
    padded = len(frames) - measured
    note = (
        f"{padded} of {len(frames)} frames are interpolated between {measured} measured ones"
        if padded
        else ""
    )
    return {
        "format": FORMAT,
        "run": Path(run_dir).name,
        "world": world,
        "position": asdict(position),
        "never_in_training": proof,
        "frames": frames,
        "measured_frames": measured,
        "milestones": find_milestones(Path(run_dir), milestones),
        "note": note,
    }


def write_film(film: dict, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(out, json.dumps(film, indent=1) + "\n")
    return out


def read_film(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
