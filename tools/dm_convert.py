"""Convert DeepMind's searchless_chess checkpoint to npz and save JAX reference log-probs (plan P8, E0(1)).

Run ONLY with the throwaway conversion venv on D: (Python 3.12, jax/jaxlib 0.11.2, orbax-checkpoint 0.12.5,
dm-haiku, chess, jaxtyping), never with the project venv:

    D:/blink/dm/convert/Scripts/python.exe tools/dm_convert.py

DeepMind's training_utils is never imported (current JAX has no jax.sharding.PositionalSharding). Steps:
1. fetch DeepMind's src/transformer.py, tokenizer.py and utils.py at a pinned commit into <out>/src, each
   checked against a pinned sha256 (a file already there is re-checked, never trusted blindly);
2. restore `params` and `params_ema` with orbax straight to numpy (explicit RestoreArgs, so the 2024
   sharding file is not needed) and check every name and shape against the parameters DeepMind's own
   model definition creates under hk.transform;
3. write <out>/<size>-params.npz and <size>-params_ema.npz keyed by the flattened Haiku names
   (module/param, e.g. multi_head_dot_product_attention_3/linear_2/w), in Haiku's [in, out] layout;
4. run DeepMind's own forward pass (transformer_decoder under hk.transform, jitted, JAX on CPU) on 100
   fixed positions x every legal move, tokenized by DeepMind's tokenizer, and save the 128 return-bucket
   log-probs of both parameter sets to <out>/<size>-jax-logits.npz for the port's parity test.
"""

import argparse
import csv
import functools
import hashlib
import importlib.util
import io
import json
import os
import sys
import types
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np

DM_COMMIT = "90ae0e6b121673fc3079aaeffa047580bb600c0a"  # google-deepmind/searchless_chess main, 2025-01-10
DM_RAW = "https://raw.githubusercontent.com/google-deepmind/searchless_chess/{commit}/src/{name}"
DM_SOURCES = {
    "transformer.py": "7359830c02a3ab92a849dad46b71320d0a6c5b8c4d575640136917b1c3fcc6a2",
    "tokenizer.py": "26c0822aead13e9e3dcc772ef07c59f83cc0232bd2f327122f5ee9f6bb000f11",
    "utils.py": "331275099b0981f3c69cbbb696efe97ff73ecd0bf45ab441adf1fca72f29a7f7",
}
KINDS = ("params", "params_ema")
SIZES = {"9M": (8, 256, 8)}  # num_layers, embedding_dim, num_heads (src/engines/constants.py)
STEP = 6_400_000
NUM_RETURN_BUCKETS = 128
N_POSITIONS = 100
BATCH = 256
# Hand-picked rows the puzzle positions may miss: en passant for both sides, every castling pattern,
# promotions with captures, 2- and 3-digit counters, and the 218-legal-move position.
SPECIAL_FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "rnbqkbnr/ppp1pppp/8/8/P2pP3/8/1PPP1PPP/RNBQKBNR b KQkq e3 0 3",
    "r3k2r/1P4P1/8/8/8/8/6p1/R3K2R w KQkq - 0 40",
    "r3k2r/6P1/8/8/8/8/1p4p1/R3K2R b KQkq - 0 41",
    "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R w Kq - 6 9",
    "r3k2r/pppq1ppp/2npbn2/4p3/4P3/2NPBN2/PPPQ1PPP/R3K2R b Qk - 7 9",
    "8/8/4k3/8/8/3K4/6R1/8 w - - 57 112",
    "8/5k2/8/8/8/8/1K6/7q b - - 99 250",
    "8/8/4k3/8/8/3K4/6R1/8 w - - 120 999",
    "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w K - 13 70",
)


class Predictor(NamedTuple):
    """Stand-in for searchless_chess.src.constants.Predictor, the only name transformer.py uses from it."""

    initial_params: object
    predict: object


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_sources(dest: Path, commit: str = DM_COMMIT) -> None:
    """DeepMind's model files at the pinned commit; refuses any file whose sha256 differs."""
    dest.mkdir(parents=True, exist_ok=True)
    for name, digest in DM_SOURCES.items():
        path = dest / name
        if not path.is_file():
            with urllib.request.urlopen(DM_RAW.format(commit=commit, name=name), timeout=60) as response:
                path.write_bytes(response.read())
        found = _sha256(path.read_bytes())
        if found != digest:
            raise RuntimeError(f"{path} has sha256 {found}, expected {digest} (commit {commit})")


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_dm_modules(src: Path) -> tuple[types.ModuleType, types.ModuleType, types.ModuleType]:
    """transformer, tokenizer and utils from DeepMind's files, with a stub for their constants module.

    The real constants module imports apache_beam and grain for the training pipeline; transformer.py only
    needs its Predictor type, so the stub keeps the model code itself byte-identical to DeepMind's.
    """
    package = types.ModuleType("searchless_chess")
    package.__path__ = []
    subpackage = types.ModuleType("searchless_chess.src")
    subpackage.__path__ = []
    constants = types.ModuleType("searchless_chess.src.constants")
    constants.Predictor = Predictor
    sys.modules.update(
        {
            "searchless_chess": package,
            "searchless_chess.src": subpackage,
            "searchless_chess.src.constants": constants,
        }
    )
    tokenizer = _load_module("searchless_chess.src.tokenizer", src / "tokenizer.py")
    utils = _load_module("searchless_chess.src.utils", src / "utils.py")
    transformer = _load_module("searchless_chess.src.transformer", src / "transformer.py")
    return transformer, tokenizer, utils


def flatten(tree: dict, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
    """(module/param, array) for every leaf of a nested parameter dict, in sorted name order."""
    for key in sorted(tree):
        name = f"{prefix}/{key}" if prefix else str(key)
        value = tree[key]
        if isinstance(value, dict):
            yield from flatten(value, name)
        else:
            yield name, np.asarray(value)


def restore(path: Path) -> dict[str, np.ndarray]:
    """One orbax parameter tree as {flattened Haiku name: float32 numpy array}."""
    import jax
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    tree_metadata = checkpointer.metadata(path).item_metadata
    tree_metadata = getattr(tree_metadata, "tree", tree_metadata)
    restore_args = jax.tree.map(
        lambda _: ocp.RestoreArgs(restore_type=np.ndarray),
        tree_metadata,
        is_leaf=lambda leaf: not isinstance(leaf, dict),
    )
    tree = checkpointer.restore(path, args=ocp.args.PyTreeRestore(restore_args=restore_args))
    return {name: array.astype(np.float32, copy=False) for name, array in flatten(tree)}


def to_haiku(flat: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """{module: {param: array}}, the two-level dict hk.transform's apply expects."""
    params: dict[str, dict[str, np.ndarray]] = {}
    for name, array in flat.items():
        module, _, param = name.rpartition("/")
        params.setdefault(module, {})[param] = array
    return params


def dm_config(transformer: types.ModuleType, tokenizer: types.ModuleType, utils, size: str):
    """The TransformerConfig of src/engines/constants.py for an action-value model of this size."""
    num_layers, embedding_dim, num_heads = SIZES[size]
    return transformer.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=NUM_RETURN_BUCKETS,
        pos_encodings=transformer.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer.SEQUENCE_LENGTH + 2,
        num_heads=num_heads,
        num_layers=num_layers,
        embedding_dim=embedding_dim,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )


def check_against_definition(flat: dict[str, np.ndarray], model) -> None:
    """Every restored name and shape equals what DeepMind's own model definition initialises."""
    import jax

    expected = dict(flatten(model.init(jax.random.PRNGKey(1), np.ones((1, 1), dtype=np.uint32))))
    if set(expected) != set(flat):
        raise RuntimeError(
            f"parameter names differ: missing {sorted(set(expected) - set(flat))}, "
            f"extra {sorted(set(flat) - set(expected))}"
        )
    bad = {
        name: (flat[name].shape, array.shape)
        for name, array in expected.items()
        if flat[name].shape != array.shape
    }
    if bad:
        raise RuntimeError(f"parameter shapes differ (restored, defined): {bad}")


def write_npz(path: Path, arrays: dict) -> str:
    """np.savez to a .tmp, then os.replace; returns the sha256 of the written file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)
    return _sha256(path.read_bytes())


def solver_positions(puzzles_csv: Path, n: int) -> list[str]:
    """The first n puzzle positions shown to the solver (PGN end + the opponent's first move).

    Their halfmove and fullmove counters come from the real game, as in DeepMind's own puzzle scorer.
    """
    import chess
    import chess.pgn

    fens = []
    with open(puzzles_csv, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if len(fens) >= n:
                break
            game = chess.pgn.read_game(io.StringIO(row["PGN"]))
            board = game.end().board()
            board.push(chess.Move.from_uci(row["Moves"].split(" ")[0]))
            fens.append(board.fen())
    return fens


def positions(puzzles_csv: Path) -> list[str]:
    import chess

    special = [chess.Board(fen).fen() for fen in SPECIAL_FENS]
    return special + solver_positions(puzzles_csv, N_POSITIONS - len(special))


def build_rows(
    fens: list[str], tokenizer: types.ModuleType, utils: types.ModuleType
) -> dict[str, np.ndarray]:
    """ActionValueEngine.analyse's rows for every position: 77 FEN tokens, the action, a dummy bucket."""
    import chess

    tokens, offsets, ucis, actions, sequences = [], [0], [], [], []
    for fen in fens:
        board = chess.Board(fen)
        ordered = sorted(board.legal_moves, key=lambda move: utils.MOVE_TO_ACTION[move.uci()])
        fen_tokens = tokenizer.tokenize(board.fen()).astype(np.int32)
        tokens.append(fen_tokens.astype(np.uint8))
        for move in ordered:
            action = utils.MOVE_TO_ACTION[move.uci()]
            ucis.append(move.uci())
            actions.append(action)
            sequences.append(np.concatenate([fen_tokens, [action, 0]]).astype(np.int32))
        offsets.append(len(ucis))
    return {
        "fens": np.array(fens),
        "tokens": np.stack(tokens),
        "offsets": np.array(offsets, dtype=np.int64),
        "moves": np.array(ucis),
        "actions": np.array(actions, dtype=np.int32),
        "sequences": np.stack(sequences),
    }


def forward(apply, params: dict, sequences: np.ndarray) -> np.ndarray:
    """DeepMind's predict_fn(sequences)[:, -1] in fixed-size padded batches (one jit compile)."""
    out = []
    for start in range(0, len(sequences), BATCH):
        chunk = sequences[start : start + BATCH]
        padded = np.pad(chunk, ((0, BATCH - len(chunk)), (0, 0)))
        out.append(np.asarray(apply(params, None, padded))[: len(chunk), -1])
    return np.concatenate(out).astype(np.float32)


def convert(ckpt: Path, out: Path, puzzles_csv: Path, size: str) -> dict:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import haiku as hk
    import jax
    import orbax.checkpoint as ocp

    src = out / "src"
    fetch_sources(src)
    transformer, tokenizer, utils = load_dm_modules(src)
    config = dm_config(transformer, tokenizer, utils, size)
    model = hk.transform(functools.partial(transformer.transformer_decoder, config=config))
    apply = jax.jit(model.apply)
    rows = build_rows(positions(puzzles_csv), tokenizer, utils)
    _, bucket_values = utils.get_uniform_buckets_edges_values(NUM_RETURN_BUCKETS)
    report = {"size": size, "step": STEP, "dm_commit": DM_COMMIT, "jax": jax.__version__}
    report["orbax"] = getattr(ocp, "__version__", "?")
    logits = {}
    for kind in KINDS:
        flat = restore(ckpt / kind)
        check_against_definition(flat, model)
        sha = write_npz(out / f"{size}-{kind}.npz", flat)
        log_probs = forward(apply, to_haiku(flat), rows["sequences"])
        logits[f"log_probs_{kind}"] = log_probs
        report[kind] = {
            "tensors": len(flat),
            "parameters": int(sum(a.size for a in flat.values())),
            "sha256": sha,
        }
    meta = {**report, "positions": len(rows["fens"]), "rows": len(rows["moves"])}
    arrays = {key: value for key, value in rows.items() if key != "sequences"}
    arrays.update(logits, bucket_values=bucket_values.astype(np.float64), meta=np.array(json.dumps(meta)))
    report["logits_sha256"] = write_npz(out / f"{size}-jax-logits.npz", arrays)
    report["positions"], report["rows"] = meta["positions"], meta["rows"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", choices=sorted(SIZES), default="9M")
    parser.add_argument("--ckpt", type=Path, default=None, help="default <out>/<size>/6400000")
    parser.add_argument("--out", type=Path, default=Path("D:/blink/dm"))
    parser.add_argument("--puzzles", type=Path, default=Path("D:/blink/downloads/puzzles.csv"))
    args = parser.parse_args()
    ckpt = args.ckpt or args.out / args.size / str(STEP)
    print(json.dumps(convert(ckpt, args.out, args.puzzles, args.size), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
