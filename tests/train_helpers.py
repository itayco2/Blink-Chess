"""Shared helpers for the model, training and dashboard tests (kept out of conftest.py on purpose)."""

import dataclasses
import threading
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from blink.data.parse import Rejected, parse_line
from blink.model.config import ModelConfig, TrainConfig

FIXTURE = Path(__file__).parent / "fixtures" / "eval_lines_first100.jsonl"


@lru_cache(maxsize=1)
def _parsed_fixture() -> np.ndarray:
    rows = []
    for line in FIXTURE.read_bytes().splitlines():
        try:
            rows.append(parse_line(line))
        except Rejected:
            continue
    return np.stack(rows)


def fixture_records() -> np.ndarray:
    """Every parseable row of the 100-line CC0 fixture, as ROOT_DTYPE records (a fresh copy)."""
    return _parsed_fixture().copy()


def tiny_model_config(**overrides) -> ModelConfig:
    return dataclasses.replace(ModelConfig(d_model=64, n_layers=1, n_heads=2, head_dim=32), **overrides)


def tiny_train_config(**overrides) -> TrainConfig:
    base = TrainConfig(
        model=tiny_model_config(),
        batch_size=32,
        steps=200,
        peak_lr=1e-3,
        warmup_steps=10,
        metrics_every=50,
        eval_every=100,
        val_size=64,
        ckpt_every_steps=100,
        ckpt_every_minutes=0.0,
    )
    return dataclasses.replace(base, **overrides)


@contextmanager
def held_open(*paths: Path, release_after_s: float = 0.15) -> Iterator[None]:
    """A reader (the live dashboard, a tail, a scanner) holding files open for a moment.

    Python opens files on Windows without delete-sharing, so while these handles are open an
    os.replace onto any of the paths fails with PermissionError; the handles close on a timer.
    """
    with ExitStack() as stack:
        handles = [stack.enter_context(open(path, encoding="utf-8")) for path in paths]
        timer = threading.Timer(release_after_s, lambda: [handle.close() for handle in handles])
        timer.start()
        try:
            yield
        finally:
            timer.join()
