"""Shared helpers for the model, training and dashboard tests (kept out of conftest.py on purpose)."""

import dataclasses
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
