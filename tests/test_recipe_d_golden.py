"""Recipe D's golden trajectory: new options at their defaults must not change what D trains (PF66).

P5 runs from a frozen worktree while arms a06, a07, a08 and a10 are built. Moving the worktree to a
newer commit mid-sweep is safe only if Recipe D still trains exactly as before. This test pins the
per-step losses of a small Recipe D run (GAB-lite, children, weights, soft targets, micro-batches,
clip auto) on CPU. The numbers are this machine's floats, so it is marked local.

Regenerate deliberately, and only for an intended change to D: BLINK_WRITE_GOLDEN=1 pytest this file.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_mixed_training import children_from  # noqa: E402
from train_helpers import fixture_records, tiny_model_config, tiny_train_config  # noqa: E402

from blink.train import loop  # noqa: E402
from blink.train.source import InMemorySource, mixed_source  # noqa: E402

pytestmark = [pytest.mark.torch, pytest.mark.local]
GOLDEN = Path(__file__).parent / "fixtures" / "recipe_d_golden.json"
STEPS = 20


def _recipe_d_losses(run_dir: Path) -> list[list[float]]:
    cfg = tiny_train_config(
        model=tiny_model_config(gab=True),
        steps=STEPS,
        warmup_steps=5,
        batch_size=16,
        child_frac=0.25,
        micro_batch=8,
        clip_norm="auto",
        metrics_every=1,
        eval_every=1000,
        ckpt_every_steps=1000,
    )
    records = fixture_records()
    roots = InMemorySource(records[:48], 12, seed=3).batches
    children = InMemorySource(children_from(records[48:]), 4, seed=4).batches
    weights = mixed_source(roots, children, lambda r: (r["fen_hash"] % 7).astype(np.float32) / 3 + 0.2)
    spec = loop.RunSpec(run_dir=run_dir, world="0123456789ab", device="cpu")
    loop.train(cfg, spec, weights, val=records, log=lambda _: None)
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    return [[row["step"], row["loss_policy"], row["loss_value"], row["grad_norm"]] for row in rows]


def test_recipe_d_trains_exactly_as_it_did_when_p5_started(tmp_path):
    losses = _recipe_d_losses(tmp_path / "run")
    assert len(losses) == STEPS
    if os.environ.get("BLINK_WRITE_GOLDEN") == "1":
        GOLDEN.write_text(json.dumps(losses, indent=1) + "\n", encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert [row[0] for row in losses] == [row[0] for row in golden]
    np.testing.assert_allclose(np.array(losses)[:, 1:], np.array(golden)[:, 1:], rtol=1e-4, atol=0)
