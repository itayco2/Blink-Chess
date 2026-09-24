import itertools
import sys
import types

import numpy as np
import pytest

from blink.data.record import CHILD_DTYPE, ROOT_DTYPE
from blink.model.config import TrainConfig
from blink.train.source import InMemorySource, as_step_data, mixed_source, shard_source


def _records(n: int) -> np.ndarray:
    records = np.zeros(n, dtype=ROOT_DTYPE)
    records["fen_hash"] = np.arange(n)
    return records


def _hashes(batches) -> list[list[int]]:
    return [batch["fen_hash"].tolist() for batch in batches]


def test_seeking_to_a_step_equals_skipping_to_it():
    source = InMemorySource(_records(40), batch_size=8, seed=5)
    straight = _hashes(itertools.islice(source.batches(0), 23))
    assert _hashes(itertools.islice(source.batches(17), 6)) == straight[17:]


def test_an_epoch_visits_every_record_once_in_a_new_order_each_epoch():
    source = InMemorySource(_records(40), batch_size=8, seed=5)
    first, second = (list(itertools.islice(source.batches(start), 5)) for start in (0, 5))
    assert sorted(np.concatenate(first)["fen_hash"].tolist()) == list(range(40))
    assert sorted(np.concatenate(second)["fen_hash"].tolist()) == list(range(40))
    assert _hashes(first) != _hashes(second)


def test_every_batch_is_full_and_of_root_dtype():
    batches = list(itertools.islice(InMemorySource(_records(21), batch_size=8, seed=0).batches(0), 7))
    assert all(len(b) == 8 and b.dtype == ROOT_DTYPE for b in batches)


def test_a_batch_larger_than_the_data_is_refused():
    with pytest.raises(ValueError):
        InMemorySource(_records(4), batch_size=8, seed=0)


class _FakeShardLoader:
    """Records the arguments and yields batches whose fen_hash is the batch number in the stream."""

    calls: list[dict] = []

    def __init__(self, paths, batch_size, seed, loop=True, start_batch=0, **kwargs):
        self.batch_size, self.start_batch = batch_size, start_batch
        self.dtype = kwargs.get("dtype", ROOT_DTYPE)
        _FakeShardLoader.calls.append(
            {"paths": paths, "batch_size": batch_size, "seed": seed, "loop": loop, **kwargs}
        )

    def __iter__(self):
        for step in itertools.count(self.start_batch):
            batch = np.zeros(self.batch_size, dtype=self.dtype)
            batch["fen_hash"] = step
            yield batch


@pytest.fixture
def fake_loader(monkeypatch):
    _FakeShardLoader.calls = []
    fake = types.ModuleType("blink.data.loader")
    fake.ShardLoader = _FakeShardLoader
    monkeypatch.setitem(sys.modules, "blink.data.loader", fake)
    return _FakeShardLoader


def test_the_shard_source_starts_the_loader_at_the_resume_step(fake_loader, tmp_path):
    paths = [tmp_path / "train_000.bin"]
    batches = list(itertools.islice(shard_source(paths, batch_size=4, seed=9)(3), 2))
    assert [int(b["fen_hash"][0]) for b in batches] == [3, 4]
    assert fake_loader.calls == [{"paths": paths, "batch_size": 4, "seed": 9, "loop": True}]


def test_a_child_shard_source_asks_the_loader_for_child_records(fake_loader, tmp_path):
    paths = [tmp_path / "train_c000.bin"]
    batch = next(shard_source(paths, batch_size=3, seed=1, dtype=CHILD_DTYPE)(7))
    assert batch.dtype == CHILD_DTYPE and int(batch["fen_hash"][0]) == 7
    assert fake_loader.calls[0]["dtype"] == CHILD_DTYPE


def _children(n: int) -> np.ndarray:
    children = np.zeros(n, dtype=CHILD_DTYPE)
    children["fen_hash"] = np.arange(n) + 1000
    return children


def test_a_mixed_step_holds_717_roots_and_307_children_at_batch_1024():
    cfg = TrainConfig(batch_size=1024, child_frac=0.3)
    roots = InMemorySource(_records(4000), cfg.roots_per_step, seed=1).batches
    children = InMemorySource(_children(4000), cfg.children_per_step, seed=2).batches
    step = next(mixed_source(roots, children)(0))
    assert (len(step.roots), len(step.children)) == (717, 307)
    assert step.roots.dtype == ROOT_DTYPE and step.children.dtype == CHILD_DTYPE


def test_the_mixed_source_resumes_at_the_same_roots_children_and_weights():
    roots = InMemorySource(_records(70), 7, seed=1).batches
    children = InMemorySource(_children(30), 3, seed=2).batches

    def weigher(records):
        return (records["fen_hash"] % 5).astype(np.float32) + 0.5

    source = mixed_source(roots, children, weigher)
    straight = list(itertools.islice(source(0), 12))
    resumed = list(itertools.islice(source(9), 3))
    for a, b in zip(straight[9:], resumed, strict=True):
        assert np.array_equal(a.roots, b.roots) and np.array_equal(a.children, b.children)
        assert np.array_equal(a.root_weight, b.root_weight) and np.array_equal(a.child_weight, b.child_weight)
    assert np.array_equal(straight[0].root_weight, weigher(straight[0].roots))


def test_a_root_only_source_has_no_children_and_unit_weights():
    step = as_step_data(_records(5))
    assert len(step.children) == 0 and step.children.dtype == CHILD_DTYPE
    assert step.root_weight is None and step.child_weight is None
    assert as_step_data(step) is step
