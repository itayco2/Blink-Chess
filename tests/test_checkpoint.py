import pickle

import pytest

torch = pytest.importorskip("torch")

from blink.train import checkpoint  # noqa: E402
from blink.train.checkpoint import (  # noqa: E402
    checkpoint_name,
    latest_checkpoint,
    list_checkpoints,
    load_checkpoint,
    save_checkpoint,
    step_of,
)

pytestmark = pytest.mark.torch


class NotAllowed:
    """A class torch.load(weights_only=True) must refuse to rebuild."""


def test_checkpoint_names_are_zero_padded_to_nine_digits():
    assert checkpoint_name(42) == "ckpt_000000042.pt"
    assert step_of(checkpoint.Path("ckpt_000000042.pt")) == 42


def test_checkpoints_sort_numerically(tmp_path):
    steps = (1_000, 200, 30, 999_999_999, 1_000_000_000, 5)
    for step in steps:
        (tmp_path / checkpoint_name(step)).write_bytes(b"x")
    (tmp_path / (checkpoint_name(7) + ".tmp")).write_bytes(b"partial")
    (tmp_path / "ckpt_notanumber.pt").write_bytes(b"x")
    (tmp_path / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    assert [step_of(p) for p in list_checkpoints(tmp_path)] == sorted(steps)
    assert step_of(latest_checkpoint(tmp_path)) == 1_000_000_000


def test_an_empty_run_has_no_latest_checkpoint(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    assert latest_checkpoint(tmp_path / "missing") is None


def test_save_keeps_only_the_last_three(tmp_path):
    for step in (100, 200, 300, 400, 500):
        save_checkpoint(tmp_path, step, {"step": step}, keep_last=3)
    assert [step_of(p) for p in list_checkpoints(tmp_path)] == [300, 400, 500]


def test_a_crash_between_write_and_replace_leaves_the_previous_checkpoint_loadable(tmp_path, monkeypatch):
    save_checkpoint(tmp_path, 100, {"step": 100, "w": torch.ones(3)})

    def power_cut(src, dst):
        raise OSError("power cut between write and replace")

    monkeypatch.setattr(checkpoint.os, "replace", power_cut)
    with pytest.raises(OSError):
        save_checkpoint(tmp_path, 200, {"step": 200, "w": torch.zeros(3)})
    monkeypatch.undo()

    assert (tmp_path / (checkpoint_name(200) + ".tmp")).exists()
    latest = latest_checkpoint(tmp_path)
    assert step_of(latest) == 100
    state = load_checkpoint(latest)
    assert state["step"] == 100 and torch.equal(state["w"], torch.ones(3))

    save_checkpoint(tmp_path, 300, {"step": 300})
    assert not list(tmp_path.glob("*.tmp"))


def test_a_locked_target_is_retried_before_giving_up(tmp_path, monkeypatch):
    real_replace = checkpoint.os.replace
    failures = {"left": 2}

    def flaky(src, dst):
        if failures["left"]:
            failures["left"] -= 1
            raise PermissionError(32, "locked by a scanner")
        real_replace(src, dst)

    monkeypatch.setattr(checkpoint.os, "replace", flaky)
    monkeypatch.setattr(checkpoint, "RETRY_SLEEP_S", 0.0)
    path = save_checkpoint(tmp_path, 10, {"step": 10})
    assert load_checkpoint(path)["step"] == 10


def test_loading_refuses_arbitrary_pickled_objects(tmp_path):
    path = tmp_path / checkpoint_name(1)
    torch.save({"payload": NotAllowed()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)
