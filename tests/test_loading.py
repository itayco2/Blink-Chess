import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train_helpers import fixture_records, tiny_train_config  # noqa: E402

from blink.board.encode import unpack  # noqa: E402
from blink.model import loading  # noqa: E402
from blink.model.config import config_from_dict  # noqa: E402
from blink.model.transformer import BlinkNet  # noqa: E402
from blink.train import loop  # noqa: E402
from blink.train.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    """A 30-step run under its own BLINK_HOME, shared by the tests in this module."""
    home = tmp_path_factory.mktemp("home")
    records = fixture_records()
    cfg = tiny_train_config(steps=30, warmup_steps=3, batch_size=16, ckpt_every_steps=15)

    def source(start_step):
        while True:
            yield records[:16]

    spec = loop.RunSpec(run_dir=home / "runs" / "tiny", world="0123456789ab", device="cpu")
    loop.train(cfg, spec, source, val=None, log=lambda _: None)
    return home


@pytest.fixture
def home(trained_run, monkeypatch):
    monkeypatch.setenv("BLINK_HOME", str(trained_run))
    return trained_run


def _codes() -> np.ndarray:
    return unpack(fixture_records()["board"][:4])


def _reference(weights: str, home) -> np.ndarray:
    state = load_checkpoint(latest_checkpoint(home / "runs" / "tiny"))
    model = BlinkNet(config_from_dict(state["config"]).model)
    model.load_state_dict(state[weights])
    with torch.no_grad():
        return model(torch.from_numpy(_codes().astype(np.int64)))[0].numpy()


def test_a_run_selector_loads_the_latest_raw_weights(home):
    evaluator = loading.load_evaluator("run:tiny", device="cpu")
    np.testing.assert_allclose(
        evaluator.evaluate(_codes()).policy_logits, _reference("model", home), atol=1e-6
    )


def test_the_ema_suffix_loads_the_ema_weights(home):
    ema = loading.load_evaluator("run:tiny:ema", device="cpu").evaluate(_codes()).policy_logits
    np.testing.assert_allclose(ema, _reference("ema", home), atol=1e-6)
    assert not np.allclose(ema, _reference("model", home), atol=1e-6)


def test_a_path_selector_loads_that_checkpoint(home):
    path = latest_checkpoint(home / "runs" / "tiny")
    evaluator = loading.load_evaluator(str(path), device="cpu")
    np.testing.assert_allclose(
        evaluator.evaluate(_codes()).policy_logits, _reference("model", home), atol=1e-6
    )


def test_a_run_selector_pinned_to_its_checkpoint_loads_the_same_weights(home):
    """`blink eval puzzles` resolves run:<name>:ema once and loads that file (pinned_selector)."""
    path, which = loading.resolve_selector("run:tiny:ema")
    pinned = loading.pinned_selector(path, which)
    assert loading.resolve_selector(pinned) == (path, "ema") and pinned.endswith(".pt:ema")
    assert loading.resolve_selector(loading.pinned_selector(path, "model")) == (path, "model")
    ema = loading.load_evaluator(pinned, device="cpu").evaluate(_codes()).policy_logits
    np.testing.assert_allclose(ema, _reference("ema", home), atol=1e-6)


def test_ship_and_release_selectors_resolve_under_blink_home(home):
    source = latest_checkpoint(home / "runs" / "tiny")
    ship = home / "ship" / loading.WEIGHTS_FILE
    release = home / "ship" / "releases" / "model-v1" / loading.WEIGHTS_FILE
    for target in (ship, release):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    assert loading.resolve_selector("ship") == (ship, "model")
    assert loading.resolve_selector("release:model-v1") == (release, "model")
    expected = _reference("model", home)
    for selector in ("ship", "release:model-v1"):
        got = loading.load_evaluator(selector, device="cpu").evaluate(_codes()).policy_logits
        np.testing.assert_allclose(got, expected, atol=1e-6)


def test_a_slim_weights_file_with_only_a_model_config_loads(home, tmp_path):
    state = load_checkpoint(latest_checkpoint(home / "runs" / "tiny"))
    slim = tmp_path / "slim.pt"
    torch.save({"config": state["config"]["model"], "model": state["ema"]}, slim)
    got = loading.load_evaluator(str(slim), device="cpu").evaluate(_codes()).policy_logits
    np.testing.assert_allclose(got, _reference("ema", home), atol=1e-6)


def test_load_model_returns_an_eval_mode_blinknet(home):
    model = loading.load_model("run:tiny:ema", device="cpu")
    assert isinstance(model, BlinkNet) and not model.training


@pytest.mark.parametrize(
    "selector",
    ["run:", "run:../x", "run:tiny:raw", "release:../../x", "nonsense", "weights.bin", "run:a:ema:b"],
)
def test_malformed_selectors_are_refused(home, selector):
    with pytest.raises(ValueError):
        loading.resolve_selector(selector)


def test_a_run_without_checkpoints_is_a_clear_error(home):
    with pytest.raises(FileNotFoundError, match="missing"):
        loading.load_evaluator("run:missing", device="cpu")
    with pytest.raises(FileNotFoundError):
        loading.load_evaluator(str(home / "nope.pt"), device="cpu")
