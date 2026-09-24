import pytest

torch = pytest.importorskip("torch")

from blink.train import film  # noqa: E402

pytestmark = pytest.mark.torch


def test_the_film_plan_is_step_0_nineteen_geometric_ema_steps_and_the_final_weights():
    plan = film.frame_plan(100_000)
    assert len(plan) == 21
    assert plan[0] == (0, "init") and plan[-1] == (100_000, "final")
    ema = [step for step, kind in plan if kind == "ema"]
    assert ema == [round(250 * (100_000 / 250) ** (k / 19)) for k in range(19)]
    assert ema[0] == 250 and ema == sorted(set(ema)) and ema[-1] < 100_000


def test_a_short_run_drops_duplicate_and_late_frame_steps():
    plan = film.frame_plan(260)
    steps = [step for step, _ in plan]
    assert steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == 260
    assert all(0 < step < 260 for step, kind in plan if kind == "ema")


def _frame(tmp_path, step: int, world: str = "0123456789ab") -> None:
    film.save_frame(tmp_path, step, kind="ema", weights={"w": torch.zeros(1)}, world=world, config={})


def test_film_frames_sort_numerically_and_share_one_world(tmp_path):
    for step in (100_000, 9, 0, 1_000_000_000, 10, 250):
        _frame(tmp_path, step)
    frames = film.list_frames(tmp_path)
    assert [film.step_of(p) for p in frames] == [0, 9, 10, 250, 100_000, 1_000_000_000]
    assert all(p.parent == tmp_path / "film" for p in frames)
    assert film.frames_world(frames) == "0123456789ab"

    _frame(tmp_path, 11, world="ffffffffffff")
    with pytest.raises(ValueError, match="2 worlds"):
        film.frames_world(film.list_frames(tmp_path))


def test_a_frame_is_a_slim_weights_file_the_model_loader_accepts(tmp_path):
    from blink.model.config import TrainConfig, config_to_dict
    from blink.model.loading import load_model
    from blink.model.transformer import BlinkNet

    cfg = TrainConfig()
    model = BlinkNet(cfg.model)
    path = film.save_frame(
        tmp_path, 250, kind="ema", weights=model.state_dict(), world="w", config=config_to_dict(cfg)
    )
    loaded = load_model(str(path), device="cpu")
    for name, tensor in model.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], tensor)
    state = torch.load(path, weights_only=True)
    assert (state["step"], state["kind"], state["world"]) == (250, "ema", "w")


def test_frames_after_a_resume_point_are_removed(tmp_path):
    for step in (0, 250, 300, 400):
        _frame(tmp_path, step)
    film.drop_after(tmp_path, 300)
    assert [film.step_of(p) for p in film.list_frames(tmp_path)] == [0, 250, 300]
