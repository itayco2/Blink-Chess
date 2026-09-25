"""tools/p7_v2_driver.py: P6 v2's flagship choreography (EVAL.md PR-2), driven against a fake machine.

Nothing here trains or starts a process: the fake host answers each blink command the way the real one
would and writes the files the real command leaves behind (checkpoints, the branch's final row, the
steps `train calibrate --write` sets). Step numbers are checked against exact fraction arithmetic.
"""

import importlib.util
import json
import math
import shutil
import sys
from fractions import Fraction
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("p7_v2_driver", REPO / "tools" / "p7_v2_driver.py")
driver = importlib.util.module_from_spec(SPEC)
sys.modules["p7_v2_driver"] = driver
SPEC.loader.exec_module(driver)

RATE = "2621.3"  # a true M rate (PR-5 expects about 2,621 samples/s at micro 256)
ARMS = {"a01": 0.55115, "a02": 0.5529, "a03": 0.5536}  # final full-valprobe EMA VAA of the seeds
VALPROBE = 20_000


def exact_steps(hours: float, rate: str, batch: int = 1024) -> int:
    return math.floor(Fraction(hours) * 3600 * Fraction(rate) / batch)


def exact_plan(rate: str) -> dict[str, int]:
    rung, total = exact_steps(6, rate), exact_steps(120, rate)
    cooldown = round(Fraction(rung) / 5)
    return {"long": total, "rung": rung, "cooldown": cooldown, "start": rung - cooldown}


# ---------------------------------------------------------------- the arithmetic


def _train() -> dict:
    return driver.train_table(REPO / "configs" / "long.toml")


def test_the_repo_config_chain_gives_the_schedule_the_plan_reads():
    train = _train()
    assert (train["batch_size"], train["cooldown_frac"], train["warmup_steps"]) == (1024, 0.2, 2000)
    assert train["vaa_reference"] == ""


def test_the_analysis_numbers_come_back_at_the_bench_rate():
    """47,301 is where a 59,126-step M run starts its cooldown; the branch is 11,825 steps."""
    plan = driver.make_plan(2803.05, _train(), 120.0, 6.0)
    assert (plan.rung_steps, plan.rung_cooldown, plan.rung_start) == (59_126, 11_825, 47_301)
    assert plan.rung_steps == exact_steps(6, "2803.05") and plan.long_steps == exact_steps(120, "2803.05")
    assert plan.first_check == 59_127 > plan.rung_start


def test_pr5_flagship_steps_match_the_analysis():
    """floor(120 x 3600 x R / 1024): 1,105,734 at the true 2,621/s, 1,182,515 at the bench's 2,803/s."""
    assert driver.steps_for(120.0, 2621.0, 1024) == 1_105_734
    assert driver.steps_for(120.0, 2803.0, 1024) == 1_182_515


@pytest.mark.parametrize("rate", ["1658.0", "2214.7", "2621.3", "2710.0", "2803.05", "9938.2", "10690.7"])
def test_the_plan_matches_exact_arithmetic_and_stays_inside_the_stable_phase(rate):
    plan = driver.make_plan(float(rate), _train(), 120.0, 6.0)
    exact = exact_plan(rate)
    assert (plan.long_steps, plan.rung_steps) == (exact["long"], exact["rung"])
    assert (plan.rung_cooldown, plan.rung_start) == (exact["cooldown"], exact["start"])
    assert plan.rung_start + plan.rung_cooldown == plan.rung_steps
    assert driver.cooldown_start(plan.rung_steps, 0.2) == plan.rung_start  # the branch's own schedule
    assert plan.rung_steps == plan.long_steps // 20  # 6 h is 5% of 120 h: the rung fits long.toml's steps
    assert 2000 < plan.rung_start < plan.first_check == math.floor(0.05 * plan.long_steps + 0.5)


def test_pr2_s_literal_rung_can_be_kept_whatever_the_true_rate():
    """At the true ~2,621/s the rung is 55,296 steps; --rung-steps 59126 keeps PR-2's 47,301 + 11,825."""
    derived = driver.make_plan(2621.44, _train(), 120.0, 6.0)
    assert (derived.rung_steps, derived.rung_start, derived.rung_cooldown) == (55_296, 44_237, 11_059)
    literal = driver.make_plan(2621.44, _train(), 120.0, 6.0, rung_steps=59_126)
    assert (literal.rung_steps, literal.rung_start, literal.rung_cooldown) == (59_126, 47_301, 11_825)
    assert literal.long_steps == derived.long_steps and literal.rung_start < literal.first_check


def test_the_driver_branches_at_pr2_s_literal_numbers_when_told_to(tmp_path):
    s = _settings(tmp_path, rung_steps=59_126)
    host = FakeHost(s)
    assert driver.drive(s, host) == driver.EXIT_DONE
    assert host.args("leg1")[-2:] == ["--max-steps", "47301"]
    branch = host.args("branch")
    assert (
        branch[branch.index("--from-step") + 1] == "47301"
        and branch[branch.index("--preview-steps") + 1] == "11825"
    )


def test_the_plan_refuses_a_rung_that_reaches_the_first_check_or_starts_in_the_warmup():
    with pytest.raises(ValueError, match="5% check"):
        driver.make_plan(2621.3, _train(), 120.0, 8.0)  # 0.8 x 8 h > 6 h: past the 5% check
    with pytest.raises(ValueError, match="warmup"):
        driver.make_plan(50.0, _train(), 120.0, 6.0)


def test_the_rate_is_read_from_the_calibration_output_with_its_printed_precision():
    assert driver.read_rate("R_true = 2,621.44 samples/s\nsteps = 1,105,920") == (2621.44, 0.005)
    assert driver.read_rate("R_true: 2621 samples/s") == (2621.0, 0.5)
    assert driver.read_rate("r_true 10 then\nR_true 2,700.5") == (2700.5, 0.05)  # the last one counts
    assert driver.read_rate("median 2,700 samples/s") is None


def test_calibrate_is_recognised_from_p7prep_s_help_and_not_from_main_s():
    assert driver.calibrate_supported(P7PREP_TRAIN_HELP)
    assert not driver.calibrate_supported(MAIN_TRAIN_HELP)
    assert driver.calibrate_supported("usage: blink train calibrate [-h] --config CONFIG [--write]")


def test_the_rate_is_read_from_p7prep_s_calibration_lines():
    out = (
        "R_true 2,621.44 samples/s: 4,587,520 samples in 1,750.1 s over 29 train-phase intervals\n"
        "steps = floor(120 x 3600 x 2,621.44 / 1024) = 1,105,921\n"
    )
    assert driver.read_rate(out) == (2621.44, 0.005)


def test_the_written_steps_must_agree_with_the_printed_rate():
    written = driver.steps_for(120.0, 2621.4437, 1024)  # what the command computed from the exact rate
    assert driver.steps_agree(2621.44, 0.005, 120.0, 1024, written)
    assert not driver.steps_agree(2621.44, 0.005, 120.0, 1024, driver.steps_for(120.0, 2622.0, 1024))


# ---------------------------------------------------------------- long.toml edits


def test_setting_the_reference_changes_that_one_value_and_keeps_comments_and_line_endings(tmp_path):
    path = tmp_path / "long.toml"
    shutil.copy(REPO / "configs" / "long.toml", path)
    before = driver.tomllib.loads(path.read_text(encoding="utf-8"))
    driver.set_train_string(path, "vaa_reference", "size-m")
    raw = path.read_bytes()
    after = driver.tomllib.loads(raw.decode("utf-8"))
    assert after == {**before, "train": {**before["train"], "vaa_reference": "size-m"}}
    assert b'vaa_reference = "size-m"' in raw and b"\r\n" not in raw
    line = next(x for x in raw.decode().splitlines() if x.startswith("vaa_reference"))
    assert "#" in line  # the comment after the value survives


def test_a_crlf_file_stays_crlf(tmp_path):
    path = tmp_path / "long.toml"
    path.write_bytes(b'base = "m.toml"\r\n\r\n[train]\r\nsteps = 10\r\nvaa_reference = ""\r\n')
    driver.set_train_string(path, "vaa_reference", "size-m")
    assert (
        path.read_bytes() == b'base = "m.toml"\r\n\r\n[train]\r\nsteps = 10\r\nvaa_reference = "size-m"\r\n'
    )


def test_a_file_without_the_key_is_refused_and_left_alone(tmp_path):
    path = tmp_path / "long.toml"
    path.write_text("[train]\nsteps = 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="vaa_reference"):
        driver.set_train_string(path, "vaa_reference", "size-m")
    assert path.read_text(encoding="utf-8") == "[train]\nsteps = 10\n"


# ---------------------------------------------------------------- the guard


def test_the_guard_compares_with_the_seeds_mean_at_2_sample_sigma():
    values = list(ARMS.values())
    mean = sum(values) / 3
    sigma = math.sqrt(sum((v - mean) ** 2 for v in values) / 2)
    ok = driver.guard_verdict(mean - 2 * sigma, values)
    assert (
        ok["passed"] and ok["delta"] == pytest.approx(-2 * sigma) and ok["sigma_ema"] == pytest.approx(sigma)
    )
    bad = driver.guard_verdict(mean - 2 * sigma - 1e-6, values)
    assert not bad["passed"] and bad["threshold"] == pytest.approx(-2 * sigma)
    assert driver.guard_verdict(0.60, values)["delta"] == pytest.approx(0.60 - mean)


def _run(runs: Path, name: str, steps: int, final: float, roots: int = VALPROBE) -> None:
    run = runs / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps({"config": {"steps": steps, "batch_size": 1024}}), "utf-8")
    rows = [
        {"step": steps - 376, "ema_vaa": 0.9, "vaa_set": "subset", "vaa_n": 2000},
        {"step": steps, "ema_vaa": final, "vaa_set": "full", "vaa_n": roots, "check": "100%"},
    ]
    (run / "evals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    for folder in ("eval", "logs", "runs", "ops"):
        (home / folder).mkdir(parents=True)
    arms = {}
    for name, vaa in ARMS.items():
        _run(home / "runs", f"abl-{name}", 56_376, vaa)
        arms[name] = {"run": f"abl-{name}", "status": "finished", "steps": 56_376}
    arms["a04"] = {"run": "abl-a04", "status": "running"}
    (home / "eval" / "ablations.json").write_text(json.dumps({"arms": arms}), encoding="utf-8")
    return home


def _settings(tmp_path: Path, **overrides) -> "driver.Settings":
    repo = tmp_path / "repo"
    (repo / "configs").mkdir(parents=True, exist_ok=True)
    for name in ("long.toml", "m.toml", "recipe.toml", "sweep.toml"):
        shutil.copy(REPO / "configs" / name, repo / "configs" / name)
    home = tmp_path / "home" if (tmp_path / "home").is_dir() else _home(tmp_path)
    values = {"repo": repo, "python": Path("python.exe"), "home": home, "data": home / "data" / "v1",
              "keeper": home / "ops" / driver.KEEPER}  # fmt: skip
    return driver.Settings(**{**values, **overrides})


def test_the_guard_reads_the_seeds_from_ablations_json_and_the_branch_s_final_row(tmp_path):
    s = _settings(tmp_path)
    _run(s.runs, "size-m", 55_293, 0.58)
    verdict = driver.guard(s)
    assert verdict["passed"] and verdict["branch"] == {"run": "size-m", "step": 55_293, "ema_vaa": 0.58}
    assert verdict["arms"]["a02"] == {"step": 56_376, "ema_vaa": 0.5529} and verdict["vaa_n"] == VALPROBE
    assert verdict["mean"] == pytest.approx(sum(ARMS.values()) / 3)


def test_the_guard_refuses_an_unfinished_seed_a_different_valprobe_or_a_branch_short_of_its_end(tmp_path):
    s = _settings(tmp_path)
    _run(s.runs, "size-m", 55_293, 0.58, roots=19_000)
    with pytest.raises(ValueError, match="different valprobes"):
        driver.guard(s)
    rows = (s.runs / "size-m" / "evals.jsonl").read_text(encoding="utf-8").splitlines()
    (s.runs / "size-m" / "evals.jsonl").write_text(rows[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="size-m has no full-valprobe EMA VAA at its last step 55,293"):
        driver.guard(s)
    with pytest.raises(ValueError, match="a04 has not finished"):
        driver.guard(driver.Settings(**{**s.__dict__, "arms": ("a01", "a04")}))


# ---------------------------------------------------------------- the whole choreography


# `python -m blink.cli train calibrate --help`, abridged, as p7prep (26e9649) and main (a043ee5) print it
P7PREP_TRAIN_HELP = """usage: blink train [-h] [--config CONFIG] [--run RUN]
                   [--data DATA | --source-raw SOURCE_RAW]
                   [--lr-scale LR_SCALE] [--preview-cooldown PREVIEW_COOLDOWN]
                   [--preview-steps PREVIEW_STEPS]
                   [--preview-name PREVIEW_NAME] [--from-step FROM_STEP]
                   [--device {cuda,cpu}] [--max-steps MAX_STEPS]
                   [--steps STEPS] [--write] [--from-run FROM_RUN]
                   [{calibrate}]

positional arguments:
  {calibrate}           calibrate: PR-5's calibration of a flagship config
                        (sets its steps with --write)
"""
MAIN_TRAIN_HELP = """usage: blink train [-h] [--config CONFIG] --run RUN
                   (--data DATA | --source-raw SOURCE_RAW)
                   [--lr-scale LR_SCALE] [--preview-cooldown PREVIEW_COOLDOWN]
                   [--from-step FROM_STEP] [--device {cuda,cpu}]
                   [--max-steps MAX_STEPS]
"""


class FakeHost:
    """Answers blink commands as the real ones would and writes what they leave behind."""

    def __init__(self, s, rate: str = RATE, **options) -> None:
        self.s, self.rate = s, rate
        self.options = {"calibrate": True, "busy": [], "n_star": "m", "branch_vaa": 0.585, "leg1": 0,
                        "branch": 0, "print_rate": True, "write_steps": None, "raise_in": None,
                        "command_lines": [], **options}  # fmt: skip
        self.log: list[tuple] = []  # every call and event, in order

    def reference(self) -> str:
        return driver.train_table(self.s.config_path)["vaa_reference"]

    def run(self, step: str, args: list[str]) -> tuple[int, str]:
        self.log.append(("run", step, list(args), self.reference()))
        if self.options["raise_in"] == step:
            raise OSError(f"disk gone in {step}")
        return getattr(self, "_" + step.replace("-", "_"))(args)

    def command_lines(self) -> list[str]:
        self.log.append(("ps",))
        return [*self.options["busy"], *self.options["command_lines"]]

    def stop_endgame_screen(self) -> None:
        self.log.append(("stop-screen",))

    def start_keeper(self, script) -> None:
        self.log.append(("keeper", str(script)))

    def steps(self) -> list[str]:
        return [entry[1] for entry in self.log if entry[0] == "run"]

    def args(self, step: str) -> list[str]:
        return next(entry[2] for entry in self.log if entry[0] == "run" and entry[1] == step)

    def _probe_branch(self, args):
        return 0, P7PREP_TRAIN_HELP

    def _probe_calibrate(self, args):
        return 0, P7PREP_TRAIN_HELP if self.options["calibrate"] else MAIN_TRAIN_HELP

    def _preflight_choose(self, args):
        return 0, f"rule prior\n  m: eligible\nN* = {self.options['n_star']} (P6 v2 prior)\n"

    _choose = _preflight_choose

    def _calibrate(self, args):
        steps = self.options["write_steps"] or exact_steps(120, self.rate)
        path = self.s.config_path
        text = path.read_text(encoding="utf-8")
        old = next(line for line in text.splitlines() if line.startswith("steps = "))
        path.write_text(text.replace(old, f"steps = {steps}"), encoding="utf-8")
        rate = float(self.rate)  # the lines p7prep's calibrate prints (blink.train.calibrate.describe)
        shown = f"R_true {rate:,.2f} samples/s: 4,587,520 samples in 1,750.1 s\n"
        formula = f"steps = floor(120 x 3600 x {rate:,.2f} / 1024) = {steps:,}\n"
        head = "calibrate: 2,000 steps of configs/long.toml as run calib-long-x (film off)\n"
        return 0, head + (shown if self.options["print_rate"] else "") + formula

    def _leg1(self, args):
        run = self.s.runs / "long"
        run.mkdir(parents=True, exist_ok=True)
        if self.options["leg1"] == 0:
            (run / driver.checkpoint_name(int(args[args.index("--max-steps") + 1]))).write_bytes(b"")
        return self.options["leg1"], ""

    def _branch(self, args):
        if self.options["branch"] == 0:
            total = int(args[args.index("--from-step") + 1]) + int(args[args.index("--preview-steps") + 1])
            _run(self.s.runs, "size-m", total, self.options["branch_vaa"])
        return self.options["branch"], ""

    def _launch(self, args):
        return 0, "launched p7-long: pid 4242 (cmd.exe), python [4243]\n  logs ...\n"


def _status(s) -> dict:
    return json.loads(s.path(".status.json").read_text(encoding="utf-8"))


def test_the_driver_runs_the_v2_sequence_and_hands_the_flagship_size_m_as_its_reference(tmp_path):
    s = _settings(tmp_path)
    host = FakeHost(s)
    assert driver.drive(s, host) == driver.EXIT_DONE
    expected = ["probe-branch", "preflight-choose", "probe-calibrate", "calibrate", "leg1", "branch",
                "choose", "launch"]  # fmt: skip
    assert host.steps() == expected
    plan = exact_plan(RATE)
    assert host.args("leg1") == [
        "supervise", "--bench-size", "m", "--", "train", "--config", "configs/long.toml", "--run", "long",
        "--data", str(s.data), "--max-steps", str(plan["start"]),
    ]  # fmt: skip
    assert host.args("branch") == [
        "supervise", "--bench-size", "m", "--", "train", "--run", "long", "--data", str(s.data),
        "--from-step", str(plan["start"]), "--preview-steps", str(plan["cooldown"]),
        "--preview-name", "size-m",
    ]  # fmt: skip
    assert host.args("launch") == [
        "ops", "launch", "--name", "p7-long", "--", "supervise", "--bench-size", "m", "--", "train",
        "--config", "configs/long.toml", "--run", "long", "--data", str(s.data), "--resume",
    ]  # fmt: skip
    assert host.args("calibrate") == [
        "train", "calibrate", "--config", "configs/long.toml", "--steps", "2000", "--write",
        "--data", str(s.data),
    ]  # fmt: skip
    references = {entry[1]: entry[3] for entry in host.log if entry[0] == "run"}
    assert references["leg1"] == references["branch"] == references["calibrate"] == ""
    assert references["launch"] == "size-m" and driver.train_table(s.config_path)["steps"] == plan["long"]
    assert (
        _status(s)["state"] == "done" and f"{plan['start']:,} + {plan['cooldown']:,}" in _status(s)["detail"]
    )
    guard = json.loads((s.home / "eval" / "size_guard.json").read_text(encoding="utf-8"))
    assert guard["passed"] and guard["branch"]["step"] == plan["rung"]


def test_the_endgame_screen_stops_before_the_calibration_and_the_keeper_restarts(tmp_path):
    s = _settings(tmp_path)
    host = FakeHost(s)
    driver.drive(s, host)
    kinds = [entry[1] if entry[0] == "run" else entry[0] for entry in host.log]
    assert kinds.index("stop-screen") < kinds.index("calibrate")
    assert kinds.count("keeper") == 2  # before leg 1 and after the relaunch (none was running)
    assert kinds.index("keeper") < kinds.index("calibrate") and kinds[-1] == "keeper"
    assert "relaunch it" in _status(s)["detail"]  # the screen stays stopped until the agent restarts it


def test_a_running_keeper_is_left_alone(tmp_path):
    s = _settings(tmp_path)
    keeper = r"powershell -NoProfile -File D:\blink\ops\keep_training_priority.ps1"
    host = FakeHost(s, command_lines=[keeper])
    assert driver.drive(s, host) == driver.EXIT_DONE
    assert not any(entry[0] == "keeper" for entry in host.log)


def test_without_train_calibrate_the_driver_fails_clearly_and_trains_nothing(tmp_path):
    s = _settings(tmp_path)
    host = FakeHost(s, calibrate=False)
    assert driver.drive(s, host) == driver.EXIT_FAILED
    status = _status(s)
    assert status["state"] == "failed" and status["step"] == "calibrate"
    assert "blink train calibrate" in status["detail"] and "--rate" in status["detail"]
    assert "leg1" not in host.steps() and "calibrate" not in host.steps()
    assert not any(entry[0] == "stop-screen" for entry in host.log)


def test_a_given_rate_skips_the_calibration(tmp_path):
    s = _settings(tmp_path, rate=2621.3, rate_eps=0.05)
    text = s.config_path.read_text(encoding="utf-8")
    old = next(line for line in text.splitlines() if line.startswith("steps = "))
    s.config_path.write_text(text.replace(old, f"steps = {exact_steps(120, RATE)}"), encoding="utf-8")
    host = FakeHost(s, calibrate=False)
    assert driver.drive(s, host) == driver.EXIT_DONE
    assert "calibrate" not in host.steps() and "probe-calibrate" not in host.steps()
    assert (
        not any(entry[0] == "stop-screen" for entry in host.log) and "relaunch it" not in _status(s)["detail"]
    )


def test_a_rounded_printed_rate_is_pinned_to_the_written_steps():
    """The command computes steps from its exact rate and prints it to 2 decimals: the plan follows the
    written steps, and the rung is the exact rate's."""
    written = exact_steps(120, "2621.4437")
    rate = driver.pinned_rate(2621.44, written, 120.0, 1024)
    assert driver.steps_for(120.0, rate, 1024) == written and abs(rate - 2621.44) < 0.005
    plan = driver.make_plan(rate, _train(), 120.0, 6.0)
    assert plan.long_steps == written and plan.rung_steps == exact_steps(6, "2621.4437") == written // 20
    assert driver.pinned_rate(2621.4437, written, 120.0, 1024) == 2621.4437  # already on them: unchanged


def test_the_plan_follows_long_toml_s_steps_when_the_printed_rate_is_rounded(tmp_path):
    s = _settings(tmp_path)
    exact = "2621.4437"
    host = FakeHost(s, rate=exact)  # prints 2,621.44 and writes the exact rate's steps
    assert driver.drive(s, host) == driver.EXIT_DONE
    plan = json.loads(s.path(".state.json").read_text(encoding="utf-8"))["plan"]
    assert plan["long_steps"] == exact_steps(120, exact) == driver.train_table(s.config_path)["steps"]
    assert host.args("leg1")[-1] == str(exact_plan(exact)["start"])


def test_a_busy_gpu_or_a_v1_sweep_toml_refuses_before_anything_runs(tmp_path):
    s = _settings(tmp_path)
    busy = FakeHost(s, busy=["python.exe -m blink.cli sweep ablations"])
    assert driver.drive(s, busy) == driver.EXIT_FAILED
    assert busy.steps() == [] and "holds the GPU" in _status(s)["detail"]
    toml = s.repo / s.sweep_config
    toml.write_text(
        toml.read_text(encoding="utf-8").replace('rule = "prior"', 'rule = "vaa"'), encoding="utf-8"
    )
    v1 = FakeHost(s)
    assert driver.drive(s, v1) == driver.EXIT_FAILED
    assert v1.steps() == [] and "not merged" in _status(s)["detail"]


def test_an_n_star_other_than_m_stops_before_training(tmp_path):
    s = _settings(tmp_path)
    host = FakeHost(s, n_star="s")
    assert driver.drive(s, host) == driver.EXIT_FAILED
    assert host.steps() == ["probe-branch", "preflight-choose"] and "N* = s" in _status(s)["detail"]


def test_a_reference_set_before_the_branch_exists_is_refused(tmp_path):
    s = _settings(tmp_path)
    driver.set_train_string(s.config_path, "vaa_reference", "size-m")
    host = FakeHost(s)
    assert driver.drive(s, host) == driver.EXIT_FAILED
    assert "calibrate" not in host.steps() and "before the branch exists" in _status(s)["detail"]


def test_a_failed_guard_pauses_the_flagship_at_the_rung_s_start_and_sets_gate_p7_vaa(tmp_path):
    s = _settings(tmp_path)
    host = FakeHost(s, branch_vaa=0.545)  # the seeds' mean is 0.5526 and 2 sigma about 0.0025
    assert driver.drive(s, host) == driver.EXIT_PAUSED
    assert host.steps()[-1] == "branch" and "choose" not in host.steps()
    beat = json.loads((s.runs / "long" / "heartbeat.json").read_text(encoding="utf-8"))
    assert beat["state"] == "paused" and beat["stopped"] == "paused: P7-VAA"
    status = _status(s)
    assert status["state"] == "paused" and f"step {exact_plan(RATE)['start']:,}" in status["detail"]
    assert driver.train_table(s.config_path)["vaa_reference"] == ""
    assert json.loads((s.home / "eval" / "size_guard.json").read_text(encoding="utf-8"))["passed"] is False


def test_a_rerun_after_a_reboot_in_leg_1_resumes_it_without_calibrating_again(tmp_path):
    s = _settings(tmp_path)
    first = FakeHost(s, leg1=1)  # supervise gave up (say a reboot killed it)
    assert driver.drive(s, first) == driver.EXIT_FAILED and _status(s)["step"] == "leg1"
    (s.runs / "long" / driver.checkpoint_name(20_000)).write_bytes(b"")
    again = FakeHost(s)
    assert driver.drive(s, again) == driver.EXIT_DONE
    assert "calibrate" not in again.steps() and again.args("leg1")[-1] == "--resume"


def test_checkpoints_without_a_recorded_rate_are_never_recalibrated(tmp_path):
    s = _settings(tmp_path)
    (s.runs / "long").mkdir()
    (s.runs / "long" / driver.checkpoint_name(9_000)).write_bytes(b"")
    host = FakeHost(s)
    assert driver.drive(s, host) == driver.EXIT_FAILED
    assert "calibrate" not in host.steps() and "pass --rate" in _status(s)["detail"]


def test_a_crashed_branch_is_resumed_and_a_branch_of_another_plan_is_refused(tmp_path):
    s = _settings(tmp_path)
    assert driver.drive(s, FakeHost(s, branch=1)) == driver.EXIT_FAILED
    plan = exact_plan(RATE)
    (s.runs / "size-m").mkdir(exist_ok=True)
    (s.runs / "size-m" / driver.checkpoint_name(plan["start"] + 500)).write_bytes(b"")
    again = FakeHost(s)
    assert driver.drive(s, again) == driver.EXIT_DONE and again.args("branch")[-1] == "--resume"
    assert "leg1" not in again.steps()  # leg 1 already reached the rung's start
    other = _settings(tmp_path / "other")
    _run(other.runs, "size-m", 59_126, 0.58)
    (other.runs / "long").mkdir()
    (other.runs / "long" / driver.checkpoint_name(plan["start"])).write_bytes(b"")
    driver.save_state(other, {"rate": 2621.3, "rate_eps": 0.05})
    host = FakeHost(other)
    host._calibrate([])  # long.toml's steps as the recorded calibration wrote them
    assert driver.drive(other, host) == driver.EXIT_FAILED
    assert "branched for 59,126 steps" in _status(other)["detail"]


def test_a_calibration_without_r_true_or_with_other_steps_fails_clearly(tmp_path):
    s = _settings(tmp_path)
    assert driver.drive(s, FakeHost(s, print_rate=False)) == driver.EXIT_FAILED
    assert "no R_true" in _status(s)["detail"]
    other = _settings(tmp_path / "other")
    host = FakeHost(other, write_steps=exact_steps(120, "2803.05"))
    assert driver.drive(other, host) == driver.EXIT_FAILED
    assert _status(other)["step"] == "calibrate" and "is not floor(120" in _status(other)["detail"]
    assert "leg1" not in host.steps()
    assert "rate" not in driver.load_state(other)  # nothing half-done is trusted: a rerun calibrates again
    again = FakeHost(other)
    assert driver.drive(other, again) == driver.EXIT_DONE and "calibrate" in again.steps()


def test_a_launched_flagship_makes_a_rerun_a_no_op(tmp_path):
    s = _settings(tmp_path)
    assert driver.drive(s, FakeHost(s)) == driver.EXIT_DONE
    done = _status(s)
    assert driver.load_state(s)["launched"].startswith("launched p7-long: pid 4242")
    again = FakeHost(s)
    assert driver.drive(s, again) == driver.EXIT_DONE
    assert again.log == [] and _status(s) == done


def test_an_unexpected_error_is_recorded_in_the_status_and_raised(tmp_path):
    s = _settings(tmp_path)
    with pytest.raises(OSError, match="disk gone"):
        driver.drive(s, FakeHost(s, raise_in="leg1"))
    status = _status(s)
    assert status["state"] == "failed" and status["step"] == "leg1" and "disk gone" in status["detail"]


def test_busy_commands_are_the_ones_that_train():
    lines = [
        r"C:\py\python.exe -m blink.cli supervise --bench-size m -- train --run long",
        r"C:\py\python.exe -m blink.cli eval endgames --sf-procs 4",
        r"C:\py\python.exe C:\dev\blink-run\tools\p7_v2_driver.py",
        r"C:\py\python.exe -m blink.cli sweep ablations",
    ]
    assert driver.busy_commands(lines) == [lines[0], lines[3]]


def test_the_dry_run_prints_the_plan_and_the_commands(tmp_path, capsys):
    s = _settings(tmp_path)
    argv = ["--dry-run", "--rate", "2803.05", "--repo", str(s.repo), "--home", str(s.home)]
    assert driver.main(argv) == driver.EXIT_DONE
    out = capsys.readouterr().out
    assert "59,126" in out and "47,301" in out and "11,825" in out and "--preview-name size-m" in out
    assert not s.path(".status.json").exists()
