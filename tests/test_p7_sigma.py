"""configs/long.toml's vaa_sigma is sigma_EMA (tools/p7_guard.py), and the P6 v2 driver checks it.

The flagship's 25% and 50% checks compare against 2 x vaa_sigma; PR-2's guard uses 2 sigma_EMA, the
sample sd of D's seeds' 100% check EMA VAA. Both must be the same number, so the driver refuses to start
unless long.toml's value is the guard's to its printed precision, and records both in size_guard.json.
"""

import json
import statistics
from pathlib import Path

import pytest
from test_p7_v2_driver import ARMS, PR2, REPO, FakeHost, _settings, _status, driver

REAL_HOME = Path(r"D:\blink")
SIGMA = statistics.stdev(ARMS.values())  # 0.0016023...: a01 0.55115, a02 0.5529, a03 0.5497


def test_the_repo_s_long_toml_sets_vaa_sigma_to_sigma_ema_with_its_source():
    literal = driver.train_literal(REPO / "configs" / "long.toml", "vaa_sigma")
    value, eps = driver.parse_number(literal)
    assert abs(value - SIGMA) <= eps and eps <= 5e-8  # 0.0016023, to 7 decimals
    line = next(
        x for x in (REPO / "configs" / "long.toml").read_text("utf-8").splitlines() if "vaa_sigma =" in x
    )
    assert "sigma_EMA" in line and "a01-a03" in line.replace("a01, a02, a03", "a01-a03")


def test_the_real_seeds_give_that_sigma(tmp_path):
    rows = driver.guard_verdict(0.56, list(ARMS.values()))
    assert rows["sigma_ema"] == pytest.approx(0.0016023420358961945, rel=1e-12)


def test_a_vaa_sigma_other_than_sigma_ema_refuses_before_anything_runs(tmp_path):
    s = _settings(tmp_path)
    text = s.config_path.read_text(encoding="utf-8")
    old = next(line for line in text.splitlines() if line.startswith("vaa_sigma ="))
    s.config_path.write_text(text.replace(old, "vaa_sigma = 0.01"), encoding="utf-8")
    host = FakeHost(s)
    assert driver.drive(s, host) == driver.EXIT_FAILED
    status = _status(s)
    assert status["step"] == "preflight" and host.steps() == []
    assert "vaa_sigma is 0.01, not sigma_EMA 0.0016023" in status["detail"]
    assert "set vaa_sigma = 0.0016023" in status["detail"] and "a03 0.5497" in status["detail"]


def test_a_value_rounded_to_its_printed_digits_is_accepted(tmp_path):
    s = _settings(tmp_path)
    text = s.config_path.read_text(encoding="utf-8")
    old = next(line for line in text.splitlines() if line.startswith("vaa_sigma ="))
    s.config_path.write_text(text.replace(old, "vaa_sigma = 0.0016"), encoding="utf-8")  # +- 0.00005
    assert driver.check_sigma(s)["long_toml"] == 0.0016
    s.config_path.write_text(text.replace(old, "vaa_sigma = 0.001600"), encoding="utf-8")  # 0.001602 is
    with pytest.raises(driver.StepFailed, match="not sigma_EMA"):
        driver.check_sigma(s)


@pytest.mark.parametrize("literal", ["0", "0.0", "0.00", "0.002", "-0.0016023", "0.0016023e0"])
def test_a_zero_negative_or_coarsely_printed_vaa_sigma_is_refused(tmp_path, literal):
    """0 would fail the flagship's 25% and 50% checks on any noise dip (ema_vaa >= previous - 2 sigma), and
    '0.002' is sigma_EMA only to one digit: the literal must be positive and printed to within 5% of it."""
    s = _settings(tmp_path)
    text = s.config_path.read_text(encoding="utf-8")
    old = next(line for line in text.splitlines() if line.startswith("vaa_sigma ="))
    s.config_path.write_text(text.replace(old, f"vaa_sigma = {literal}"), encoding="utf-8")
    if literal == "0.0016023e0":  # the same value in another spelling, printed as precisely: accepted
        assert driver.check_sigma(s)["long_toml"] == pytest.approx(SIGMA, abs=5e-8)
        return
    with pytest.raises(driver.StepFailed, match="not sigma_EMA"):
        driver.check_sigma(s)


def test_unfinished_seeds_refuse_the_check_rather_than_skip_it(tmp_path):
    s = _settings(tmp_path)
    path = s.home / "eval" / "ablations.json"
    arms = json.loads(path.read_text(encoding="utf-8"))
    arms["arms"]["a03"]["status"] = "running"
    path.write_text(json.dumps(arms), encoding="utf-8")
    with pytest.raises(driver.StepFailed, match="a03 has not finished"):
        driver.check_sigma(s)


def test_size_guard_json_records_long_toml_s_vaa_sigma_beside_sigma_ema(tmp_path):
    s = _settings(tmp_path)
    assert driver.drive(s, FakeHost(s)) == driver.EXIT_DONE
    guard = json.loads((s.home / "eval" / "size_guard.json").read_text(encoding="utf-8"))
    assert guard["vaa_sigma"]["sigma_ema"] == pytest.approx(SIGMA) == guard["sigma_ema"]
    assert guard["vaa_sigma"]["long_toml"] == pytest.approx(SIGMA, abs=5e-8)
    assert guard["vaa_sigma"]["seeds"] == ARMS and guard["branch"]["step"] == PR2["rung"]


@pytest.mark.local
@pytest.mark.skipif(
    not (REAL_HOME / "eval" / "ablations.json").is_file(), reason="needs D:/blink (read only)"
)
def test_the_repo_s_vaa_sigma_matches_the_seeds_in_the_real_ablations_json():
    """Read only: ablations.json names the seeds' runs, whose evals.jsonl hold the 100% check rows."""
    s = driver.Settings(repo=REPO, python=Path("python.exe"), home=REAL_HOME, data=REAL_HOME / "data" / "v1")
    record = driver.check_sigma(s)
    assert record["seeds"] == ARMS and record["sigma_ema"] == pytest.approx(SIGMA)
