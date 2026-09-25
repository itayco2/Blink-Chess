"""tools/p7_v2_driver.py under PR-6: the PC is Itay's by day, so the driver waits out the Pause Blink flag.

No step starts while BLINK_HOME/PAUSE is up; a calibration with a user pause inside is never used (a
fresh one runs once the flag goes); and the relaunched flagship counts as verified only once it trains
past the rung start, with the minutes it spends paused by the user never counted toward the 10.
"""

import json

import pytest
from test_p7_v2_driver import PR2, FakeHost, _settings, _status, driver

# isort: split
import p7_machine  # tools/: on sys.path once test_p7_v2_driver is imported


def _flag(s):
    return s.home / "PAUSE"


def _lift_after(sleeps: int, seen: list):
    """An on_sleep hook: note the driver's status while it waits, remove the flag after `sleeps` sleeps."""

    def hook(host):
        seen.append(_status(host.s))
        if sum(1 for entry in host.log if entry[0] == "sleep") >= sleeps:
            _flag(host.s).unlink(missing_ok=True)

    return hook


def test_the_constants_are_blink_s_own():
    from blink.train import supervise, userpause

    assert p7_machine.PAUSED_USER == userpause.PAUSED_USER and p7_machine.PAUSE_FLAG == userpause.FLAG_NAME
    assert p7_machine.EXIT_USER_PAUSE == supervise.EXIT_USER_PAUSE
    assert p7_machine.RESUMER_ENV == userpause.RESUMER_ENV


def test_nothing_starts_while_the_pause_flag_is_up(tmp_path):
    s, seen = _settings(tmp_path), []
    _flag(s).write_text("paused", encoding="utf-8")
    host = FakeHost(s, on_sleep=_lift_after(3, seen))
    assert driver.drive(s, host) == driver.EXIT_DONE
    first_run = next(i for i, entry in enumerate(host.log) if entry[0] in ("run", "ps"))
    assert [entry[0] for entry in host.log[:first_run]] == ["sleep"] * 3
    assert seen[0]["state"] == "paused: user" and seen[0]["step"] == "preflight"
    assert "Resume Blink" in seen[0]["detail"]


def test_a_flag_raised_between_steps_holds_the_next_one(tmp_path):
    s, seen = _settings(tmp_path), []

    def pause_after_leg1(host, step):
        if step == "leg1":
            _flag(host.s).write_text("paused", encoding="utf-8")

    host = FakeHost(s, on_run=pause_after_leg1, on_sleep=_lift_after(2, seen))
    assert driver.drive(s, host) == driver.EXIT_DONE
    order = [entry[1] if entry[0] == "run" else entry[0] for entry in host.log]
    assert order.index("leg1") < order.index("sleep") < order.index("branch")
    assert {status["step"] for status in seen} == {"branch"}


def test_a_calibration_paused_by_the_user_is_never_used_and_a_fresh_one_runs(tmp_path):
    """`blink train calibrate` exits 75 when the Pause button stops it; the driver waits for Resume."""
    s, seen = _settings(tmp_path), []

    def pause_first_calibration(host, step):
        if step == "calibrate" and len(host.options["calibrate_codes"]) == 1:
            _flag(host.s).write_text("paused", encoding="utf-8")

    host = FakeHost(s, calibrate_codes=[75, 0], on_run=pause_first_calibration, on_sleep=_lift_after(4, seen))
    assert driver.drive(s, host) == driver.EXIT_DONE
    assert host.steps().count("calibrate") == 2
    assert all(status["step"] == "calibrate" and status["state"] == "paused: user" for status in seen)
    assert "never used" in seen[0]["detail"] or "Resume Blink" in seen[0]["detail"]
    assert driver.load_state(s)["rate"] == pytest.approx(2621.3)


def test_the_driver_tells_only_its_calibration_that_it_reruns_one_the_user_paused(tmp_path):
    """`blink train calibrate` stops for the Pause button (freeing the GPU) only when its caller says it
    reruns a paused calibration (userpause.RESUMER_ENV); run by hand it trains on through the flag."""
    s = _settings(tmp_path)
    host = FakeHost(s, calibrate_codes=[75, 0])
    assert driver.drive(s, host) == driver.EXIT_DONE
    envs = dict(host.envs)
    assert [env for step, env in host.envs if step == "calibrate"] == [{p7_machine.RESUMER_ENV: "1"}] * 2
    assert all(env == {} for step, env in envs.items() if step != "calibrate")


class RowsHost(FakeHost):
    """A calibration that exits 0 but whose run's rows came from two trainer sessions (an older
    calibrate that did not stop for the flag)."""

    def _calibrate(self, args):
        if not self.options.get("rows_written"):
            self.options["rows_written"] = True
            run = self.s.runs / "calib-long-old"
            run.mkdir(parents=True)
            rows = [{"step": 600, "session": 1.0}, {"step": 650, "session": 2.0}]
            (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return 0, "calibrate: 2,000 steps of configs/long.toml as run calib-long-old (film off)\n"
        return super()._calibrate(args)


def test_a_calibration_whose_rows_show_a_pause_is_rerun_as_a_fresh_run(tmp_path):
    s = _settings(tmp_path)
    host = RowsHost(s)
    assert driver.drive(s, host) == driver.EXIT_DONE
    assert host.steps().count("calibrate") == 2


def test_a_calibration_refused_for_a_slow_window_fails_and_is_not_rerun(tmp_path):
    s = _settings(tmp_path)
    run = s.runs / "calib-long-0"
    run.mkdir(parents=True)
    refused = {"written": False, "refused": {"kind": "slow interval", "detail": "steps 1,200-1,250 ..."}}
    (run / "calibration.json").write_text(json.dumps(refused), encoding="utf-8")
    host = FakeHost(s, calibrate_codes=[2])
    assert driver.drive(s, host) == driver.EXIT_FAILED
    assert host.steps().count("calibrate") == 1 and _status(s)["step"] == "calibrate"
    assert _status(s)["detail"] == "exit 2: steps 1,200-1,250 ..."  # calibration.json's reason


def test_done_only_once_the_resumed_flagship_trains_past_the_rung_start(tmp_path):
    s = _settings(tmp_path)
    assert driver.drive(s, FakeHost(s)) == driver.EXIT_DONE
    status = _status(s)
    assert status["state"] == "done" and f"running at step {PR2['start'] + 150:,}" in status["detail"]


def test_a_relaunch_that_never_trains_fails_after_ten_unpaused_minutes_with_the_supervisor_state(tmp_path):
    s = _settings(tmp_path)
    host = FakeHost(s, resumed_step=PR2["start"])  # running, but still at the step leg 1 stopped at
    assert driver.drive(s, host) == driver.EXIT_FAILED
    status = _status(s)
    assert status["step"] == "verify" and "supervisor running" in status["detail"]
    assert f"did not train past step {PR2['start']:,} in 10 minutes" in status["detail"]
    polls = [entry for entry in host.log if entry[0] == "sleep"]
    assert len(polls) == driver.VERIFY_S / driver.VERIFY_POLL_S
    assert driver.load_state(s)["launched"]  # launched all the same: a rerun must not launch it twice


def test_a_user_pause_during_the_verification_is_alive_and_never_counted(tmp_path):
    """Itay pauses right after the relaunch and resumes 20 minutes later: still verified, not failed."""
    s = _settings(tmp_path)

    def resume_after_20_minutes(host):
        if sum(1 for entry in host.log if entry[0] == "sleep") == 120:
            beat = {"state": "running", "step": PR2["start"] + 40, "time": host.now}
            (host.s.runs / "long" / "heartbeat.json").write_text(json.dumps(beat), encoding="utf-8")

    host = FakeHost(s, resumed="paused: user", on_sleep=resume_after_20_minutes)
    assert driver.drive(s, host) == driver.EXIT_DONE
    assert f"running at step {PR2['start'] + 40:,}" in _status(s)["detail"]


def test_a_relaunched_supervisor_that_stopped_fails_the_driver_at_once(tmp_path):
    s = _settings(tmp_path)
    host = FakeHost(s, resumed="stopped")
    assert driver.drive(s, host) == driver.EXIT_FAILED
    assert "supervisor is stopped" in _status(s)["detail"] and not any(e[0] == "sleep" for e in host.log)


def test_leg_one_s_old_records_never_verify_the_relaunch(tmp_path):
    """Leg 1 left heartbeat.json and supervisor.json behind ("finished"): older than the launch, ignored."""
    s = _settings(tmp_path)

    class Stale(FakeHost):
        def _launch(self, args):
            code, out = super()._launch(args)
            run = self.s.runs / "long"
            old = {"state": "finished", "status": "finished", "started": self.now - 3600}
            (run / "supervisor.json").write_text(json.dumps(old), encoding="utf-8")
            return code, out

    assert driver.drive(s, Stale(s)) == driver.EXIT_DONE  # the fresh heartbeat verifies it
