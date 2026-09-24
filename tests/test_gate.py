import pytest

from blink.checks import CheckResult, exit_code, format_results


def test_gate_prints_ok_fail_skip_and_names_the_fix():
    results = [
        CheckResult("python", "ok", "3.12.10"),
        CheckResult("torch", "FAIL", "2.14.0+cpu", fix="uv sync --extra train --extra compile"),
        CheckResult("stockfish", "skip", "not downloaded yet"),
    ]
    lines = format_results(results).splitlines()
    assert lines[0].startswith("ok ")
    assert lines[1].startswith("FAIL")
    assert "fix: uv sync --extra train --extra compile" in lines[1]
    assert lines[2].startswith("skip")
    assert exit_code(results) == 1


def test_gate_exits_zero_when_nothing_failed():
    results = [CheckResult("python", "ok", "3.12.10"), CheckResult("vram", "WARN", "4.9 GB free")]
    assert exit_code(results) == 0


def test_a_fail_without_a_fix_is_rejected():
    with pytest.raises(ValueError, match="fix"):
        CheckResult("torch", "FAIL", "broken")


def test_a_partial_download_is_a_skip(tmp_path):
    from blink.gate import check_file

    part = tmp_path / "db.zst"
    part.write_bytes(b"x" * 10)
    assert check_file("eval DB", part, 100, "download in progress").status == "skip"


def test_an_oversized_download_fails_because_it_can_only_be_corrupt(tmp_path):
    """PF42: a zombie writer grew files past their planned size; that must never pass as 'in progress'."""
    from blink.gate import check_file

    big = tmp_path / "db.zst"
    big.write_bytes(b"x" * 101)
    result = check_file("eval DB", big, 100, "download in progress")
    assert result.status == "FAIL"
    assert "re-download" in result.fix


def test_an_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="status"):
        CheckResult("torch", "maybe", "?")
