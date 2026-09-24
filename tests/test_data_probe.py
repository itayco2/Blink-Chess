"""The probe: what N frames of the eval DB look like before anything is packed."""

import json

import orjson
import pytest
from data_fakes import BAD_LINES, fixture_lines, lines_text, synthetic_lines, write_pzstd

from blink.data import probe


def _line(fen: str, pvs: list[dict], depth: int = 30) -> bytes:
    return orjson.dumps({"fen": fen, "evals": [{"pvs": pvs, "depth": depth}]})


CASTLING_LINES = [
    _line("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", [{"cp": 20, "line": "e1h1 e8a8"}]),
    _line("r3k2r/8/8/8/8/8/8/R3K2R b KQkq -", [{"cp": -20, "line": "e8a8 e1h1"}, {"cp": 0, "line": "a8b8"}]),
]
MATE_LINE = _line("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - -", [{"mate": 1, "line": "d1d8"}], depth=12)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    lines = fixture_lines() + synthetic_lines(600, seed=5) + CASTLING_LINES + [MATE_LINE] + BAD_LINES
    path = tmp_path_factory.mktemp("probe") / "db.jsonl.zst"
    write_pzstd(path, lines_text(lines), frame_bytes=30_000)
    return path, len(lines)


def test_probe_counts_lines_rejects_and_legal_best_moves(source):
    path, n_lines = source
    report = probe.probe(path, frames=None, workers=1, check_every=3)
    assert report["lines"] == n_lines
    assert report["rejects"] == {"bad_row": 2, "no_evals": 1, "illegal_best_move": 1}
    assert report["errors"] == {}
    assert report["parsed"] == n_lines - 4
    assert report["best_move_legal_pct"] == pytest.approx(100 * report["parsed"] / (report["parsed"] + 1))
    check = report["sample_check"]
    assert check["checked"] >= report["parsed"] // 3 - 20
    assert check["legal_pct"] == check["canonical_uci_pct"] == 100.0


def test_probe_histograms_and_shares_cover_every_parsed_line(source):
    path, _ = source
    report = probe.probe(path, frames=None, workers=1, check_every=3)
    parsed = report["parsed"]
    assert sum(report["npv_hist"].values()) == sum(report["depth_hist"].values()) == parsed
    assert report["side_to_move"]["white"] + report["side_to_move"]["black"] == pytest.approx(1.0)
    assert 0 < report["mate_share"] < 1
    assert report["shallow_nonmate_share"] >= 0
    assert report["frames"] > 3 and report["end"] == "eof"


def test_probe_counts_castling_best_moves_written_king_takes_rook():
    counts = probe.probe_lines(CASTLING_LINES + [MATE_LINE], check_every=1)
    assert counts.castling == 2
    assert counts.checked == counts.legal == counts.canonical == 3
    assert counts.mate == 1 and counts.white == 2 and counts.black == 1
    assert counts.alt_dropped == 0


def test_merging_counts_adds_every_field():
    a = probe.probe_lines(CASTLING_LINES, check_every=1)
    b = probe.probe_lines([MATE_LINE, BAD_LINES[0]], check_every=1)
    both = probe.merge(a, b)
    assert both.lines == 4 and both.parsed == 3 and both.castling == 2 and both.mate == 1
    assert both.rejects == {"bad_row": 1}
    assert sum(both.npv.values()) == 3


def test_an_unexpected_parser_exception_is_reported_as_an_error(monkeypatch):
    def explode(line):
        raise KeyError("surprise")

    monkeypatch.setattr(probe.parse, "parse_line", explode)
    counts = probe.probe_lines([MATE_LINE], check_every=1)
    assert counts.errors == {"KeyError": 1} and counts.error_samples


def test_probe_report_is_written_as_json_and_summarised(tmp_path, source):
    path, _ = source
    out = tmp_path / "probe.json"
    report = probe.probe(path, frames=2, workers=1)
    probe.write_report(report, out)
    assert json.loads(out.read_text(encoding="utf-8")) == report
    text = probe.summary(report)
    assert "lines" in text and "legal" in text and "lines/s" in text
    assert report["frames"] == 2 and report["end"] == "limit"
