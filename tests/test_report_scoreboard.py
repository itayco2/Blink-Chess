"""The README scoreboard: made from results/*.json only, spliced between two markers, checked byte-exact."""

from dataclasses import replace

import pytest
from test_report_fixtures import FIXTURE_DIR, diagnostics_rows, lichess, results, strength_rows, write_bundle

from blink.report import scoreboard as sb


def _block(tmp_path, **kwargs) -> str:
    return sb.render_block(sb.load_bundle(write_bundle(tmp_path / "results", **kwargs)))


def _row_line(block: str, agent: str) -> str:
    return next(line for line in block.splitlines() if line.startswith(f"| {agent} |"))


def _table_rows(block: str, heading: str) -> list[str]:
    lines = block.splitlines()
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("### "):
            break
        if line.startswith("| ") and not line.startswith("|---"):
            rows.append(line)
    return rows[1:]


def test_the_block_holds_the_headline_table_the_no_search_box_both_tables_and_the_band_table_in_order(
    tmp_path,
):
    block = _block(tmp_path)
    marks = [
        "| headline number |",
        "**No search.**",
        sb.TABLE1_HEADING,
        sb.TABLE2_HEADING,
        sb.BANDS_HEADING,
    ]
    positions = [block.index(mark) for mark in marks]
    assert positions == sorted(positions)
    assert block.startswith("| headline number |") and block.endswith("\n")


def test_every_elo_cell_shows_its_interval_and_game_count(tmp_path):
    block = _block(tmp_path)
    assert "1850 +/- 35 (4,100 games)" in _row_line(block, "**Blink-M (value)**")
    assert "1702 +/- 38 (4,100 games)" in _row_line(block, "Blink-M (policy)")
    assert "1790 +/- 30 (1,000 games)" in _row_line(block, "DM-9M")
    for row in _table_rows(block, sb.TABLE1_HEADING):
        elo_cell = row.split(" | ")[6]
        assert elo_cell == "-" or ("+/-" in elo_cell and "games)" in elo_cell)


def test_paper_numbers_appear_only_in_the_paper_reported_column(tmp_path):
    block = _block(tmp_path)
    cells = _row_line(block, "DM-270M").strip("| ").split(" | ")
    header = (
        next(line for line in block.splitlines() if line.startswith("| agent | params"))
        .strip("| ")
        .split(" | ")
    )
    paper = header.index("paper-reported (scale named)")
    assert cells[paper] == "2895 Lichess blitz vs humans (paper, 2024)"
    assert all("2895" not in cell for i, cell in enumerate(cells) if i != paper)


def test_ratings_below_1320_are_labelled_extrapolated(tmp_path):
    block = _block(tmp_path)
    assert "512 +/- 61 (1,000 games), extrapolated" in _row_line(block, "random")
    assert "1104 +/- 44 (1,000 games), extrapolated" in _row_line(block, "MLP")
    assert "extrapolated" not in _row_line(block, "DM-9M")


def test_the_shipped_row_is_bold_and_carries_the_lichess_rating(tmp_path):
    block = _block(tmp_path)
    line = _row_line(block, "**Blink-M (value)**")
    assert "1950 +/- 124 (2 RD), 231 games, 2026-10-11" in line
    assert "1950" not in _row_line(block, "Blink-M (policy)")


def test_lichess_cells_say_accruing_until_the_rating_is_publishable(tmp_path):
    block = _block(tmp_path, lichess_obj=lichess(n=40, rd=120))
    assert "rating accruing (40 games, RD 120)" in block
    assert "2400" not in block and "+/- 240" not in block


def test_the_no_search_box_counts_public_moves_and_violations(tmp_path):
    block = _block(tmp_path)
    assert (
        "**No search.** Across 123,456 public moves in 2,051 games: at most 1 network call and legal+1 "
        "positions each (0 violations; largest batch 38 rows)." in block
    )
    assert "812 with 0 rows (R2 mate now)" in block


def test_the_headline_table_names_its_pool_and_its_caveats(tmp_path):
    block = _block(tmp_path)
    assert "| 1850 +/- 35 (4,100 games) |" in block
    assert sb.ELO_CAVEAT in block
    assert "120.3 flagship / 181.2 total GPU-h, 39.8 GPU-board kWh" in block
    assert "80.1% (79.3 to 80.9)" in block


def test_the_diagnostics_and_band_tables_show_fractions_as_percent(tmp_path):
    block = _block(tmp_path)
    row = _table_rows(block, sb.TABLE2_HEADING)[0]
    assert row.startswith("| **Blink-M** | **value** | 51.2 / 78.1 / 87.4% | 60.3% | 77.1% | 0.412 | 0.121 |")
    assert "1905 (1880 to 1931)" in row
    bands = _table_rows(block, sb.BANDS_HEADING)
    assert bands[0] == "| **Blink-M** | **value** | 97.2% | 91.0% | 80.3% | 62.4% | 41.0% |"


def test_a_fraction_column_holding_a_percent_is_refused(tmp_path):
    rows = (replace(diagnostics_rows()[0], top1=51.2), *diagnostics_rows()[1:])
    with pytest.raises(sb.ScoreboardError, match="top1"):
        _block(tmp_path, results_obj=results(diagnostics=rows))


def test_the_generated_block_is_ascii_only(tmp_path):
    assert _block(tmp_path).isascii()


def test_the_scoreboard_refuses_without_its_results_files(tmp_path):
    folder = write_bundle(tmp_path / "r", skip=("results.json",))
    with pytest.raises(sb.ScoreboardError, match="results.json"):
        sb.load_bundle(folder)
    folder = write_bundle(tmp_path / "s", skip=("compute.json",))
    with pytest.raises(sb.ScoreboardError, match="report compute"):
        sb.load_bundle(folder)
    assert sb.load_bundle(write_bundle(tmp_path / "t", skip=("lichess.json",))).lichess is None


def test_a_shipped_agent_matches_its_strength_row_with_or_without_the_mode_suffix(tmp_path):
    bare = tuple(replace(r, agent="Blink-M") if r.agent == "Blink-M (value)" else r for r in strength_rows())
    block = _block(tmp_path, results_obj=results(strength=bare))
    assert "| **Blink-M** |" in block


README = "# Title\n\nStory.\n\n## Scoreboard\n\n{start}\nold\n{end}\n\n## Run the tests\n"


def _readme(tmp_path, body: str = "old\n") -> str:
    path = tmp_path / "README.md"
    path.write_text(README.format(start=sb.START, end=sb.END).replace("old\n", body), encoding="utf-8")
    return path


def test_write_replaces_only_the_text_between_the_markers(tmp_path):
    path = _readme(tmp_path)
    sb.write_readme(path, "new block\n")
    text = path.read_text(encoding="utf-8")
    assert text == README.format(start=sb.START, end=sb.END).replace("old\n", "new block\n")
    assert sb.check_readme(path, "new block\n") == []


def test_check_fails_when_the_readme_block_differs_by_one_byte(tmp_path):
    path = _readme(tmp_path, "new block \n")
    problems = sb.check_readme(path, "new block\n")
    assert problems and "differs" in problems[0]


def test_check_fails_when_the_markers_are_missing_or_repeated(tmp_path):
    path = tmp_path / "README.md"
    path.write_text("# Title\n", encoding="utf-8")
    assert "marker" in sb.check_readme(path, "x\n")[0]
    path.write_text(f"{sb.START}\n{sb.END}\n{sb.START}\n{sb.END}\n", encoding="utf-8")
    assert "marker" in sb.check_readme(path, "x\n")[0]
    with pytest.raises(sb.ScoreboardError, match="marker"):
        sb.write_readme(path, "x\n")


def test_blink_report_scoreboard_write_then_check_round_trips_and_detects_drift(tmp_path, capsys):
    from blink import cli

    path = _readme(tmp_path)
    args = ["report", "scoreboard", "--results", str(FIXTURE_DIR), "--readme", str(path)]
    assert cli.main([*args, "--check"]) == 1
    assert cli.main([*args, "--write"]) == 0
    assert cli.main([*args, "--check"]) == 0
    path.write_text(path.read_text(encoding="utf-8").replace("1850", "1851"), encoding="utf-8")
    assert cli.main([*args, "--check"]) == 1
    assert "differs" in capsys.readouterr().err


def test_blink_report_scoreboard_without_a_flag_prints_the_block(tmp_path, capsys):
    from blink import cli

    assert cli.main(["report", "scoreboard", "--results", str(FIXTURE_DIR)]) == 0
    assert capsys.readouterr().out.startswith("| headline number |")


def test_a_row_without_a_non_gab_count_shows_its_total_alone(tmp_path):
    assert "| MLP | 0.53M |" in _block(tmp_path)


def test_numbers_in_prose_skip_code_links_comments_and_small_counts():
    text = (
        "## 3. The loop\n\nIt solved 80.1% of 10,000 puzzles in 21 frames, seen 573.4M positions, "
        "`--games 100` [paper](https://arxiv.org/abs/2402.04494) <!-- 777 --> and step 3.\n"
    )
    assert sb.numbers_in(text) == ["80.1%", "10,000", "21", "573.4M"]


def test_a_prose_number_is_measured_when_it_rounds_scales_or_percents_a_results_value(tmp_path):
    folder = write_bundle(tmp_path / "results")
    values = sb.measured_values(folder)
    assert sb.is_measured("80.1%", values)  # dm_puzzles_pct
    assert sb.is_measured("51.2%", values)  # top1 = 0.512, as a percent
    assert sb.is_measured("573.4M", values)  # positions_seen
    assert sb.is_measured("4,100", values)  # elo_games
    assert sb.is_measured("0.41", values)  # kendall_tau_b 0.412, rounded
    assert not sb.is_measured("81.0%", values)
    assert not sb.is_measured("2895", values)  # paper text is not a measurement


def test_markers_must_each_sit_on_their_own_line(tmp_path):
    path = tmp_path / "README.md"
    path.write_text(f"# T\n{sb.START}{sb.END}\n", encoding="utf-8")
    assert "own line" in sb.check_readme(path, "x\n")[0]
    path.write_text(f"# T\n{sb.END}\n{sb.START}\n", encoding="utf-8")
    assert "own line" in sb.check_readme(path, "x\n")[0]


def test_a_results_file_off_its_schema_is_a_scoreboard_error(tmp_path):
    folder = write_bundle(tmp_path / "a")
    (folder / "nosearch.json").write_text('{"games": 1}', encoding="utf-8")
    with pytest.raises(sb.ScoreboardError, match="nosearch.json lacks"):
        sb.load_bundle(folder)
    (folder / "results.json").write_text('{"schema_version": 7}', encoding="utf-8")
    with pytest.raises(sb.ScoreboardError, match="schema"):
        sb.load_bundle(folder)
