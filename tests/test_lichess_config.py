"""The Lichess bot configs: every lookup off (N3), the plan's rated and casual settings, no token (P9)."""

import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath

import pytest

from blink import cli, uci
from blink.lichess import config as botconfig
from blink.lichess import config_check
from blink.report import results_schema

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "deploy" / "lichess" / "config.template.yml"
CASUAL_TEMPLATE = ROOT / "deploy" / "lichess" / "config.casual.yml"
SHA = hashlib.sha256(b"blink weights").hexdigest()


def load_template(path: Path = TEMPLATE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def console_scripts() -> dict[str, str]:
    with open(ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


def lichess_bot_flags(engine_options: dict) -> list[str]:
    """lichess-bot appends each engine_options entry to the engine command as --key=value."""
    return [f"--{key}={value}" for key, value in engine_options.items()]


def fake_weights(tmp_path: Path) -> Path:
    path = tmp_path / "blink.pt"
    path.write_bytes(b"blink weights")
    return path


def generate(tmp_path: Path, **overrides) -> dict[str, dict]:
    weights = fake_weights(tmp_path)
    spec = botconfig.BotSpec(
        model=overrides.pop("model", "ship"),
        mode=overrides.pop("mode", "value"),
        sha=overrides.pop("sha", None),
        casual_model=overrides.pop("casual_model", "run:long:ema"),
    )
    written = botconfig.generate(spec, tmp_path / "bot", resolve=lambda selector: weights, **overrides)
    return {kind: json.loads(path.read_text(encoding="utf-8")) for kind, path in written.items()}


def all_configs(tmp_path: Path) -> list[dict]:
    generated = generate(tmp_path)
    return [load_template(), load_template(CASUAL_TEMPLATE), generated["rated"], generated["casual"]]


def test_the_bot_engine_is_a_console_script_the_project_installs():
    engine = load_template()["engine"]
    script = PurePosixPath(engine["name"])
    assert script.suffix == ".exe"
    assert PurePosixPath(engine["dir"]).parts[-2:] == (".venv", "Scripts")
    target = console_scripts().get(script.stem)
    assert target is not None, f"pyproject.toml [project.scripts] has no {script.stem!r}"
    module_name, _, function_name = target.partition(":")
    assert getattr(importlib.import_module(module_name), function_name) is uci.main


def test_the_bot_engine_options_are_flags_blink_uci_accepts(tmp_path):
    for config in all_configs(tmp_path):
        engine = config["engine"]
        args = uci.build_parser().parse_args(lichess_bot_flags(engine["engine_options"]))
        assert args.mode in ("policy", "value") and args.device in ("cpu", "cuda")
        assert args.log is not None and args.random is False


def test_lichess_config_disables_every_lookup(tmp_path):
    for config in all_configs(tmp_path):
        assert config_check.lookup_problems(config) == []
        engine = config["engine"]
        assert engine["polyglot"]["enabled"] is False
        assert all(engine["online_moves"][s]["enabled"] is False for s in config_check.ONLINE_SOURCES)
        assert all(engine["lichess_bot_tbs"][t]["enabled"] is False for t in config_check.LOCAL_TABLEBASES)
        assert engine["draw_or_resign"]["resign_for_egtb_minus_two"] is False
        assert engine["draw_or_resign"]["offer_draw_for_egtb_zero"] is False
        for section in ("engine", "correspondence"):
            assert (config[section]["ponder"], config[section]["uci_ponder"]) == (False, False)
        assert not any("syzygy" in option.lower() for option in engine.get("uci_options", {}))


@pytest.mark.parametrize(
    ("path", "value", "problem"),
    [
        (("engine", "polyglot", "enabled"), True, "engine.polyglot.enabled"),
        (
            ("engine", "online_moves", "online_egtb", "enabled"),
            True,
            "engine.online_moves.online_egtb.enabled",
        ),
        (("engine", "lichess_bot_tbs", "gaviota", "enabled"), True, "engine.lichess_bot_tbs.gaviota.enabled"),
        (
            ("engine", "draw_or_resign", "resign_for_egtb_minus_two"),
            True,
            "engine.draw_or_resign.resign_for_egtb_minus_two",
        ),
        (("correspondence", "uci_ponder"), True, "correspondence.uci_ponder"),
        (("engine", "uci_options", "SyzygyPath"), "./syzygy/", "engine.uci_options.SyzygyPath"),
    ],
)
def test_each_lookup_switched_on_is_named(path, value, problem):
    config = copy.deepcopy(load_template())
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert problem in config_check.lookup_problems(config)


def test_a_missing_switch_counts_as_on_because_lichess_bot_defaults_some_to_true():
    config = copy.deepcopy(load_template())
    del config["engine"]["draw_or_resign"]["offer_draw_for_egtb_zero"]
    assert config_check.lookup_problems(config) == ["engine.draw_or_resign.offer_draw_for_egtb_zero"]


def test_the_template_never_holds_a_token():
    for path in (TEMPLATE, CASUAL_TEMPLATE):
        assert "token" not in load_template(path)
        assert "lip_" not in path.read_text(encoding="utf-8")


def test_lichess_config_sets_abort_time_30_and_puts_concurrency_under_challenge(tmp_path):
    rated = generate(tmp_path)["rated"]
    assert rated["abort_time"] == 30  # lichess-bot's code default for a missing key is 20
    assert "concurrency" not in rated and "games_reserved_for_humans" not in rated
    challenge = rated["challenge"]
    assert (challenge["concurrency"], challenge["games_reserved_for_humans"]) == (2, 1)
    assert (challenge["preference"], challenge["bullet_requires_increment"]) == ("human", True)
    assert challenge["max_simultaneous_games_per_user"] == 1
    broken = copy.deepcopy(rated)
    broken["concurrency"] = broken["challenge"].pop("concurrency")
    found = botconfig.problems(broken, "rated", frozenset())
    assert any("challenge.concurrency" in p for p in found)
    assert any("top level" in p for p in found)


def test_the_rated_config_matchmakes_rated_blitz_only_with_the_plan_settings(tmp_path):
    rated = generate(tmp_path)["rated"]
    match = rated["matchmaking"]
    assert match["allow_matchmaking"] is True and match["challenge_mode"] == "rated"
    assert (match["challenge_initial_time"], match["challenge_increment"]) == ([180, 300], [0, 2, 3])
    assert (match["challenge_timeout"], match["opponent_rating_difference"]) == (2, 300)
    assert match["challenge_filter"] == "fine"
    assert (rated["challenge"]["time_controls"], rated["challenge"]["modes"]) == (["blitz"], ["rated"])
    assert rated["engine"]["engine_options"]["device"] == "cuda"
    draw_or_resign = rated["engine"]["draw_or_resign"]
    assert (draw_or_resign["resign_enabled"], draw_or_resign["offer_draw_enabled"]) == (False, False)
    assert rated["pgn_directory"] == "D:/blink/lichess/pgn"
    # every base and increment pair estimates (base + 40 x increment) to between 180 and 420 s: blitz
    assert all(
        180 <= b + 40 * i <= 479
        for b in match["challenge_initial_time"]
        for i in match["challenge_increment"]
    )


def test_the_casual_config_allows_only_itayco2_with_matchmaking_off_on_cpu(tmp_path):
    casual = generate(tmp_path)["casual"]
    assert casual["challenge"]["allow_list"] == ["itayco2"]
    assert casual["challenge"]["modes"] == ["casual"]
    assert casual["matchmaking"]["allow_matchmaking"] is False
    options = casual["engine"]["engine_options"]
    assert (options["model"], options["device"]) == ("run:long:ema", "cpu")
    assert casual["abort_time"] == 30
    assert botconfig.problems(casual, "casual", frozenset()) == []


def test_config_passes_only_declared_uci_options(tmp_path):
    declared = botconfig.declared_uci_options()
    for config in all_configs(tmp_path):
        assert set(config["engine"].get("uci_options", {})) <= declared
    with_defaults = copy.deepcopy(generate(tmp_path)["rated"])
    with_defaults["engine"]["uci_options"] = {"Move Overhead": 100, "Threads": 4, "Hash": 512}
    found = botconfig.problems(with_defaults, "rated", declared)
    assert {p for p in found if "uci_options" in p} == {
        "engine.uci_options.Move Overhead is not an option blink-uci declares",
        "engine.uci_options.Threads is not an option blink-uci declares",
        "engine.uci_options.Hash is not an option blink-uci declares",
    }


def test_declared_uci_options_are_read_from_a_uci_handshake():
    lines = ["id name X", "option name Move Overhead type spin default 10 min 0 max 5000", "uciok"]
    assert botconfig.parse_uci_options(lines) == frozenset({"Move Overhead"})
    assert botconfig.declared_uci_options() == frozenset()  # blink-uci declares no options today


def test_bot_config_points_at_the_shipped_sha_and_mode(tmp_path):
    configs = generate(tmp_path, model=str(tmp_path / "blink.pt"), mode="value")
    rated, casual = configs["rated"], configs["casual"]
    shipped = results_schema.Shipped(agent="Blink-M", mode="value", sha=SHA)
    assert rated["blink"]["sha"] == shipped.sha
    assert rated["engine"]["engine_options"]["mode"] == shipped.mode
    assert rated["engine"]["engine_options"]["model"] == rated["blink"]["model"] == str(tmp_path / "blink.pt")
    assert botconfig.problems(rated, "rated", frozenset(), shipped=shipped, weights_sha=SHA) == []
    other_mode = results_schema.Shipped(agent="Blink-M", mode="policy", sha=SHA)
    assert any("mode" in p for p in botconfig.problems(rated, "rated", frozenset(), shipped=other_mode))
    other_sha = results_schema.Shipped(agent="Blink-M", mode="value", sha="0" * 64)
    assert any("sha" in p for p in botconfig.problems(rated, "rated", frozenset(), shipped=other_sha))
    swapped_weights = botconfig.problems(rated, "rated", frozenset(), weights_sha="f" * 64)
    assert any("sha256" in p for p in swapped_weights)
    # the G5 casual smoke runs the preview model on CPU during P7, so it is exempt
    assert casual["engine"]["engine_options"]["model"] == "run:long:ema"
    assert botconfig.problems(casual, "casual", frozenset(), shipped=other_sha) == []


def test_generate_refuses_a_sha_that_does_not_match_the_weights_file(tmp_path):
    with pytest.raises(botconfig.ConfigError, match="does not match"):
        generate(tmp_path, sha="0" * 12)
    assert generate(tmp_path, sha=SHA[:12])["rated"]["blink"]["sha"] == SHA


def test_generate_needs_a_sha_when_the_weights_file_cannot_be_found(tmp_path):
    spec = botconfig.BotSpec(model="dm:9M", mode="value")
    with pytest.raises(botconfig.ConfigError, match="--sha"):
        botconfig.generate(spec, tmp_path, resolve=lambda selector: None)
    written = botconfig.generate(spec, tmp_path, kinds=("casual",), resolve=lambda selector: None)
    assert set(written) == {"casual"} and not (tmp_path / "config.yml").exists()


@pytest.mark.parametrize(
    ("kind", "path", "value", "problem"),
    [
        ("rated", ("abort_time",), 20, "abort_time"),
        ("rated", ("challenge", "games_reserved_for_humans"), 0, "challenge.games_reserved_for_humans"),
        ("rated", ("matchmaking", "challenge_mode"), "random", "matchmaking.challenge_mode"),
        ("rated", ("challenge", "time_controls"), ["bullet", "blitz"], "challenge.time_controls"),
        ("rated", ("engine", "engine_options", "device"), "cpu", "engine.engine_options.device"),
        ("casual", ("challenge", "allow_list"), [], "challenge.allow_list"),
        ("casual", ("matchmaking", "allow_matchmaking"), True, "matchmaking.allow_matchmaking"),
        ("casual", ("challenge", "concurrency"), True, "challenge.concurrency"),  # true is not 1
    ],
)
def test_check_config_names_each_setting_a_config_breaks(tmp_path, kind, path, value, problem):
    config = copy.deepcopy(generate(tmp_path)[kind])
    node = config
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    assert any(p.startswith(problem) for p in botconfig.problems(config, kind, frozenset()))


def test_engine_options_that_blink_uci_would_reject_are_named(tmp_path):
    config = copy.deepcopy(generate(tmp_path)["rated"])
    config["engine"]["engine_options"]["random"] = None
    config["engine"]["engine_options"]["mode"] = "both"
    found = botconfig.problems(config, "rated", frozenset())
    assert any("random" in p for p in found) and any("blink-uci" in p for p in found)


def test_a_token_anywhere_in_a_config_is_refused(tmp_path):
    config = copy.deepcopy(generate(tmp_path)["rated"])
    config["token"] = "lip_" + "A" * 20
    config["greeting"] = {"hello": "lio_" + "B" * 32}
    found = botconfig.problems(config, "rated", frozenset())
    assert any(p.startswith("token") for p in found)
    assert sum("token" in p for p in found) >= 2


def test_generated_configs_are_ascii_json_that_yaml_reads_the_same(tmp_path):
    weights = fake_weights(tmp_path)
    spec = botconfig.BotSpec(model="ship", mode="policy")
    written = botconfig.generate(spec, tmp_path / "bot", resolve=lambda selector: weights)
    for path in written.values():
        text = path.read_text(encoding="utf-8")
        assert text.isascii() and "\t" not in text and "token" not in json.loads(text)
        assert "lip_" not in text and "lio_" not in text
        assert json.loads(text) == botconfig.load_config(path)


def test_the_cli_generates_both_configs_and_check_config_passes_them(monkeypatch, tmp_path, capsys):
    weights = fake_weights(tmp_path)
    monkeypatch.setattr(botconfig, "weights_file", lambda selector: weights)  # torch-free too
    monkeypatch.setattr(botconfig, "engine_present", lambda path: True)  # no bot install on CI
    out = tmp_path / "bot"
    argv = ["lichess", "config", "--model", str(weights), "--mode", "policy", "--sha", SHA]
    assert cli.main([*argv, "--casual-model", "run:preview:ema", "--out-dir", str(out)]) == 0
    assert {p.name for p in out.iterdir()} == {"config.yml", "config.casual.yml"}
    for name in ("config.yml", "config.casual.yml"):
        assert cli.main(["lichess", "check-config", "--config", str(out / name), "--results", "none"]) == 0
    assert "0 problems" in capsys.readouterr().out


def test_check_config_exits_1_and_lists_every_problem(tmp_path, capsys):
    config = copy.deepcopy(generate(tmp_path)["rated"])
    config["abort_time"] = 20
    config["engine"]["polyglot"]["enabled"] = True
    bad = tmp_path / "bad.yml"
    bad.write_text(json.dumps(config), encoding="utf-8")
    assert cli.main(["lichess", "check-config", "--config", str(bad), "--results", "none"]) == 1
    printed = capsys.readouterr().out
    assert "abort_time" in printed and "engine.polyglot.enabled" in printed


def test_check_config_compares_against_the_shipped_model_in_results_json(tmp_path, capsys):
    weights = fake_weights(tmp_path)
    out = tmp_path / "bot"
    botconfig.generate(botconfig.BotSpec(str(weights), "policy", SHA), out, resolve=lambda s: weights)
    results = results_schema.Results(
        strength=(),
        diagnostics=(),
        shipped=results_schema.Shipped(agent="Blink-M", mode="value", sha=SHA),
        eval_md_sha="e" * 12,
        generated_at="2026-10-08T12:00:00+03:00",
    )
    results_path = tmp_path / "results.json"
    results_path.write_text(results_schema.to_json(results), encoding="utf-8")
    argv = ["lichess", "check-config", "--config", str(out / "config.yml"), "--results", str(results_path)]
    assert cli.main(argv) == 1
    assert "shipped mode" in capsys.readouterr().out


def test_a_config_file_that_is_not_a_mapping_is_refused_with_a_clear_error(tmp_path):
    bad = tmp_path / "list.yml"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(botconfig.ConfigError, match="mapping"):
        botconfig.load_config(bad)
    assert cli.main(["lichess", "check-config", "--config", str(bad), "--results", "none"]) == 2


def test_engine_options_that_are_not_a_mapping_are_named(tmp_path):
    config = copy.deepcopy(generate(tmp_path)["rated"])
    config["engine"]["engine_options"] = ["--model=ship"]
    assert any(
        "engine.engine_options must be a mapping" in p
        for p in botconfig.problems(config, "rated", frozenset())
    )


# ---------------------------------------------------------------- one decision log per engine process
# lichess-bot starts one blink-uci per game and passes every engine the same static flags, so at
# challenge.concurrency 2 two engines run at once; a shared --log file would take appends from both.

DECISION_SCRIPT = "".join(
    ["uci\n", "isready\n", "ucinewgame\n"]
    + [f"position startpos{tail}\ngo movetime 50\n" for tail in ("", " moves e2e4", " moves e2e4 e7e5")]
    + ["quit\n"]
)


def test_every_bot_config_gives_each_engine_process_its_own_decision_log(tmp_path):
    for config in all_configs(tmp_path):
        log = config["engine"]["engine_options"]["log"]
        assert uci.PROCESS_FIELD in log, log


@pytest.mark.parametrize("kind", ["rated", "casual"])
def test_a_decision_log_shared_by_games_running_at_once_is_named(tmp_path, kind):
    config = copy.deepcopy(generate(tmp_path)[kind])
    config["engine"]["engine_options"]["log"] = "D:/blink/lichess/decisions.jsonl"
    config["challenge"]["concurrency"] = 2
    found = botconfig.problems(config, kind, frozenset())
    assert any(p.startswith("engine.engine_options.log") and uci.PROCESS_FIELD in p for p in found), found


def test_one_game_at_a_time_may_keep_a_single_decision_log(tmp_path):
    config = copy.deepcopy(generate(tmp_path)["casual"])
    config["engine"]["engine_options"]["log"] = "D:/blink/lichess/casual-decisions.jsonl"
    assert botconfig.problems(config, "casual", frozenset()) == []


def test_the_process_field_in_the_log_path_becomes_the_utc_start_time_and_the_pid():
    folder = Path("D:/blink/lichess/decisions")
    expanded = uci.log_path(folder / f"{uci.PROCESS_FIELD}.jsonl", pid=4242, now=0.0)
    assert expanded == folder / "19700101T000000Z-4242.jsonl"
    assert uci.log_path(folder / "one.jsonl", pid=4242, now=0.0) == folder / "one.jsonl"


def test_two_engines_started_at_once_from_the_rated_flags_write_two_whole_logs(tmp_path):
    options = {**load_template()["engine"]["engine_options"], "device": "cpu"}
    options["log"] = str(tmp_path / "decisions" / PurePosixPath(options["log"]).name)
    command = [sys.executable, "-m", "blink.uci", *lichess_bot_flags(options), "--random"]
    pipes = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    engines = [
        subprocess.Popen(command, **pipes, text=True, encoding="utf-8", cwd=ROOT, env=env) for _ in range(2)
    ]
    for engine in engines:  # both engines get their game before either is read, as two live games do
        engine.stdin.write(DECISION_SCRIPT)
        engine.stdin.close()
    outputs = [engine.communicate(timeout=180) for engine in engines]
    for engine, (out, err) in zip(engines, outputs, strict=True):
        assert engine.returncode == 0, err
        assert out.count("bestmove") == 3
    logs = sorted((tmp_path / "decisions").iterdir())
    # the PID is the interpreter's own, which a Windows venv launcher or console script runs as a child
    assert len(logs) == 2 and len({path.stem.rsplit("-", 1)[1] for path in logs}) == 2
    assert all(re.fullmatch(r"\d{8}T\d{6}Z-\d+\.jsonl", path.name) for path in logs), logs
    for path in logs:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert [(r["game"], r["ply"]) for r in records] == [("g1", 0), ("g1", 1), ("g1", 2)]


# ---------------------------------------------------------------- the rated check fails closed
# check-config is the gate before the rated bot starts: anything it cannot verify is a problem, not a
# note, so a config whose engine cannot start or whose model cannot be checked never reads 0 problems.


def rated_file(tmp_path: Path) -> tuple[Path, Path]:
    weights = fake_weights(tmp_path)
    spec = botconfig.BotSpec(str(weights), "value", SHA)
    written = botconfig.generate(spec, tmp_path / "bot", kinds=("rated",), resolve=lambda s: weights)
    return written["rated"], weights


def results_file(tmp_path: Path, shipped: results_schema.Shipped | None) -> Path:
    results = results_schema.Results(
        strength=(),
        diagnostics=(),
        shipped=shipped,
        eval_md_sha="e" * 12,
        generated_at="2026-10-08T12:00:00+03:00",
    )
    path = tmp_path / "results.json"
    path.write_text(results_schema.to_json(results), encoding="utf-8")
    return path


def check(path: Path, weights: Path | None, results: Path | None = None, exe: bool = True):
    return botconfig.check_file(path, results=results, resolve=lambda s: weights, exists=lambda p: exe)


def test_a_rated_check_passes_only_against_the_shipped_model_in_results_json(tmp_path):
    path, weights = rated_file(tmp_path)
    shipped = results_schema.Shipped(agent="Blink-M", mode="value", sha=SHA)
    assert check(path, weights, results_file(tmp_path, shipped)).problems == ()


def test_a_rated_check_fails_closed_without_results_json_or_a_shipped_model_in_it(tmp_path):
    path, weights = rated_file(tmp_path)
    missing = check(path, weights, tmp_path / "absent.json").problems
    assert any("absent.json" in p and "--results none" in p for p in missing), missing
    unshipped = check(path, weights, results_file(tmp_path, None)).problems
    assert any("no shipped model" in p for p in unshipped), unshipped
    skipped = check(path, weights, None)  # --results none: skipped on purpose, and said so
    assert skipped.problems == () and any("--results none" in n for n in skipped.notes)


def test_a_rated_check_fails_when_the_weights_file_cannot_be_found(tmp_path):
    path, _ = rated_file(tmp_path)
    found = check(path, None).problems
    assert any("weights file" in p and "cannot be found" in p for p in found), found


@pytest.mark.parametrize("kind", ["rated", "casual"])
def test_check_config_fails_when_the_engine_exe_does_not_exist(tmp_path, kind):
    weights = fake_weights(tmp_path)
    spec = botconfig.BotSpec(str(weights), "value", SHA)
    path = botconfig.generate(spec, tmp_path / "bot", kinds=(kind,), resolve=lambda s: weights)[kind]
    found = check(path, weights, exe=False).problems
    assert any("blink-uci.exe" in p and "does not exist" in p for p in found), found


def test_a_rated_config_records_the_full_sha256_of_its_weights(tmp_path):
    rated = copy.deepcopy(generate(tmp_path)["rated"])
    rated["blink"]["sha"] = SHA[:12]
    assert any("64 hex" in p for p in botconfig.problems(rated, "rated", frozenset()))
    spec = botconfig.BotSpec(model="C:/nowhere/missing.pt", mode="value", sha="abcdef1")
    with pytest.raises(botconfig.ConfigError, match="64 hex"):
        botconfig.generate(spec, tmp_path / "out", kinds=("rated",), resolve=lambda selector: None)
    assert not (tmp_path / "out" / "config.yml").exists()
