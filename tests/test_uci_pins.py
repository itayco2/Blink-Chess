"""blink-uci under lichess-bot: the rated engine refuses any weights but the pinned sha256 (plan P9).

lichess-bot starts a fresh blink-uci for every game, for days, from a selector (`ship`) whose file
can be overwritten. `--sha` makes each engine hash its weights file before the UCI handshake and
exit 2 on a mismatch, so the bot plays the evaluated model or no game at all.
"""

import hashlib
import io

import pytest

from blink import uci

WEIGHTS = b"blink weights"
SHA = hashlib.sha256(WEIGHTS).hexdigest()
HANDSHAKE = "uci\nquit\n"


def run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = uci.main(argv, stdin=io.StringIO(HANDSHAKE), stdout=out)
    return code, out.getvalue()


@pytest.fixture
def weights(tmp_path):
    path = tmp_path / "blink.pt"
    path.write_bytes(WEIGHTS)
    return path


@pytest.mark.torch
def test_blink_uci_starts_when_its_weights_hash_to_the_pinned_sha(weights):
    code, out = run(["--model", str(weights), "--sha", SHA])
    assert code == 0 and "uciok" in out


@pytest.mark.torch
def test_blink_uci_refuses_weights_whose_sha256_is_not_the_pinned_one(weights, capsys):
    code, out = run(["--model", str(weights), "--sha", "0" * 64])
    assert code == 2 and out == ""  # nothing reaches lichess-bot, so its engine check fails at startup
    err = capsys.readouterr().err
    assert "refusing to start" in err and SHA[:12] in err


@pytest.mark.torch
def test_blink_uci_refuses_a_pinned_model_whose_weights_file_is_missing(tmp_path, capsys):
    code, out = run(["--model", str(tmp_path / "gone.pt"), "--sha", SHA])
    assert code == 2 and out == "" and "no weights file" in capsys.readouterr().err


def test_a_sha_pin_needs_a_blink_weights_file_not_a_random_network(capsys):
    code, out = run(["--random", "--sha", SHA])
    assert code == 2 and out == "" and "--sha" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["abcdef1", "G" * 64, SHA.upper() + "0"])
def test_the_sha_flag_takes_only_a_full_lowercase_sha256(bad):
    with pytest.raises(SystemExit):
        uci.build_parser().parse_args(["--sha", bad])
