import re

import pytest

from blink.train import world


def test_the_world_id_is_12_hex_characters():
    assert re.fullmatch(r"[0-9a-f]{12}", world.world_id("manifest-sha"))


def test_the_world_id_changes_when_any_input_changes():
    base = world.world_id("m", blocklist_sha="b", split_rule="s", contract="c")
    assert base == world.world_id("m", blocklist_sha="b", split_rule="s", contract="c")
    assert base != world.world_id("m2", blocklist_sha="b", split_rule="s", contract="c")
    assert base != world.world_id("m", blocklist_sha="b2", split_rule="s", contract="c")
    assert base != world.world_id("m", blocklist_sha="b", split_rule="s2", contract="c")
    assert base != world.world_id("m", blocklist_sha="b", split_rule="s", contract="c2")


def test_the_contract_hash_is_a_stable_sha1():
    assert re.fullmatch(r"[0-9a-f]{40}", world.contract_hash())
    assert world.contract_hash() == world.contract_hash()


def test_the_world_guard_passes_the_same_world_and_refuses_another():
    world.require_same_world(found="abc123abc123", expected="abc123abc123")
    with pytest.raises(world.WorldMismatch, match="abc123abc123"):
        world.require_same_world(found="abc123abc123", expected="ffffffffffff")
