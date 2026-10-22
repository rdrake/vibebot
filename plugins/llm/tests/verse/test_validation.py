import pytest
from llm.verse.validation import validate_payload


def test_validate_payload_ok():
    assert validate_payload("add_entity", {"kind": "npc", "name": "Bob"}) is None


def test_validate_payload_missing_field():
    assert "name" in (validate_payload("add_entity", {"kind": "npc"}) or "")


def test_validate_payload_update_entity():
    assert validate_payload("update_entity", {"entity_id": 3, "summary": "x"}) is None
    assert validate_payload("update_entity", {"entity_id": "x", "summary": "y"}) is not None


def test_validate_payload_unknown_op():
    assert validate_payload("nope", {}) is not None


@pytest.mark.parametrize("entity_ids", [[1, 2], []])
def test_validate_payload_add_event_accepts_int_entity_ids(entity_ids):
    """add_event takes a str summary and a list of ints; an event with no
    actors (empty list) is allowed."""
    assert validate_payload("add_event", {"summary": "x", "entity_ids": entity_ids}) is None


@pytest.mark.parametrize(
    "entity_ids",
    [["bob"], [True], [1.5], [1, "2"], "", "12", {}, {1: 1}, 3],
)
def test_validate_payload_add_event_rejects_non_int_list(entity_ids):
    """entity_ids must be a list whose every element is a non-bool int;
    anything else would be json-dumped into events.entity_ids as junk."""
    assert (
        validate_payload("add_event", {"summary": "x", "entity_ids": entity_ids})
        == "entity_ids not list[int]"
    )
