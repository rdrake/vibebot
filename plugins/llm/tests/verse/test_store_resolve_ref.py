import pytest
from llm.verse.store import VerseStore


def test_resolve_ref_numeric_name_is_not_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "7")  # literally named "7"
    assert store.resolve_ref("7") == eid  # name, not id  # noqa: UP031
    assert store.resolve_ref("#%d" % eid) == eid  # explicit id form  # noqa: UP031


def test_resolve_ref_name_with_digit_tail_is_a_name_not_an_id(tmp_path):
    """Only '#<digits>' is an id. A name like 'x1' ends in digits but is still a
    name; reading its tail as id #1 would make @versedit edit the wrong entity."""
    store = VerseStore(tmp_path, "#chan")
    first = store.add_entity("npc", "Archie")
    named_x1 = store.add_entity("npc", "x1")
    assert first == 1  # the id the digit tail would wrongly resolve to
    assert store.resolve_ref("x1") == named_x1


def test_resolve_ref_hash_without_digits_is_an_unknown_name(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Archie")
    with pytest.raises(LookupError, match="no active entity named"):
        store.resolve_ref("#nope")
