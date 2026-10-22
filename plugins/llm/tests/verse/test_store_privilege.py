import pytest
from llm.verse.store import VerseStore

VALID_SOURCES = {"operator", "loom", "llm", "crosspoll", "avatar"}


def _store(tmp_path):
    return VerseStore(tmp_path, "#chan")


def test_invalid_source_rejected(tmp_path):
    store = _store(tmp_path)
    with store.write_transaction() as conn, pytest.raises(ValueError, match="source"):
        store._apply_op_inline(
            conn, op="add_event", payload={"summary": "x", "entity_ids": []}, source="bogus"
        )


def test_destructive_op_blocked_for_non_operator(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    ev = store.add_event(summary="hi", entity_ids=[eid], source="loom")
    with store.write_transaction() as conn, pytest.raises(PermissionError):
        store._apply_op_inline(conn, op="delete_event", payload={"event_id": ev}, source="llm")


def test_destructive_op_allowed_for_operator(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    ev = store.add_event(summary="hi", entity_ids=[eid], source="operator")
    with store.write_transaction() as conn:
        store._apply_op_inline(conn, op="delete_event", payload={"event_id": ev}, source="operator")
    assert store.recent_events(limit=10) == []


# ---------------------------------------------------------------------------
# _apply_op_inline op-specific error branches (update_entity, set_status,
# set_pinned, edit_event, delete_event, delete_relation).
# ---------------------------------------------------------------------------


def _apply(store, **kw):
    with store.write_transaction() as conn:
        return store._apply_op_inline(conn, source="operator", **kw)


def test_update_entity_rejects_kind_change(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    with pytest.raises(ValueError, match="cannot change kind"):
        _apply(store, op="update_entity", payload={"entity_id": eid, "kind": "place"})


def test_update_entity_requires_name_or_summary(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    with pytest.raises(ValueError, match="name and/or summary"):
        _apply(store, op="update_entity", payload={"entity_id": eid})


def test_update_entity_missing_entity(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(LookupError, match="does not exist"):
        _apply(store, op="update_entity", payload={"entity_id": 999, "name": "X"})


def test_update_entity_sets_name_and_summary(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    _apply(store, op="update_entity", payload={"entity_id": eid, "name": "Rob", "summary": "new"})
    ent = store.get_entity(eid)
    assert ent.name == "Rob"
    assert ent.summary == "new"


def test_update_entity_place_rename_relocates_avatars(tmp_path):
    """Renaming a place rewrites avatar 'location' attributes that pointed at
    the old name (case-insensitively) — otherwise every avatar standing there
    is stranded at a ghost name and loses the aging occupied-place protection."""
    store = _store(tmp_path)
    place_id = store.add_entity("place", "The Clearing", "a glade")
    here = store.add_entity("avatar", "alice")
    store.set_attribute(here, "location", "the clearing")  # case differs
    elsewhere = store.add_entity("avatar", "bob")
    store.set_attribute(elsewhere, "location", "The Tavern")

    _apply(store, op="update_entity", payload={"entity_id": place_id, "name": "The Glade"})

    assert store.get_attribute(here, "location") == "The Glade"
    assert store.get_attribute(elsewhere, "location") == "The Tavern"  # untouched


def test_update_entity_place_rename_skips_ambiguous_homonym(tmp_path):
    """When another ACTIVE place shares the old name, the location rewrite is
    ambiguous (location is stored by name, not id) — rewriting would teleport
    avatars standing at the homonym place. Skip the rewrite in that case rather
    than mislocate unrelated avatars."""
    store = _store(tmp_path)
    cave1 = store.add_entity("place", "Cave", "the first cave")
    store.add_entity("place", "Cave", "a different, same-named cave")  # homonym
    at_homonym = store.add_entity("avatar", "bob")
    store.set_attribute(at_homonym, "location", "Cave")

    _apply(store, op="update_entity", payload={"entity_id": cave1, "name": "Deep Cave"})

    # The homonym-place avatar is NOT teleported to "Deep Cave".
    assert store.get_attribute(at_homonym, "location") == "Cave"


def test_update_entity_place_summary_only_edit(tmp_path):
    """A summary-only edit of a place is not a rename: it must succeed and
    leave avatars standing there where they are."""
    store = _store(tmp_path)
    place_id = store.add_entity("place", "The Clearing", "a glade")
    here = store.add_entity("avatar", "alice")
    store.set_attribute(here, "location", "The Clearing")

    _apply(store, op="update_entity", payload={"entity_id": place_id, "summary": "a sunny glade"})

    ent = store.get_entity(place_id)
    assert ent.name == "The Clearing"
    assert ent.summary == "a sunny glade"
    assert store.get_attribute(here, "location") == "The Clearing"


def test_update_entity_npc_rename_leaves_locations_alone(tmp_path):
    """The location rewrite is scoped to kind='place' renames — renaming an
    NPC that happens to share a place's name must not move anyone."""
    store = _store(tmp_path)
    store.add_entity("place", "Echo")
    npc = store.add_entity("npc", "Echo")
    avatar = store.add_entity("avatar", "alice")
    store.set_attribute(avatar, "location", "Echo")

    _apply(store, op="update_entity", payload={"entity_id": npc, "name": "Reverb"})

    assert store.get_attribute(avatar, "location") == "Echo"


def test_set_status_invalid_status(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    with pytest.raises(ValueError, match="invalid status"):
        _apply(store, op="set_status", payload={"entity_id": eid, "status": "frozen"})


def test_set_pinned_unpin_removes_attribute(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    _apply(store, op="set_pinned", payload={"entity_id": eid, "pinned": True})
    assert "pinned" in store.list_attributes(eid)
    _apply(store, op="set_pinned", payload={"entity_id": eid, "pinned": False})
    assert "pinned" not in store.list_attributes(eid)


def test_edit_event_missing(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(LookupError, match="event_id 999"):
        _apply(store, op="edit_event", payload={"event_id": 999, "summary": "nope"})


def test_delete_relation_removes(tmp_path):
    store = _store(tmp_path)
    a = store.add_entity("npc", "A")
    b = store.add_entity("npc", "B")
    rid = _apply(
        store,
        op="add_relation",
        payload={"from_id": a, "to_id": b, "kind": "knows", "note": ""},
    )
    _apply(store, op="delete_relation", payload={"relation_id": rid})
    assert store.list_relations(a) == []


# --- resolve_ref --------------------------------------------------------


def test_resolve_ref_unknown_id(tmp_path):
    store = _store(tmp_path)
    with pytest.raises(LookupError, match="no entity #5"):
        store.resolve_ref("#5")


def test_resolve_ref_by_id_and_name(tmp_path):
    store = _store(tmp_path)
    eid = store.add_entity("npc", "Bob")
    assert store.resolve_ref(f"#{eid}") == eid
    assert store.resolve_ref("Bob") == eid


def test_loom_cannot_forge_author_locked(tmp_path):
    store = VerseStore(tmp_path, "#priv")
    h = store.add_entity("npc", "Harry")
    with pytest.raises(ValueError):
        store.apply_direct(
            op="set_attribute",
            payload={"entity_id": h, "key": "author_locked", "value": "1"},
            source="loom",
            provenance="test",
        )
