from llm.verse.store import VerseStore


def test_list_canon_returns_only_active_pinned(tmp_path):
    """Pinned filtering (formerly list_pinned_entities, now folded into
    list_canon_entities): unpinned entities are excluded, and retiring a
    pinned entity drops it from the roster."""
    store = VerseStore(tmp_path, "#chan")
    a = store.add_entity("npc", "Archie", "stinky")
    store.add_entity("npc", "Bob", "plain")
    store.apply_direct(
        op="set_pinned", payload={"entity_id": a, "pinned": True}, source="operator", provenance="t"
    )
    pinned = store.list_canon_entities()
    assert [e.name for e in pinned] == ["Archie"]
    store.apply_direct(
        op="set_status",
        payload={"entity_id": a, "status": "retired"},
        source="operator",
        provenance="t",
    )
    assert store.list_canon_entities() == []


def test_active_name_exists(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Archie")
    assert store.active_name_exists("archie") is True
    assert store.active_name_exists("nobody") is False


class TestAuthorLocked:
    def test_list_canon_unions_pinned_and_author_locked(self, store):
        h = store.add_entity("npc", "Harry", "year 8")
        t = store.add_entity("npc", "Toby", "year 9")
        store.set_attribute(t, "pinned", "1")
        store.set_author_locked(h, True)
        assert {e.name for e in store.list_canon_entities()} == {"Harry", "Toby"}

    def test_unlock_removes_from_canon(self, store):
        h = store.add_entity("npc", "Harry")
        store.set_author_locked(h, True)
        store.set_author_locked(h, False)
        assert store.list_canon_entities() == []
