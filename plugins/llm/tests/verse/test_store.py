"""Tests for the verse store."""

from __future__ import annotations

import concurrent.futures
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from llm.verse.store import db_path_for_channel

from .conftest import add_alias, link_avatar


class TestDbPathForChannel:
    def test_lowercases_and_sanitizes(self, verse_db_dir: Path) -> None:
        result = db_path_for_channel(verse_db_dir, "#Foo")
        assert result.parent == verse_db_dir
        assert result.name.startswith("_foo_")
        assert result.suffix == ".db"

    def test_folds_irc_case_variants(self, verse_db_dir: Path) -> None:
        """IRC channel names are case-insensitive (rfc1459), so '#AfterNet'
        typed from a PM must open the same DB as '#afternet'."""
        assert db_path_for_channel(verse_db_dir, "#AfterNet") == db_path_for_channel(
            verse_db_dir, "#afternet"
        )
        assert db_path_for_channel(verse_db_dir, "#a[b]") == db_path_for_channel(
            verse_db_dir, "#a{b}"
        )

    def test_lowercase_name_unchanged(self, verse_db_dir: Path) -> None:
        """Prod's existing file must keep its name across the case-fold fix."""
        assert db_path_for_channel(verse_db_dir, "#afternet").name == "_afternet_2de47b99.db"

    def test_strips_funky_characters(self, verse_db_dir: Path) -> None:
        result = db_path_for_channel(verse_db_dir, "#foo!bar/baz")
        assert "!" not in result.name
        assert "/" not in result.name


class TestVerseStoreInit:
    def test_creates_db_with_schema(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        with store.read_connection() as conn:
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            assert row is not None
            assert row[0] >= 1

    def test_idempotent_init(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        VerseStore(verse_db_dir, "#afnet")
        VerseStore(verse_db_dir, "#afnet")
        store = VerseStore(verse_db_dir, "#afnet")
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
            assert count == 1

    def test_migrate_backfills_event_actor_for_legacy_unversioned_db(
        self, verse_db_dir: Path
    ) -> None:
        """A pre-versioning DB — events present, no schema_version row, empty
        event_actor — must run the v2->v3 backfill on open, not skip straight to
        stamping the latest version. Otherwise every historical entity's event
        history silently vanishes from prompts."""
        import sqlite3

        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#legacy")
        eid = store.add_entity("npc", "Alice", "a traveller")
        store.add_event(summary="Alice arrived", entity_ids=[eid], source="operator")
        path = store.path

        # add_event dual-writes event_actor; simulate a legacy DB by dropping the
        # version row and clearing event_actor while leaving events (and their
        # entity_ids JSON) intact.
        raw = sqlite3.connect(path)
        try:
            assert raw.execute("SELECT COUNT(*) FROM event_actor").fetchone()[0] == 1
            raw.execute("DELETE FROM schema_version")
            raw.execute("DELETE FROM event_actor")
            raw.commit()
        finally:
            raw.close()

        # Re-open: _migrate must detect the data and run the upgrade chain,
        # rebuilding event_actor from events.entity_ids.
        VerseStore(verse_db_dir, "#legacy")

        raw = sqlite3.connect(path)
        try:
            assert raw.execute("SELECT COUNT(*) FROM event_actor").fetchone()[0] == 1
            assert raw.execute("SELECT entity_id FROM event_actor").fetchone()[0] == eid
        finally:
            raw.close()


class TestEntityCrud:
    def test_add_returns_id_and_persists(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "A mysterious stranger")
        assert isinstance(eid, int)
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity.id == eid
        assert entity.kind == "npc"
        assert entity.name == "Alice"
        assert entity.summary == "A mysterious stranger"
        assert entity.status == "active"
        assert isinstance(entity.created_at, float)
        assert isinstance(entity.updated_at, float)
        assert abs(entity.created_at - entity.updated_at) < 1.0

    def test_get_unknown_returns_none(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        assert store.get_entity(99999) is None

    def test_find_by_name_case_insensitive(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test entity")
        for variant in ("alice", "ALICE", "Alice"):
            result = store.find_entity_by_name(variant)
            assert result is not None, f"Should find entity with name variant {variant!r}"
            assert result.id == eid

    def test_find_by_name_with_kind_filter(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        place_id = store.add_entity("place", "Crossroads", "A dusty crossroads")
        _npc_id = store.add_entity("npc", "Crossroads", "A person named Crossroads")
        result = store.find_entity_by_name("Crossroads", kind="place")
        assert result is not None
        assert result.id == place_id
        assert result.kind == "place"

    def test_set_status_updates_updated_at(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Bob", "A test npc")
        original = store.get_entity(eid)
        assert original is not None
        time.sleep(0.01)
        store.set_status(eid, "retired")
        updated = store.get_entity(eid)
        assert updated is not None
        assert updated.status == "retired"
        assert updated.updated_at > original.created_at

    def test_set_status_unknown_id_silent(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.set_status(99999, "retired")  # must not raise

    def test_list_entities_by_kind_filters_by_status(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_entity("place", "Town Square", "Central plaza")
        store.add_entity("place", "Harbor", "The docks")
        retired_id = store.add_entity("place", "Old Mill", "Abandoned mill")
        store.set_status(retired_id, "retired")

        active = store.list_entities_by_kind("place")
        assert len(active) == 2

        all_places = store.list_entities_by_kind("place", status=None)
        assert len(all_places) == 3

        retired = store.list_entities_by_kind("place", status="retired")
        assert len(retired) == 1
        assert retired[0].name == "Old Mill"

    def test_list_entities_by_kind_orders_by_updated_at_desc(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        id_a = store.add_entity("place", "Alpha", "First added")
        time.sleep(0.01)
        id_b = store.add_entity("place", "Beta", "Second added")

        results = store.list_entities_by_kind("place")
        assert len(results) == 2
        assert results[0].id == id_b
        assert results[1].id == id_a


class TestAttributesCrud:
    def test_get_missing_returns_none(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        assert store.get_attribute(eid, "nonexistent") is None

    def test_set_overwrites_existing(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        store.set_attribute(eid, "color", "red")
        store.set_attribute(eid, "color", "blue")
        assert store.get_attribute(eid, "color") == "blue"
        attrs = store.list_attributes(eid)
        assert list(attrs.keys()).count("color") == 1

    def test_list_attributes_returns_dict(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        store.set_attribute(eid, "color", "red")
        store.set_attribute(eid, "mood", "cheerful")
        store.set_attribute(eid, "weapon", "staff")
        attrs = store.list_attributes(eid)
        assert attrs == {"color": "red", "mood": "cheerful", "weapon": "staff"}

    def test_list_attributes_empty(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        assert store.list_attributes(eid) == {}

    def test_attributes_cascade_on_entity_delete(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("npc", "Alice", "Test npc")
        store.set_attribute(eid, "color", "red")
        store.set_attribute(eid, "mood", "happy")
        # manually delete the entity to trigger ON DELETE CASCADE
        with store.write_transaction() as conn:
            conn.execute("DELETE FROM entities WHERE id = ?", (eid,))
        assert store.list_attributes(eid) == {}


class TestRelationsCrud:
    def test_list_no_filter_returns_all(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations()
        assert len(rels) == 3
        # ordered by id ASC
        assert rels[0].id < rels[1].id < rels[2].id

    def test_list_filtered_by_from_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations(from_id=a)
        assert len(rels) == 2
        assert all(r.from_id == a for r in rels)

    def test_list_filtered_by_to_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations(to_id=c)
        assert len(rels) == 2
        assert all(r.to_id == c for r in rels)

    def test_list_filtered_by_kind(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        allied = store.list_relations(kind="allied_with")
        assert len(allied) == 2
        hates = store.list_relations(kind="hates")
        assert len(hates) == 1

    def test_list_filtered_by_combo(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        c = store.add_entity("npc", "Carol", "")
        store.add_relation(a, b, "allied_with")
        store.add_relation(a, c, "hates")
        store.add_relation(b, c, "allied_with")
        rels = store.list_relations(from_id=a, kind="allied_with")
        assert len(rels) == 1
        assert rels[0].from_id == a
        assert rels[0].to_id == b
        assert rels[0].kind == "allied_with"

    def test_list_relations_empty_when_no_match(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.add_entity("npc", "Alice", "")
        b = store.add_entity("npc", "Bob", "")
        store.add_relation(a, b, "allied_with")
        assert store.list_relations(kind="nonexistent_kind") == []


class TestEventsCrud:
    def test_add_returns_id_and_persists(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_event("A dragon appeared", [1, 2], "avatar")
        assert isinstance(eid, int)
        assert eid > 0
        events = store.recent_events()
        assert len(events) == 1
        ev = events[0]
        assert ev.id == eid
        assert ev.summary == "A dragon appeared"
        assert ev.source == "avatar"
        assert ev.entity_ids == (1, 2)
        assert isinstance(ev.ts, float)

    def test_entity_ids_empty_default(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("Empty entity event", [], "loom")
        events = store.recent_events()
        assert events[0].entity_ids == ()

    def test_recent_events_survives_corrupt_entity_ids(self, verse_db_dir: Path) -> None:
        """A row with malformed entity_ids JSON must not crash recent_events:
        the bad row's entity_ids degrade to empty and valid rows still load."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("good event", [1, 2], "avatar")
        bad_id = store.add_event("bad event", [3], "avatar")
        # Corrupt the entity_ids JSON for the bad row directly.
        with store.write_transaction() as conn:
            conn.execute("UPDATE events SET entity_ids = ? WHERE id = ?", ("}{nope", bad_id))

        events = store.recent_events()  # must not raise

        summaries = {e.summary: e.entity_ids for e in events}
        assert summaries["good event"] == (1, 2)
        assert summaries["bad event"] == ()

    def test_recent_events_newest_first(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        id1 = store.add_event("First", [], "avatar")
        time.sleep(0.01)
        id2 = store.add_event("Second", [], "avatar")
        time.sleep(0.01)
        id3 = store.add_event("Third", [], "avatar")
        events = store.recent_events(limit=10)
        assert [ev.id for ev in events] == [id3, id2, id1]

    def test_recent_events_limit(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(5):
            store.add_event(f"Event {i}", [], "loom")
        events = store.recent_events(limit=2)
        assert len(events) == 2

    def test_recent_events_exclude_sources_multiple(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.add_event("Avatar event", [], "avatar")
        store.add_event("Loom event", [], "loom")
        store.add_event("Crosspoll event", [], "crosspoll")
        events = store.recent_events(exclude_sources=("crosspoll", "loom"))
        assert len(events) == 1
        assert events[0].source == "avatar"

    def test_recent_events_require_active_entity_drops_dead_only(self, verse_db_dir: Path) -> None:
        """require_active_entity=True excludes events whose every referenced
        entity is retired (dead lore), but keeps events with at least one
        active entity and keeps entity-less narration."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        live = store.add_entity("npc", "live", "")
        dead = store.add_entity("npc", "dead", "")
        store.set_status(dead, "retired")
        store.add_event("dead-only", [dead], "loom")
        store.add_event("live-only", [live], "loom")
        store.add_event("mixed", [live, dead], "loom")
        store.add_event("narration", [], "loom")

        out = store.recent_events(limit=10, require_active_entity=True)
        assert {ev.summary for ev in out} == {"live-only", "mixed", "narration"}

    def test_recent_events_require_active_entity_default_keeps_dead(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        dead = store.add_entity("npc", "dead", "")
        store.set_status(dead, "retired")
        store.add_event("dead-only", [dead], "loom")
        assert len(store.recent_events(limit=10)) == 1

    def test_recent_events_require_active_entity_limit_counts_post_filter(
        self, verse_db_dir: Path
    ) -> None:
        """limit counts surviving rows: dead-only events interleaved among
        live ones must not eat limit slots."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        live = store.add_entity("npc", "live", "")
        dead = store.add_entity("npc", "dead", "")
        store.set_status(dead, "retired")
        for i in range(3):
            store.add_event(f"dead {i}", [dead], "loom")
            store.add_event(f"live {i}", [live], "loom")
        out = store.recent_events(limit=2, require_active_entity=True)
        assert len(out) == 2
        assert all("live" in ev.summary for ev in out)


class TestAvatarLinkCrud:
    def test_link_with_account(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Alice", "")
        link_avatar(store, eid, nick="Alice", account="alice@network")
        assert store.find_avatar_by_account("alice@network") == eid
        assert store.find_avatar_by_account("Alice@Network") is None  # case-sensitive

    def test_find_by_nick_case_insensitive(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Alice", "")
        link_avatar(store, eid, nick="Alice")
        for variant in ("ALICE", "alice", "aLiCe"):
            assert store.find_avatar_by_nick(variant) == eid

    def test_unlink_atomicity(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid = store.add_entity("avatar", "Bob", "")
        link_avatar(store, eid, nick="Bob", account="bob@net")
        store.unlink_avatar(eid)
        # Both effects visible after single call
        assert store.find_avatar_by_nick("Bob") is None
        assert store.find_avatar_by_account("bob@net") is None
        entity = store.get_entity(eid)
        assert entity is not None
        assert entity.status == "retired"

    def test_unlink_unknown_id_silent(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.unlink_avatar(99999)  # must not raise

    def test_two_avatars_distinct_links(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        eid_a = store.add_entity("avatar", "Alice", "")
        eid_b = store.add_entity("avatar", "Bob", "")
        link_avatar(store, eid_a, nick="Alice")
        link_avatar(store, eid_b, nick="Bob")
        assert store.find_avatar_by_nick("alice") == eid_a
        assert store.find_avatar_by_nick("bob") == eid_b
        store.unlink_avatar(eid_a)
        assert store.find_avatar_by_nick("alice") is None
        assert store.find_avatar_by_nick("bob") == eid_b
        entity_b = store.get_entity(eid_b)
        assert entity_b is not None
        assert entity_b.status == "active"


class TestOptInAvatar:
    def test_new_user_creates_avatar_and_default_place(self, verse_db_dir: Path) -> None:
        from llm.verse.store import AvatarOptInResult, VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        result = store.opt_in_avatar("alice", None, "curious traveller")

        assert isinstance(result, AvatarOptInResult)
        assert result.was_already_opted_in is False
        assert result.place_name == "The Clearing"
        assert result.scene_text.startswith("You step into The Clearing.")
        assert "A quiet woodland clearing where new stories begin." in result.scene_text

        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.kind == "avatar"
        assert entity.status == "active"
        assert entity.name == "alice"
        assert entity.summary == "curious traveller"

        places = store.list_entities_by_kind("place")
        assert len(places) == 1

        assert store.find_avatar_by_nick("alice") == result.entity_id
        assert store.get_attribute(result.entity_id, "location") == "The Clearing"

    def test_second_user_lands_at_existing_place(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.opt_in_avatar("alice", None, "first in")
        bob_result = store.opt_in_avatar("bob", None, "second in")

        places = store.list_entities_by_kind("place")
        assert len(places) == 1

        assert bob_result.place_name == "The Clearing"
        assert store.get_attribute(bob_result.entity_id, "location") == "The Clearing"

    def test_already_opted_in_active_returns_existing(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", None, "curious traveller")
        second = store.opt_in_avatar("alice", None, "different text")

        assert second.was_already_opted_in is True
        assert second.entity_id == first.entity_id
        assert second.place_name == "The Clearing"
        assert "The Clearing" in second.scene_text

        # Only one avatar entity
        avatars = store.list_entities_by_kind("avatar")
        assert len(avatars) == 1

    def test_re_opt_in_preserves_moved_location(self, verse_db_dir: Path) -> None:
        """GIVEN an active avatar that has moved away from its starter place
        WHEN the user re-runs opt-in THEN the avatar is NOT teleported back: the
        committed 'location' is preserved.

        Re-opt-in is meant to be an idempotent no-op ("you are already opted
        in"); it must not silently overwrite the avatar's location with the
        activity-best place. ``set_attribute(eid, 'location', ...)`` is the exact
        engine write ``verse_move`` performs.
        """
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", None, "curious traveller")
        assert store.get_attribute(first.entity_id, "location") == "The Clearing"

        # Alice moves elsewhere (same write path as verse_move).
        store.set_attribute(first.entity_id, "location", "The Tower")

        # Re-opt-in must not relocate her, even though "The Clearing" is still
        # the only/activity-best place the placement logic would pick.
        second = store.opt_in_avatar("alice", None, "back again")

        assert second.was_already_opted_in is True
        assert second.entity_id == first.entity_id
        assert store.get_attribute(first.entity_id, "location") == "The Tower"
        assert second.place_name == "The Tower"

    def test_retired_avatar_reactivates(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", None, "curious traveller")
        eid = first.entity_id

        # Soft pause: retire entity but leave link
        store.set_status(eid, "retired")

        result = store.opt_in_avatar("alice", None, "coming back")

        assert result.was_already_opted_in is False
        assert result.entity_id == eid

        entity = store.get_entity(eid)
        assert entity is not None
        assert entity.status == "active"

    def test_retired_avatar_via_unlink_then_reopt_creates_new(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", None, "curious traveller")
        old_eid = first.entity_id

        # Hard goodbye: removes link AND retires
        store.unlink_avatar(old_eid)

        result = store.opt_in_avatar("alice", None, "starting fresh")

        assert result.was_already_opted_in is False
        assert result.entity_id != old_eid

    def test_account_lookup_prefers_account_over_nick(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("alice", "alice@net", "curious traveller")
        eid = first.entity_id

        # Nick changed on IRC side, account stable
        result = store.opt_in_avatar("aliceX", "alice@net", "new session")

        assert result.was_already_opted_in is True
        assert result.entity_id == eid

        # Link nick should be updated
        assert store.find_avatar_by_nick("aliceX") == eid

    def test_opt_in_survives_cross_account_nick_collision(self, verse_db_dir: Path) -> None:
        """A second account claiming a nick another avatar's link holds must not crash.

        avatar_link.nick is UNIQUE NOT NULL. When an identified user renames to a
        nick already recorded on another avatar's link, opt_in's link UPDATE hit
        sqlite3.IntegrityError and crashed @verseopt in. The opting-in user must
        still be served (resolved by account); the contested nick stays with its
        current owner.
        """
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        a = store.opt_in_avatar("alice", "alice@net", "the traveller")
        b = store.opt_in_avatar("bob", "bob@net", "the smith")

        # alice@net renames their IRC nick to "bob" (held by bob@net's link).
        result = store.opt_in_avatar("bob", "alice@net", "still the traveller")

        # No crash; still alice@net's own avatar (resolved by account).
        assert result.entity_id == a.entity_id
        assert result.was_already_opted_in is True
        # The contested nick stays mapped to bob@net's avatar.
        assert store.find_avatar_by_nick("bob") == b.entity_id

    def test_opt_in_does_not_hijack_another_accounts_nick_avatar(self, verse_db_dir: Path) -> None:
        """A new account opting in on a recycled nick must NOT take over the
        avatar owned by the nick's previous authenticated holder.

        bob@net opts in as 'bob'. The nick 'bob' is later recycled to a
        different authenticated user alice@net who has no avatar of their own.
        The nick fallback resolves to bob@net's avatar, but relinking it would
        transfer bob@net's lore/history/account to alice@net. Instead alice@net
        gets a fresh avatar, the live nick follows them, and bob@net keeps their
        avatar (still resolvable by account, self-healing on next opt-in).
        """
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        b = store.opt_in_avatar("bob", "bob@net", "the smith")

        result = store.opt_in_avatar("bob", "alice@net", "a newcomer")

        # alice@net gets their OWN new avatar, not bob@net's — no takeover.
        assert result.entity_id != b.entity_id
        assert result.was_already_opted_in is False
        # bob@net still owns their original avatar (resolved by account).
        assert store.find_avatar_by_account("bob@net") == b.entity_id
        # The live nick now points at the newcomer's avatar.
        assert store.find_avatar_by_nick("bob") == result.entity_id

    def test_anon_opt_in_does_not_hijack_authed_nick_avatar(self, verse_db_dir: Path) -> None:
        """An UNauthenticated user reusing an authenticated user's nick must not
        be handed that user's avatar, and opt-in must not crash on the UNIQUE
        nick constraint."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        b = store.opt_in_avatar("bob", "bob@net", "the smith")

        result = store.opt_in_avatar("bob", None, "just passing through")

        assert result.entity_id != b.entity_id
        assert result.was_already_opted_in is False
        assert store.find_avatar_by_account("bob@net") == b.entity_id

    def test_opt_in_nick_lookup_is_indexed_not_full_scan(self, verse_db_dir: Path) -> None:
        """The nick-fallback avatar lookup must seek an index, not scan avatar_link.

        opt_in_avatar resolves an existing avatar by nick when no account match
        is found. Wrapping the column in LOWER() (``WHERE LOWER(al.nick)=LOWER(?)``)
        is non-sargable and forces a full O(n) SCAN of avatar_link while the write
        lock is held. The lookup must instead use a case-insensitive index seek.
        """
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        store.opt_in_avatar("alice", None, "curious traveller")

        # Mirror the exact predicate opt_in_avatar uses for the nick fallback.
        sql = (
            "SELECT al.entity_id, e.status FROM avatar_link al"
            " JOIN entities e ON e.id = al.entity_id"
            " WHERE al.nick = ? COLLATE NOCASE"
        )
        with store.read_connection() as conn:
            plan = conn.execute("EXPLAIN QUERY PLAN " + sql, ("alice",)).fetchall()
        detail = " | ".join(row[3] for row in plan)
        # RED before fix: idx_avatar_link_nick is BINARY, so this SCANs avatar_link.
        # GREEN after fix: a NOCASE index turns it into a SEARCH ... (nick=?).
        assert "SCAN avatar_link" not in detail and "SCAN al" not in detail, detail
        assert "SEARCH" in detail and "nick" in detail.lower(), detail

    def test_opt_in_nick_fallback_case_insensitive_resolves_same_avatar(
        self, verse_db_dir: Path
    ) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        first = store.opt_in_avatar("Alice", None, "curious traveller")
        # Different-case nick, still no account -> must hit the nick fallback and
        # resolve to the same (already active) avatar.
        again = store.opt_in_avatar("aLiCe", None, "second visit")
        assert again.entity_id == first.entity_id
        assert again.was_already_opted_in is True
        assert len(store.list_entities_by_kind("avatar")) == 1

    def test_concurrent_opt_in_distinct_nicks_one_place(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#concurrent-optin")

        def do_opt_in(nick: str):  # type: ignore[return]
            return store.opt_in_avatar(nick, None, f"{nick} instruct")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(do_opt_in, ["alice", "bob"]))

        # Both succeed
        assert all(r is not None for r in results)
        # Only one place
        assert len(store.list_entities_by_kind("place")) == 1
        # Both land at same place
        assert results[0].place_name == results[1].place_name
        # Distinct entity ids
        assert results[0].entity_id != results[1].entity_id

    def test_place_selection_uses_most_recent_event(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        # Pre-create two places
        store.add_entity("place", "Place A", "Summary of A.")
        place_b_id = store.add_entity("place", "Place B", "Summary of B.")

        # Add event referencing place B — makes it the most recently referenced
        store.add_event("Something happened at B", [place_b_id], "loom")

        result = store.opt_in_avatar("alice", None, "curious traveller")
        assert result.place_name == "Place B"
        assert store.get_attribute(result.entity_id, "location") == "Place B"
        # The user-visible scene text describes the chosen place.
        assert "You step into Place B. Summary of B." in result.scene_text

    def test_account_avatar_beats_unowned_nick_avatar(self, verse_db_dir: Path) -> None:
        """An authenticated user who renames onto an anonymous user's nick
        resumes their OWN avatar (found by account); the anonymous avatar on
        that nick is not taken over."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        anon = store.opt_in_avatar("bob", None, "anonymous bob")
        mine = store.opt_in_avatar("alice", "alice@net", "the traveller")

        result = store.opt_in_avatar("bob", "alice@net", "still the traveller")

        assert result.entity_id == mine.entity_id
        assert result.was_already_opted_in is True
        assert store.find_avatar_by_account("alice@net") == mine.entity_id
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT account FROM avatar_link WHERE entity_id = ?", (anon.entity_id,)
            ).fetchone()
        assert row == (None,)

    def test_authenticating_claims_own_anonymous_avatar(self, verse_db_dir: Path) -> None:
        """Someone who opted in before authenticating keeps their avatar when they
        opt in again authenticated: the unowned nick-matched avatar is claimed
        and linked to their account, not parked in favour of a fresh one."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        anon = store.opt_in_avatar("alice", None, "curious traveller")

        result = store.opt_in_avatar("alice", "alice@net", "now authenticated")

        assert result.entity_id == anon.entity_id
        assert result.was_already_opted_in is True
        assert store.find_avatar_by_account("alice@net") == anon.entity_id
        assert len(store.list_entities_by_kind("avatar")) == 1


class TestProposalsCRUD:
    def test_list_proposals_filters_and_decodes(self, verse_db_dir: Path) -> None:
        import json as _json
        import time as _time
        import uuid as _uuid

        from llm.verse.store import Proposal, VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        p1 = _uuid.uuid4().hex
        t1 = _time.time()
        with store.write_transaction() as conn:
            conn.execute(
                "INSERT INTO proposals (id, created_at, cycle_id, op, payload, confidence, "
                "provenance, status, reviewer, reviewed_at) VALUES (?,?,?,?,?,?,?,'pending',NULL,NULL)",
                (p1, t1, "c-1", "add_event", _json.dumps({"summary": "first"}), 0.9, ""),
            )
        _time.sleep(0.005)
        p2 = _uuid.uuid4().hex
        t2 = _time.time()
        with store.write_transaction() as conn:
            conn.execute(
                "INSERT INTO proposals (id, created_at, cycle_id, op, payload, confidence, "
                "provenance, status, reviewer, reviewed_at) VALUES (?,?,?,?,?,?,?,'pending',NULL,NULL)",
                (p2, t2, "c-2", "add_event", _json.dumps({"summary": "second"}), 0.5, ""),
            )
        rows = store.list_proposals()
        assert [r.id for r in rows] == [p2, p1]
        assert isinstance(rows[0], Proposal)
        assert rows[0].payload == {"summary": "second"}
        assert [r.id for r in store.list_proposals(status="pending")] == [p2, p1]
        assert [r.id for r in store.list_proposals(cycle_id="c-1")] == [p1]


class TestWriteLockConcurrency:
    def test_concurrent_add_entity_yields_unique_ids(self, verse_db_dir: Path) -> None:
        """50 concurrent add_entity calls across 8 threads — all rows persist
        with unique IDs, none are lost or duplicated."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#concurrent")

        def worker(i: int) -> int:
            return store.add_entity(kind="npc", name=f"worker_{i}", summary="")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            ids = list(pool.map(worker, range(50)))

        assert len(set(ids)) == 50, "duplicate ids returned"
        assert all(i > 0 for i in ids), "non-positive id returned"

        names = [e.name for e in store.list_entities_by_kind("npc")]
        assert sorted(names) == sorted(f"worker_{i}" for i in range(50))

    def test_concurrent_set_attribute_no_lost_writes(self, verse_db_dir: Path) -> None:
        """Concurrent upserts to the same entity-key pair — last writer wins;
        no transaction is dropped (count of distinct values seen during the
        test <= thread count, no SQL errors)."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#concurrent2")
        eid = store.add_entity(kind="place", name="arena", summary="")
        errors: list[BaseException] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            try:
                store.set_attribute(eid, "round", str(i))
            except BaseException as e:
                with lock:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(50)))

        assert errors == []
        # The final value is one of {0,...,49}; we don't care which.
        final = store.get_attribute(eid, "round")
        assert final is not None
        assert int(final) in range(50)


class TestEventsOlderThan:
    def test_returns_oldest_first(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        ids: list[int] = []
        for ts in (10.0, 20.0, 30.0):
            ids.append(
                insert_event_at(
                    store,
                    summary=f"e{ts}",
                    entity_ids=[],
                    source="loom",
                    ts=ts,
                )
            )
        rows = store.events_older_than(cutoff_ts=25.0)
        assert [r.id for r in rows] == [ids[0], ids[1]]
        assert [r.ts for r in rows] == [10.0, 20.0]

    def test_empty_when_no_events_below_cutoff(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=100.0)
        assert store.events_older_than(cutoff_ts=50.0) == []

    def test_includes_all_sources(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="a", entity_ids=[], source="avatar", ts=5.0)
        insert_event_at(store, summary="b", entity_ids=[], source="loom", ts=6.0)
        insert_event_at(store, summary="c", entity_ids=[], source="crosspoll", ts=7.0)
        rows = store.events_older_than(cutoff_ts=10.0)
        assert {r.source for r in rows} == {"avatar", "loom", "crosspoll"}


class TestReplaceEventsWithLoreDigest:
    def test_replaces_atomically_and_returns_new_id(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        old_ids = [
            insert_event_at(
                store,
                summary=f"e{i}",
                entity_ids=[],
                source="avatar",
                ts=float(i),
            )
            for i in range(5)
        ]
        new_id = store.replace_events_with_lore_digest(
            delete_ids=old_ids,
            summary="A digest of five small events.",
            entity_ids=(),
            ts=100.0,
        )
        assert new_id > 0
        # surviving rows: only the new digest event
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id, summary, source FROM events ORDER BY id ASC").fetchall()
        assert len(rows) == 1
        assert rows[0][0] == new_id
        assert rows[0][1] == "A digest of five small events."
        assert rows[0][2] == "llm"

    def test_rolls_back_when_digest_insert_fails(
        self, verse_db_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        oid = insert_event_at(store, summary="e", entity_ids=[], source="avatar", ts=1.0)

        def boom(*_a, **_k):
            raise sqlite3.IntegrityError("forced")

        # The DELETE has already run inside the transaction when the insert fails.
        monkeypatch.setattr(store, "add_event", boom)
        with pytest.raises(sqlite3.IntegrityError):
            store.replace_events_with_lore_digest(
                delete_ids=[oid], summary="x", entity_ids=(), ts=2.0
            )
        # original event still present, no digest row created
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id FROM events").fetchall()
        assert [r[0] for r in rows] == [oid]

    def test_entity_ids_are_json_encoded(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#afnet")
        new_id = store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="d",
            entity_ids=(1, 2, 3),
            ts=10.0,
        )
        with store.read_connection() as conn:
            row = conn.execute("SELECT entity_ids FROM events WHERE id=?", (new_id,)).fetchone()
        assert json.loads(row[0]) == [1, 2, 3]


class TestInlineHelpers:
    """The former _*_inline pairs are now single methods taking conn=...;
    these tests prove they compose into a caller's open transaction."""

    def test_add_entity_runs_on_caller_conn(self, verse_db_dir: Path) -> None:
        """Caller opens its own write_transaction, calls add_entity(conn=...),
        and a sibling INSERT in the same tx — all without lock reentry."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#inline")
        with store.write_transaction() as conn:
            eid = store.add_entity("npc", "ghost", "a wisp of vapour", conn=conn)
            # Sibling INSERT in the same tx proves we hold the same conn.
            conn.execute(
                "INSERT INTO attributes (entity_id, key, value) VALUES (?, 'inline_marker', '1')",
                (eid,),
            )
        assert store.find_entity_by_name("ghost", kind="npc") is not None
        assert store.get_attribute(eid, "inline_marker") == "1"

    def test_set_attribute_upserts_on_caller_conn(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#inline")
        eid = store.add_entity("npc", "moss", "")
        with store.write_transaction() as conn:
            store.set_attribute(eid, "k", "v1", conn=conn)
            store.set_attribute(eid, "k", "v2", conn=conn)
        assert store.get_attribute(eid, "k") == "v2"

    def test_add_event_writes_on_caller_conn(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#inline")
        eid = store.add_entity("avatar", "alice", "")
        with store.write_transaction() as conn:
            ev_id = store.add_event("alice waved", [eid], "avatar", conn=conn)
        events = store.recent_events(limit=10)
        assert any(e.id == ev_id and e.summary == "alice waved" for e in events)

    def test_set_status_updates_on_caller_conn(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#inline")
        eid = store.add_entity("npc", "ghost", "")
        with store.write_transaction() as conn:
            store.set_status(eid, "retired", conn=conn)
        # status reads via raw SQL since find_entity_by_name's exact filtering
        # is documented per design §2 to ship in Phase 1, not here.
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "retired"


class TestFindActiveEntityByName:
    def test_avatar_wins_over_npc(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        npc_id = store.add_entity("npc", "Andrew", "")
        avatar_id = store.add_entity("avatar", "Andrew", "")
        result = store.find_active_entity_by_name("Andrew")
        assert result is not None and result.id == avatar_id
        # Sanity: legacy kind-filtered call still finds the npc
        legacy = store.find_entity_by_name("Andrew", kind="npc")
        assert legacy is not None and legacy.id == npc_id

    def test_case_insensitive(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        eid = store.add_entity("avatar", "andrew", "")
        result = store.find_active_entity_by_name("ANDREW")
        assert result is not None and result.id == eid

    def test_skips_retired(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        eid = store.add_entity("npc", "ghost", "")
        store.set_status(eid, "retired")
        assert store.find_active_entity_by_name("ghost") is None

    def test_conn_variant_runs_on_caller_conn(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        store.add_entity("npc", "moss", "")
        with store.read_connection() as conn:
            result = store.find_active_entity_by_name("moss", conn=conn)
        assert result is not None and result.kind == "npc"

    def test_npc_beats_item(self, verse_db_dir: Path) -> None:
        """avatar > npc > item > place precedence — verify mid-tier."""
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#prec")
        item_id = store.add_entity("item", "shard", "")
        npc_id = store.add_entity("npc", "shard", "")
        result = store.find_active_entity_by_name("shard")
        assert result is not None and result.id == npc_id
        # Ensure the item still exists — we picked, not deleted.
        with store.read_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM entities WHERE id IN (?, ?)",
                (item_id, npc_id),
            ).fetchone()[0]
        assert count == 2


class TestListEntitiesWithAttribute:
    def test_returns_matching_entities(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#attr")
        a = store.add_entity("npc", "alpha", "")
        b = store.add_entity("npc", "beta", "")
        store.add_entity("npc", "gamma", "")  # no attribute
        store.set_attribute(a, "auto_created", "1")
        store.set_attribute(b, "auto_created", "1")
        rows = store.list_entities_with_attribute(key="auto_created", value="1", status="active")
        assert {e.id for e in rows} == {a, b}

    def test_status_filter(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#attr")
        a = store.add_entity("npc", "alpha", "")
        store.set_attribute(a, "auto_created", "1")
        store.set_status(a, "retired")
        active = store.list_entities_with_attribute(key="auto_created", value="1", status="active")
        assert active == []

    def test_value_filter(self, verse_db_dir: Path) -> None:
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#attr")
        a = store.add_entity("npc", "alpha", "")
        store.set_attribute(a, "auto_created", "0")
        rows = store.list_entities_with_attribute(key="auto_created", value="1", status="active")
        assert rows == []


class TestEventActorAllSites:
    def test_add_event_populates_event_actor(self, store):
        a = store.add_entity("npc", "Harry")
        ev = store.add_event("Harry did a thing", [a, 99999], source="avatar")
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT entity_id FROM event_actor WHERE event_id=?", (ev,)
            ).fetchall()
        assert rows == [(a,)]  # 99999 has no entities row -> FK-safe skip

    def test_loom_digest_populates_event_actor(self, store):
        a = store.add_entity("npc", "Harry")
        ev = store.replace_events_with_lore_digest(
            delete_ids=[], summary="loom digest", entity_ids=[a, 99999], ts=123.0
        )
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT entity_id FROM event_actor WHERE event_id=?", (ev,)
            ).fetchall()
        assert rows == [(a,)]  # loom-digest site routes through the single writer

    def test_apply_direct_add_event_populates_event_actor(self, store):
        a = store.add_entity("npc", "Harry")
        store.apply_direct(
            op="add_event",
            payload={"summary": "Harry returned", "entity_ids": [a]},
            source="operator",
            provenance="test",
        )
        # events_for_entities() arrives in Task 6; assert via SQL here (plan ordering note).
        with store.read_connection() as conn:
            rows = conn.execute(
                "SELECT ea.entity_id FROM event_actor ea "
                "JOIN events ev ON ev.id = ea.event_id "
                "WHERE ev.summary = 'Harry returned'"
            ).fetchall()
        assert rows == [(a,)]


class TestAliases:
    def test_resolve_alias_case_insensitive(self, store):
        t = store.add_entity("npc", "Toby")
        add_alias(store, t, "Tobes")
        assert store.find_entity_by_name_or_alias("tobes").id == t

    def test_exact_active_name_beats_alias(self, store):
        real = store.add_entity("npc", "Tobes")
        other = store.add_entity("npc", "Toby")
        add_alias(store, other, "Tobes")
        assert store.find_entity_by_name_or_alias("Tobes").id == real


class TestSceneRetrieval:
    def test_match_word_boundary_and_alias(self, store):
        h = store.add_entity("npc", "Harry")
        t = store.add_entity("npc", "Toby")
        add_alias(store, t, "Tobes")
        store.add_entity("npc", "Di")  # 2 chars -> skipped (too short)
        got = {e.id for e in store.match_entities_in_text("did Harry and Tobes fight? ask Di.")}
        assert got == {h, t}

    def test_match_three_char_name(self, store):
        """Only names of 2 chars or fewer are skipped; a 3-char NPC is in the scene."""
        b = store.add_entity("npc", "Bob")
        assert {e.id for e in store.match_entities_in_text("ask Bob about it")} == {b}

    def test_match_handles_punctuation_and_avoids_common_words(self, store):
        w = store.add_entity("npc", "Will")
        assert store.match_entities_in_text("I will go") == []  # common-word, not the name
        assert {e.id for e in store.match_entities_in_text("Will, run!")} == {w}

    def test_relations_for_one_hop(self, store):
        h = store.add_entity("npc", "Harry")
        t = store.add_entity("npc", "Toby")
        store.add_relation(h, t, "rival_of", "since year 7")
        assert any(
            r.from_name == "Harry" and r.to_name == "Toby" and r.kind == "rival_of"
            for r in store.relations_for([h])
        )

    def test_events_for_entities_active_only(self, store):
        h = store.add_entity("npc", "Harry")
        g = store.add_entity("npc", "Ghost")
        store.add_event("Harry won", [h], source="avatar")
        store.add_event("Ghost faded", [g], source="avatar")
        store.set_status(g, "retired")
        sums = [e.summary for e in store.events_for_entities([h, g], limit=10)]
        assert "Harry won" in sums and "Ghost faded" not in sums

    def test_apply_direct_event_retrievable_via_events_for_entities(self, store):
        # closes the Task-2 loop: author/operator add_event must be retrievable
        a = store.add_entity("npc", "Harry")
        store.apply_direct(
            op="add_event",
            payload={"summary": "Harry returned", "entity_ids": [a]},
            source="operator",
            provenance="test",
        )
        evs = store.events_for_entities([a], limit=10)
        assert any("returned" in e.summary for e in evs)


class TestOptInCorruptEventTolerance:
    def test_opt_in_survives_corrupt_entity_ids_row(self, store) -> None:
        """A corrupt events.entity_ids row must not crash avatar opt-in
        (starter-place scoring parses every event's entity_ids)."""
        store.add_entity("place", "The Tavern", "A busy tavern.")
        with store.write_transaction() as conn:
            conn.execute(
                "INSERT INTO events (ts, summary, entity_ids, source)"
                " VALUES (1.0, 'corrupt', ?, 'operator')",
                ('["abc", null]',),
            )

        result = store.opt_in_avatar("alice", None, "curious traveller")

        assert result.entity_id is not None
        assert "Tavern" in result.scene_text
