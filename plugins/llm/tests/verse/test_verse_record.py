"""Tests for VerseStore.record_user_event (verse_record's atomic DB path)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from llm.verse.store import VerseStore


@pytest.fixture
def store(tmp_path: Path) -> VerseStore:
    return VerseStore(tmp_path, "#record")


def _opt_in(store: VerseStore, nick: str = "alice") -> int:
    """Convenience: opt a nick into the verse and return the avatar entity id."""
    result = store.opt_in_avatar(nick, account=None, instruct_text=f"{nick} instruct")
    return result.entity_id


class TestRecordUserEvent:
    def test_all_new_names_create_npcs_and_link_event(self, store: VerseStore) -> None:
        """GIVEN unknown actor names WHEN record_user_event called THEN
        each name becomes an active npc with auto_created='1', and the
        event row's entity_ids list starts with the caller's avatar."""
        alice_id = _opt_in(store)
        event_id = store.record_user_event(
            actor_id=alice_id,
            summary="stinky dan threw a guff grenade at Andrew",
            actor_names=["stinky dan", "Andrew"],
            now=lambda: 100.0,
        )
        # NPC rows exist
        dan = store.find_active_entity_by_name("stinky dan")
        andrew = store.find_active_entity_by_name("Andrew")
        assert dan is not None and dan.kind == "npc"
        assert andrew is not None and andrew.kind == "npc"
        # auto_created marker on each
        assert store.get_attribute(dan.id, "auto_created") == "1"
        assert store.get_attribute(andrew.id, "auto_created") == "1"
        # last_seen_ts = now
        assert store.get_attribute(dan.id, "last_seen_ts") == "100.0"
        assert store.get_attribute(andrew.id, "last_seen_ts") == "100.0"
        # Event row links caller first, then actors in order
        events = store.recent_events(limit=10)
        ev = next(e for e in events if e.id == event_id)
        assert list(ev.entity_ids) == [alice_id, dan.id, andrew.id]
        assert ev.source == "avatar"
        assert ev.summary == "stinky dan threw a guff grenade at Andrew"

    def test_avatar_actor_not_tagged_or_bumped(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, "alice")
        andrew_id = _opt_in(store, "andrew")
        store.record_user_event(
            actor_id=alice_id,
            summary="alice greeted Andrew",
            actor_names=["andrew"],
            now=lambda: 100.0,
        )
        assert store.get_attribute(andrew_id, "auto_created") is None
        assert store.get_attribute(andrew_id, "last_seen_ts") is None
        events = store.recent_events(limit=10)
        assert any(list(e.entity_ids) == [alice_id, andrew_id] for e in events)

    def test_existing_npc_reused_and_heartbeat_updated(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        dan_id = store.add_entity("npc", "dan", "")
        store.set_attribute(dan_id, "auto_created", "1")
        store.set_attribute(dan_id, "last_seen_ts", "50.0")

        store.record_user_event(
            actor_id=alice_id,
            summary="dan returned",
            actor_names=["dan"],
            now=lambda: 200.0,
        )
        rows = [e for e in store.list_entities_by_kind("npc") if e.name == "dan"]
        assert len(rows) == 1
        assert store.get_attribute(dan_id, "last_seen_ts") == "200.0"

    def test_repeated_calls_one_row_latest_timestamp(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        for ts in (100.0, 200.0, 300.0):
            store.record_user_event(
                actor_id=alice_id,
                summary="dan reappears",
                actor_names=["dan"],
                now=lambda ts=ts: ts,
            )
        rows = [e for e in store.list_entities_by_kind("npc") if e.name == "dan"]
        assert len(rows) == 1
        assert store.get_attribute(rows[0].id, "last_seen_ts") == "300.0"

    def test_concurrent_record_same_actor_one_row(
        self, store: VerseStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two threads racing record_user_event for the same unknown actor
        produce exactly one entity row. The Python `threading.Lock` inside
        `write_transaction()` is what enforces this — both threads serialise
        behind it; the second thread sees the just-created npc on its find
        and skips the auto-create branch.

        A `time.sleep(0.01)` is monkey-patched between find and add to widen
        the contention window for the lock-holding thread (without it, the
        test would still pass but with a smaller probability of catching a
        genuine ordering bug). The sleep lands INSIDE `write_transaction`,
        after the lock is acquired — so this test exercises lock-held
        serialisation, not a SQLite-level race. That's intentional: the
        Python lock IS the safety mechanism for single-process callers (see
        v2.3 design doc, §4.2 race-test scope clarification). Multi-process
        contention is out of scope.

        Both threads start at a Barrier so they enter their respective
        record_user_event calls at the same instant."""
        import threading

        alice_id = _opt_in(store)

        real_find = store.find_active_entity_by_name
        barrier = threading.Barrier(2)

        def slow_find(name, *, conn=None):  # noqa: ANN001
            result = real_find(name, conn=conn)
            time.sleep(0.01)
            return result

        monkeypatch.setattr(store, "find_active_entity_by_name", slow_find)

        results: list[int] = []

        def call(seq: int) -> None:
            barrier.wait()
            eid = store.record_user_event(
                actor_id=alice_id,
                summary=f"event {seq}",
                actor_names=["zorp"],
                now=lambda: 100.0 + seq,
            )
            results.append(eid)

        t1 = threading.Thread(target=call, args=(1,))
        t2 = threading.Thread(target=call, args=(2,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(results) == 2
        zorps = [e for e in store.list_entities_by_kind("npc") if e.name == "zorp"]
        assert len(zorps) == 1, f"expected exactly one 'zorp' entity, got {len(zorps)}"

    def test_retired_auto_npc_reactivated_by_name(self, store: VerseStore) -> None:
        """An aged-out auto-created npc is REACTIVATED on re-mention (same id,
        status flips back to active, heartbeat refreshed) instead of spawning a
        duplicate row. This is the fix for the duplicate-cluster churn: aging
        retires auto npcs, and re-mention used to create a brand-new id."""
        alice_id = _opt_in(store)
        dan_id = store.add_entity("npc", "dan", "")
        store.set_attribute(dan_id, "auto_created", "1")
        store.set_attribute(dan_id, "last_seen_ts", "50.0")
        store.set_status(dan_id, "retired")

        event_id = store.record_user_event(
            actor_id=alice_id,
            summary="dan crawls back out of the void",
            actor_names=["dan"],
            now=lambda: 300.0,
        )

        rows = [e for e in store.list_entities_by_kind("npc", status=None) if e.name == "dan"]
        assert len(rows) == 1, "must reuse the retired auto-npc, not duplicate it"
        assert rows[0].id == dan_id
        assert rows[0].status == "active"
        assert store.get_attribute(dan_id, "last_seen_ts") == "300.0"
        ev = next(e for e in store.recent_events(limit=10) if e.id == event_id)
        assert dan_id in ev.entity_ids

    def test_retired_entity_not_rehydrated(self, store: VerseStore) -> None:
        # A retired npc WITHOUT the auto_created marker (e.g. operator canon, or
        # the npc shell left behind by an opted-out avatar) is NOT reactivated:
        # reactivation is scoped to auto_created npcs so the avatar lifecycle and
        # deliberate operator retirements are preserved.
        alice_id = _opt_in(store)
        old_id = store.add_entity("npc", "ghost", "")
        store.set_status(old_id, "retired")

        store.record_user_event(
            actor_id=alice_id,
            summary="ghost reappeared",
            actor_names=["ghost"],
            now=lambda: 100.0,
        )
        ghosts = [e for e in store.list_entities_by_kind("npc", status=None) if e.name == "ghost"]
        statuses = sorted(e.status for e in ghosts)
        assert statuses == ["active", "retired"]
        new_ghost = next(e for e in ghosts if e.status == "active")
        assert new_ghost.id != old_id
        assert store.get_attribute(new_ghost.id, "auto_created") == "1"

    def test_actors_resolved_case_insensitively(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, "alice")
        andrew_id = _opt_in(store, "andrew")
        store.record_user_event(
            actor_id=alice_id,
            summary="alice waved at ANDREW",
            actor_names=["ANDREW"],
            now=lambda: 100.0,
        )
        events = store.recent_events(limit=5)
        latest = events[0]
        assert list(latest.entity_ids) == [alice_id, andrew_id]
        assert store.get_attribute(andrew_id, "auto_created") is None

    def test_opt_out_then_record_then_reopt_in_three_row_state(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, "alice")
        avatar_v1 = _opt_in(store, "andrew")
        store.unlink_avatar(avatar_v1)

        store.record_user_event(
            actor_id=alice_id,
            summary="Andrew was seen",
            actor_names=["Andrew"],
            now=lambda: 100.0,
        )
        npc = store.find_active_entity_by_name("Andrew")
        assert npc is not None and npc.kind == "npc"

        avatar_v2 = _opt_in(store, "andrew")
        assert avatar_v2 != avatar_v1
        store.record_user_event(
            actor_id=alice_id,
            summary="Andrew is back",
            actor_names=["Andrew"],
            now=lambda: 200.0,
        )
        events = store.recent_events(limit=2)
        latest = events[0]
        assert avatar_v2 in latest.entity_ids
        assert npc.id not in latest.entity_ids
        assert store.get_attribute(avatar_v2, "last_seen_ts") is None
        assert store.get_attribute(npc.id, "last_seen_ts") == "100.0"

    def test_retired_actor_id_raises(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, "alice")
        store.unlink_avatar(alice_id)
        with pytest.raises(ValueError, match="not an active entity"):
            store.record_user_event(
                actor_id=alice_id,
                summary="alice did something",
                actor_names=["bob"],
                now=lambda: 100.0,
            )
        assert store.recent_events(limit=10) == []
        assert store.find_active_entity_by_name("bob") is None


class TestVerseRecordDispatch:
    def test_dispatch_happy_path(self, store: VerseStore) -> None:
        """dispatch_verse_tool_call routes 'verse_record' to
        record_user_event and surfaces event_id in the result payload."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            dispatch_verse_tool_call,
        )

        alice_id = _opt_in(store)
        result = dispatch_verse_tool_call(
            store,
            alice_id,
            "verse_record",
            {"summary": "alice waved", "actors": ["bob"]},
        )
        assert isinstance(result, VerseDispatchResult)
        assert result.ok is True
        assert result.error is None
        assert result.payload is not None
        assert result.payload["status"] == "ok"
        assert isinstance(result.payload["event_id"], int)

    def test_dispatch_empty_summary_returns_error(self, store: VerseStore) -> None:
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        n_events_before = len(store.recent_events(limit=100))
        result = dispatch_verse_tool_call(
            store,
            alice_id,
            "verse_record",
            {"summary": "   ", "actors": ["bob"]},
        )
        assert result.ok is False
        assert result.error == "summary required"
        assert len(store.recent_events(limit=100)) == n_events_before

    def test_dispatch_too_long_summary_returns_error(self, store: VerseStore) -> None:
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        summary = "x" * 250
        result = dispatch_verse_tool_call(
            store,
            alice_id,
            "verse_record",
            {"summary": summary, "actors": []},
        )
        assert result.ok is False
        assert result.error is not None
        assert "too long" in result.error
        assert "250" in result.error

    def test_dispatch_actors_filter_then_slice(self, store: VerseStore) -> None:
        """Mixed-type input ['alice', 42, 'bob'] with max_actors=2 yields
        actors ['alice', 'bob'], not ['alice'] (the 42 must NOT eat a slot)."""
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        result = dispatch_verse_tool_call(
            store,
            alice_id,
            "verse_record",
            {
                "summary": "test mixed",
                "actors": ["alice_npc", 42, "bob_npc"],
                "_max_actors": 2,
            },
        )
        assert result.ok is True
        assert store.find_active_entity_by_name("alice_npc") is not None
        assert store.find_active_entity_by_name("bob_npc") is not None

    def test_dispatch_actors_empty_or_whitespace_filtered(self, store: VerseStore) -> None:
        """actors=['', '  ', 'alice'] processes only 'alice'; no
        empty-name entities created."""
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        result = dispatch_verse_tool_call(
            store,
            alice_id,
            "verse_record",
            {"summary": "filter ws", "actors": ["", "  ", "alice_w"]},
        )
        assert result.ok is True
        assert store.find_active_entity_by_name("alice_w") is not None
        assert store.find_active_entity_by_name("") is None
        assert store.find_active_entity_by_name("  ") is None

    def test_dispatch_truncates_to_max_actors(self, store: VerseStore) -> None:
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        raw = [f"npc{i}" for i in range(20)]
        result = dispatch_verse_tool_call(
            store,
            alice_id,
            "verse_record",
            {"summary": "many", "actors": raw, "_max_actors": 5},
        )
        assert result.ok is True
        for i in range(5):
            assert store.find_active_entity_by_name(f"npc{i}") is not None
        for i in range(5, 20):
            assert store.find_active_entity_by_name(f"npc{i}") is None

    @pytest.mark.parametrize("actors", ["bob", {"a": 1}])
    def test_dispatch_non_array_actors_rejected_and_writes_nothing(
        self, store: VerseStore, actors: object
    ) -> None:
        """A non-array ``actors`` is refused with a model-facing error. A string
        must not be iterated per character into NPCs 'b', 'o', 'b' — canon
        pollution is hard to undo."""
        import json

        from llm.verse.avatar import dispatch_verse_tool_call, make_verse_extra_handlers

        alice_id = _opt_in(store)
        npcs_before = store.list_entities_by_kind("npc")
        events_before = len(store.recent_events(limit=100))

        result = dispatch_verse_tool_call(
            store, alice_id, "verse_record", {"summary": "s", "actors": actors}
        )
        assert result.ok is False
        assert "array" in (result.error or "")

        payload = json.loads(
            make_verse_extra_handlers(store, alice_id)["verse_record"](
                {"summary": "s", "actors": actors}
            ).content
        )
        assert payload["status"] == "error"
        assert "array" in payload["error"]

        assert store.list_entities_by_kind("npc") == npcs_before
        assert len(store.recent_events(limit=100)) == events_before

    def test_dispatch_db_write_failure_is_not_ok(
        self, store: VerseStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DB failure inside record_user_event must reach the model as a
        failure, so it does not narrate an event that was never written."""
        import json
        import sqlite3

        from llm.verse.avatar import dispatch_verse_tool_call, make_verse_extra_handlers

        alice_id = _opt_in(store)
        events_before = len(store.recent_events(limit=100))

        def boom(**_kw):
            raise sqlite3.OperationalError("disk I/O error")

        monkeypatch.setattr(store, "record_user_event", boom)

        result = dispatch_verse_tool_call(
            store, alice_id, "verse_record", {"summary": "alice waved", "actors": ["bob"]}
        )
        assert result.ok is False
        assert result.error == "could not record event"

        payload = json.loads(
            make_verse_extra_handlers(store, alice_id)["verse_record"](
                {"summary": "alice waved"}
            ).content
        )
        assert payload["status"] == "error"
        assert payload["error"] == "could not record event"
        assert len(store.recent_events(limit=100)) == events_before


class TestMaxActorsNotModelOverridable:
    def test_handler_overwrites_model_supplied_max_actors(self, store: VerseStore) -> None:
        """_max_actors is server-side plumbing: a tool call smuggling its own
        value in the (model-controlled) args dict must not raise the cap."""
        import json

        from llm.verse.avatar import make_verse_extra_handlers

        alice_id = _opt_in(store)
        handlers = make_verse_extra_handlers(store, alice_id, max_actors=2)
        raw = [f"smuggle{i}" for i in range(6)]

        res = handlers["verse_record"]({"summary": "s", "actors": raw, "_max_actors": 99})

        assert json.loads(res.content)["status"] == "ok"
        for i in range(2):
            assert store.find_active_entity_by_name(f"smuggle{i}") is not None
        for i in range(2, 6):
            assert store.find_active_entity_by_name(f"smuggle{i}") is None
