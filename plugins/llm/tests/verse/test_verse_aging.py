"""Tests for verse/aging.age_auto_created_entities."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest
from llm.verse.store import VerseStore


@pytest.fixture
def store(tmp_path: Path) -> VerseStore:
    return VerseStore(tmp_path, "#aging")


SECONDS_PER_DAY = 86400.0


class TestAgeAutoCreatedEntities:
    def test_retire_after_days_zero_disables(self, store: VerseStore) -> None:
        from llm.verse.aging import AgingOutcome, age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")
        outcome = age_auto_created_entities(store, retire_after_days=0, now=lambda: 1e9)
        assert outcome == AgingOutcome(scanned=0, retired=0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_retires_past_cutoff(self, store: VerseStore) -> None:
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "100.0")  # very stale
        now = 100.0 + 30 * SECONDS_PER_DAY  # 30 days later

        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: now)
        assert outcome.scanned == 1
        assert outcome.retired == 1
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "retired"

    def test_keeps_recent(self, store: VerseStore) -> None:
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "moss", "")
        store.set_attribute(eid, "auto_created", "1")
        last_seen = 1000.0
        store.set_attribute(eid, "last_seen_ts", str(last_seen))
        now = last_seen + 5 * SECONDS_PER_DAY  # 5 days < 14-day cutoff

        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: now)
        assert outcome == (1, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_missing_last_seen_ts_is_left_untouched(self, store: VerseStore) -> None:
        """An auto_created entity with NO last_seen_ts attribute at all is
        scanned but never retired (no heartbeat -> no decision). Recovered
        state may lack the key entirely; aging must not crash or retire on
        absence."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        # Deliberately do NOT set last_seen_ts.
        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome.scanned == 1
        assert outcome.retired == 0
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_malformed_last_seen_ts_warns_and_skips(
        self, store: VerseStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-numeric last_seen_ts (corrupt/recovered state) is logged at
        WARNING and the entity is skipped: scanned but not retired, and never
        crashes on float() of garbage."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "not-a-number")
        with caplog.at_level(logging.WARNING, logger="llm.verse.aging"):
            outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome.scanned == 1
        assert outcome.retired == 0
        assert any("malformed last_seen_ts" in r.message for r in caplog.records)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_skips_manually_created(self, store: VerseStore) -> None:
        """An NPC without auto_created='1' must never be touched, even if
        last_seen_ts is past cutoff."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "manual", "")
        store.set_attribute(eid, "last_seen_ts", "0.0")
        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome == (0, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_skips_avatar_kind_defensively(self, store: VerseStore) -> None:
        """Even if a bug somewhere stamps auto_created='1' on an avatar,
        aging must not retire it. The kind!='avatar' guard is defensive."""
        from llm.verse.aging import age_auto_created_entities

        avatar_id = store.add_entity("avatar", "alice", "")
        store.set_attribute(avatar_id, "auto_created", "1")
        store.set_attribute(avatar_id, "last_seen_ts", "0.0")
        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome == (0, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (avatar_id,)).fetchone()
        assert row[0] == "active"

    def test_skips_pinned_entities(self, store: VerseStore) -> None:
        """A pinned auto_created NPC is explicit operator canon and must
        never be auto-retired, even when silent well past the cutoff.
        Pinned entities are protected, not aging candidates: scanned stays
        0 and the entity remains active."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "assgas archie", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")  # ancient
        store.set_attribute(eid, "pinned", "1")

        outcome = age_auto_created_entities(store, retire_after_days=14, now=lambda: 1e9)
        assert outcome == (0, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_digest_insert_bumps_last_seen(self, store: VerseStore) -> None:
        """When replace_events_with_lore_digest inserts a digest, every non-avatar
        entity in entity_ids has last_seen_ts bumped to ts (the heartbeat now
        rides on add_event itself). The bump is on the same conn as the
        INSERT — atomic with the digest write."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")  # stale

        digest_ts = 1000.0
        store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="ghost remained",
            entity_ids=(eid,),
            ts=digest_ts,
        )
        # Heartbeat fired
        assert store.get_attribute(eid, "last_seen_ts") == str(digest_ts)
        # Aging now sees a fresh entity → keeps it
        outcome = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: digest_ts + SECONDS_PER_DAY
        )
        assert outcome == (1, 0)
        with store.read_connection() as conn:
            row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
        assert row[0] == "active"

    def test_digest_truncated_entities_correctly_age(self, store: VerseStore) -> None:
        """Setup: _MAX_DIGEST_ENTITY_IDS + 8 auto-created NPCs all with
        last_seen_ts=0. Insert a digest with all of them in entity_ids; the
        digest layer truncates to _MAX_DIGEST_ENTITY_IDS. Aging then runs
        and the 8 truncated-out NPCs are correctly retired."""
        from llm.verse.aging import age_auto_created_entities
        from llm.verse.compaction import _MAX_DIGEST_ENTITY_IDS

        n_total = _MAX_DIGEST_ENTITY_IDS + 8
        ids: list[int] = []
        for i in range(n_total):
            eid = store.add_entity("npc", f"ghost{i}", "")
            store.set_attribute(eid, "auto_created", "1")
            store.set_attribute(eid, "last_seen_ts", "0.0")
            ids.append(eid)

        # digest_ts must be > retire_after_days so truncated-out entities
        # (last_seen_ts=0) age past cutoff while bumped survivors stay
        # fresh.
        digest_ts = 30 * SECONDS_PER_DAY
        union_ids_truncated = ids[:_MAX_DIGEST_ENTITY_IDS]
        store.replace_events_with_lore_digest(
            delete_ids=[],
            summary="many ghosts",
            entity_ids=union_ids_truncated,
            ts=digest_ts,
        )

        outcome = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: digest_ts + 5 * SECONDS_PER_DAY
        )
        assert outcome.retired == 8
        survivors_status = []
        truncated_status = []
        with store.read_connection() as conn:
            for i, eid in enumerate(ids):
                row = conn.execute("SELECT status FROM entities WHERE id=?", (eid,)).fetchone()
                if i < _MAX_DIGEST_ENTITY_IDS:
                    survivors_status.append(row[0])
                else:
                    truncated_status.append(row[0])
        assert set(survivors_status) == {"active"}
        assert set(truncated_status) == {"retired"}

    def test_applied_add_event_bumps_last_seen(self, store: VerseStore) -> None:
        """apply_direct(add_event) — verse_edit's add_event op — DOES auto-bump
        last_seen_ts on linked non-avatar entities: any event write counts as
        activity, so an NPC interacted with only via events must not age out.
        Aging then keeps the entity alive with no explicit heartbeat."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")

        store.apply_direct(
            op="add_event",
            payload={"summary": "ghost lurked", "entity_ids": [eid]},
            source="llm",
            provenance="test",
        )
        # The event write IS the heartbeat — last_seen_ts is bumped
        last_seen = float(store.get_attribute(eid, "last_seen_ts") or "0")
        assert last_seen > 0.0

        # Aging keeps the entity alive without any explicit heartbeat
        keep = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: last_seen + SECONDS_PER_DAY
        )
        assert keep.retired == 0

    def test_verse_act_event_keeps_npc_alive(self, store: VerseStore) -> None:
        """Plain add_event (the verse_act path) bumps last_seen_ts on a linked
        NPC but never on a linked avatar — an NPC interacted with exclusively
        via verse_act must not age out despite the activity."""
        npc_id = store.add_entity("npc", "ghost", "")
        store.set_attribute(npc_id, "auto_created", "1")
        store.set_attribute(npc_id, "last_seen_ts", "0.0")
        avatar_id = store.add_entity("avatar", "alice", "")

        store.add_event("alice examines ghost", [avatar_id, npc_id], "avatar")

        assert float(store.get_attribute(npc_id, "last_seen_ts") or "0") > 0.0
        # Avatars are lifecycle-managed by opt-in/out, not aging heartbeats.
        assert store.get_attribute(avatar_id, "last_seen_ts") is None

    def test_applied_set_attribute_bumps_last_seen(self, store: VerseStore) -> None:
        """apply_direct(set_attribute) does NOT auto-bump last_seen_ts; an explicit
        heartbeat is required. After the heartbeat, aging keeps the entity alive."""
        from llm.verse.aging import age_auto_created_entities

        eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(eid, "auto_created", "1")
        store.set_attribute(eid, "last_seen_ts", "0.0")

        store.apply_direct(
            op="set_attribute",
            payload={"entity_id": eid, "key": "mood", "value": "wary"},
            source="llm",
            provenance="test",
        )
        # Step 3: op must NOT auto-bump — last_seen_ts stays at "0.0"
        assert store.get_attribute(eid, "last_seen_ts") == "0.0"

        # Step 4: explicit heartbeat
        store.set_attribute(eid, "last_seen_ts", str(time.time()))

        # Step 5: bump recorded
        last_seen = float(store.get_attribute(eid, "last_seen_ts") or "0")
        assert last_seen > 0.0

        # Step 6: aging keeps the recently-heartbeated entity alive
        keep = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: last_seen + SECONDS_PER_DAY
        )
        assert keep.retired == 0

    def test_applied_add_relation_bumps_both_endpoints(self, store: VerseStore) -> None:
        """apply_direct(add_relation) does NOT auto-bump last_seen_ts on either
        endpoint; explicit heartbeats are required. After the heartbeats, aging
        keeps both entities alive."""
        from llm.verse.aging import age_auto_created_entities

        a = store.add_entity("npc", "alpha", "")
        b = store.add_entity("npc", "beta", "")
        for eid in (a, b):
            store.set_attribute(eid, "auto_created", "1")
            store.set_attribute(eid, "last_seen_ts", "0.0")

        store.apply_direct(
            op="add_relation",
            payload={"from_id": a, "to_id": b, "kind": "ally"},
            source="llm",
            provenance="test",
        )
        # Step 3: op must NOT auto-bump either endpoint
        assert store.get_attribute(a, "last_seen_ts") == "0.0"
        assert store.get_attribute(b, "last_seen_ts") == "0.0"

        # Step 4: explicit heartbeats on both endpoints
        now_ts = str(time.time())
        store.set_attribute(a, "last_seen_ts", now_ts)
        store.set_attribute(b, "last_seen_ts", now_ts)

        # Step 5: bumps recorded on both
        last_seen_a = float(store.get_attribute(a, "last_seen_ts") or "0")
        last_seen_b = float(store.get_attribute(b, "last_seen_ts") or "0")
        assert last_seen_a > 0.0
        assert last_seen_b > 0.0

        # Step 6: aging keeps both recently-heartbeated entities alive
        keep = age_auto_created_entities(
            store, retire_after_days=14, now=lambda: last_seen_a + SECONDS_PER_DAY
        )
        assert keep.retired == 0

    def test_add_event_with_invalid_ref_skips_bad_id_bumps_real(self, store: VerseStore) -> None:
        """apply_direct(add_event) with a nonexistent entity id in entity_ids
        still succeeds: the bad id is silently dropped from event_actor AND
        gets no heartbeat (it has no entity row to bump), while the real
        linked NPC is bumped — a hallucinated co-actor must not crash the
        write or cost the real actor its heartbeat."""
        real_eid = store.add_entity("npc", "ghost", "")
        store.set_attribute(real_eid, "auto_created", "1")
        store.set_attribute(real_eid, "last_seen_ts", "0.0")
        nonexistent_id = real_eid + 999_999

        store.apply_direct(
            op="add_event",
            payload={
                "summary": "phantom event",
                "entity_ids": [real_eid, nonexistent_id],
            },
            source="llm",
            provenance="test",
        )
        # The real entity got the heartbeat; the phantom id wrote nothing.
        assert float(store.get_attribute(real_eid, "last_seen_ts") or "0") > 0.0
        with store.read_connection() as conn:
            rows = conn.execute("SELECT entity_id FROM event_actor ORDER BY entity_id").fetchall()
            attr_rows = conn.execute(
                "SELECT 1 FROM attributes WHERE entity_id=?", (nonexistent_id,)
            ).fetchall()
        assert [r[0] for r in rows] == [real_eid]
        assert attr_rows == []


class TestAgingExemptsAuthorLocked:
    def test_author_locked_npc_not_retired(self, store):
        h = store.add_entity("npc", "Harry")
        store.set_attribute(h, "auto_created", "1")
        store.set_attribute(h, "last_seen_ts", "0.0")  # ancient
        store.set_author_locked(h, True)
        from llm.verse.aging import age_auto_created_entities

        age_auto_created_entities(store, retire_after_days=1, now=lambda: time.time())
        assert store.get_entity(h).status == "active"


class TestAgingExemptsOccupiedPlace:
    def test_occupied_place_survives_aging(self, store: VerseStore) -> None:
        """A stale auto_created place that is still an active avatar's
        location must never be auto-retired: retiring it would strand the
        avatar at a ghost location. The place is exempt (not scanned)."""
        from llm.verse.aging import age_auto_created_entities

        place_id = store.add_entity("place", "The Clearing", "a quiet glade")
        store.set_attribute(place_id, "auto_created", "1")
        store.set_attribute(place_id, "last_seen_ts", "0")  # ancient

        avatar_id = store.add_entity("avatar", "wanderer", "")
        store.set_attribute(avatar_id, "location", "The Clearing")

        outcome = age_auto_created_entities(store, retire_after_days=30, now=lambda: 1e9)
        assert store.get_entity(place_id).status == "active"
        assert outcome.retired == 0
        # Exemption must sit BEFORE `scanned += 1` (same as pinned/author_locked),
        # so an occupied place is not even counted as scanned.
        assert outcome.scanned == 0

    def test_unoccupied_stale_place_is_retired(self, store: VerseStore) -> None:
        """Same ancient auto_created place, but with NO avatar standing in
        it, is retired normally. Guards against the occupancy exemption
        being too broad."""
        from llm.verse.aging import age_auto_created_entities

        place_id = store.add_entity("place", "The Clearing", "a quiet glade")
        store.set_attribute(place_id, "auto_created", "1")
        store.set_attribute(place_id, "last_seen_ts", "0")  # ancient

        outcome = age_auto_created_entities(store, retire_after_days=30, now=lambda: 1e9)
        assert store.get_entity(place_id).status == "retired"
        assert outcome.retired == 1
        assert outcome.scanned == 1  # not exempt → counted and retired
