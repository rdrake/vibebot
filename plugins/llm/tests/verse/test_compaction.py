"""Tests for the daily verse retention compaction helper."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_verse_compaction_model_key_registered() -> None:
    """The new global key exists with the old loomModel default."""
    import llm.config  # noqa: F401 — import side effect registers the values
    from supybot import conf

    val = conf.supybot.plugins.LLM.verseCompactionModel()
    assert val == "anthropic/claude-haiku-5-5"


def test_relocated_client_types_importable_from_compaction():
    from llm.verse.compaction import (
        ProviderVerseClient,
        VerseCallUsage,
        VerseModelClient,
    )

    usage = VerseCallUsage(prompt_tokens=1, completion_tokens=2, cost=0.0)
    assert usage.completion_tokens == 2
    # Structural check only. Do NOT use isinstance(ProviderVerseClient(),
    # VerseModelClient): VerseModelClient is a plain Protocol (not
    # @runtime_checkable), so isinstance() would raise TypeError, not pass.
    assert callable(getattr(ProviderVerseClient, "call", None))
    assert hasattr(VerseModelClient, "call")


@pytest.fixture
def verse_db_dir(tmp_path: Path) -> Path:
    d = tmp_path / "verse"
    d.mkdir()
    return d


def test_lore_digest_is_stamped_llm(store):
    """Compaction digests are now source='llm', not 'loom'."""
    e1 = store.add_entity("npc", "Aldous", "")
    ev1 = store.apply_direct(
        op="add_event",
        payload={"summary": "old deed one", "entity_ids": [e1]},
        source="avatar",
        provenance="test",
    )
    ev2 = store.apply_direct(
        op="add_event",
        payload={"summary": "old deed two", "entity_ids": [e1]},
        source="avatar",
        provenance="test",
    )
    store.replace_events_with_lore_digest(
        delete_ids=[ev1, ev2],
        summary="A digest of Aldous's old deeds.",
        entity_ids=[e1],
        ts=1000.0,
    )
    with store.read_connection() as conn:
        rows = conn.execute("SELECT source FROM events WHERE summary LIKE 'A digest%'").fetchall()
    assert rows and all(r[0] == "llm" for r in rows)


class _FakeClient:
    def __init__(self, content: str = "A digest of past events.") -> None:
        self.content = content
        self.calls: list[dict] = []

    def call(self, *, op: str, model: str, messages: list[dict[str, str]]):
        from llm.verse.compaction import VerseCallUsage

        self.calls.append({"op": op, "model": model, "messages": messages})
        return self.content, VerseCallUsage(prompt_tokens=10, completion_tokens=20, cost=0.0)


class TestCompactVerse:
    def test_skips_when_retention_zero(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=1.0)
        out = compact_verse(
            store,
            retention_days=0,
            min_keep_events=20,
            model="anthropic/claude-haiku-5-5",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 1_000_000.0,
        )
        assert out.state == "skipped_disabled"
        assert out.total_events == 0
        assert out.kept_in_digest == 0

    def test_skips_when_below_min_keep(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        insert_event_at(store, summary="x", entity_ids=[], source="loom", ts=1.0)
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 100_000_000.0,
        )
        assert out.state == "skipped_below_floor"
        assert out.total_events == 1
        assert out.kept_in_digest == 0

    def test_skips_when_no_old_events(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"e{i}",
                entity_ids=[],
                source="loom",
                ts=1_000_000.0 - i,
            )
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 1_000_000.0,
        )
        assert out.state == "skipped_no_events"
        assert out.total_events == 25
        assert out.kept_in_digest == 0

    def test_compacts_old_events_into_single_digest(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )
        for i in range(25):
            insert_event_at(
                store,
                summary=f"new{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 1.0,
            )
        usage_calls: list[dict] = []
        client = _FakeClient(content="Past events: a wood, a brook, a whisper.")
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="anthropic/claude-haiku-5-5",
            client=client,
            log_usage=lambda **kw: usage_calls.append(kw),
            now=lambda: now,
        )
        assert out.state == "compacted"
        assert out.total_events == 50
        assert out.kept_in_digest == 0  # entity_ids=[] in setup
        with store.read_connection() as conn:
            rows = conn.execute("SELECT summary, source FROM events ORDER BY ts ASC").fetchall()
        assert len(rows) == 26
        # The digest is stamped at now() so it lands as the NEWEST row
        # (last in ASC order). Stamping at max(batch.ts) used to put it
        # ahead of the cutoff and re-summarise it the next day.
        assert rows[-1][1] == "llm"
        assert "Past events" in rows[-1][0]
        assert client.calls and client.calls[0]["op"] == "compact"
        assert len(usage_calls) == 1
        assert usage_calls[0]["op"] == "compact"

    def test_digest_is_not_re_summarised_on_next_day(self, verse_db_dir: Path) -> None:
        """Regression: a digest stamped at max(batch.ts) used to sit
        older than the retention cutoff, so the next daily pass picked
        it up via ``events_older_than`` and re-summarised it. Now: the
        digest is stamped at now(), so the next pass (one day later)
        finds nothing eligible.
        """
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        t0 = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=t0 - 60 * seconds_per_day,
            )
        client = _FakeClient(content="A digest.")
        out1 = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=client,
            log_usage=lambda **kw: None,
            now=lambda: t0,
        )
        assert out1.state == "compacted"
        assert out1.total_events == 25
        assert out1.kept_in_digest == 0  # entity_ids=[] in setup

        # One day later, no new events arrive. The digest stamped at
        # t0 is now 1 day old, well within the 30-day retention. With
        # min_keep_events=1 the floor check passes (only the digest
        # remains), so we must reach the events_older_than query and
        # find nothing eligible.
        out2 = compact_verse(
            store,
            retention_days=30,
            min_keep_events=1,
            model="m",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: t0 + seconds_per_day,
        )
        assert out2.state == "skipped_no_events"
        assert out2.total_events == 1  # only the digest survived
        assert out2.kept_in_digest == 0

    def test_long_backlog_only_deletes_what_was_summarised(self, verse_db_dir: Path) -> None:
        """If there are 500 old events and the per-pass cap is 200,
        exactly 200 originals are deleted; 300 survive for the next pass.
        Regression test for the v1 plan bug where ALL olds were deleted
        but only the last 200 were shown to the model."""
        from llm.verse.compaction import _MAX_EVENTS_PER_PASS, compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(500):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day - i,
            )
        client = _FakeClient(content="A long-ago digest.")
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        assert out.state == "compacted"
        assert out.total_events == 500
        assert out.kept_in_digest == 0  # entity_ids=[] in setup
        assert _MAX_EVENTS_PER_PASS == 200
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 500 - _MAX_EVENTS_PER_PASS + 1

    def test_bullet_trim_only_deletes_events_in_prompt(self, verse_db_dir: Path) -> None:
        """When the bullet block exceeds _MAX_BULLET_BLOCK_CHARS, the NEWEST
        bullets are dropped from the prompt so the oldest-fitting prefix is
        compacted this pass (forward progress — the oldest lore is never
        stranded). Regression: previously the delete_ids list still referenced
        the FULL batch, so events the LLM never saw got deleted. Now: delete_ids
        matches exactly the events whose bullets remain in the prompt; the
        trimmed (newest) events survive for the next pass.
        """
        from llm.verse.compaction import (
            _MAX_BULLET_BLOCK_CHARS,
            _MAX_SUMMARY_CHARS_PER_EVENT,
            compact_verse,
        )
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")

        # Long summaries cap at _MAX_SUMMARY_CHARS_PER_EVENT (240).
        # Bullet line is "- " + summary -> ~242 chars per line.
        # 70 events * (242 + 1 newline) ~= 17000 chars > 16000 cap.
        n_events = 70
        long_summary = "x" * (_MAX_SUMMARY_CHARS_PER_EVENT * 2)
        ids: list[int] = []
        for i in range(n_events):
            eid = insert_event_at(
                store,
                summary=f"e{i:03d}-{long_summary}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day - (n_events - i),
            )
            ids.append(eid)

        client = _FakeClient(content="A digest.")
        out = compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        assert out.state == "compacted"
        assert out.total_events == n_events
        assert out.kept_in_digest == 0  # entity_ids=[] in setup

        # The bullet block sent to the LLM should be capped.
        bullets = client.calls[0]["messages"][1]["content"]
        assert len(bullets) <= _MAX_BULLET_BLOCK_CHARS
        kept_bullet_lines = bullets.count("\n") + 1 if bullets else 0
        assert kept_bullet_lines < n_events  # i.e. some were trimmed

        # After compaction, surviving events = (n_events - kept) originals
        # + 1 digest. The bullet block trims the NEWEST events off the end so
        # the OLDEST-fitting prefix is compacted + deleted this pass — that is
        # what guarantees forward progress (the oldest lore never gets stranded
        # and re-served every pass). So survivors are the NEWEST events.
        with store.read_connection() as conn:
            rows = conn.execute("SELECT id, source FROM events ORDER BY ts ASC, id ASC").fetchall()
        # 1 digest row (source='llm') + (n_events - kept_bullet_lines) survivors.
        survivor_ids = [r[0] for r in rows if r[1] != "llm"]
        digest_rows = [r for r in rows if r[1] == "llm"]
        assert len(digest_rows) == 1
        # Number of deletions equals number of bullet lines actually shown.
        assert len(survivor_ids) == n_events - kept_bullet_lines
        # Survivors are the NEWEST events (end-trimmed off the prompt); the
        # oldest were compacted into the digest and deleted.
        assert survivor_ids == ids[kept_bullet_lines:]

    def test_per_event_summary_cap_truncates_long_summaries(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import (
            _MAX_SUMMARY_CHARS_PER_EVENT,
            compact_verse,
        )
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        long_summary = "x" * 5000
        for _ in range(25):
            insert_event_at(
                store,
                summary=long_summary,
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )
        client = _FakeClient()
        compact_verse(
            store,
            retention_days=30,
            min_keep_events=20,
            model="m",
            client=client,
            log_usage=lambda **kw: None,
            now=lambda: now,
        )
        assert client.calls
        bullets = client.calls[0]["messages"][1]["content"]
        for line in bullets.splitlines():
            assert len(line) <= _MAX_SUMMARY_CHARS_PER_EVENT + 4

    def test_entity_ids_truncation_logs_when_capped(self, verse_db_dir: Path, caplog) -> None:
        import json as _json
        import logging

        from llm.verse.compaction import _MAX_DIGEST_ENTITY_IDS, compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"e{i}",
                entity_ids=list(range(i * 4, i * 4 + 4)),
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )
        with caplog.at_level(logging.INFO, logger="llm.verse.compaction"):
            compact_verse(
                store,
                retention_days=30,
                min_keep_events=20,
                model="m",
                client=_FakeClient(),
                log_usage=lambda **kw: None,
                now=lambda: now,
            )
        assert any("entity_ids truncated" in r.message for r in caplog.records)
        with store.read_connection() as conn:
            row = conn.execute(
                "SELECT entity_ids FROM events WHERE source='llm' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        assert len(_json.loads(row[0])) == _MAX_DIGEST_ENTITY_IDS

    def test_client_failure_aborts_without_data_loss(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import compact_verse
        from llm.verse.store import VerseStore

        from .conftest import insert_event_at

        seconds_per_day = 86400
        now = 100_000_000.0
        store = VerseStore(verse_db_dir, "#afnet")
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now - 60 * seconds_per_day,
            )

        class Bomb:
            def call(self, **kw):
                raise RuntimeError("model down")

        with pytest.raises(RuntimeError):
            compact_verse(
                store,
                retention_days=30,
                min_keep_events=20,
                model="m",
                client=Bomb(),
                log_usage=lambda **kw: None,
                now=lambda: now,
            )
        with store.read_connection() as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 25


class TestCompactionOutcomeShape:
    def test_returns_namedtuple(self, verse_db_dir: Path) -> None:
        from llm.verse.compaction import CompactionOutcome, compact_verse
        from llm.verse.store import VerseStore

        store = VerseStore(verse_db_dir, "#shape")
        outcome = compact_verse(
            store,
            retention_days=0,  # trips skipped_disabled
            min_keep_events=20,
            model="m",
            client=_FakeClient(),
            log_usage=lambda **kw: None,
            now=lambda: 0.0,
        )
        assert isinstance(outcome, CompactionOutcome)
        assert outcome.state == "skipped_disabled"
        assert outcome.total_events == 0
        assert outcome.kept_in_digest == 0


class TestNextLocalTime:
    def test_returns_today_when_hhmm_in_future(self) -> None:
        import time as _t

        from llm.verse.compaction import _next_local_time

        # construct a "now" at local 03:00, ask for 10:00
        struct = _t.struct_time((2026, 5, 8, 3, 0, 0, 4, 128, -1))
        now_ts = _t.mktime(struct)
        out = _next_local_time("10:00", now=lambda: now_ts)
        assert out > now_ts
        assert (out - now_ts) < 86400  # under one day away

    def test_returns_tomorrow_when_hhmm_already_passed(self) -> None:
        import time as _t

        from llm.verse.compaction import _next_local_time

        struct = _t.struct_time((2026, 5, 8, 14, 0, 0, 4, 128, -1))
        now_ts = _t.mktime(struct)
        out = _next_local_time("10:00", now=lambda: now_ts)
        assert (out - now_ts) > 0
        assert (out - now_ts) < 86400  # under one day away

    def test_malformed_hhmm_falls_back_to_one_hour(self) -> None:
        from llm.verse.compaction import _next_local_time

        out = _next_local_time("not-a-time", now=lambda: 1000.0)
        assert 3590.0 < (out - 1000.0) < 3610.0

    def test_out_of_range_hhmm_falls_back(self) -> None:
        from llm.verse.compaction import _next_local_time

        out = _next_local_time("25:99", now=lambda: 1000.0)
        assert 3590.0 < (out - 1000.0) < 3610.0
