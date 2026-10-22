"""Tests for SQLite persistence layer."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from llm.persistence import (
    LLMDatabase,
    PendingTaskRow,
    ReminderRow,
    UsageRank,
    lease_until,
)


def test_existing_reminder_gets_empty_action_prompt(tmp_path: Path) -> None:
    """GIVEN a v8 reminder WHEN opened with v9 code THEN action_prompt defaults to empty string."""
    db_path = str(tmp_path / "test.db")
    fire_at = time.time() + 300
    created_at = time.time()

    # Create a v8-style schema reminder row (no action_prompt/account columns).
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE reminders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_name TEXT UNIQUE NOT NULL,
            nick TEXT NOT NULL,
            channel TEXT NOT NULL,
            message TEXT NOT NULL,
            fire_at REAL NOT NULL,
            created_at REAL NOT NULL
        );
    """)
    conn.execute(
        "INSERT INTO reminders (event_name, nick, channel, message, fire_at, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("evt-v8", "alice", "#general", "check build", fire_at, created_at),
    )
    conn.commit()
    conn.execute("PRAGMA user_version = 8")
    conn.commit()
    conn.close()

    db = LLMDatabase(db_path)
    reminders = db.load_pending_reminders()
    assert len(reminders) == 1
    assert reminders[0].event_name == "evt-v8"
    assert reminders[0].action_prompt == ""
    assert reminders[0].account is None
    db.close()


class TestDatabaseInit:
    """Test database initialization and schema creation."""

    def test_connect_raises_after_close(self, tmp_path: Path) -> None:
        """GIVEN a closed database WHEN a connection is requested THEN
        _connect raises instead of silently reopening.

        Teardown backstop: a straggler worker or untracked daemon thread
        touching the DB after die() must fail loudly, not reopen a
        connection to a database the process is finished with.
        """
        db = LLMDatabase(str(tmp_path / "test.db"))
        db.close()
        with pytest.raises(RuntimeError, match="closed"):
            db._connect()

    def test_wal_mode_enabled(self, test_db: LLMDatabase) -> None:
        """GIVEN a database WHEN connected THEN WAL journal mode is active."""
        conn = test_db._connect()
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert row is not None
            assert row[0] == "wal"
        finally:
            conn.close()

    def test_idempotent_init(self, tmp_path: Path) -> None:
        """GIVEN an existing database WHEN opened again THEN no error."""
        db_path = str(tmp_path / "test.db")
        db1 = LLMDatabase(db_path)
        db1.save_reminder("evt1", "nick", "#chan", "msg", time.time() + 60)
        db1.close()

        # Open the same database again — should not raise or lose data
        db2 = LLMDatabase(db_path)
        reminders = db2.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].event_name == "evt1"
        db2.close()


class TestSchemaMigration:
    """Test schema version migration from v1 to v2."""

    def test_migration_preserves_existing_usage_data(self, tmp_path: Path) -> None:
        """GIVEN a v1 database with usage data WHEN opened with v2 code THEN data preserved with defaults."""
        db_path = str(tmp_path / "test.db")

        # Manually create a v1 database (no prompt/status/error_detail columns)
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                nick TEXT NOT NULL,
                channel TEXT NOT NULL,
                command TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0.0
            );
        """)
        conn.execute(
            "INSERT INTO usage "
            "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (1000000.0, "alice", "#general", "ask", "gpt-4", 100, 50, 0.01),
        )
        conn.commit()
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()

        # Open with LLMDatabase — should run v2 migration
        db = LLMDatabase(db_path)

        # Verify original data is preserved
        summary = db.get_usage_summary()
        assert summary.total_requests == 1
        assert summary.total_prompt_tokens == 100
        assert summary.total_completion_tokens == 50
        assert summary.total_cost == pytest.approx(0.01)

        # Verify new columns exist with defaults
        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT prompt, status, error_detail FROM usage WHERE nick = 'alice'"
            ).fetchone()
            assert row is not None
            assert row[0] == ""  # prompt default
            assert row[1] == "success"  # status default
            assert row[2] == ""  # error_detail default
        finally:
            conn.close()
        db.close()

    def test_migrate_recovers_from_partial_alter_block(self, tmp_path: Path) -> None:
        """GIVEN a DB whose v3 block crashed after its first ADD COLUMN committed
        but before ``user_version`` advanced (delivery_state present, the rest
        missing, user_version still 2) WHEN opened THEN migration completes
        instead of raising 'duplicate column name' and wedging startup.

        ``user_version`` is stamped only at the end of ``_migrate``, so a mid-
        block crash re-runs the whole block on restart; the ADD COLUMNs must be
        idempotent. ``pending_tasks`` is the LIVE timeout-recovery queue, so this
        is the highest-impact instance of the class.
        """
        from llm.persistence import SCHEMA_VERSION

        db_path = str(tmp_path / "partial.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE pending_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                nick TEXT NOT NULL,
                reply_target TEXT NOT NULL,
                is_channel INTEGER NOT NULL,
                prompt_preview TEXT NOT NULL,
                model TEXT NOT NULL,
                request_data TEXT NOT NULL DEFAULT '{}',
                submitted_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL,
                claimed_until REAL NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT ''
            );
            -- the first ADD COLUMN of the v3 block committed before the crash:
            ALTER TABLE pending_tasks
                ADD COLUMN delivery_state TEXT NOT NULL DEFAULT 'pending';
        """)
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        # Must NOT raise sqlite3.OperationalError('duplicate column name').
        db = LLMDatabase(db_path)
        conn = db._connect()
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(pending_tasks)")}
            assert {
                "delivery_state",
                "result_payload",
                "last_delivery_error",
                "delivery_attempt_count",
                "origin_request_id",
            } <= cols
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        finally:
            conn.close()
        db.close()

    def test_migrate_is_fully_idempotent_on_rerun_from_zero(self, tmp_path: Path) -> None:
        """GIVEN a fully-migrated DB whose ``user_version`` was reset to 0
        (simulating a crash before the version stamp) WHEN ``_migrate`` re-runs
        every version-gated block against the already-current schema THEN it
        neither raises (duplicate ADD / missing DROP) nor changes the schema.

        Exercises BOTH idempotency helpers across every block, including the
        DROP-COLUMN branches that the partial-v3 case does not reach.
        """
        from llm.persistence import SCHEMA_VERSION

        db_path = str(tmp_path / "rerun.db")
        db = LLMDatabase(db_path)
        conn = db._connect()
        try:
            before = {row[1] for row in conn.execute("PRAGMA table_info(pending_tasks)")}
            conn.execute("PRAGMA user_version = 0")
            conn.commit()

            db._migrate()  # must be a clean no-op, not a crash

            after = {row[1] for row in conn.execute("PRAGMA table_info(pending_tasks)")}
            assert after == before
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        finally:
            conn.close()
        db.close()


# NOTE: round-trip, fire_at-ordering, and recurrence-mutual-exclusion
# invariants for reminders (and the parallel scheduled_llm_task family
# below) are now covered by test_persistence_scheduled_properties.py.
# The cases here are kept as executable specifications and to cover
# paths the property tests intentionally do not parameterize: the 24h
# EXPIRY_THRESHOLD cutoff, the IntegrityError on duplicate event_name,
# and the agenda's point lookups and in-place reschedule.
class TestReminderPersistence:
    """Test reminder CRUD operations."""

    def test_save_and_load_reminder(self, test_db: LLMDatabase) -> None:
        """GIVEN a database WHEN a reminder is saved THEN it can be loaded."""
        fire_at = time.time() + 300
        row_id = test_db.save_reminder("test_event", "alice", "#general", "Check build", fire_at)

        assert isinstance(row_id, int)
        assert row_id > 0

        reminders = test_db.load_pending_reminders()
        assert len(reminders) == 1

        r = reminders[0]
        assert isinstance(r, ReminderRow)
        assert r.event_name == "test_event"
        assert r.nick == "alice"
        assert r.channel == "#general"
        assert r.message == "Check build"
        assert r.fire_at == fire_at

    def test_delete_reminder_returns_true(self, test_db: LLMDatabase) -> None:
        """GIVEN a saved reminder WHEN deleted THEN returns True."""
        test_db.save_reminder("evt1", "alice", "#chan", "msg", time.time() + 60)

        assert test_db.delete_reminder("evt1") is True

    def test_delete_nonexistent_reminder_returns_false(self, test_db: LLMDatabase) -> None:
        """GIVEN no reminders WHEN deleting nonexistent event THEN returns False."""
        assert test_db.delete_reminder("no_such_event") is False

    def test_load_excludes_reminders_older_than_24h(self, test_db: LLMDatabase) -> None:
        """GIVEN a reminder >24h overdue WHEN loading pending THEN it is excluded."""
        # fire_at was 25 hours ago — should be excluded
        old_fire_at = time.time() - (25 * 3600)
        test_db.save_reminder("old_event", "alice", "#chan", "old msg", old_fire_at)

        reminders = test_db.load_pending_reminders()
        assert len(reminders) == 0

    def test_load_includes_reminders_less_than_24h_overdue(self, test_db: LLMDatabase) -> None:
        """GIVEN a reminder <24h overdue WHEN loading pending THEN it is included."""
        # fire_at was 23 hours ago — should still be included for delivery on restart
        recent_fire_at = time.time() - (23 * 3600)
        test_db.save_reminder("recent_event", "alice", "#chan", "recent msg", recent_fire_at)

        reminders = test_db.load_pending_reminders()
        assert len(reminders) == 1
        assert reminders[0].event_name == "recent_event"

    def test_agenda_lookups_and_in_place_move(self, test_db: LLMDatabase) -> None:
        """GIVEN reminders WHEN the agenda reads and moves them THEN rows keep their id."""
        now = time.time()
        test_db.save_reminder("due", "alice", "#chan", "a", now - 5, recurrence_seconds=60)
        test_db.save_reminder("later", "alice", "#chan", "b", now + 300)

        assert [r.event_name for r in test_db.load_due_reminders(now)] == ["due"]
        assert [name for name, _at in test_db.reminder_fire_times()] == ["due", "later"]
        before = test_db.get_reminder("due")
        assert before is not None
        test_db.update_reminder_fire_at("due", now + 600, chain_position=2)
        after = test_db.get_reminder("due")
        assert after == before._replace(fire_at=now + 600, chain_position=2)
        assert test_db.get_reminder("missing") is None

    @pytest.mark.parametrize(
        "recurrence",
        [{"recurrence_seconds": 3600}, {"recurrence_rrule": "FREQ=DAILY;BYHOUR=9"}],
    )
    def test_recurring_chain_survives_a_long_outage(
        self, test_db: LLMDatabase, recurrence: dict
    ) -> None:
        """GIVEN a recurring row 25h overdue WHEN loading THEN it is not
        skipped, so its owner still sees the chain after an outage."""
        test_db.save_reminder(
            "chain", "alice", "#chan", "stretch", time.time() - (25 * 3600), **recurrence
        )

        assert [r.event_name for r in test_db.load_pending_reminders()] == ["chain"]

    def test_unique_event_name_constraint(self, test_db: LLMDatabase) -> None:
        """GIVEN a saved reminder WHEN saving with same event_name THEN IntegrityError."""
        test_db.save_reminder("dup_event", "alice", "#chan", "first", time.time() + 60)

        with pytest.raises(sqlite3.IntegrityError):
            test_db.save_reminder("dup_event", "bob", "#other", "second", time.time() + 120)


class TestUsageTracking:
    """Test usage logging and aggregation."""

    def test_log_usage_and_summarize(self, test_db: LLMDatabase) -> None:
        """GIVEN logged usage WHEN summarizing THEN totals are correct."""
        test_db.log_usage("alice", "#chan", "ask", "gpt-4", 100, 50, 0.01)
        test_db.log_usage("bob", "#chan", "ask", "gpt-4", 200, 100, 0.02)

        summary = test_db.get_usage_summary()
        assert summary.total_requests == 2
        assert summary.total_prompt_tokens == 300
        assert summary.total_completion_tokens == 150
        assert summary.total_cost == pytest.approx(0.03)

    def test_summary_with_since_filter(self, test_db: LLMDatabase) -> None:
        """GIVEN old and new usage WHEN summarizing with since THEN only counts recent."""
        # Insert an "old" record by manipulating time
        conn = test_db._connect()
        try:
            old_time = time.time() - 7200  # 2 hours ago
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#chan", "ask", "gpt-4", 100, 50, 0.01),
            )
            conn.commit()
        finally:
            conn.close()

        # Insert a recent record
        test_db.log_usage("bob", "#chan", "ask", "gpt-4", 200, 100, 0.02)

        # Filter to only the last hour
        since = time.time() - 3600
        summary = test_db.get_usage_summary(since=since)
        assert summary.total_requests == 1
        assert summary.total_prompt_tokens == 200
        assert summary.total_completion_tokens == 100
        assert summary.total_cost == pytest.approx(0.02)

    def test_empty_summary_returns_zeros(self, test_db: LLMDatabase) -> None:
        """GIVEN no usage records WHEN summarizing THEN returns zeros."""
        summary = test_db.get_usage_summary()
        assert summary.total_requests == 0
        assert summary.total_prompt_tokens == 0
        assert summary.total_completion_tokens == 0
        assert summary.total_cost == pytest.approx(0.0)

    def test_usage_by_nick_sorted_by_cost(self, test_db: LLMDatabase) -> None:
        """GIVEN usage from multiple nicks WHEN querying by nick THEN sorted by cost desc."""
        test_db.log_usage("alice", "#chan", "ask", "gpt-4", 100, 50, 0.01)
        test_db.log_usage("bob", "#chan", "ask", "gpt-4", 200, 100, 0.05)
        test_db.log_usage("charlie", "#chan", "ask", "gpt-4", 50, 25, 0.03)

        breakdown = test_db.get_usage_by_nick()
        assert len(breakdown) == 3
        assert breakdown[0].name == "bob"
        assert breakdown[0].total_cost == pytest.approx(0.05)
        assert breakdown[1].name == "charlie"
        assert breakdown[1].total_cost == pytest.approx(0.03)
        assert breakdown[2].name == "alice"
        assert breakdown[2].total_cost == pytest.approx(0.01)

    def test_usage_by_channel_sorted_by_cost(self, test_db: LLMDatabase) -> None:
        """GIVEN usage in multiple channels WHEN querying by channel THEN sorted by cost desc."""
        test_db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.01)
        test_db.log_usage("alice", "#dev", "ask", "gpt-4", 200, 100, 0.05)
        test_db.log_usage("bob", "#general", "code", "gpt-4", 50, 25, 0.02)

        breakdown = test_db.get_usage_by_channel()
        assert len(breakdown) == 2
        assert breakdown[0].name == "#dev"
        assert breakdown[0].total_cost == pytest.approx(0.05)
        assert breakdown[1].name == "#general"
        assert breakdown[1].total_cost == pytest.approx(0.03)

    def test_usage_by_nick_respects_limit(self, test_db: LLMDatabase) -> None:
        """GIVEN many nicks WHEN querying with limit THEN only top N returned."""
        for i in range(10):
            test_db.log_usage(f"user{i}", "#chan", "ask", "gpt-4", 100, 50, 0.01 * (i + 1))

        breakdown = test_db.get_usage_by_nick(limit=3)
        assert len(breakdown) == 3
        # Should be the top 3 by cost (user9, user8, user7)
        assert breakdown[0].name == "user9"
        assert breakdown[1].name == "user8"
        assert breakdown[2].name == "user7"

    def test_usage_by_dimension_respects_since(self, test_db: LLMDatabase) -> None:
        """GIVEN last month's spend and this month's WHEN the @usage report
        asks for the top users and channels since month start THEN only this
        month's spend is counted, not all-time spend labelled as this month."""
        test_db.log_usage("alice", "#old", "ask", "gpt-4", 100, 50, 0.50)
        with test_db._write_txn() as conn:
            conn.execute("UPDATE usage SET timestamp = timestamp - 40 * 86400")
        test_db.log_usage("bob", "#new", "ask", "gpt-4", 200, 100, 0.02)

        since = time.time() - 86400
        by_nick = test_db.get_usage_by_nick(since=since)
        by_channel = test_db.get_usage_by_channel(since=since)

        assert [(b.name, b.total_requests) for b in by_nick] == [("bob", 1)]
        assert [(b.name, b.total_requests) for b in by_channel] == [("#new", 1)]
        # Without since, both rows count.
        assert {b.name for b in test_db.get_usage_by_nick()} == {"alice", "bob"}


class TestRoundTrip:
    """Integration tests for full persistence round-trips."""

    def test_concurrent_writes_thread_safety(self, tmp_path: Path) -> None:
        """GIVEN multiple threads WHEN writing concurrently THEN no corruption."""
        db = LLMDatabase(str(tmp_path / "test.db"))
        errors: list[Exception] = []

        def write_usage(n: int) -> None:
            try:
                for _i in range(20):
                    db.log_usage(f"user{n}", "#test", "ask", "model", 10, 5, 0.001)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_usage, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        summary = db.get_usage_summary()
        assert summary.total_requests == 100  # 5 threads * 20 writes
        db.close()


class TestFilteredUsageSummary:
    """Test channel- and nick-filtered usage summaries."""

    def test_channel_summary_filters_to_channel(self, test_db: LLMDatabase) -> None:
        """GIVEN usage in two channels WHEN querying one THEN only that channel counted."""
        test_db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.01)
        test_db.log_usage("bob", "#general", "ask", "gpt-4", 200, 100, 0.02)
        test_db.log_usage("alice", "#dev", "ask", "gpt-4", 300, 150, 0.05)

        summary = test_db.get_usage_summary_for_channel("#general")
        assert summary.total_requests == 2
        assert summary.total_cost == pytest.approx(0.03)

    def test_channel_summary_with_since_filter(self, test_db: LLMDatabase) -> None:
        """GIVEN old and new usage WHEN filtering by since THEN only recent counted."""
        conn = test_db._connect()
        try:
            old_time = time.time() - 7200
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#test", "ask", "gpt-4", 100, 50, 0.01),
            )
            conn.commit()
        finally:
            conn.close()
        test_db.log_usage("bob", "#test", "ask", "gpt-4", 200, 100, 0.02)

        since = time.time() - 3600
        summary = test_db.get_usage_summary_for_channel("#test", since=since)
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.02)

    def test_nick_summary_filters_to_nick(self, test_db: LLMDatabase) -> None:
        """GIVEN usage from two nicks WHEN querying one THEN only that nick counted."""
        test_db.log_usage("alice", "#test", "ask", "gpt-4", 100, 50, 0.01)
        test_db.log_usage("bob", "#test", "ask", "gpt-4", 200, 100, 0.05)

        summary = test_db.get_usage_summary_for_nick("alice")
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.01)

    def test_nick_summary_scoped_to_channel(self, test_db: LLMDatabase) -> None:
        """GIVEN usage in two channels WHEN querying nick with channel THEN scoped."""
        test_db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.01)
        test_db.log_usage("alice", "#dev", "ask", "gpt-4", 200, 100, 0.05)

        summary = test_db.get_usage_summary_for_nick("alice", channel="#general")
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.01)

    def test_nick_summary_with_since_and_channel(self, test_db: LLMDatabase) -> None:
        """GIVEN old and new usage WHEN filtering by since and channel THEN both applied."""
        conn = test_db._connect()
        try:
            old_time = time.time() - 7200
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#test", "ask", "gpt-4", 100, 50, 0.01),
            )
            conn.commit()
        finally:
            conn.close()
        test_db.log_usage("alice", "#test", "ask", "gpt-4", 200, 100, 0.02)
        test_db.log_usage("alice", "#other", "ask", "gpt-4", 300, 150, 0.03)

        since = time.time() - 3600
        summary = test_db.get_usage_summary_for_nick("alice", since=since, channel="#test")
        assert summary.total_requests == 1
        assert summary.total_cost == pytest.approx(0.02)


class TestUsageRanking:
    """Test rank computation for channels and nicks.

    Monotone-rank, valid-range, unused→0, and zero-cost short-circuit
    invariants are covered by ``test_persistence_usage_properties.py``.
    What remains here exercises code paths the property tests do not
    parameterize: empty DB, ``since=`` filter, and ``channel=`` scoping.
    """

    def test_channel_rank_empty_db(self, test_db: LLMDatabase) -> None:
        """GIVEN no usage data WHEN ranking THEN rank is 0 total is 0."""
        rank = test_db.get_channel_rank("#any")
        assert rank == UsageRank(rank=0, total=0)

    def test_channel_rank_with_since(self, test_db: LLMDatabase) -> None:
        """GIVEN old and new usage WHEN ranking with since THEN only recent counted."""
        conn = test_db._connect()
        try:
            old_time = time.time() - 7200
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, completion_tokens, cost) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (old_time, "alice", "#old_top", "ask", "gpt-4", 100, 50, 1.00),
            )
            conn.commit()
        finally:
            conn.close()
        test_db.log_usage("bob", "#recent", "ask", "gpt-4", 100, 50, 0.01)

        since = time.time() - 3600
        rank = test_db.get_channel_rank("#recent", since=since)
        assert rank == UsageRank(rank=1, total=1)

    def test_nick_rank_scoped_to_channel(self, test_db: LLMDatabase) -> None:
        """GIVEN usage in multiple channels WHEN ranking with channel THEN scoped."""
        # alice is top globally but not in #dev
        test_db.log_usage("alice", "#general", "ask", "gpt-4", 100, 50, 0.50)
        test_db.log_usage("alice", "#dev", "ask", "gpt-4", 100, 50, 0.01)
        test_db.log_usage("bob", "#dev", "ask", "gpt-4", 100, 50, 0.10)

        rank = test_db.get_nick_rank("alice", channel="#dev")
        assert rank == UsageRank(rank=2, total=2)

    def test_nick_rank_empty_db(self, test_db: LLMDatabase) -> None:
        """GIVEN no usage data WHEN ranking nick THEN rank is 0 total is 0."""
        rank = test_db.get_nick_rank("anyone")
        assert rank == UsageRank(rank=0, total=0)


# NOTE: lifecycle invariants (claim mutual-exclusion, lease-deadline,
# attempt_count delta with/without increment, expired-row deletion) are
# now covered by test_persistence_pending_task_properties.py. What
# remains here is the basic save/load shape spec, the not-due skip
# (which was the mutation-test target — keep until property test
# stabilizes), and the cross-process reopen path. TestDeliveryStatePersistence
# (below) covers delivery_state_filter / max_delivery_attempts paths the
# state machine intentionally does not parameterize.
class TestPendingTasks:
    """Test pending task CRUD operations and claim/release semantics."""

    def test_save_and_load_pending_task(self, test_db: LLMDatabase) -> None:
        """GIVEN a database WHEN a pending task is saved THEN it can be loaded."""
        now = time.time()
        task_id = test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#general",
            is_channel=True,
            prompt_preview="What is the weather?",
            model="gpt-4",
            request_data='{"messages": []}',
            submitted_at=now,
            expires_at=now + 60,
            next_attempt_at=now,
        )

        assert isinstance(task_id, int)
        assert task_id > 0

        tasks = test_db.load_pending_tasks()
        assert len(tasks) == 1

        t = tasks[0]
        assert isinstance(t, PendingTaskRow)
        assert t.id == task_id
        assert t.task_type == "ask"
        assert t.nick == "alice"
        assert t.reply_target == "#general"
        assert t.is_channel == 1
        assert t.prompt_preview == "What is the weather?"
        assert t.model == "gpt-4"
        assert t.request_data == '{"messages": []}'
        assert t.attempt_count == 0
        assert t.last_error == ""

    def test_claim_skips_not_due_and_claimed(self, test_db: LLMDatabase) -> None:
        """GIVEN tasks not yet due or already claimed WHEN claiming THEN skipped."""
        now = time.time()

        # Not yet due (next_attempt_at in future)
        test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="future task",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now + 60,
        )

        claimed = test_db.claim_due_pending_tasks(now, limit=10, lease_seconds=120)
        assert len(claimed) == 0

    def test_survives_reopen(self, tmp_path: Path) -> None:
        """GIVEN saved pending task WHEN DB reopened THEN task loadable."""
        db_path = str(tmp_path / "test.db")
        now = time.time()

        db1 = LLMDatabase(db_path)
        db1.save_pending_task(
            task_type="draw",
            nick="grace",
            reply_target="grace",
            is_channel=False,
            prompt_preview="dancing cat",
            model="vertex_ai/imagen-4.0-generate-001",
            request_data='{"request_id": "req-999"}',
            submitted_at=now,
            expires_at=now + 3600,
            next_attempt_at=now,
        )
        db1.close()

        db2 = LLMDatabase(db_path)
        tasks = db2.load_pending_tasks()
        assert len(tasks) == 1
        assert tasks[0].nick == "grace"
        assert tasks[0].task_type == "draw"
        assert tasks[0].is_channel == 0
        db2.close()

    def test_concurrent_claim_never_double_claims(self, tmp_path: Path) -> None:
        """GIVEN due pending tasks WHEN several threads claim concurrently THEN
        no row is ever claimed by more than one thread.

        Guards the ``BEGIN IMMEDIATE`` write-lock in ``claim_due_pending_tasks``.
        Under ``BEGIN DEFERRED`` two threads can both run the SELECT before
        either commits its UPDATE and claim the same rows -- a double provider
        call / double delivery on the live @ask/@code/@draw recovery queue. The
        sequential mutual-exclusion property test cannot see this: the
        ``claimed_until <= now`` filter alone makes a *second sequential* claim
        disjoint, so only real concurrency exercises the lock. Repeated across
        rounds so the race window is hit reliably; correct code stays disjoint
        every round regardless of scheduling.
        """

        def claimer(
            db: LLMDatabase,
            barrier: threading.Barrier,
            now: float,
            limit: int,
            claimed: list[list[int]],
            errors: list[Exception],
            guard: threading.Lock,
        ) -> None:
            try:
                barrier.wait()
                rows = db.claim_due_pending_tasks(now, limit=limit, lease_seconds=300)
                with guard:
                    claimed.append([row.id for row in rows])
            except Exception as exc:
                with guard:
                    errors.append(exc)

        now = time.time()
        thread_count = 4
        tasks_per_round = 40
        rounds = 15

        for round_index in range(rounds):
            db = LLMDatabase(str(tmp_path / f"race-{round_index}.db"))
            for i in range(tasks_per_round):
                db.save_pending_task(
                    task_type="ask",
                    nick=f"u{i}",
                    reply_target="#test",
                    is_channel=True,
                    prompt_preview="x",
                    model="gpt-4",
                    request_data="{}",
                    submitted_at=now,
                    expires_at=now + 3600,
                    next_attempt_at=now,
                )

            barrier = threading.Barrier(thread_count)
            claimed: list[list[int]] = []
            errors: list[Exception] = []
            guard = threading.Lock()
            threads = [
                threading.Thread(
                    target=claimer,
                    args=(db, barrier, now, tasks_per_round, claimed, errors, guard),
                )
                for _ in range(thread_count)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert errors == [], f"round {round_index}: claim raised {errors!r}"
            id_sets = [set(ids) for ids in claimed]
            for a in range(len(id_sets)):
                for b in range(a + 1, len(id_sets)):
                    overlap = id_sets[a] & id_sets[b]
                    assert not overlap, f"round {round_index}: id claimed twice: {overlap}"
            total_claimed = sum(len(ids) for ids in claimed)
            union: set[int] = set().union(*id_sets) if id_sets else set()
            assert len(union) == total_claimed
            db.close()


class TestDeliveryStatePersistence:
    """Test delivery state transitions and filtered queries for Phase 1b."""

    def _save_task(self, db: LLMDatabase, now: float, **overrides: object) -> int:
        """Helper to save a pending task with sensible defaults."""
        defaults: dict[str, object] = {
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello",
            "model": "gpt-4",
            "request_data": "{}",
            "submitted_at": now,
            "expires_at": now + 120,
            "next_attempt_at": now,
        }
        defaults.update(overrides)
        return db.save_pending_task(**defaults)  # type: ignore[arg-type]

    def test_update_task_for_delivery_sets_ready(self, test_db: LLMDatabase) -> None:
        """GIVEN a pending task WHEN update_task_for_delivery called THEN delivery_state='ready'."""
        now = time.time()
        task_id = self._save_task(test_db, now)

        test_db.update_task_for_delivery(
            task_id,
            delivery_state="ready",
            result_payload='{"content": "hello world"}',
        )

        tasks = test_db.load_pending_tasks()
        assert len(tasks) == 1
        assert tasks[0].delivery_state == "ready"
        assert tasks[0].result_payload == '{"content": "hello world"}'

    def test_claim_filters_by_delivery_state(self, test_db: LLMDatabase) -> None:
        """GIVEN tasks with different delivery_states WHEN claiming with filter THEN only matching tasks returned."""
        now = time.time()

        # pending task (provider phase)
        self._save_task(test_db, now, nick="pending_nick", next_attempt_at=now - 5)
        # ready task (delivery phase)
        ready_id = self._save_task(test_db, now, nick="ready_nick", next_attempt_at=now - 5)
        test_db.update_task_for_delivery(ready_id, "ready", '{"content": "result"}')

        # Provider claim: should only get pending
        provider_claimed = test_db.claim_due_pending_tasks(
            now, limit=10, lease_seconds=120, delivery_state_filter="pending"
        )
        assert len(provider_claimed) == 1
        assert provider_claimed[0].nick == "pending_nick"

    def test_claim_delivery_ready_and_retrying(self, test_db: LLMDatabase) -> None:
        """GIVEN ready and retrying tasks WHEN delivery claim THEN both returned."""
        now = time.time()

        ready_id = self._save_task(test_db, now, nick="ready_nick", next_attempt_at=now - 5)
        test_db.update_task_for_delivery(ready_id, "ready", '{"content": "r1"}')

        retrying_id = self._save_task(test_db, now, nick="retrying_nick", next_attempt_at=now - 5)
        test_db.update_task_for_delivery(retrying_id, "retrying", '{"content": "r2"}')

        delivery_claimed = test_db.claim_due_pending_tasks(
            now,
            limit=10,
            lease_seconds=120,
            delivery_state_filter=("ready", "retrying"),
        )
        assert len(delivery_claimed) == 2

    def test_update_delivery_attempt(self, test_db: LLMDatabase) -> None:
        """GIVEN a ready task WHEN delivery fails THEN retrying with incremented attempt count."""
        now = time.time()
        task_id = self._save_task(test_db, now)
        test_db.update_task_for_delivery(task_id, "ready", '{"content": "result"}')

        test_db.update_delivery_attempt(
            task_id,
            delivery_state="retrying",
            last_delivery_error="queueMsg failed",
            delivery_attempt_count=1,
            next_attempt_at=now + 15,
        )

        tasks = test_db.load_pending_tasks()
        assert tasks[0].delivery_state == "retrying"
        assert tasks[0].last_delivery_error == "queueMsg failed"
        assert tasks[0].delivery_attempt_count == 1
        assert tasks[0].next_attempt_at == now + 15

    def test_delivery_failed_not_auto_claimed(self, test_db: LLMDatabase) -> None:
        """GIVEN a delivery_failed task WHEN claiming for delivery THEN not returned."""
        now = time.time()
        task_id = self._save_task(test_db, now, next_attempt_at=now - 5)
        test_db.update_task_for_delivery(task_id, "delivery_failed", '{"content": "result"}')

        claimed = test_db.claim_due_pending_tasks(
            now,
            limit=10,
            lease_seconds=120,
            delivery_state_filter=("ready", "retrying"),
        )
        assert len(claimed) == 0

    def test_delivery_claim_skips_exhausted_attempts(self, test_db: LLMDatabase) -> None:
        """GIVEN retrying rows at/under cap WHEN claiming THEN exhausted rows are excluded."""
        now = time.time()

        retriable_id = self._save_task(test_db, now, nick="retriable", next_attempt_at=now - 5)
        test_db.update_task_for_delivery(retriable_id, "ready", '{"content": "r1"}')
        test_db.update_delivery_attempt(
            retriable_id,
            delivery_state="retrying",
            last_delivery_error="net",
            delivery_attempt_count=9,
            next_attempt_at=now - 5,
        )

        exhausted_id = self._save_task(test_db, now, nick="exhausted", next_attempt_at=now - 5)
        test_db.update_task_for_delivery(exhausted_id, "ready", '{"content": "r2"}')
        test_db.update_delivery_attempt(
            exhausted_id,
            delivery_state="retrying",
            last_delivery_error="net",
            delivery_attempt_count=10,
            next_attempt_at=now - 5,
        )

        claimed = test_db.claim_due_pending_tasks(
            now,
            limit=10,
            lease_seconds=120,
            delivery_state_filter=("ready", "retrying"),
            max_delivery_attempts=10,
        )
        assert len(claimed) == 1
        assert claimed[0].nick == "retriable"

    def test_expired_deletes_all_delivery_states_past_ttl(self, test_db: LLMDatabase) -> None:
        """GIVEN expired tasks in any delivery_state WHEN expiry sweep THEN all past-TTL rows reaped.

        ``expires_at`` is the whole-task hard deadline ("stop retrying after
        this"). A result that reached ready/retrying/delivery_failed but can no
        longer be delivered (e.g. the bot left the channel) must still be reaped
        at its TTL — otherwise it re-polls every 30s forever (queue leak +
        perpetual wakeup). Only rows still within their TTL survive.
        """
        now = time.time()

        self._save_task(
            test_db,
            now,
            nick="expired_pending",
            submitted_at=now - 120,
            expires_at=now - 10,
            next_attempt_at=now - 60,
        )

        ready_id = self._save_task(
            test_db,
            now,
            nick="expired_ready",
            submitted_at=now - 120,
            expires_at=now - 10,
            next_attempt_at=now - 60,
        )
        test_db.update_task_for_delivery(ready_id, "ready", '{"content": "result"}')

        retrying_id = self._save_task(
            test_db,
            now,
            nick="expired_retrying",
            submitted_at=now - 120,
            expires_at=now - 10,
            next_attempt_at=now - 60,
        )
        test_db.update_delivery_attempt(
            task_id=retrying_id,
            delivery_state="retrying",
            last_delivery_error="IRC delivery failed",
            delivery_attempt_count=1,
            next_attempt_at=now - 5,
        )

        failed_id = self._save_task(
            test_db,
            now,
            nick="expired_delivery_failed",
            submitted_at=now - 120,
            expires_at=now - 10,
            next_attempt_at=now - 60,
        )
        test_db.update_delivery_attempt(
            task_id=failed_id,
            delivery_state="delivery_failed",
            last_delivery_error="gave up",
            delivery_attempt_count=5,
            next_attempt_at=now - 5,
        )

        # Within TTL with a result — must survive the sweep.
        fresh_id = self._save_task(
            test_db,
            now,
            nick="fresh_ready",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )
        test_db.update_task_for_delivery(fresh_id, "ready", '{"content": "fresh"}')

        expired = test_db.delete_expired_pending_tasks(now)
        assert {r.nick for r in expired} == {
            "expired_pending",
            "expired_ready",
            "expired_retrying",
            "expired_delivery_failed",
        }

        remaining = test_db.load_pending_tasks()
        assert {r.nick for r in remaining} == {"fresh_ready"}


class TestSchemaV3Migration:
    """Test schema v3 migration adds delivery columns to pending_tasks."""

    def test_migration_from_v2_adds_delivery_columns(self, tmp_path: Path) -> None:
        """GIVEN a v2 database with pending_tasks WHEN opened with v3 code THEN new columns exist with defaults."""
        db_path = str(tmp_path / "test.db")
        now = time.time()

        # Create a v2-schema database manually
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT UNIQUE NOT NULL,
                nick TEXT NOT NULL, channel TEXT NOT NULL,
                message TEXT NOT NULL, fire_at REAL NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL, nick TEXT NOT NULL, channel TEXT NOT NULL,
                command TEXT NOT NULL, model TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0.0,
                prompt TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'success',
                error_detail TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS pending_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL, nick TEXT NOT NULL,
                reply_target TEXT NOT NULL, is_channel INTEGER NOT NULL,
                prompt_preview TEXT NOT NULL, model TEXT NOT NULL,
                request_data TEXT NOT NULL DEFAULT '{}',
                submitted_at REAL NOT NULL, expires_at REAL NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at REAL NOT NULL,
                claimed_until REAL NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE IF NOT EXISTS flagged_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account TEXT UNIQUE NOT NULL,
                flagged_at REAL NOT NULL, reason TEXT NOT NULL DEFAULT '',
                auto_flagged INTEGER NOT NULL DEFAULT 0,
                resolved_at REAL, resolved_by TEXT
            );
        """)
        # Insert a pending task with v2 schema
        conn.execute(
            "INSERT INTO pending_tasks "
            "(task_type, nick, reply_target, is_channel, prompt_preview, model, "
            "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
            "claimed_until, last_error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, '')",
            (
                "draw",
                "alice",
                "#test",
                1,
                "dancing cat",
                "vertex_ai/imagen-4.0-generate-001",
                '{"request_id": "req-1"}',
                now,
                now + 3600,
                now,
            ),
        )
        conn.commit()
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        conn.close()

        # Open with v3-capable LLMDatabase
        db = LLMDatabase(db_path)

        # Verify new columns exist with correct defaults
        conn = db._connect()
        try:
            row = conn.execute(
                "SELECT delivery_state, result_payload, last_delivery_error, "
                "delivery_attempt_count, origin_request_id FROM pending_tasks WHERE nick = 'alice'"
            ).fetchone()
            assert row is not None
            assert row[0] == "pending"  # delivery_state default
            assert row[1] == ""  # result_payload default
            assert row[2] == ""  # last_delivery_error default
            assert row[3] == 0  # delivery_attempt_count default
            assert row[4] == ""  # origin_request_id default
        finally:
            conn.close()
        db.close()

    def test_reminder_persists_structured_recurrence_fields(self, test_db: LLMDatabase) -> None:
        """GIVEN a save with structured recurrence WHEN reloaded THEN fields round-trip."""
        test_db.save_reminder(
            event_name="evt-structured",
            nick="alice",
            channel="#test",
            message="msg",
            fire_at=time.time() + 60,
            action_prompt="check the build",
            account="alice",
            chain_position=1,
            recurrence_seconds=300,
            recurrence_rrule=None,
            watch_mode=True,
        )
        rows = test_db.load_pending_reminders()
        assert len(rows) == 1
        assert rows[0].recurrence_seconds == 300
        assert rows[0].recurrence_rrule is None
        assert rows[0].watch_mode is True

    def test_reminder_persists_reply_msgid(self, test_db: LLMDatabase) -> None:
        """GIVEN a save carrying the setter's msgid WHEN reloaded THEN it round-trips for +draft/reply."""
        test_db.save_reminder(
            event_name="evt-msgid",
            nick="alice",
            channel="#test",
            message="msg",
            fire_at=time.time() + 60,
            action_prompt="check the build",
            account="alice",
            reply_msgid="abc123",
        )
        rows = test_db.load_pending_reminders()
        assert len(rows) == 1
        assert rows[0].reply_msgid == "abc123"


class TestGetNextDueTime:
    """Test get_next_due_time() for event-driven queue wakeups (Phase 2)."""

    def _save_task(self, db: LLMDatabase, now: float, **overrides: object) -> int:
        """Helper to save a pending task with sensible defaults."""
        defaults: dict[str, object] = {
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello",
            "model": "gpt-4",
            "request_data": "{}",
            "submitted_at": now,
            "expires_at": now + 120,
            "next_attempt_at": now,
        }
        defaults.update(overrides)
        return db.save_pending_task(**defaults)  # type: ignore[arg-type]

    def test_empty_queue_returns_none(self, test_db: LLMDatabase) -> None:
        """GIVEN no pending tasks WHEN get_next_due_time called THEN returns None."""
        assert test_db.get_next_due_time() is None

    def test_returns_earliest_next_attempt_at(self, test_db: LLMDatabase) -> None:
        """GIVEN multiple pending tasks WHEN get_next_due_time called THEN returns the earliest."""
        now = time.time()
        self._save_task(test_db, now, nick="later", next_attempt_at=now + 60)
        self._save_task(test_db, now, nick="sooner", next_attempt_at=now + 10)
        self._save_task(test_db, now, nick="middle", next_attempt_at=now + 30)

        result = test_db.get_next_due_time()
        assert result == now + 10

    def test_excludes_claimed_tasks(self, test_db: LLMDatabase) -> None:
        """GIVEN a claimed task WHEN get_next_due_time called THEN it is excluded."""
        now = time.time()
        self._save_task(test_db, now, nick="claimed", next_attempt_at=now + 5)
        test_db.claim_due_pending_tasks(now + 10, limit=10, lease_seconds=120)
        # The only task is now claimed — should return None
        assert test_db.get_next_due_time() is None

    def test_includes_lease_expired_claimed_tasks(self, test_db: LLMDatabase) -> None:
        """GIVEN claim lease already expired WHEN get_next_due_time THEN task is considered."""
        now = time.time()
        task_id = self._save_task(test_db, now, nick="leased", next_attempt_at=now + 5)

        # Claim the row first, then force claimed_until into the past.
        test_db.claim_due_pending_tasks(now + 10, limit=10, lease_seconds=120)
        conn = test_db._connect()
        try:
            conn.execute(
                "UPDATE pending_tasks SET claimed_until = ? WHERE id = ?",
                (time.time() - 1, task_id),
            )
            conn.commit()
        finally:
            conn.close()

        assert test_db.get_next_due_time() == pytest.approx(now + 5)

    def test_includes_pending_and_delivery_states(self, test_db: LLMDatabase) -> None:
        """GIVEN tasks in pending, ready, and retrying states WHEN get_next_due_time THEN all considered."""
        now = time.time()

        # pending task due later
        self._save_task(test_db, now, nick="pending", next_attempt_at=now + 60)
        # ready task due sooner
        ready_id = self._save_task(test_db, now, nick="ready", next_attempt_at=now + 20)
        test_db.update_task_for_delivery(ready_id, "ready", '{"content": "r1"}')
        # retrying task due soonest
        retrying_id = self._save_task(test_db, now, nick="retrying", next_attempt_at=now + 10)
        test_db.update_task_for_delivery(retrying_id, "retrying", '{"content": "r2"}')

        assert test_db.get_next_due_time() == now + 10

    def test_excludes_terminal_delivery_states(self, test_db: LLMDatabase) -> None:
        """GIVEN tasks in delivery_failed state WHEN get_next_due_time THEN excluded."""
        now = time.time()

        failed_id = self._save_task(test_db, now, nick="failed", next_attempt_at=now + 5)
        test_db.update_task_for_delivery(failed_id, "delivery_failed", '{"content": "r1"}')

        assert test_db.get_next_due_time() is None


class TestLogUsageExtended:
    """Test the extended log_usage parameters (prompt, status, error_detail)."""

    def test_log_usage_stores_prompt_and_status(self, test_db: LLMDatabase) -> None:
        """GIVEN new log_usage params WHEN logging with prompt/status/error_detail THEN stored."""
        test_db.log_usage(
            "alice",
            "#test",
            "ask",
            "gpt-4",
            100,
            50,
            0.01,
            prompt="tell me a joke",
            status="content_blocked",
            error_detail="moderation filter triggered",
        )

        conn = test_db._connect()
        try:
            row = conn.execute(
                "SELECT prompt, status, error_detail FROM usage WHERE nick = 'alice'"
            ).fetchone()
            assert row is not None
            assert row[0] == "tell me a joke"
            assert row[1] == "content_blocked"
            assert row[2] == "moderation filter triggered"
        finally:
            conn.close()

    def test_log_usage_defaults_to_success(self, test_db: LLMDatabase) -> None:
        """GIVEN no new params WHEN logging usage THEN defaults applied."""
        test_db.log_usage("bob", "#test", "code", "gpt-4", 200, 100, 0.02)

        conn = test_db._connect()
        try:
            row = conn.execute(
                "SELECT prompt, status, error_detail FROM usage WHERE nick = 'bob'"
            ).fetchone()
            assert row is not None
            assert row[0] == ""
            assert row[1] == "success"
            assert row[2] == ""
        finally:
            conn.close()


class TestConversationPersistence:
    """Test conversation persistence methods.

    Round-trip, upsert (no duplicates), and case-insensitive
    save/delete are covered by ``test_persistence_conversation_properties.py``.
    What remains here exercises paths the property tests do not cover:
    the bulk ``delete_all_conversations`` and the corrupt-JSON load path.
    """

    def test_delete_all_conversations(self, test_db: LLMDatabase) -> None:
        """GIVEN multiple conversations WHEN delete_all THEN all are removed."""
        test_db.save_conversation("user1", "#chan", [], 1000.0)
        test_db.save_conversation("user2", "#chan", [], 1000.0)
        test_db.delete_all_conversations()

        assert len(test_db.load_conversations()) == 0

    def test_load_removes_multiple_corrupt_rows_atomically(self, test_db: LLMDatabase) -> None:
        """GIVEN several corrupt rows interleaved with valid ones WHEN load THEN
        all corrupt rows are removed and every valid row is returned."""
        conn = test_db._connect()
        test_db.save_conversation("good1", "#chan", [{"role": "user", "content": "a"}], 1000.0)
        for bad in ("bad1", "bad2", "bad3"):
            conn.execute(
                "INSERT OR REPLACE INTO conversations (nick, channel, messages, last_activity) "
                "VALUES (?, ?, ?, ?)",
                (bad, "#chan", "}{not json", 1000.0),
            )
        conn.commit()
        test_db.save_conversation("good2", "#chan", [{"role": "user", "content": "b"}], 1000.0)

        loaded = test_db.load_conversations()
        assert {row[0] for row in loaded} == {"good1", "good2"}

        # Corrupt rows are gone from the DB, not just filtered from the result.
        remaining = {r[0] for r in conn.execute("SELECT nick FROM conversations").fetchall()}
        assert remaining == {"good1", "good2"}


class TestMemoryPersistence:
    """Test memory persistence methods."""

    def test_save_and_get_memory(self, test_db: LLMDatabase) -> None:
        """GIVEN a saved memory WHEN get_memories THEN it is returned."""
        row_id = test_db.save_memory("user1", "likes Python", "#test")
        memories = test_db.get_memories("user1")
        assert len(memories) == 1
        assert memories[0].id == row_id
        assert memories[0].nick == "user1"
        assert memories[0].fact == "likes Python"
        assert memories[0].source_channel == "#test"

    def test_save_memory_lowercases_nick(self, test_db: LLMDatabase) -> None:
        """GIVEN mixed-case nick WHEN saved THEN stored lowercased."""
        test_db.save_memory("Alice", "fact", "#test")
        memories = test_db.get_memories("alice")
        assert len(memories) == 1
        assert memories[0].nick == "alice"

    def test_get_memories_ordered_newest_first(self, test_db: LLMDatabase) -> None:
        """GIVEN multiple memories WHEN get_memories THEN ordered newest first."""
        test_db.save_memory("user1", "first fact", "#test")
        test_db.save_memory("user1", "second fact", "#test")
        memories = test_db.get_memories("user1")
        assert len(memories) == 2
        assert memories[0].fact == "second fact"
        assert memories[1].fact == "first fact"

    def test_delete_memory_by_id(self, test_db: LLMDatabase) -> None:
        """GIVEN two memories WHEN one deleted THEN only one remains."""
        id1 = test_db.save_memory("user1", "fact one", "#test")
        test_db.save_memory("user1", "fact two", "#test")
        result = test_db.delete_memory("user1", id1)
        assert result is True
        memories = test_db.get_memories("user1")
        assert len(memories) == 1
        assert memories[0].fact == "fact two"

    def test_update_memory(self, test_db: LLMDatabase) -> None:
        """GIVEN a saved memory WHEN update_memory THEN fact text is changed."""
        row_id = test_db.save_memory("user1", "likes Python", "#test")
        result = test_db.update_memory("user1", row_id, "likes Rust")
        assert result is True
        memories = test_db.get_memories("user1")
        assert len(memories) == 1
        assert memories[0].fact == "likes Rust"

    def test_update_memory_wrong_nick(self, test_db: LLMDatabase) -> None:
        """GIVEN memory for alice WHEN bob tries to update THEN fails."""
        row_id = test_db.save_memory("alice", "secret", "#test")
        result = test_db.update_memory("bob", row_id, "changed")
        assert result is False
        assert test_db.get_memories("alice")[0].fact == "secret"

    def test_delete_memory_wrong_nick(self, test_db: LLMDatabase) -> None:
        """GIVEN memory for alice WHEN bob tries to delete THEN fails."""
        row_id = test_db.save_memory("alice", "secret", "#test")
        result = test_db.delete_memory("bob", row_id)
        assert result is False
        assert len(test_db.get_memories("alice")) == 1

    def test_delete_all_memories(self, test_db: LLMDatabase) -> None:
        """GIVEN three memories WHEN delete_all THEN all removed."""
        test_db.save_memory("user1", "a", "#test")
        test_db.save_memory("user1", "b", "#test")
        test_db.save_memory("user1", "c", "#test")
        count = test_db.delete_all_memories("user1")
        assert count == 3
        assert test_db.get_memories("user1") == []


class TestMemoryCleanupState:
    """Test memory cleanup counter methods."""

    def test_increment_memory_saves_accumulates(self, test_db: LLMDatabase) -> None:
        """GIVEN two increments WHEN get THEN returns 2."""
        test_db.increment_memory_saves("user1")
        assert test_db.increment_memory_saves("user1") == 2

    def test_increment_memory_saves_case_insensitive(self, test_db: LLMDatabase) -> None:
        """GIVEN mixed case nick WHEN increment THEN stored lowercased."""
        test_db.increment_memory_saves("Alice")
        assert test_db.increment_memory_saves("alice") == 2

    def test_reset_memory_saves(self, test_db: LLMDatabase) -> None:
        """GIVEN incremented counter WHEN reset THEN returns to 0."""
        test_db.increment_memory_saves("user1")
        test_db.increment_memory_saves("user1")
        test_db.reset_memory_saves("user1")
        assert test_db.increment_memory_saves("user1") == 1


class TestMemoryCandidates:
    """Tests for the multi-stage memory candidate table."""

    def test_add_and_get_returns_candidate(self, test_db: LLMDatabase) -> None:
        """GIVEN one candidate WHEN listed THEN it is returned with mentions=1."""
        cid = test_db.add_memory_candidate("user1", "uses Arch", "#test")
        candidates = test_db.get_memory_candidates("user1")
        assert len(candidates) == 1
        assert candidates[0].id == cid
        assert candidates[0].fact == "uses Arch"
        assert candidates[0].mentions == 1
        assert candidates[0].first_seen == candidates[0].last_seen
        assert candidates[0].source_channel == "#test"

    def test_get_orders_by_mentions_then_last_seen(self, test_db: LLMDatabase) -> None:
        """GIVEN candidates with different mention counts WHEN listed THEN highest first."""
        a = test_db.add_memory_candidate("u", "low", "#c")
        b = test_db.add_memory_candidate("u", "high", "#c")
        test_db.reinforce_memory_candidate(b, "u")
        test_db.reinforce_memory_candidate(b, "u")
        ordered = test_db.get_memory_candidates("u")
        assert [c.id for c in ordered] == [b, a]

    def test_reinforce_increments_mentions(self, test_db: LLMDatabase) -> None:
        """GIVEN a candidate WHEN reinforced THEN mentions counter goes up."""
        cid = test_db.add_memory_candidate("user1", "likes coffee", "#test")
        new_count = test_db.reinforce_memory_candidate(cid, "user1")
        assert new_count == 2
        candidates = test_db.get_memory_candidates("user1")
        assert candidates[0].mentions == 2
        assert candidates[0].last_seen >= candidates[0].first_seen

    def test_reinforce_can_update_fact_text(self, test_db: LLMDatabase) -> None:
        """GIVEN reinforce called with new text WHEN listed THEN fact is updated."""
        cid = test_db.add_memory_candidate("user1", "rough draft", "#test")
        test_db.reinforce_memory_candidate(cid, "user1", fact="polished")
        candidates = test_db.get_memory_candidates("user1")
        assert candidates[0].fact == "polished"

    def test_reinforce_wrong_owner_is_noop(self, test_db: LLMDatabase) -> None:
        """GIVEN one user's candidate WHEN another reinforces THEN nothing changes."""
        cid = test_db.add_memory_candidate("alice", "private", "#test")
        assert test_db.reinforce_memory_candidate(cid, "bob") == 0
        assert test_db.get_memory_candidates("alice")[0].mentions == 1

    def test_delete_candidate(self, test_db: LLMDatabase) -> None:
        """GIVEN a candidate WHEN deleted THEN it disappears."""
        cid = test_db.add_memory_candidate("user1", "x", "#test")
        assert test_db.delete_memory_candidate(cid, "user1") is True
        assert test_db.get_memory_candidates("user1") == []

    def test_delete_candidate_wrong_owner(self, test_db: LLMDatabase) -> None:
        """GIVEN a candidate WHEN another user tries to delete THEN no-op."""
        cid = test_db.add_memory_candidate("alice", "secret", "#test")
        assert test_db.delete_memory_candidate(cid, "bob") is False
        assert len(test_db.get_memory_candidates("alice")) == 1

    def test_prune_removes_only_old_rows(self, test_db: LLMDatabase) -> None:
        """GIVEN candidates older than cutoff WHEN pruned THEN only those removed."""
        old = test_db.add_memory_candidate("user1", "old", "#test")
        fresh = test_db.add_memory_candidate("user1", "fresh", "#test")
        # Age the old row by rewriting last_seen directly.
        conn = test_db._connect()
        conn.execute("UPDATE memory_candidates SET last_seen = 0 WHERE id = ?", (old,))
        conn.commit()
        removed = test_db.prune_memory_candidates("user1", older_than=1.0)
        assert removed == 1
        remaining = [c.id for c in test_db.get_memory_candidates("user1")]
        assert remaining == [fresh]

    def test_delete_all_candidates(self, test_db: LLMDatabase) -> None:
        """GIVEN multiple candidates WHEN delete_all THEN every one is removed."""
        test_db.add_memory_candidate("user1", "a", "#test")
        test_db.add_memory_candidate("user1", "b", "#test")
        assert test_db.delete_all_memory_candidates("user1") == 2
        assert test_db.get_memory_candidates("user1") == []


class TestUserInstructions:
    """Tests for user_instructions table CRUD."""

    def test_save_and_get_instruction(self, test_db: LLMDatabase) -> None:
        """GIVEN saved instruction WHEN queried THEN returns text."""
        test_db.save_instruction("testnick", "You are Captain Picard.")
        assert test_db.get_instruction("testnick") == "You are Captain Picard."

    def test_save_instruction_overwrites(self, test_db: LLMDatabase) -> None:
        """GIVEN existing instruction WHEN saved again THEN overwrites."""
        test_db.save_instruction("testnick", "old")
        test_db.save_instruction("testnick", "new")
        assert test_db.get_instruction("testnick") == "new"

    def test_delete_instruction(self, test_db: LLMDatabase) -> None:
        """GIVEN existing instruction WHEN deleted THEN returns True and clears."""
        test_db.save_instruction("testnick", "text")
        assert test_db.delete_instruction("testnick") is True
        assert test_db.get_instruction("testnick") is None

    def test_delete_missing_instruction_returns_false(self, test_db: LLMDatabase) -> None:
        """GIVEN no instruction set WHEN cleared THEN False, so @instruct clear
        answers 'No instruction to clear.' instead of a fabricated 'cleared'."""
        assert test_db.delete_instruction("nobody") is False
        test_db.save_instruction("testnick", "text")
        assert test_db.delete_instruction("testnick") is True
        assert test_db.delete_instruction("testnick") is False


class TestUserTimezones:
    """Tests for user_timezones table CRUD."""

    def test_save_get_overwrite_case_insensitive(self, test_db: LLMDatabase) -> None:
        test_db.save_user_timezone("RDrake", "America/Toronto")
        test_db.save_user_timezone("rdrake", "Europe/London")
        assert test_db.get_user_timezone("RDRAKE") == "Europe/London"

    def test_delete(self, test_db: LLMDatabase) -> None:
        test_db.save_user_timezone("testnick", "UTC")
        assert test_db.delete_user_timezone("testnick") is True
        assert test_db.delete_user_timezone("testnick") is False
        assert test_db.get_user_timezone("testnick") is None


class TestUserClockOffsets:
    """Tests for user_clock_offsets, the stored CTCP TIME answers."""

    def test_get_returns_none_when_never_probed(self, test_db: LLMDatabase) -> None:
        assert test_db.get_clock_offset("testnick") is None

    def test_save_get_overwrite_case_insensitive(self, test_db: LLMDatabase) -> None:
        test_db.save_clock_offset("Rubin", -4 * 3600)
        test_db.save_clock_offset("rubin", 3600)
        offset, probed_at = test_db.get_clock_offset("RUBIN")
        assert offset == 3600
        assert probed_at > 0

    def test_silence_is_stored_as_none(self, test_db: LLMDatabase) -> None:
        test_db.save_clock_offset("rubin", None)
        assert test_db.get_clock_offset("rubin")[0] is None


class TestUserAvatarPersonas:
    """Tests for user_avatar_personas table CRUD."""

    def test_save_and_get_persona(self, test_db: LLMDatabase) -> None:
        """GIVEN saved persona WHEN queried THEN returns text."""
        test_db.save_avatar_persona("testnick", "moss-covered tree spirit")
        assert test_db.get_avatar_persona("testnick") == "moss-covered tree spirit"

    def test_save_persona_overwrites(self, test_db: LLMDatabase) -> None:
        """GIVEN existing persona WHEN saved again THEN overwrites."""
        test_db.save_avatar_persona("testnick", "old")
        test_db.save_avatar_persona("testnick", "new")
        assert test_db.get_avatar_persona("testnick") == "new"

    def test_delete_persona(self, test_db: LLMDatabase) -> None:
        """GIVEN existing persona WHEN deleted THEN returns True and clears."""
        test_db.save_avatar_persona("testnick", "text")
        assert test_db.delete_avatar_persona("testnick") is True
        assert test_db.get_avatar_persona("testnick") is None

    def test_delete_missing_persona_returns_false(self, test_db: LLMDatabase) -> None:
        """GIVEN no persona set WHEN cleared THEN False, so the user sees
        'No persona to clear.' instead of 'Persona cleared.'."""
        assert test_db.delete_avatar_persona("nobody") is False
        test_db.save_avatar_persona("testnick", "text")
        assert test_db.delete_avatar_persona("testnick") is True
        assert test_db.delete_avatar_persona("testnick") is False

    def test_persona_independent_of_instruction(self, test_db: LLMDatabase) -> None:
        """GIVEN instruction set WHEN persona queried THEN returns None.

        The split is the whole point: setting one must not leak into the other.
        """
        test_db.save_instruction("testnick", "ask voice")
        assert test_db.get_avatar_persona("testnick") is None
        test_db.save_avatar_persona("testnick", "verse voice")
        assert test_db.get_instruction("testnick") == "ask voice"
        assert test_db.get_avatar_persona("testnick") == "verse voice"


class TestLoadPendingTasksTypeFilter:
    """Test load_pending_tasks with the optional task_type filter."""

    def test_filter_by_type_returns_only_matching(self, test_db: LLMDatabase) -> None:
        """GIVEN ask and draw tasks WHEN load_pending_tasks(task_type='ask') THEN only ask returned."""
        now = time.time()
        test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="question",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )
        test_db.save_pending_task(
            task_type="draw",
            nick="bob",
            reply_target="#test",
            is_channel=True,
            prompt_preview="a cat",
            model="dall-e-3",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )

        ask_tasks = test_db.load_pending_tasks(task_type="ask")
        assert len(ask_tasks) == 1
        assert ask_tasks[0].task_type == "ask"
        assert ask_tasks[0].nick == "alice"


class TestDeletePendingTask:
    """Test delete_pending_task found and not-found cases."""

    def test_delete_existing_task_returns_true(self, test_db: LLMDatabase) -> None:
        """GIVEN a saved task WHEN delete_pending_task called THEN returns True."""
        now = time.time()
        task_id = test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="question",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )

        assert test_db.delete_pending_task(task_id) is True
        assert test_db.load_pending_tasks() == []

    def test_delete_same_task_twice_returns_false(self, test_db: LLMDatabase) -> None:
        """GIVEN a deleted task WHEN delete_pending_task called again THEN returns False."""
        now = time.time()
        task_id = test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="question",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 120,
            next_attempt_at=now,
        )

        assert test_db.delete_pending_task(task_id) is True
        assert test_db.delete_pending_task(task_id) is False


class TestZeroCostRank:
    """Test rank computation when usage exists but cost is zero."""

    def test_zero_cost_usage_has_rank_one(self, test_db: LLMDatabase) -> None:
        """GIVEN usage with zero cost WHEN get_nick_rank THEN rank is 1 (not 0)."""
        test_db.log_usage("alice", "#test", "ask", "gpt-4", 10, 5, 0.0)

        rank = test_db.get_nick_rank("alice")
        assert rank == UsageRank(rank=1, total=1)

    def test_zero_cost_ranked_below_nonzero(self, test_db: LLMDatabase) -> None:
        """GIVEN one nick with cost and one with zero cost WHEN ranking THEN zero-cost is rank 2."""
        test_db.log_usage("alice", "#test", "ask", "gpt-4", 10, 5, 0.05)
        test_db.log_usage("bob", "#test", "ask", "gpt-4", 10, 5, 0.0)

        rank_bob = test_db.get_nick_rank("bob")
        assert rank_bob == UsageRank(rank=2, total=2)

        rank_alice = test_db.get_nick_rank("alice")
        assert rank_alice == UsageRank(rank=1, total=2)


class TestPendingTasksAccountColumn:
    def test_save_with_account(self, test_db):
        task_id = test_db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="hi",
            model="gpt-4",
            request_data="{}",
            submitted_at=1000.0,
            expires_at=2000.0,
            next_attempt_at=1000.0,
            account="alice_acct",
        )
        assert task_id > 0
        rows = test_db.load_pending_tasks()
        assert len(rows) == 1
        assert rows[0].account == "alice_acct"


def test_save_scheduled_llm_task_rejects_duplicate_event_name(tmp_path: Path) -> None:
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="dup",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=0.0,
    )
    with pytest.raises(sqlite3.IntegrityError):
        db.save_scheduled_llm_task(
            event_name="dup",
            creator_nick="n",
            account=None,
            channel="#x",
            network="afternet",
            wire_msg=":n!u@h PRIVMSG #x :@ask hi2",
            prompt="p2",
            fire_at=0.0,
        )


@pytest.mark.parametrize(
    "recurrence_seconds,recurrence_rrule",
    [
        (3600, None),
        (None, "FREQ=WEEKLY;BYDAY=MO"),
    ],
)
@pytest.mark.parametrize("watch_mode", [True, False])
def test_scheduled_llm_task_full_column_round_trip(
    tmp_path: Path,
    recurrence_seconds: int | None,
    recurrence_rrule: str | None,
    watch_mode: bool,
) -> None:
    """Every scheduled-task column survives save -> load carrying its own value.

    The INSERT positional list (``save_scheduled_llm_task``) and the ``row[N]``
    map (``_row_to_scheduled_llm_task``) are two independent 15-position
    mappings. Other tests assert only a handful of fields, so a
    ``network``<->``wire_msg`` or ``reply_target``<->``watch_mode``
    transposition would pass the suite while silently corrupting the live
    scheduled-task restore path (``rehydrate_msg`` consumes ``wire_msg`` and
    ``network``). Distinct, type-distinguishable values for every column make
    any transposition fail here. Parametrized over both recurrence kinds
    because they are mutually exclusive, and over watch_mode: an ordinary
    "in 30m check X" task (watch_mode False) that reloads as a watch has its
    negative-result reply suppressed, and the user hears nothing.
    """
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))

    fire_at = time.time() + 100_000.0  # far future -> distinguishable from created_at
    before = time.time()
    row_id = db.save_scheduled_llm_task(
        event_name="rt_event_distinct",
        creator_nick="creatorNick",
        account="accountValue",
        channel="#channelchan",
        network="netDistinct",
        wire_msg=":creatorNick!u@h PRIVMSG #channelchan :@ask payload",
        prompt="promptPayload",
        fire_at=fire_at,
        recurrence_seconds=recurrence_seconds,
        recurrence_rrule=recurrence_rrule,
        chain_position=7,
        watch_mode=watch_mode,
        reply_target="#replytarget",
    )
    after = time.time()

    row = db.get_scheduled_llm_task("rt_event_distinct")
    assert row is not None
    assert row.id == row_id
    assert row.event_name == "rt_event_distinct"
    assert row.creator_nick == "creatorNick"
    assert row.account == "accountValue"
    assert row.channel == "#channelchan"
    assert row.network == "netDistinct"
    assert row.wire_msg == ":creatorNick!u@h PRIVMSG #channelchan :@ask payload"
    assert row.prompt == "promptPayload"
    assert row.fire_at == fire_at
    assert before <= row.created_at <= after  # created_at not swapped with fire_at
    assert row.recurrence_seconds == recurrence_seconds
    assert row.recurrence_rrule == recurrence_rrule
    assert row.chain_position == 7
    assert row.watch_mode is watch_mode
    assert row.reply_target == "#replytarget"

    # The agenda's due scan reads via the same column map; it must agree
    # field-for-field with the point lookup.
    loaded = db.load_due_scheduled_llm_tasks(row.fire_at)
    assert loaded == [row]


def test_owner_listing_excludes_stale_one_shots_only(tmp_path: Path) -> None:
    """Mirror reminders: a one-shot past EXPIRY_THRESHOLD is excluded, a
    recurring row however overdue is not."""
    from llm.persistence import EXPIRY_THRESHOLD_SECONDS

    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    now = time.time()
    db.save_scheduled_llm_task(
        event_name="future",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=now + 60,
    )
    db.save_scheduled_llm_task(
        event_name="recent_overdue",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=now - 60,
    )
    db.save_scheduled_llm_task(
        event_name="ancient",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=now - EXPIRY_THRESHOLD_SECONDS - 60,
    )
    db.save_scheduled_llm_task(
        event_name="ancient_chain",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=now - EXPIRY_THRESHOLD_SECONDS - 60,
        recurrence_seconds=3600,
    )
    rows = db.load_scheduled_llm_tasks_for(account=None, nick="n")
    names = {r.event_name for r in rows}
    assert names == {"future", "recent_overdue", "ancient_chain"}


def test_count_scheduled_llm_tasks_for_account_then_nick(tmp_path: Path) -> None:
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="ev1",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#x",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=time.time() + 1,
    )
    db.save_scheduled_llm_task(
        event_name="ev2",
        creator_nick="rdrake_alt",
        account="rdrake_a",
        channel="#x",
        network="afternet",
        wire_msg=":rdrake_alt!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=time.time() + 2,
    )
    db.save_scheduled_llm_task(
        event_name="ev3",
        creator_nick="anon",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":anon!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=time.time() + 3,
    )
    db.save_scheduled_llm_task(
        event_name="ev4",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#other",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #other :@ask hi",
        prompt="p",
        fire_at=time.time() + 4,
    )
    # Account match is channel-scoped and case-insensitive.
    assert db.count_scheduled_llm_tasks_for(account="RDRAKE_A", nick="anything", channel="#x") == 2
    assert (
        db.count_scheduled_llm_tasks_for(account="rdrake_a", nick="anything", channel="#other") == 1
    )
    # When account is None, count by nick.
    assert db.count_scheduled_llm_tasks_for(account=None, nick="anon", channel="#x") == 1
    assert db.count_scheduled_llm_tasks_for(account=None, nick="rdrake", channel="#x") == 0


def test_update_scheduled_llm_task_fire_at(tmp_path: Path) -> None:
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="ev",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=time.time() + 1,
    )
    future = time.time() + 9
    db.update_scheduled_llm_task_fire_at("ev", future, chain_position=2)
    row = db.get_scheduled_llm_task("ev")
    assert row is not None
    assert row.fire_at == future
    assert row.chain_position == 2


def test_delete_scheduled_llm_task_returns_bool(tmp_path: Path) -> None:
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="ev",
        creator_nick="n",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=time.time() + 1,
    )
    assert db.delete_scheduled_llm_task("ev") is True
    assert db.delete_scheduled_llm_task("ev") is False


def test_load_scheduled_llm_tasks_for_owner(tmp_path: Path) -> None:
    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    db.save_scheduled_llm_task(
        event_name="ev_acct",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#x",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=time.time() + 1,
    )
    db.save_scheduled_llm_task(
        event_name="ev_nick",
        creator_nick="anon",
        account=None,
        channel="#x",
        network="afternet",
        wire_msg=":anon!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=time.time() + 2,
    )
    rows = db.load_scheduled_llm_tasks_for(account="RDRAKE_A", nick="anything")
    assert {r.event_name for r in rows} == {"ev_acct"}
    rows = db.load_scheduled_llm_tasks_for(account=None, nick="ANON")
    assert {r.event_name for r in rows} == {"ev_nick"}


def test_load_scheduled_llm_tasks_for_target_by_nick_or_account(tmp_path: Path) -> None:
    """@remind admin list/del <nick> finds another user's live tasks by the
    nick that scheduled them or the account they were stored under,
    case-insensitively; an upcoming one-shot is listed, an expired one is not."""
    from llm.persistence import EXPIRY_THRESHOLD_SECONDS

    db = LLMDatabase(str(tmp_path / "llm.sqlite"))
    now = time.time()
    db.save_scheduled_llm_task(
        event_name="upcoming",
        creator_nick="Alice",
        account="acct",
        channel="#x",
        network="afternet",
        wire_msg=":Alice!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=now + 3600,
    )
    db.save_scheduled_llm_task(
        event_name="expired",
        creator_nick="Alice",
        account="acct",
        channel="#x",
        network="afternet",
        wire_msg=":Alice!u@h PRIVMSG #x :@ask hi",
        prompt="p",
        fire_at=now - 2 * EXPIRY_THRESHOLD_SECONDS,
    )

    for target in ("alice", "ACCT"):
        rows = db.load_scheduled_llm_tasks_for_target(target)
        assert [r.event_name for r in rows] == ["upcoming"]
    assert db.load_scheduled_llm_tasks_for_target("nobody") == []


def test_write_txn_rolls_back_on_error(tmp_path: Path) -> None:
    db = LLMDatabase(str(tmp_path / "t.db"))
    db.save_memory("alice", "fact-1", "#chan")

    # Trigger an IntegrityError mid-block by violating a UNIQUE/PK constraint.
    with pytest.raises(sqlite3.IntegrityError), db._write_txn() as conn:
        conn.execute(
            "INSERT INTO memories (id, nick, fact, source_channel, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (1, "bob", "x", "#c", 0.0),
        )

    # Existing row still present; bob's row never committed.
    rows = db.get_memories("alice")
    assert len(rows) == 1
    assert db.get_memories("bob") == []


class TestNickCasingUnification:
    """Instruction/persona accessors lowercase nick like the memories family."""

    def test_instruction_case_insensitive_roundtrip(self, test_db: LLMDatabase) -> None:
        test_db.save_instruction("Bob", "be terse")
        assert test_db.get_instruction("BOB") == "be terse"
        assert test_db.delete_instruction("bob") is True

    def test_persona_case_insensitive_roundtrip(self, test_db: LLMDatabase) -> None:
        test_db.save_avatar_persona("Bob", "a tree spirit")
        assert test_db.get_avatar_persona("BOB") == "a tree spirit"
        assert test_db.delete_avatar_persona("bob") is True

    def test_v17_migration_dedupes_and_lowercases(self, tmp_path: Path) -> None:
        """Pre-v17 case-forked rows collapse to the newest, lowercased."""
        db = LLMDatabase(str(tmp_path / "m.db"))
        with db._write_txn() as conn:
            conn.execute(
                "INSERT INTO user_instructions (nick, instruction, updated_at) VALUES "
                "('Bob', 'old', 1.0), ('BOB', 'new', 2.0)"
            )
            conn.execute("PRAGMA user_version = 16")
        db.close()

        db2 = LLMDatabase(str(tmp_path / "m.db"))
        assert db2.get_instruction("bob") == "new"
        conn = db2._connect()
        rows = conn.execute("SELECT nick FROM user_instructions").fetchall()
        assert [r[0] for r in rows] == ["bob"]


class TestLeaseOwnership:
    """Writes that carry a lease land only while the writer still holds it.

    The TLC counterexample (docs/formal/PendingTasks.tla): the poll die()
    leaves running across an @reload outlives its lease, the new instance
    claims the row and settles it ``completed``, and the old poll's late
    ``failed_terminal`` overwrote that — the user was told a paid-for result
    failed.
    """

    def _claimed(self, db: LLMDatabase, now: float) -> tuple[int, float, float]:
        task_id = db.save_pending_task(
            task_type="draw",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="a corgi",
            model="m",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 3600,
            next_attempt_at=now,
        )
        (row,) = db.claim_due_pending_tasks(now, limit=1, lease_seconds=120)
        stale = lease_until(now, 120)
        # The lease runs out mid-call and the next instance claims the row.
        (row,) = db.claim_due_pending_tasks(now + 200, limit=1, lease_seconds=120)
        return task_id, stale, lease_until(now + 200, 120)

    def test_a_taken_over_lease_cannot_overwrite_the_new_owners_result(
        self, test_db: LLMDatabase
    ) -> None:
        now = 1_000_000.0
        task_id, stale, fresh = self._claimed(test_db, now)

        assert test_db.update_task_for_delivery(
            task_id, "ready", '{"status": "completed"}', lease=fresh
        )
        assert not test_db.update_task_for_delivery(
            task_id, "ready", '{"status": "failed_terminal"}', lease=stale
        )
        assert not test_db.release_pending_task(task_id, now + 300, "late", lease=stale)

        (row,) = test_db.load_pending_tasks("draw")
        assert row.result_payload == '{"status": "completed"}'
        assert row.attempt_count == 0

    def test_the_holder_of_the_lease_still_writes(self, test_db: LLMDatabase) -> None:
        now = 1_000_000.0
        task_id, _stale, fresh = self._claimed(test_db, now)
        assert test_db.release_pending_task(task_id, now + 300, "transient", lease=fresh)
        (row,) = test_db.load_pending_tasks("draw")
        assert row.claimed_until == 0 and row.attempt_count == 1


class TestPendingTaskUsageBooking:
    """A stashed retry's spend is booked exactly once, whatever its end state."""

    @staticmethod
    def _ready_task(
        db: LLMDatabase,
        *,
        account: str | None = None,
        cost: float = 0.02,
        prompt_tokens: int = 100,
    ) -> int:
        import json

        now = time.time()
        task_id = db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            prompt_preview="write me an epic",
            model="m",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 60,
            next_attempt_at=now,
            account=account,
        )
        payload = {
            "status": "completed",
            "content": "ok",
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 10,
        }
        db.update_task_for_delivery(task_id, "ready", json.dumps(payload))
        return task_id

    @staticmethod
    def _usage(db: LLMDatabase) -> list[tuple]:
        return (
            db._connect()
            .execute("SELECT nick, channel, command, cost, status, error_detail FROM usage")
            .fetchall()
        )

    def test_books_once_across_repeated_calls(self, test_db: LLMDatabase) -> None:
        """GIVEN a delivery that is retried (or re-delivered after a failed ack)
        WHEN every attempt books THEN one row lands, under the storage key."""
        task_id = self._ready_task(test_db)

        assert test_db.book_pending_task_usage(task_id, status="success") is True
        assert test_db.book_pending_task_usage(task_id, status="success") is False
        assert test_db.book_pending_task_usage(task_id, status="error") is False

        assert self._usage(test_db) == [("!alice", "#chan", "ask", 0.02, "success", "")]

    def test_books_under_the_account_when_authenticated(self, test_db: LLMDatabase) -> None:
        task_id = self._ready_task(test_db, account="alice_acct")
        test_db.book_pending_task_usage(task_id, status="success")
        assert self._usage(test_db)[0][0] == "alice_acct"

    @pytest.mark.parametrize(("cost", "prompt_tokens"), [(0.02, 0), (0.0, 100)])
    def test_books_any_recorded_spend(
        self, test_db: LLMDatabase, cost: float, prompt_tokens: int
    ) -> None:
        """GIVEN a retry that recorded only a cost (billed without tokens) or
        only tokens WHEN booked THEN the spend still lands; only a retry that
        recorded neither is skipped."""
        task_id = self._ready_task(test_db, cost=cost, prompt_tokens=prompt_tokens)
        assert test_db.book_pending_task_usage(task_id, status="success") is True
        assert self._usage(test_db) == [("!alice", "#chan", "ask", cost, "success", "")]

    def test_no_row_without_spend(self, test_db: LLMDatabase) -> None:
        task_id = self._ready_task(test_db, cost=0.0, prompt_tokens=0)
        assert test_db.book_pending_task_usage(task_id, status="success") is False
        assert self._usage(test_db) == []

    def test_gone_task_books_nothing(self, test_db: LLMDatabase) -> None:
        assert test_db.book_pending_task_usage(999, status="success") is False

    def test_expiry_books_an_unbooked_billed_task(self, test_db: LLMDatabase) -> None:
        """GIVEN a billed retry that never got delivered WHEN its TTL passes
        THEN the sweep books it as an error before deleting it."""
        self._ready_task(test_db)
        assert len(test_db.delete_expired_pending_tasks(time.time() + 3600)) == 1
        assert self._usage(test_db) == [
            ("!alice", "#chan", "ask", 0.02, "error", "expired before delivery")
        ]

    def test_expiry_skips_an_already_booked_task(self, test_db: LLMDatabase) -> None:
        """GIVEN delivery_failed already booked the task WHEN it expires THEN
        no second row."""
        task_id = self._ready_task(test_db)
        test_db.book_pending_task_usage(task_id, status="error", error_detail="delivery failed")
        test_db.delete_expired_pending_tasks(time.time() + 3600)
        assert [row[4:] for row in self._usage(test_db)] == [("error", "delivery failed")]


class TestNickWatches:
    """notify_when_online's stored watches: per-requester listing and the
    7-day lapse."""

    def test_list_is_scoped_to_the_requester(self, test_db: LLMDatabase) -> None:
        """GIVEN two people's watches WHEN one asks for their list (and the
        per-person cap counts it) THEN only their own watches; without a
        requester, every watched nick on the network."""
        test_db.add_nick_watch("afternet", "larry", "rdrake", "#test")
        test_db.add_nick_watch("afternet", "eve", "Bob", "#test")
        test_db.add_nick_watch("othernet", "mallory", "bob", "#test")

        assert test_db.list_nick_watches("afternet", "BOB") == ["eve"]
        assert test_db.list_nick_watches("afternet", "rdrake") == ["larry"]
        assert test_db.list_nick_watches("afternet") == ["eve", "larry"]

    def test_lapsed_watch_is_dropped_before_listing(self, test_db: LLMDatabase) -> None:
        """GIVEN a watch older than max_age WHEN listed with max_age THEN it is
        gone from the list, from the cap count and from the table, so the
        reconnect MONITOR resend skips it too."""
        test_db.add_nick_watch("afternet", "larry", "rdrake", "#test")
        with test_db._write_txn() as conn:
            conn.execute("UPDATE nick_watches SET created_at = created_at - 8 * 86400")
        test_db.add_nick_watch("afternet", "eve", "rdrake", "#test")

        week = 7 * 86400
        assert test_db.list_nick_watches("afternet", "rdrake", max_age=week) == ["eve"]
        assert test_db.list_nick_watches("afternet") == ["eve"]
        rows = test_db._connect().execute("SELECT nick FROM nick_watches").fetchall()
        assert rows == [("eve",)]
