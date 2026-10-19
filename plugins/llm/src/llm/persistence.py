"""SQLite persistence layer for LLM plugin.

Provides thread-safe database operations for reminders, usage tracking,
and pending task queue.  Uses thread-local connections with WAL mode for
concurrent read performance without the overhead of reconnecting on every call.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple

# Schema version for future migrations
SCHEMA_VERSION = 23

# Reminders older than 24 hours past their fire_at are considered expired
EXPIRY_THRESHOLD_SECONDS = 86400  # 24 hours


class ReminderRow(NamedTuple):
    """A reminder loaded from the database."""

    id: int
    event_name: str
    nick: str
    channel: str
    message: str
    action_prompt: str
    account: str | None
    fire_at: float
    created_at: float
    chain_position: int
    recurrence_seconds: int | None
    recurrence_rrule: str | None
    watch_mode: bool
    reply_msgid: str = ""


class ScheduledLlmTaskRow(NamedTuple):
    """A scheduled LLM task loaded from the database."""

    id: int
    event_name: str
    creator_nick: str
    account: str | None
    channel: str
    network: str
    wire_msg: str
    prompt: str
    fire_at: float
    created_at: float
    recurrence_seconds: int | None
    recurrence_rrule: str | None
    chain_position: int
    watch_mode: bool
    reply_target: str | None = None

    def rehydrate_msg(self):
        """Build a fresh ``IrcMsg`` from the persisted wire string."""
        from supybot.ircmsgs import IrcMsg

        return IrcMsg(s=self.wire_msg)


class UsageSummary(NamedTuple):
    """Aggregated usage statistics."""

    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float


class UsageBreakdown(NamedTuple):
    """Usage statistics grouped by a dimension (nick or channel)."""

    name: str
    total_requests: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost: float


class UsageRank(NamedTuple):
    """Rank position within a leaderboard.

    rank=0 means the entry has no usage data; rank=1 is the top spender.
    """

    rank: int  # 1-based position, 0 = no data
    total: int  # total entries in the leaderboard


def lease_until(now: float, lease_seconds: float) -> float:
    """The ``claimed_until`` a claim at ``now`` writes: the lease's identity.

    ``claim_due_pending_tasks`` stamps rows with this, and a claimer that
    passes the same value back as ``lease=`` to ``release_pending_task`` or
    ``update_task_for_delivery`` only writes while it still holds the row.
    One definition so both sides compute the same float.
    """
    return now + lease_seconds


class PendingTaskRow(NamedTuple):
    """A pending task loaded from the database."""

    id: int
    task_type: str  # ask|code|draw
    nick: str
    reply_target: str  # channel name or PM nick
    is_channel: int  # 1 channel, 0 PM
    prompt_preview: str
    model: str
    request_data: str  # JSON blob
    submitted_at: float
    expires_at: float
    attempt_count: int
    next_attempt_at: float
    claimed_until: float
    last_error: str
    delivery_state: str  # pending|ready|retrying|delivered|delivery_failed|expired|failed_terminal
    result_payload: str  # JSON blob of delivery content
    last_delivery_error: str
    delivery_attempt_count: int
    origin_request_id: str
    account: str | None


class MemoryRow(NamedTuple):
    """A long-term memory (fact) about a user."""

    id: int
    nick: str
    fact: str
    source_channel: str
    created_at: float


class MemoryCandidate(NamedTuple):
    """A provisional fact about a user, awaiting enough mentions to promote.

    Candidates accumulate ``mentions`` each time the extractor reinforces
    them. Once ``mentions`` reaches ``memoryPromotionThreshold`` the row is
    moved into ``memories``. Untouched candidates are pruned after
    ``memoryCandidateTTLDays``.
    """

    id: int
    nick: str
    fact: str
    mentions: int
    first_seen: float
    last_seen: float
    source_channel: str


class LLMDatabase:
    """SQLite database for LLM plugin persistence.

    Uses thread-local connections for thread safety: each thread gets its
    own long-lived connection (created lazily on first use).  WAL mode is
    set once per connection for concurrent read performance.
    """

    def __init__(self, db_path: str) -> None:
        """Create or open the database and run schema migration.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()
        # Set by close() (die()/__del__). A straggler worker or untracked
        # daemon thread that touches the DB after teardown then fails loudly
        # in _connect() instead of silently reopening a connection to a
        # database the process is finished with.
        self._closed = False
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        """Return a thread-local connection, creating one if needed.

        Reuses connections within the same thread to avoid the overhead of
        opening a new connection (and re-setting WAL mode) on every call.
        WAL mode is persistent on the database file so it only needs to be
        set once per connection.

        If the cached connection was closed externally (e.g. by a caller
        that obtained it via ``_connect()`` and called ``conn.close()``),
        a fresh connection is created transparently.

        Returns:
            A sqlite3.Connection with WAL journal mode.
        """
        if self._closed:
            raise RuntimeError("LLMDatabase is closed")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                # Cheapest possible liveness probe — never hits disk.
                conn.execute("SELECT 1")
                return conn
            except sqlite3.ProgrammingError:
                self._local.conn = None
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        self._local.conn = conn
        return conn

    @contextmanager
    def _write_txn(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection, commit on clean exit, rollback+raise on exception.

        Use for any write that relies on sqlite3's implicit transaction. Callers
        that issue their own ``BEGIN IMMEDIATE`` (e.g. ``claim_due_pending_tasks``)
        must NOT use this helper -- they manage commit/rollback themselves.

        Reads should not use this helper; it adds a commit on every successful
        SELECT, which is harmless but obscures intent.
        """
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    @staticmethod
    def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}

    @staticmethod
    def _add_column_if_missing(
        conn: sqlite3.Connection, table: str, column: str, coldef: str
    ) -> None:
        """Idempotent ``ALTER TABLE ... ADD COLUMN``.

        ``user_version`` is stamped only once, at the very end of ``_migrate``,
        and ``ALTER`` runs in DDL autocommit — so a crash between two ADD COLUMNs
        in one block leaves columns added while the version stays unadvanced. On
        restart the block re-runs from the top and a plain ``ADD COLUMN`` would
        raise 'duplicate column name', aborting plugin construction on every
        future boot. Skipping already-present columns makes each block safe to
        re-run regardless of where a prior run died. ``table``/``column`` are
        module-internal literals (no injection surface).
        """
        if column not in LLMDatabase._table_columns(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {coldef}")

    @staticmethod
    def _drop_column_if_present(conn: sqlite3.Connection, table: str, column: str) -> None:
        """Idempotent ``ALTER TABLE ... DROP COLUMN`` — same re-run-safety
        rationale as :meth:`_add_column_if_missing` (a re-run after the drop
        committed would otherwise raise 'no such column')."""
        if column in LLMDatabase._table_columns(conn, table):
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")

    def _migrate(self) -> None:
        """Run schema migration to create tables and indexes.

        Uses SQLite's ``PRAGMA user_version`` to track which migrations have
        been applied.  New tables are created with ``CREATE TABLE IF NOT EXISTS``
        so the initial DDL is always safe to re-run, and incremental column /
        table additions go through ``_add_column_if_missing`` /
        ``_drop_column_if_present`` so a block half-applied by a mid-migration
        crash (before ``user_version`` is stamped) re-runs cleanly instead of
        wedging startup with a 'duplicate column name' error.
        """
        conn = self._connect()
        # --- v1 baseline: create core tables ---------------------------
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_name TEXT UNIQUE NOT NULL,
                nick TEXT NOT NULL,
                channel TEXT NOT NULL,
                message TEXT NOT NULL,
                fire_at REAL NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reminders_fire_at
                ON reminders(fire_at);

            CREATE TABLE IF NOT EXISTS usage (
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
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp
                ON usage(timestamp);
            CREATE INDEX IF NOT EXISTS idx_usage_nick
                ON usage(nick);
            CREATE INDEX IF NOT EXISTS idx_usage_channel
                ON usage(channel);

            CREATE TABLE IF NOT EXISTS pending_tasks (
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
            CREATE INDEX IF NOT EXISTS idx_pending_tasks_expires_at
                ON pending_tasks(expires_at);
            CREATE INDEX IF NOT EXISTS idx_pending_tasks_due
                ON pending_tasks(next_attempt_at, claimed_until);
            CREATE INDEX IF NOT EXISTS idx_pending_tasks_type
                ON pending_tasks(task_type);
        """)
        conn.commit()

        # --- version-gated migrations ----------------------------------
        row = conn.execute("PRAGMA user_version").fetchone()
        current_version = row[0] if row else 0

        if current_version < 2:
            self._add_column_if_missing(conn, "usage", "prompt", "prompt TEXT NOT NULL DEFAULT ''")
            self._add_column_if_missing(
                conn, "usage", "status", "status TEXT NOT NULL DEFAULT 'success'"
            )
            self._add_column_if_missing(
                conn, "usage", "error_detail", "error_detail TEXT NOT NULL DEFAULT ''"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_nick_status ON usage(nick, status)")
            conn.commit()

        if current_version < 3:
            self._add_column_if_missing(
                conn,
                "pending_tasks",
                "delivery_state",
                "delivery_state TEXT NOT NULL DEFAULT 'pending'",
            )
            self._add_column_if_missing(
                conn,
                "pending_tasks",
                "result_payload",
                "result_payload TEXT NOT NULL DEFAULT ''",
            )
            self._add_column_if_missing(
                conn,
                "pending_tasks",
                "last_delivery_error",
                "last_delivery_error TEXT NOT NULL DEFAULT ''",
            )
            self._add_column_if_missing(
                conn,
                "pending_tasks",
                "delivery_attempt_count",
                "delivery_attempt_count INTEGER NOT NULL DEFAULT 0",
            )
            self._add_column_if_missing(
                conn,
                "pending_tasks",
                "origin_request_id",
                "origin_request_id TEXT NOT NULL DEFAULT ''",
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_tasks_delivery_state "
                "ON pending_tasks(delivery_state)"
            )
            conn.commit()

        if current_version < 4:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    nick TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    last_activity REAL NOT NULL,
                    PRIMARY KEY (nick, channel)
                );
            """)
            conn.commit()

        if current_version < 5:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nick TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    source_channel TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_nick
                    ON memories(nick);
            """)
            conn.commit()

        if current_version < 6:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_cleanup_state (
                    nick TEXT PRIMARY KEY,
                    saves_since_cleanup INTEGER NOT NULL DEFAULT 0
                );
            """)
            conn.commit()

        if current_version < 7:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_instructions (
                    nick TEXT PRIMARY KEY,
                    instruction TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            conn.commit()

        if current_version < 8:
            self._add_column_if_missing(conn, "pending_tasks", "account", "account TEXT")
            conn.commit()

        if current_version < 9:
            self._add_column_if_missing(
                conn, "reminders", "action_prompt", "action_prompt TEXT NOT NULL DEFAULT ''"
            )
            self._add_column_if_missing(conn, "reminders", "account", "account TEXT")
            conn.commit()

        if current_version < 10:
            # Per-chain caps: each reminder belongs to a chain (started by
            # a chat-level set, extended by action-fire reschedules). Old
            # rows backfill chain_id = event_name (single-fire chain),
            # chain_position = 1, chain_started_at = created_at.
            self._add_column_if_missing(
                conn, "reminders", "chain_id", "chain_id TEXT NOT NULL DEFAULT ''"
            )
            self._add_column_if_missing(
                conn, "reminders", "chain_position", "chain_position INTEGER NOT NULL DEFAULT 1"
            )
            self._add_column_if_missing(
                conn, "reminders", "chain_started_at", "chain_started_at REAL NOT NULL DEFAULT 0"
            )
            conn.execute("UPDATE reminders SET chain_id = event_name WHERE chain_id = ''")
            conn.execute(
                "UPDATE reminders SET chain_started_at = created_at WHERE chain_started_at = 0"
            )
            conn.commit()

        if current_version < 11:
            # chain_id was stored on every row but never used as a lookup
            # key — chain_position and chain_started_at carry the cap and
            # TTL semantics. Drop the unused column.
            self._drop_column_if_present(conn, "reminders", "chain_id")
            conn.commit()

        if current_version < 12:
            # B0.5 strategy: graceful degradation. Existing rows have NULL
            # recurrence_seconds/recurrence_rrule and watch_mode=0; the
            # legacy LLM-tool reschedule path keeps them firing via
            # parenthetical parsing in action_prompt until they exhaust
            # naturally. New rows populate the structured columns directly
            # (B2 parser, B4 mechanical reschedule). The 30-day chain TTL
            # is also retired here — the 50-fire chain_position cap remains
            # the sole runaway guard.
            self._drop_column_if_present(conn, "reminders", "chain_started_at")
            self._add_column_if_missing(
                conn, "reminders", "recurrence_seconds", "recurrence_seconds INTEGER"
            )
            self._add_column_if_missing(
                conn, "reminders", "recurrence_rrule", "recurrence_rrule TEXT"
            )
            self._add_column_if_missing(
                conn, "reminders", "watch_mode", "watch_mode INTEGER NOT NULL DEFAULT 0"
            )
            conn.commit()

        if current_version < 13:
            # Task 3 (Limnoria bridge Phase 2): native LLM tool
            # ``schedule_llm_task`` and friends. One row per active
            # schedule. Persists wire-format msg so the fire closure can
            # rebuild a fresh IrcMsg without relying on pickle (msg.tags
            # would be lost over IrcMsg.__reduce__).
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scheduled_llm_tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_name TEXT UNIQUE NOT NULL,
                    creator_nick TEXT NOT NULL,
                    account TEXT,
                    channel TEXT NOT NULL,
                    network TEXT NOT NULL,
                    wire_msg TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    fire_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    recurrence_seconds INTEGER,
                    recurrence_rrule TEXT,
                    chain_position INTEGER NOT NULL DEFAULT 1,
                    watch_mode INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_fire_at
                    ON scheduled_llm_tasks(fire_at);
                CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_account
                    ON scheduled_llm_tasks(account);
                CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_creator_nick
                    ON scheduled_llm_tasks(creator_nick);
                CREATE INDEX IF NOT EXISTS idx_scheduled_llm_tasks_owner_channel
                    ON scheduled_llm_tasks(account, creator_nick, channel);
            """)
            conn.commit()

        if current_version < 14:
            # Phase 2 follow-up B: optional cross-target delivery for
            # scheduled tasks. NULL = deliver to the originating channel
            # (legacy behavior); non-empty string = override target nick or
            # channel.
            self._add_column_if_missing(
                conn, "scheduled_llm_tasks", "reply_target", "reply_target TEXT"
            )
            conn.commit()

        if current_version < 15:
            # Multi-stage memory: facts pass through ``memory_candidates``
            # and are promoted to ``memories`` once enough mentions
            # accumulate. Untouched rows decay via TTL. See
            # _schedule_memory_extraction in plugin.py.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    nick TEXT NOT NULL,
                    fact TEXT NOT NULL,
                    mentions INTEGER NOT NULL DEFAULT 1,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    source_channel TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memory_candidates_nick
                    ON memory_candidates(nick);
            """)
            conn.commit()

        if current_version < 16:
            # Avatar persona is the verse-only counterpart to
            # user_instructions. Split out so a user's @instruct (which
            # shapes %ask everywhere) no longer bleeds into the verse
            # system prompt. See @avatar / _verse_route_for in plugin.py.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_avatar_personas (
                    nick TEXT PRIMARY KEY,
                    persona TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            conn.commit()

        if current_version < 17:
            # Nick-casing repair: the memories family always lowercased nick at
            # the DB layer, but user_instructions / user_avatar_personas
            # matched exact-case — an account whose services-reported
            # capitalization varied silently forked its rows. Dedupe keeping
            # the newest updated_at per casefolded nick, then lowercase, and
            # the accessors now lowercase on every call. Also drop
            # idx_usage_nick_status: usage.status is audit-only (write-only in
            # production queries), so the index was pure per-insert overhead.
            for table in ("user_instructions", "user_avatar_personas"):
                # Guard on existence: hand-rolled legacy DBs (and tests) may
                # carry a user_version past the CREATE-gate without the table.
                row = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
                    (table,),
                ).fetchone()
                if row is None:
                    continue
                conn.execute(
                    f"DELETE FROM {table} WHERE rowid NOT IN ("  # noqa: S608
                    f"  SELECT rowid FROM ("
                    f"    SELECT rowid, MAX(updated_at) FROM {table} GROUP BY lower(nick)"
                    f"  )"
                    f")"
                )
                conn.execute(f"UPDATE {table} SET nick = lower(nick)")  # noqa: S608
            conn.execute("DROP INDEX IF EXISTS idx_usage_nick_status")
            conn.commit()

        if current_version < 18:
            # What the user typed and what was rendered are two different
            # strings on the @animate route: a planner turn rewrites the ask
            # into a script before the video box ever sees it. The only other
            # copy lives in pending_tasks.request_data, which is deleted the
            # moment the clip is delivered — so without this column the
            # rendered wording is unrecoverable minutes after the render.
            self._add_column_if_missing(
                conn, "usage", "rendered_prompt", "rendered_prompt TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()

        if current_version < 19:
            # msgid of the message that set the reminder, so the fire can be
            # threaded under it with +draft/reply the way draw and animate
            # deliveries are. Old rows get "" and fall back to the labelled
            # "Reminder (...)" text — a fire that cannot be threaded still has
            # to say what it is answering.
            self._add_column_if_missing(
                conn, "reminders", "reply_msgid", "reply_msgid TEXT NOT NULL DEFAULT ''"
            )
            conn.commit()

        if current_version < 20:
            # IRC carries no time zone, so "tomorrow morning" parsed as UTC.
            # Keyed like user_instructions: account when identified, nick
            # otherwise.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_timezones (
                    nick TEXT PRIMARY KEY,
                    tz TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
            """)
            conn.commit()

        if current_version < 21:
            # The CTCP TIME fallback's answer, so a restart does not re-probe
            # everyone and a recurring reminder keeps its owner's offset.
            # offset_seconds NULL records a client that never answered.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_clock_offsets (
                    nick TEXT PRIMARY KEY,
                    offset_seconds INTEGER,
                    probed_at REAL NOT NULL
                );
            """)
            conn.commit()

        if current_version < 22:
            # Each speaker's latest msgid per channel, for the react tool, so
            # a restart does not forget everyone who spoke before it.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS last_msgids (
                    network TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    nick TEXT NOT NULL,
                    msgid TEXT NOT NULL,
                    seen_at REAL NOT NULL,
                    PRIMARY KEY (network, channel, nick)
                );
            """)
            conn.commit()

        if current_version < 23:
            # notify_when_online: who asked to hear when a nick signs on, and
            # where to say it. MONITOR lists die with the connection, so the
            # plugin re-sends them from here on every connect.
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS nick_watches (
                    network TEXT NOT NULL,
                    nick TEXT NOT NULL,
                    requester TEXT NOT NULL,
                    target TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (network, nick, requester)
                );
            """)
            conn.commit()

        # Stamp the schema version so future opens skip completed migrations.
        # PRAGMA statements cannot be part of executescript, nor can their value
        # be a bound parameter, so interpolate — but coerce to int so the value
        # can never be anything but a number (no SQL injection surface).
        conn.execute(f"PRAGMA user_version = {int(SCHEMA_VERSION)}")
        conn.commit()

    def close(self) -> None:
        """Close the current thread's connection and mark the DB closed.

        Called only from ``die()`` and ``__del__`` (teardown), so marking
        the whole object closed is safe — it does not interfere with the
        per-connection transparent-reconnect path in ``_connect()``, which
        operates on a caller-closed borrowed connection rather than this
        method.
        """
        self._closed = True
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def __del__(self) -> None:
        """Best-effort cleanup of the current thread's connection."""
        self.close()

    # ------------------------------------------------------------------
    # Conversation persistence
    # ------------------------------------------------------------------

    def save_conversation(
        self,
        nick: str,
        channel: str,
        messages: list[dict[str, str]],
        last_activity: float,
    ) -> None:
        """Persist a conversation's messages to the database.

        Args:
            nick: User's IRC nick (lowercased before storage).
            channel: IRC channel (lowercased before storage).
            messages: List of message dicts (role + content).
            last_activity: Timestamp of last activity.
        """
        with self._write_txn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO conversations (nick, channel, messages, last_activity) "
                "VALUES (?, ?, ?, ?)",
                (nick.lower(), channel.lower(), json.dumps(messages), last_activity),
            )

    def delete_conversation(self, nick: str, channel: str) -> None:
        """Delete a conversation from the database.

        Args:
            nick: User's IRC nick.
            channel: IRC channel.
        """
        with self._write_txn() as conn:
            conn.execute(
                "DELETE FROM conversations WHERE nick = ? AND channel = ?",
                (nick.lower(), channel.lower()),
            )

    def delete_all_conversations(self) -> None:
        """Delete all conversations from the database."""
        with self._write_txn() as conn:
            conn.execute("DELETE FROM conversations")

    def load_conversations(self) -> list[tuple[str, str, list[dict[str, str]], float]]:
        """Load all conversations from the database.

        Returns:
            List of (nick, channel, messages, last_activity) tuples.
            Rows with corrupt JSON are logged and skipped.
        """
        log = logging.getLogger("supybot.plugins.LLM")
        conn = self._connect()
        rows = conn.execute(
            "SELECT nick, channel, messages, last_activity FROM conversations"
        ).fetchall()

        result: list[tuple[str, str, list[dict[str, str]], float]] = []
        corrupt: list[tuple[str, str]] = []
        for nick, channel, messages_json, last_activity in rows:
            try:
                messages = json.loads(messages_json)
            except (json.JSONDecodeError, TypeError):
                log.warning("Skipping corrupt conversation for %s/%s", nick, channel)
                corrupt.append((nick, channel))
                continue
            result.append((nick, channel, messages, last_activity))

        # Delete all corrupt rows in a single transaction after iterating, so a
        # failure mid-cleanup can't leave a half-deleted set committed.
        if corrupt:
            with self._write_txn() as wconn:
                wconn.executemany(
                    "DELETE FROM conversations WHERE nick = ? AND channel = ?",
                    corrupt,
                )
        return result

    # ------------------------------------------------------------------
    # Reminder operations
    # ------------------------------------------------------------------

    def save_reminder(
        self,
        event_name: str,
        nick: str,
        channel: str,
        message: str,
        fire_at: float,
        *,
        action_prompt: str = "",
        account: str | None = None,
        chain_position: int = 1,
        recurrence_seconds: int | None = None,
        recurrence_rrule: str | None = None,
        watch_mode: bool = False,
        reply_msgid: str = "",
    ) -> int:
        """Save a reminder to the database.

        Args:
            event_name: Unique identifier for the reminder event.
            nick: IRC nick that created the reminder.
            channel: IRC channel the reminder was created in.
            message: Reminder message text.
            fire_at: Unix timestamp when the reminder should fire.
            action_prompt: Optional follow-up LLM prompt to run when reminder fires.
            account: Requester's resolved account name, or None if unknown.
            chain_position: 1-based position within the chain.
            recurrence_seconds: Numeric cadence (seconds) for recurring fires,
                or None for one-shot / RRULE-driven rows.
            recurrence_rrule: RFC 5545 RRULE string for calendar-driven cadences,
                or None for one-shot / numeric rows.
            watch_mode: True when the action LLM may emit ``[silent]`` to skip
                user-visible delivery for a fire (long-running watch).
            reply_msgid: msgid of the message that set the reminder, used to
                thread the fire under it. Empty when the server never
                negotiated message-tags or the setter was synthetic.

        Returns:
            The row ID of the inserted reminder.

        Raises:
            ValueError: If both ``recurrence_seconds`` and ``recurrence_rrule``
                are non-null (they are mutually exclusive).
            sqlite3.IntegrityError: If event_name already exists.
        """
        if recurrence_seconds is not None and recurrence_rrule is not None:
            raise ValueError("recurrence_seconds and recurrence_rrule are mutually exclusive")
        now = time.time()
        with self._write_txn() as conn:
            cursor = conn.execute(
                "INSERT INTO reminders "
                "(event_name, nick, channel, message, action_prompt, account, "
                "fire_at, created_at, chain_position, "
                "recurrence_seconds, recurrence_rrule, watch_mode, reply_msgid) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_name,
                    nick,
                    channel,
                    message,
                    action_prompt,
                    account,
                    fire_at,
                    now,
                    chain_position,
                    recurrence_seconds,
                    recurrence_rrule,
                    int(watch_mode),
                    reply_msgid,
                ),
            )
            assert cursor.lastrowid is not None, "INSERT must produce a lastrowid"
            return cursor.lastrowid

    def delete_reminder(self, event_name: str) -> bool:
        """Delete a reminder by event name.

        Args:
            event_name: Unique identifier for the reminder event.

        Returns:
            True if a reminder was deleted, False if not found.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM reminders WHERE event_name = ?",
                (event_name,),
            )
            return cursor.rowcount > 0

    def load_pending_reminders(self) -> list[ReminderRow]:
        """Load reminders that are still pending delivery.

        Returns reminders whose fire_at is within the last 24 hours or in the
        future. This allows delivery of slightly overdue reminders (e.g., after
        a bot restart) while excluding very old ones. Recurring chains load
        however overdue they are, so a long outage does not end them.

        Returns:
            List of ReminderRow ordered by fire_at ascending.
        """
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, event_name, nick, channel, message, action_prompt, account, "
            "fire_at, created_at, chain_position, "
            "recurrence_seconds, recurrence_rrule, watch_mode, reply_msgid "
            "FROM reminders WHERE fire_at > ? "
            "OR recurrence_seconds IS NOT NULL OR recurrence_rrule IS NOT NULL "
            "ORDER BY fire_at",
            (cutoff,),
        ).fetchall()
        # watch_mode is stored as INTEGER 0/1; expose as bool on the row.
        return [
            ReminderRow(
                id=row[0],
                event_name=row[1],
                nick=row[2],
                channel=row[3],
                message=row[4],
                action_prompt=row[5],
                account=row[6],
                fire_at=row[7],
                created_at=row[8],
                chain_position=row[9],
                recurrence_seconds=row[10],
                recurrence_rrule=row[11],
                watch_mode=bool(row[12]),
                reply_msgid=row[13],
            )
            for row in rows
        ]

    def delete_expired_reminders(self) -> int:
        """Delete one-shot reminders that are more than 24 hours overdue.

        Returns:
            Number of reminders deleted.
        """
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM reminders WHERE fire_at <= ? "
                "AND recurrence_seconds IS NULL AND recurrence_rrule IS NULL",
                (cutoff,),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # Scheduled LLM task operations
    # ------------------------------------------------------------------

    _SCHEDULED_LLM_TASK_COLUMNS = (
        "id, event_name, creator_nick, account, channel, network, wire_msg, "
        "prompt, fire_at, created_at, recurrence_seconds, recurrence_rrule, "
        "chain_position, watch_mode, reply_target"
    )

    @staticmethod
    def _row_to_scheduled_llm_task(row: tuple) -> ScheduledLlmTaskRow:
        return ScheduledLlmTaskRow(
            id=row[0],
            event_name=row[1],
            creator_nick=row[2],
            account=row[3],
            channel=row[4],
            network=row[5],
            wire_msg=row[6],
            prompt=row[7],
            fire_at=row[8],
            created_at=row[9],
            recurrence_seconds=row[10],
            recurrence_rrule=row[11],
            chain_position=row[12],
            watch_mode=bool(row[13]),
            reply_target=row[14],
        )

    @staticmethod
    def _owner_where(*, account: str | None, nick: str) -> tuple[str, list[object]]:
        """Return SQL fragment + params matching scheduled-task ownership.

        If ``account`` is set, match by account (case-insensitive). Otherwise
        match by ``creator_nick`` among rows with ``NULL`` account
        (case-insensitive).
        """
        if account is not None:
            return "lower(account) = lower(?)", [account]
        return "account IS NULL AND lower(creator_nick) = lower(?)", [nick]

    def save_scheduled_llm_task(
        self,
        event_name: str,
        creator_nick: str,
        account: str | None,
        channel: str,
        network: str,
        wire_msg: str,
        prompt: str,
        fire_at: float,
        *,
        recurrence_seconds: int | None = None,
        recurrence_rrule: str | None = None,
        chain_position: int = 1,
        watch_mode: bool = False,
        reply_target: str | None = None,
    ) -> int:
        """Save a scheduled LLM task row.

        Raises:
            ValueError: if both recurrence kinds are non-null.
            sqlite3.IntegrityError: if event_name already exists.
        """
        if recurrence_seconds is not None and recurrence_rrule is not None:
            raise ValueError("recurrence_seconds and recurrence_rrule are mutually exclusive")
        now = time.time()
        with self._write_txn() as conn:
            cursor = conn.execute(
                "INSERT INTO scheduled_llm_tasks "
                "(event_name, creator_nick, account, channel, network, wire_msg, "
                "prompt, fire_at, created_at, recurrence_seconds, recurrence_rrule, "
                "chain_position, watch_mode, reply_target) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_name,
                    creator_nick,
                    account,
                    channel,
                    network,
                    wire_msg,
                    prompt,
                    fire_at,
                    now,
                    recurrence_seconds,
                    recurrence_rrule,
                    chain_position,
                    int(watch_mode),
                    reply_target,
                ),
            )
            assert cursor.lastrowid is not None, "INSERT must produce a lastrowid"
            return cursor.lastrowid

    def update_scheduled_llm_task_fire_at(
        self,
        event_name: str,
        fire_at: float,
        *,
        chain_position: int | None = None,
    ) -> None:
        """Update fire_at (and optionally chain_position) for a row."""
        with self._write_txn() as conn:
            if chain_position is None:
                conn.execute(
                    "UPDATE scheduled_llm_tasks SET fire_at = ? WHERE event_name = ?",
                    (fire_at, event_name),
                )
            else:
                conn.execute(
                    "UPDATE scheduled_llm_tasks SET fire_at = ?, chain_position = ? "
                    "WHERE event_name = ?",
                    (fire_at, chain_position, event_name),
                )

    def delete_scheduled_llm_task(self, event_name: str) -> bool:
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM scheduled_llm_tasks WHERE event_name = ?",
                (event_name,),
            )
            return cursor.rowcount > 0

    def load_active_scheduled_llm_tasks(self) -> list[ScheduledLlmTaskRow]:
        """Load rows whose fire_at is within the last 24 hours or in the future."""
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        conn = self._connect()
        rows = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks WHERE fire_at > ? ORDER BY fire_at",
            (cutoff,),
        ).fetchall()
        return [self._row_to_scheduled_llm_task(r) for r in rows]

    def get_scheduled_llm_task(self, event_name: str) -> ScheduledLlmTaskRow | None:
        """Indexed point-lookup by event_name."""
        conn = self._connect()
        row = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks WHERE event_name = ?",
            (event_name,),
        ).fetchone()
        return self._row_to_scheduled_llm_task(row) if row else None

    def load_scheduled_llm_tasks_for_target(self, target: str) -> list[ScheduledLlmTaskRow]:
        """Active rows whose creator_nick OR account matches ``target``.

        Owner-only callers identify another user by either the nick that
        scheduled the task or the account it was stored under,
        case-insensitively.
        """
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        conn = self._connect()
        rows = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks "
            "WHERE (lower(creator_nick) = lower(?) OR lower(account) = lower(?)) "
            "AND fire_at > ? ORDER BY fire_at",
            (target, target, cutoff),
        ).fetchall()
        return [self._row_to_scheduled_llm_task(r) for r in rows]

    def load_scheduled_llm_tasks_for(
        self, *, account: str | None, nick: str
    ) -> list[ScheduledLlmTaskRow]:
        """Active rows owned by the caller. Case-insensitive Identity semantics."""
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        where, params = self._owner_where(account=account, nick=nick)
        conn = self._connect()
        rows = conn.execute(
            f"SELECT {self._SCHEDULED_LLM_TASK_COLUMNS} "
            "FROM scheduled_llm_tasks "
            f"WHERE {where} AND fire_at > ? ORDER BY fire_at",
            (*params, cutoff),
        ).fetchall()
        return [self._row_to_scheduled_llm_task(r) for r in rows]

    def count_scheduled_llm_tasks_for(self, *, account: str | None, nick: str, channel: str) -> int:
        """Count active rows owned by the caller in this channel.

        When ``account`` is non-None, count rows with that account regardless of
        nick. Otherwise count by raw nick. Comparisons are case-insensitive to
        match ``Identity.matches``.
        """
        cutoff = time.time() - EXPIRY_THRESHOLD_SECONDS
        where, params = self._owner_where(account=account, nick=nick)
        row = (
            self._connect()
            .execute(
                "SELECT COUNT(*) FROM scheduled_llm_tasks "
                f"WHERE {where} AND channel = ? AND fire_at > ?",
                (*params, channel, cutoff),
            )
            .fetchone()
        )
        return int(row[0])

    # ------------------------------------------------------------------
    # Pending task operations
    # ------------------------------------------------------------------

    _PENDING_TASK_COLUMNS = (
        "id, task_type, nick, reply_target, is_channel, prompt_preview, model, "
        "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
        "claimed_until, last_error, delivery_state, result_payload, "
        "last_delivery_error, delivery_attempt_count, origin_request_id, account"
    )

    def save_pending_task(
        self,
        task_type: str,
        nick: str,
        reply_target: str,
        is_channel: bool,
        prompt_preview: str,
        model: str,
        request_data: str,
        submitted_at: float,
        expires_at: float,
        next_attempt_at: float,
        origin_request_id: str = "",
        account: str | None = None,
    ) -> int:
        """Save a pending task to the database.

        Args:
            task_type: Command type (ask, code, draw).
            nick: IRC nick that initiated the command.
            reply_target: Channel or PM nick for delivery.
            is_channel: True if reply_target is a channel.
            prompt_preview: Truncated prompt for display.
            model: Model identifier.
            request_data: JSON-serialized request payload.
            submitted_at: Unix timestamp when originally submitted.
            expires_at: Unix timestamp after which to stop retrying.
            next_attempt_at: Unix timestamp for first retry.
            origin_request_id: Stable trace/request ID captured at request acceptance.
            account: Resolved account name at submission time, or None if the
                requester was not identified. Delivery-time logging reads this
                directly instead of doing a late nick→account lookup.

        Returns:
            The row ID of the inserted task.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "INSERT INTO pending_tasks "
                "(task_type, nick, reply_target, is_channel, prompt_preview, model, "
                "request_data, submitted_at, expires_at, attempt_count, next_attempt_at, "
                "claimed_until, last_error, origin_request_id, account) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 0, '', ?, ?)",
                (
                    task_type,
                    nick,
                    reply_target,
                    1 if is_channel else 0,
                    prompt_preview,
                    model,
                    request_data,
                    submitted_at,
                    expires_at,
                    next_attempt_at,
                    origin_request_id,
                    account,
                ),
            )
            assert cursor.lastrowid is not None, "INSERT must produce a lastrowid"
            return cursor.lastrowid

    def claim_due_pending_tasks(
        self,
        now: float,
        limit: int,
        lease_seconds: int,
        delivery_state_filter: str | tuple[str, ...] | None = None,
        max_delivery_attempts: int | None = None,
    ) -> list[PendingTaskRow]:
        """Atomically claim pending tasks that are due for retry.

        Uses BEGIN IMMEDIATE for exclusive write access so concurrent
        callers cannot claim the same rows.

        Args:
            now: Current Unix timestamp.
            limit: Maximum number of tasks to claim.
            lease_seconds: How long to hold the claim (seconds).
            delivery_state_filter: Optional filter on delivery_state. Can be a
                single state string or a tuple of states to match.
            max_delivery_attempts: Optional cap used by delivery-phase callers
                to skip rows that already exhausted retries.

        Returns:
            List of claimed PendingTaskRow objects.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")

            if delivery_state_filter is not None:
                if isinstance(delivery_state_filter, str):
                    state_clause = "AND delivery_state = ?"
                    state_params: tuple[object, ...] = (delivery_state_filter,)
                else:
                    placeholders = ",".join("?" for _ in delivery_state_filter)
                    state_clause = f"AND delivery_state IN ({placeholders})"
                    state_params = tuple(delivery_state_filter)
            else:
                state_clause = ""
                state_params = ()

            if max_delivery_attempts is not None:
                attempt_clause = "AND delivery_attempt_count < ?"
                attempt_params: tuple[object, ...] = (max_delivery_attempts,)
            else:
                attempt_clause = ""
                attempt_params = ()

            rows = conn.execute(
                f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks "
                f"WHERE next_attempt_at <= ? AND claimed_until <= ? {state_clause} {attempt_clause} "
                "ORDER BY next_attempt_at LIMIT ?",
                (now, now, *state_params, *attempt_params, limit),
            ).fetchall()

            if not rows:
                conn.commit()
                return []

            claimed_until = lease_until(now, lease_seconds)
            ids = [row[0] for row in rows]
            placeholders = ",".join("?" for _ in ids)
            conn.execute(
                f"UPDATE pending_tasks SET claimed_until = ? WHERE id IN ({placeholders})",
                [claimed_until, *ids],
            )
            conn.commit()
            return [PendingTaskRow(*row) for row in rows]
        except Exception:
            conn.rollback()
            raise

    def release_pending_task(
        self,
        task_id: int,
        next_attempt_at: float,
        last_error: str,
        increment_attempt: bool = True,
        lease: float | None = None,
    ) -> bool:
        """Release a claimed task back to the queue for later retry.

        Args:
            task_id: ID of the task to release.
            next_attempt_at: When to retry next.
            last_error: Error message from this attempt.
            increment_attempt: Whether to bump attempt_count.
            lease: The ``lease_until`` value the caller claimed the row with.
                When given, the write lands only if the row still carries it.

        Returns:
            True if the task was updated, False if not found or the lease
            was taken over.
        """
        owner_clause, owner_params = self._lease_clause(lease)
        with self._write_txn() as conn:
            if increment_attempt:
                cursor = conn.execute(
                    "UPDATE pending_tasks SET "
                    "next_attempt_at = ?, claimed_until = 0, "
                    "last_error = ?, attempt_count = attempt_count + 1 "
                    f"WHERE id = ?{owner_clause}",
                    (next_attempt_at, last_error, task_id, *owner_params),
                )
            else:
                cursor = conn.execute(
                    "UPDATE pending_tasks SET "
                    "next_attempt_at = ?, claimed_until = 0, last_error = ? "
                    f"WHERE id = ?{owner_clause}",
                    (next_attempt_at, last_error, task_id, *owner_params),
                )
            return cursor.rowcount > 0

    @staticmethod
    def _lease_clause(lease: float | None) -> tuple[str, tuple[float, ...]]:
        """``AND claimed_until = ?`` when the caller holds a lease.

        A poll that outlives its lease — the one ``die()`` leaves running
        across an @reload, whose 2s drain cannot wait out a slow provider
        call — must not overwrite a row the next instance has since claimed
        and settled. See docs/formal/PendingTasks.tla.
        """
        if lease is None:
            return "", ()
        return " AND claimed_until = ?", (lease,)

    def delete_pending_task(self, task_id: int) -> bool:
        """Delete a pending task by ID.

        Args:
            task_id: ID of the task to delete.

        Returns:
            True if a task was deleted, False if not found.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM pending_tasks WHERE id = ?",
                (task_id,),
            )
            return cursor.rowcount > 0

    def update_task_for_delivery(
        self,
        task_id: int,
        delivery_state: str,
        result_payload: str,
        lease: float | None = None,
    ) -> bool:
        """Transition a task to a delivery state with its result payload.

        Args:
            task_id: ID of the task to update.
            delivery_state: New delivery state (ready, failed_terminal, etc.).
            result_payload: JSON-serialized result for delivery.
            lease: As for ``release_pending_task``.

        Returns:
            True if the task was updated, False if not found or the lease
            was taken over.
        """
        owner_clause, owner_params = self._lease_clause(lease)
        with self._write_txn() as conn:
            cursor = conn.execute(
                "UPDATE pending_tasks SET "
                "delivery_state = ?, result_payload = ?, claimed_until = 0 "
                f"WHERE id = ?{owner_clause}",
                (delivery_state, result_payload, task_id, *owner_params),
            )
            return cursor.rowcount > 0

    def update_delivery_attempt(
        self,
        task_id: int,
        delivery_state: str,
        last_delivery_error: str,
        delivery_attempt_count: int,
        next_attempt_at: float,
    ) -> bool:
        """Record a delivery attempt outcome (success or failure).

        Args:
            task_id: ID of the task to update.
            delivery_state: New delivery state (retrying, delivery_failed).
            last_delivery_error: Error message from the delivery attempt.
            delivery_attempt_count: Updated delivery attempt count.
            next_attempt_at: When to retry delivery next.

        Returns:
            True if the task was updated, False if not found.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "UPDATE pending_tasks SET "
                "delivery_state = ?, last_delivery_error = ?, "
                "delivery_attempt_count = ?, next_attempt_at = ?, claimed_until = 0 "
                "WHERE id = ?",
                (
                    delivery_state,
                    last_delivery_error,
                    delivery_attempt_count,
                    next_attempt_at,
                    task_id,
                ),
            )
            return cursor.rowcount > 0

    def delete_expired_pending_tasks(self, now: float) -> list[PendingTaskRow]:
        """Delete pending tasks whose expires_at has passed, in any state.

        ``expires_at`` is the whole-task hard deadline ("stop retrying after
        this"). It bounds the *entire* lifecycle — provider retries AND delivery
        retries. A row that reached ready/retrying/delivery_failed but can no
        longer be delivered (e.g. the bot left the channel) must still be reaped
        once its TTL passes; otherwise it re-polls every 30s forever, leaking
        rows and firing perpetual scheduler wakeups. Delivery retry is still
        honored for any row whose TTL has not yet elapsed.

        Args:
            now: Current Unix timestamp.

        Returns:
            List of expired PendingTaskRow objects (before deletion).
        """
        with self._write_txn() as conn:
            rows = conn.execute(
                f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks WHERE expires_at <= ?",
                (now,),
            ).fetchall()
            if rows:
                ids = [row[0] for row in rows]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM pending_tasks WHERE id IN ({placeholders})",
                    ids,
                )
            return [PendingTaskRow(*row) for row in rows]

    def get_next_due_time(self) -> float | None:
        """Return the earliest next_attempt_at for actionable unclaimed tasks.

        Only considers tasks that could be processed by the scheduler:
        unclaimed (or lease-expired) rows with delivery_state in
        (pending, ready, retrying).

        Returns:
            Earliest next_attempt_at timestamp, or None if the queue is empty.
        """
        conn = self._connect()
        now = time.time()
        row = conn.execute(
            "SELECT MIN(next_attempt_at) FROM pending_tasks "
            "WHERE claimed_until <= ? "
            "AND delivery_state IN ('pending', 'ready', 'retrying')",
            (now,),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return row[0]

    def load_pending_tasks(self, task_type: str | None = None) -> list[PendingTaskRow]:
        """Load pending tasks, optionally filtered by type.

        Intended for debugging and tests.

        Args:
            task_type: Optional filter (ask, code, draw).

        Returns:
            List of PendingTaskRow ordered by submitted_at ascending.
        """
        conn = self._connect()
        if task_type is not None:
            rows = conn.execute(
                f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks "
                "WHERE task_type = ? ORDER BY submitted_at",
                (task_type,),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {self._PENDING_TASK_COLUMNS} FROM pending_tasks ORDER BY submitted_at",
            ).fetchall()
        return [PendingTaskRow(*row) for row in rows]

    def active_animate_targets(self, now: float, max_age_seconds: float) -> list[str]:
        """Reply targets with a video still rendering.

        The typing refresher's whole state, read fresh each pass instead of
        tracked in memory: a row is ``pending`` while the box renders and
        ``ready`` once the clip is in hand, so this is exactly "clips still
        rendering". Deriving it means nothing has to be released, which is
        what makes at-least-once delivery and ``delivery_failed`` harmless
        here.

        Read-only on purpose. ``claim_due_pending_tasks`` accepts a
        ``delivery_state`` filter and looks like the right tool, but it opens
        BEGIN IMMEDIATE and leases the rows it returns — calling it from a
        four-second loop would steal work from the pending-task poller.

        Args:
            now: Current epoch seconds.
            max_age_seconds: Ignore rows submitted longer ago than this. A job
                can stay pending for ``animateExpiry`` (1800s by default);
                typing should give up long before that.

        Returns:
            Distinct reply targets, sorted by target name.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT reply_target FROM pending_tasks "
            "WHERE task_type = 'animate' AND delivery_state = 'pending' "
            "AND submitted_at > ? AND reply_target <> '' "
            "ORDER BY reply_target",
            (now - max_age_seconds,),
        ).fetchall()
        return [str(row[0]) for row in rows]

    _PENDING_ANIMATE_WHERE = (
        "task_type = 'animate' AND delivery_state = 'pending' AND expires_at > ?"
    )

    def count_pending_animate(self, now: float) -> int:
        """Clips submitted and not yet rendered — the box's queue depth.

        Same predicate as ``active_animate_targets`` minus its age window:
        ``pending`` means the box has not handed the clip back yet, and an
        expired row is about to be reported as such, not rendered.

        Read-only for the same reason ``active_animate_targets`` is:
        ``claim_due_pending_tasks`` leases the rows it returns, so counting
        with it would steal work from the pending-task poller.

        Args:
            now: Current epoch seconds; rows expiring before it don't count.

        Returns:
            Number of clips still waiting on the box.
        """
        conn = self._connect()
        row = conn.execute(
            f"SELECT COUNT(*) FROM pending_tasks WHERE {self._PENDING_ANIMATE_WHERE}",
            (now,),
        ).fetchone()
        return int(row[0]) if row else 0

    def count_pending_animate_for(self, now: float, *, account: str | None, nick: str) -> int:
        """One user's share of ``count_pending_animate``.

        Identity policy matches the rest of the plugin: by account when the
        caller has one, by nick otherwise, both case-insensitive. Rows stashed
        under an account are deliberately invisible to a nick-only lookup —
        someone who signs out mid-queue gets a fresh allowance rather than
        another user's rows counted against them.

        Args:
            now: Current epoch seconds; rows expiring before it don't count.
            account: Resolved account name, or None if unidentified.
            nick: IRC nick, used only when there is no account.

        Returns:
            Number of that user's clips still waiting on the box.
        """
        conn = self._connect()
        if account:
            sql = (
                f"SELECT COUNT(*) FROM pending_tasks WHERE {self._PENDING_ANIMATE_WHERE} "
                "AND account IS NOT NULL AND lower(account) = lower(?)"
            )
            params: tuple = (now, account)
        else:
            sql = (
                f"SELECT COUNT(*) FROM pending_tasks WHERE {self._PENDING_ANIMATE_WHERE} "
                "AND (account IS NULL OR account = '') AND lower(nick) = lower(?)"
            )
            params = (now, nick)
        row = conn.execute(sql, params).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Usage operations
    # ------------------------------------------------------------------

    def log_usage(
        self,
        nick: str,
        channel: str,
        command: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float,
        prompt: str = "",
        status: str = "success",
        error_detail: str = "",
        rendered_prompt: str = "",
    ) -> None:
        """Log a usage event.

        Args:
            nick: IRC nick that triggered the command.
            channel: IRC channel or PM target.
            command: Command name (ask, code, draw).
            model: Model identifier used.
            prompt_tokens: Number of prompt tokens consumed.
            completion_tokens: Number of completion tokens generated.
            cost: Estimated cost in USD.
            prompt: The user's prompt text (for audit/flagging).
            status: Outcome status.  Known values: ``"success"``,
                ``"error"``, ``"content_blocked"``,
                ``"auth_failure"``, ``"rate_limited"``.
            error_detail: Additional error context when status is not success.
            rendered_prompt: What the generator was actually asked for, when a
                planner rewrote the user's words on the way there (@animate).
                Empty on every path that sends the prompt through unchanged.
        """
        with self._write_txn() as conn:
            conn.execute(
                "INSERT INTO usage "
                "(timestamp, nick, channel, command, model, prompt_tokens, "
                "completion_tokens, cost, prompt, status, error_detail, rendered_prompt) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    nick,
                    channel,
                    command,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost,
                    prompt,
                    status,
                    error_detail,
                    rendered_prompt,
                ),
            )

    _USAGE_SELECT = (
        "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
        "COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) FROM usage"
    )

    def _query_usage_summary(self, conditions: list[str], params: list[object]) -> UsageSummary:
        """Run the usage aggregation SELECT with optional AND-joined conditions.

        Conditions are conjunctive only. ``COUNT(*) + COALESCE`` guarantees a
        row, so no ``None``-row guard is needed.
        """
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        row = (
            self._connect()
            .execute(
                f"{self._USAGE_SELECT}{where}",
                tuple(params),
            )
            .fetchone()
        )
        return UsageSummary(
            total_requests=row[0],
            total_prompt_tokens=row[1],
            total_completion_tokens=row[2],
            total_cost=row[3],
        )

    def get_usage_summary(self, since: float | None = None) -> UsageSummary:
        """Get aggregated usage statistics.

        Args:
            since: Optional Unix timestamp to filter results (only include
                   records after this time). If None, includes all records.

        Returns:
            UsageSummary with totals for requests, tokens, and cost.
        """
        conds: list[str] = []
        params: list[object] = []
        if since is not None:
            conds.append("timestamp >= ?")
            params.append(since)
        return self._query_usage_summary(conds, params)

    def get_usage_by_nick(self, since: float | None = None, limit: int = 5) -> list[UsageBreakdown]:
        """Get usage statistics grouped by nick, sorted by cost descending.

        Args:
            since: Optional Unix timestamp filter.
            limit: Maximum number of results to return.

        Returns:
            List of UsageBreakdown sorted by total_cost descending.
        """
        return self._get_usage_by_dimension("nick", since, limit)

    def get_usage_by_channel(
        self, since: float | None = None, limit: int = 5
    ) -> list[UsageBreakdown]:
        """Get usage statistics grouped by channel, sorted by cost descending.

        Args:
            since: Optional Unix timestamp filter.
            limit: Maximum number of results to return.

        Returns:
            List of UsageBreakdown sorted by total_cost descending.
        """
        return self._get_usage_by_dimension("channel", since, limit)

    def _get_usage_by_dimension(
        self, dimension: str, since: float | None, limit: int
    ) -> list[UsageBreakdown]:
        """Get usage grouped by a dimension (nick or channel).

        Args:
            dimension: Column name to group by ("nick" or "channel").
            since: Optional Unix timestamp filter.
            limit: Maximum number of results.

        Returns:
            List of UsageBreakdown sorted by total_cost descending.
        """
        assert dimension in ("nick", "channel"), f"Invalid dimension: {dimension}"
        # dimension is always a hardcoded column name from our own code,
        # never user input, so string interpolation is safe here.
        conn = self._connect()
        if since is not None:
            rows = conn.execute(
                f"SELECT {dimension}, COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                f"COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                f"FROM usage WHERE timestamp >= ? "
                f"GROUP BY {dimension} ORDER BY SUM(cost) DESC LIMIT ?",
                (since, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT {dimension}, COUNT(*), COALESCE(SUM(prompt_tokens), 0), "
                f"COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(cost), 0.0) "
                f"FROM usage "
                f"GROUP BY {dimension} ORDER BY SUM(cost) DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            UsageBreakdown(
                name=row[0],
                total_requests=row[1],
                total_prompt_tokens=row[2],
                total_completion_tokens=row[3],
                total_cost=row[4],
            )
            for row in rows
        ]

    def get_usage_summary_for_channel(
        self, channel: str, since: float | None = None
    ) -> UsageSummary:
        """Get aggregated usage statistics for a specific channel.

        Args:
            channel: IRC channel name to filter by.
            since: Optional Unix timestamp filter.

        Returns:
            UsageSummary with totals for the given channel.
        """
        conds: list[str] = ["channel = ?"]
        params: list[object] = [channel]
        if since is not None:
            conds.append("timestamp >= ?")
            params.append(since)
        return self._query_usage_summary(conds, params)

    def get_usage_summary_for_nick(
        self, nick: str, since: float | None = None, channel: str | None = None
    ) -> UsageSummary:
        """Get aggregated usage statistics for a specific nick.

        Args:
            nick: IRC nick to filter by.
            since: Optional Unix timestamp filter.
            channel: Optional channel to further scope the query.

        Returns:
            UsageSummary with totals for the given nick (optionally in a channel).
        """
        conds: list[str] = ["nick = ?"]
        params: list[object] = [nick]
        if channel is not None:
            conds.append("channel = ?")
            params.append(channel)
        if since is not None:
            conds.append("timestamp >= ?")
            params.append(since)
        return self._query_usage_summary(conds, params)

    def get_channel_rank(self, channel: str, since: float | None = None) -> UsageRank:
        """Get the cost rank of a channel among all channels.

        Args:
            channel: IRC channel name.
            since: Optional Unix timestamp filter.

        Returns:
            UsageRank with 1-based rank (0 if channel has no usage).
        """
        return self._get_rank("channel", channel, since)

    def get_nick_rank(
        self, nick: str, since: float | None = None, channel: str | None = None
    ) -> UsageRank:
        """Get the cost rank of a nick among all nicks.

        Args:
            nick: IRC nick.
            since: Optional Unix timestamp filter.
            channel: Optional channel to scope the ranking.

        Returns:
            UsageRank with 1-based rank (0 if nick has no usage).
        """
        return self._get_rank("nick", nick, since, scope_channel=channel)

    def _get_rank(
        self,
        dimension: str,
        value: str,
        since: float | None,
        scope_channel: str | None = None,
    ) -> UsageRank:
        """Compute the cost rank of a value within a dimension.

        Uses a count-of-higher approach for SQLite compatibility (no window
        functions required): rank = number of entries with strictly higher
        total cost + 1.

        Args:
            dimension: Column to rank by ("nick" or "channel").
            value: The specific nick or channel to find the rank of.
            since: Optional Unix timestamp filter.
            scope_channel: Optional channel to scope the query (only for nick ranking).

        Returns:
            UsageRank with 1-based rank, or rank=0 if the value has no usage.
        """
        assert dimension in ("nick", "channel"), f"Invalid dimension: {dimension}"
        conn = self._connect()
        # Build WHERE clause fragments
        time_filter = "timestamp >= ?" if since is not None else None
        channel_filter = "channel = ?" if scope_channel is not None else None

        base_conditions = [c for c in (time_filter, channel_filter) if c]
        base_where = (" WHERE " + " AND ".join(base_conditions)) if base_conditions else ""
        base_params: list[object] = []
        if since is not None:
            base_params.append(since)
        if scope_channel is not None:
            base_params.append(scope_channel)

        # Total distinct entries
        total_row = conn.execute(
            f"SELECT COUNT(DISTINCT {dimension}) FROM usage{base_where}",
            tuple(base_params),
        ).fetchone()
        total = total_row[0] if total_row else 0

        # Get the value's total cost
        value_conditions = [f"{dimension} = ?"] + list(base_conditions)
        value_where = " AND ".join(value_conditions)
        value_params: list[object] = [value, *base_params]

        cost_row = conn.execute(
            f"SELECT COALESCE(SUM(cost), 0.0) FROM usage WHERE {value_where}",
            tuple(value_params),
        ).fetchone()
        value_cost = cost_row[0] if cost_row else 0.0

        if value_cost == 0.0:
            # Check if there's actually any usage for this value
            count_row = conn.execute(
                f"SELECT COUNT(*) FROM usage WHERE {value_where}",
                tuple(value_params),
            ).fetchone()
            if count_row is None or count_row[0] == 0:
                return UsageRank(rank=0, total=total)

        # Count entries with strictly higher cost
        rank_sql = (
            f"SELECT COUNT(*) FROM "
            f"(SELECT {dimension}, SUM(cost) AS total_cost "
            f"FROM usage{base_where} "
            f"GROUP BY {dimension}) sub "
            f"WHERE sub.total_cost > ?"
        )
        rank_params: list[object] = [*base_params, value_cost]
        rank_row = conn.execute(rank_sql, tuple(rank_params)).fetchone()
        rank = (rank_row[0] + 1) if rank_row else 1

        return UsageRank(rank=rank, total=total)

    # ------------------------------------------------------------------
    # Memory operations
    # ------------------------------------------------------------------

    def save_memory(self, nick: str, fact: str, source_channel: str) -> int:
        """Save a memory fact for a user.

        Args:
            nick: IRC nick (stored lowercased).
            fact: The fact to remember about the user.
            source_channel: Channel where the fact was learned.

        Returns:
            The row ID of the inserted memory.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "INSERT INTO memories (nick, fact, source_channel, created_at) VALUES (?, ?, ?, ?)",
                (nick.lower(), fact, source_channel.lower(), time.time()),
            )
            assert cursor.lastrowid is not None, "INSERT must produce a lastrowid"
            return cursor.lastrowid

    def get_memories(self, nick: str) -> list[MemoryRow]:
        """Get all memories for a user, most recent first.

        Args:
            nick: IRC nick (matched case-insensitively).

        Returns:
            List of MemoryRow ordered by created_at descending (newest first).
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, nick, fact, source_channel, created_at FROM memories "
            "WHERE nick = ? ORDER BY created_at DESC",
            (nick.lower(),),
        ).fetchall()
        return [MemoryRow(*row) for row in rows]

    def delete_memory(self, nick: str, memory_id: int) -> bool:
        """Delete a specific memory by ID and nick.

        The nick check prevents users from deleting other users' memories.

        Args:
            nick: IRC nick (must match the memory's owner).
            memory_id: Row ID of the memory to delete.

        Returns:
            True if a memory was deleted, False if not found or wrong owner.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ? AND nick = ?",
                (memory_id, nick.lower()),
            )
            return cursor.rowcount > 0

    def update_memory(self, nick: str, memory_id: int, new_fact: str) -> bool:
        """Update the fact text of a specific memory.

        The nick check prevents users from editing other users' memories.

        Args:
            nick: IRC nick (must match the memory's owner).
            memory_id: Row ID of the memory to update.
            new_fact: The new fact text.

        Returns:
            True if a memory was updated, False if not found or wrong owner.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "UPDATE memories SET fact = ? WHERE id = ? AND nick = ?",
                (new_fact, memory_id, nick.lower()),
            )
            return cursor.rowcount > 0

    def delete_all_memories(self, nick: str) -> int:
        """Delete all memories for a user.

        Args:
            nick: IRC nick whose memories should be deleted.

        Returns:
            Number of memories deleted.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE nick = ?",
                (nick.lower(),),
            )
            return cursor.rowcount

    def increment_memory_saves(self, nick: str) -> int:
        """Increment the memory-saves-since-cleanup counter for a user.

        Args:
            nick: IRC nick (stored lowercased).

        Returns:
            The new counter value after incrementing.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "INSERT INTO memory_cleanup_state (nick, saves_since_cleanup) "
                "VALUES (?, 1) "
                "ON CONFLICT(nick) DO UPDATE SET saves_since_cleanup = saves_since_cleanup + 1 "
                "RETURNING saves_since_cleanup",
                (nick.lower(),),
            )
            row = cursor.fetchone()
            return row[0] if row else 0

    def reset_memory_saves(self, nick: str) -> None:
        """Reset the memory-saves-since-cleanup counter for a user.

        Args:
            nick: IRC nick (stored lowercased).
        """
        with self._write_txn() as conn:
            conn.execute(
                "UPDATE memory_cleanup_state SET saves_since_cleanup = 0 WHERE nick = ?",
                (nick.lower(),),
            )

    # ------------------------------------------------------------------
    # Memory candidate operations
    # ------------------------------------------------------------------

    def add_memory_candidate(self, nick: str, fact: str, source_channel: str) -> int:
        """Insert a new candidate fact for a user with mentions=1.

        Args:
            nick: IRC nick (stored lowercased).
            fact: The candidate fact text.
            source_channel: Channel where the fact was first observed.

        Returns:
            The row ID of the inserted candidate.
        """
        now = time.time()
        with self._write_txn() as conn:
            cursor = conn.execute(
                "INSERT INTO memory_candidates "
                "(nick, fact, mentions, first_seen, last_seen, source_channel) "
                "VALUES (?, ?, 1, ?, ?, ?)",
                (nick.lower(), fact, now, now, source_channel.lower()),
            )
            assert cursor.lastrowid is not None, "INSERT must produce a lastrowid"
            return cursor.lastrowid

    def get_memory_candidates(self, nick: str) -> list[MemoryCandidate]:
        """Get all candidates for a user, most-mentioned first.

        Args:
            nick: IRC nick (matched case-insensitively).

        Returns:
            Candidates ordered by mentions DESC, last_seen DESC.
        """
        conn = self._connect()
        rows = conn.execute(
            "SELECT id, nick, fact, mentions, first_seen, last_seen, source_channel "
            "FROM memory_candidates WHERE nick = ? "
            "ORDER BY mentions DESC, last_seen DESC",
            (nick.lower(),),
        ).fetchall()
        return [MemoryCandidate(*row) for row in rows]

    def reinforce_memory_candidate(
        self, candidate_id: int, nick: str, fact: str | None = None
    ) -> int:
        """Increment mentions and refresh last_seen for a candidate.

        The nick check prevents one user's extraction from mutating
        another's candidates.

        Args:
            candidate_id: Row ID of the candidate.
            nick: IRC nick (must match the candidate's owner).
            fact: Optional updated fact text (LLM may rephrase as it gathers
                evidence); ``None`` leaves the existing text untouched.

        Returns:
            The new mentions value, or 0 if the candidate was not found.
        """
        with self._write_txn() as conn:
            if fact is None:
                cursor = conn.execute(
                    "UPDATE memory_candidates "
                    "SET mentions = mentions + 1, last_seen = ? "
                    "WHERE id = ? AND nick = ? "
                    "RETURNING mentions",
                    (time.time(), candidate_id, nick.lower()),
                )
            else:
                cursor = conn.execute(
                    "UPDATE memory_candidates "
                    "SET mentions = mentions + 1, last_seen = ?, fact = ? "
                    "WHERE id = ? AND nick = ? "
                    "RETURNING mentions",
                    (time.time(), fact, candidate_id, nick.lower()),
                )
            row = cursor.fetchone()
            return row[0] if row else 0

    def delete_memory_candidate(self, candidate_id: int, nick: str) -> bool:
        """Delete a candidate by ID and nick.

        Args:
            candidate_id: Row ID of the candidate.
            nick: IRC nick (must match the candidate's owner).

        Returns:
            True if a candidate was deleted, False otherwise.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_candidates WHERE id = ? AND nick = ?",
                (candidate_id, nick.lower()),
            )
            return cursor.rowcount > 0

    def prune_memory_candidates(self, nick: str, older_than: float) -> int:
        """Delete a user's candidates whose last_seen is older than a cutoff.

        Args:
            nick: IRC nick (matched case-insensitively).
            older_than: Unix timestamp; candidates with last_seen strictly
                less than this are removed.

        Returns:
            Number of candidates deleted.
        """
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_candidates WHERE nick = ? AND last_seen < ?",
                (nick.lower(), older_than),
            )
            return cursor.rowcount

    def delete_all_memory_candidates(self, nick: str) -> int:
        """Delete every candidate for a user."""
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_candidates WHERE nick = ?",
                (nick.lower(),),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    # User instruction operations
    # ------------------------------------------------------------------

    def get_instruction(self, nick: str) -> str | None:
        """Get the user's persistent instruction, or None if not set."""
        conn = self._connect()
        row = conn.execute(
            "SELECT instruction FROM user_instructions WHERE nick = ?",
            (nick.lower(),),
        ).fetchone()
        return row[0] if row else None

    def save_instruction(self, nick: str, instruction: str) -> None:
        """Save or overwrite the user's persistent instruction."""
        with self._write_txn() as conn:
            conn.execute(
                "INSERT INTO user_instructions (nick, instruction, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(nick) DO UPDATE SET instruction = excluded.instruction, "
                "updated_at = excluded.updated_at",
                (nick.lower(), instruction, time.time()),
            )

    def delete_instruction(self, nick: str) -> bool:
        """Delete the user's instruction. Returns True if one was deleted."""
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM user_instructions WHERE nick = ?",
                (nick.lower(),),
            )
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Per-user time zone (IANA name, set with @tz)
    # ------------------------------------------------------------------

    def get_user_timezone(self, nick: str) -> str | None:
        """Get the user's IANA time zone name, or None if not set."""
        conn = self._connect()
        row = conn.execute(
            "SELECT tz FROM user_timezones WHERE nick = ?",
            (nick.lower(),),
        ).fetchone()
        return row[0] if row else None

    def save_user_timezone(self, nick: str, tz: str) -> None:
        """Save or overwrite the user's time zone."""
        with self._write_txn() as conn:
            conn.execute(
                "INSERT INTO user_timezones (nick, tz, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(nick) DO UPDATE SET tz = excluded.tz, "
                "updated_at = excluded.updated_at",
                (nick.lower(), tz, time.time()),
            )

    def delete_user_timezone(self, nick: str) -> bool:
        """Delete the user's time zone. Returns True if one was deleted."""
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM user_timezones WHERE nick = ?",
                (nick.lower(),),
            )
            return cursor.rowcount > 0

    def get_clock_offset(self, nick: str) -> tuple[int | None, float] | None:
        """``(offset_seconds, probed_at)`` from the last CTCP TIME probe.

        ``offset_seconds`` is None when the client never answered; the whole
        result is None when the user was never probed.
        """
        conn = self._connect()
        row = conn.execute(
            "SELECT offset_seconds, probed_at FROM user_clock_offsets WHERE nick = ?",
            (nick.lower(),),
        ).fetchone()
        return (row[0], row[1]) if row else None

    def save_clock_offset(self, nick: str, offset_seconds: int | None) -> None:
        """Record a CTCP TIME probe's answer (None for silence)."""
        with self._write_txn() as conn:
            conn.execute(
                "INSERT INTO user_clock_offsets (nick, offset_seconds, probed_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(nick) DO UPDATE SET offset_seconds = excluded.offset_seconds, "
                "probed_at = excluded.probed_at",
                (nick.lower(), offset_seconds, time.time()),
            )

    def save_last_msgids(self, rows: list[tuple[str, str, str, str, float]], *, keep: int) -> None:
        """Upsert ``(network, channel, nick, msgid, seen_at)`` rows, then
        trim the table to the ``keep`` most recently seen."""
        with self._write_txn() as conn:
            conn.executemany(
                "INSERT INTO last_msgids (network, channel, nick, msgid, seen_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(network, channel, nick) DO UPDATE SET "
                "msgid = excluded.msgid, seen_at = excluded.seen_at",
                rows,
            )
            conn.execute(
                "DELETE FROM last_msgids WHERE rowid NOT IN "
                "(SELECT rowid FROM last_msgids ORDER BY seen_at DESC LIMIT ?)",
                (keep,),
            )

    def load_last_msgids(self, limit: int) -> list[tuple[str, str, str, str]]:
        """``(network, channel, nick, msgid)`` rows, oldest first, newest ``limit``."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT network, channel, nick, msgid FROM ("
            "SELECT network, channel, nick, msgid, seen_at FROM last_msgids "
            "ORDER BY seen_at DESC LIMIT ?) ORDER BY seen_at ASC",
            (limit,),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def add_nick_watch(self, network: str, nick: str, requester: str, target: str) -> None:
        """Record (or refresh) ``requester``'s watch on ``nick``."""
        with self._write_txn() as conn:
            conn.execute(
                "INSERT INTO nick_watches (network, nick, requester, target, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(network, nick, requester) DO UPDATE SET "
                "target = excluded.target, created_at = excluded.created_at",
                (network, nick.lower(), requester.lower(), target, time.time()),
            )

    def pop_nick_watches(
        self, network: str, nick: str, *, max_age: float | None = None
    ) -> list[tuple[str, str]]:
        """Remove every watch on ``nick``; return the ``(requester, target)``
        pairs younger than ``max_age`` seconds (all of them when None)."""
        cutoff = time.time() - max_age if max_age is not None else float("-inf")
        with self._write_txn() as conn:
            rows = conn.execute(
                "SELECT requester, target FROM nick_watches "
                "WHERE network = ? AND nick = ? AND created_at >= ?",
                (network, nick.lower(), cutoff),
            ).fetchall()
            conn.execute(
                "DELETE FROM nick_watches WHERE network = ? AND nick = ?",
                (network, nick.lower()),
            )
        return [(r[0], r[1]) for r in rows]

    def delete_nick_watch(self, network: str, nick: str, requester: str) -> bool:
        with self._write_txn() as conn:
            cur = conn.execute(
                "DELETE FROM nick_watches WHERE network = ? AND nick = ? AND requester = ?",
                (network, nick.lower(), requester.lower()),
            )
        return cur.rowcount > 0

    def list_nick_watches(
        self, network: str, requester: str | None = None, *, max_age: float | None = None
    ) -> list[str]:
        """Watched nicks on ``network`` (one requester's, or everyone's),
        dropping any older than ``max_age`` seconds first."""
        if max_age is not None:
            with self._write_txn() as conn:
                conn.execute(
                    "DELETE FROM nick_watches WHERE created_at < ?", (time.time() - max_age,)
                )
        conn = self._connect()
        if requester is None:
            rows = conn.execute(
                "SELECT DISTINCT nick FROM nick_watches WHERE network = ? ORDER BY nick",
                (network,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT nick FROM nick_watches WHERE network = ? AND requester = ? ORDER BY nick",
                (network, requester.lower()),
            ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Avatar persona operations (verse-only, separate from user_instructions)
    # ------------------------------------------------------------------

    def get_avatar_persona(self, nick: str) -> str | None:
        """Get the user's avatar persona, or None if not set."""
        conn = self._connect()
        row = conn.execute(
            "SELECT persona FROM user_avatar_personas WHERE nick = ?",
            (nick.lower(),),
        ).fetchone()
        return row[0] if row else None

    def save_avatar_persona(self, nick: str, persona: str) -> None:
        """Save or overwrite the user's avatar persona."""
        with self._write_txn() as conn:
            conn.execute(
                "INSERT INTO user_avatar_personas (nick, persona, updated_at) "
                "VALUES (?, ?, ?) "
                "ON CONFLICT(nick) DO UPDATE SET persona = excluded.persona, "
                "updated_at = excluded.updated_at",
                (nick.lower(), persona, time.time()),
            )

    def delete_avatar_persona(self, nick: str) -> bool:
        """Delete the user's avatar persona. Returns True if one was deleted."""
        with self._write_txn() as conn:
            cursor = conn.execute(
                "DELETE FROM user_avatar_personas WHERE nick = ?",
                (nick.lower(),),
            )
            return cursor.rowcount > 0
