"""Tests for ConversationContext."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from llm.context import ContextConfig, Conversation, ConversationContext
from llm.persistence import LLMDatabase

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestConversationContext:
    """Test conversation context functionality."""

    # NOTE: per-user isolation, per-channel isolation, case-insensitive
    # lookup, and max_messages trim are now covered by
    # test_context_properties.py::TestConversationContextStateMachine.

    def test_context_time_expiry(self) -> None:
        """GIVEN context WHEN timeout expires THEN context cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=1, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")

        # Backdate last_activity so it appears expired
        key = ("user1", "#channel")
        ctx._conversations[key].last_activity -= 120

        messages = ctx.get_messages("user1", "#channel")
        assert len(messages) == 0

    def test_get_messages_max_age_returns_fresh(self) -> None:
        """GIVEN recent activity WHEN max_age_seconds set THEN messages returned."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")

        messages = ctx.get_messages("user1", "#channel", max_age_seconds=60)
        assert len(messages) == 1

    def test_get_messages_max_age_drops_stale(self) -> None:
        """GIVEN stale activity WHEN max_age_seconds set THEN empty returned."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        # Backdate last_activity beyond the max age window
        ctx._conversations[("user1", "#channel")].last_activity -= 120

        messages = ctx.get_messages("user1", "#channel", max_age_seconds=60)
        assert messages == []

        # Without the window, same conversation is still returned (not expired)
        messages_unfiltered = ctx.get_messages("user1", "#channel")
        assert len(messages_unfiltered) == 1

    def test_get_channel_messages_max_age(self) -> None:
        """GIVEN stale channel activity WHEN max_age_seconds set THEN empty."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "hello")
        ctx._channel_contexts["#channel"].last_activity -= 120

        stale = ctx.get_channel_messages("#channel", max_age_seconds=60)
        assert stale == []

        fresh = ctx.get_channel_messages("#channel")
        assert len(fresh) == 1

    def test_context_clear_specific_user(self) -> None:
        """GIVEN context WHEN clear specific user THEN only that context cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello from user1")
        ctx.add_message("user2", "#channel", "user", "Hello from user2")

        cleared = ctx.clear("user1", "#channel")
        assert cleared is True

        # User1 context should be cleared
        assert len(ctx.get_messages("user1", "#channel")) == 0
        # User2 context should remain
        assert len(ctx.get_messages("user2", "#channel")) == 1

    def test_context_clear_nonexistent(self) -> None:
        """GIVEN context WHEN clear nonexistent user THEN returns False."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        cleared = ctx.clear("nonexistent", "#channel")
        assert cleared is False

    def test_context_clear_all(self) -> None:
        """GIVEN context WHEN clear all THEN all contexts cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel1", "user", "Hello")
        ctx.add_message("user2", "#channel2", "user", "World")

        count = ctx.clear_all()
        assert count == 2

        assert len(ctx.get_messages("user1", "#channel1")) == 0
        assert len(ctx.get_messages("user2", "#channel2")) == 0

    def test_context_disabled(self) -> None:
        """GIVEN context disabled WHEN operations THEN no storage."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=False)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        messages = ctx.get_messages("user1", "#channel")

        assert len(messages) == 0

    def test_context_stats(self) -> None:
        """GIVEN context WHEN get stats THEN accurate counts returned."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        ctx.add_message("user1", "#channel", "assistant", "Hi")
        ctx.add_message("user2", "#channel", "user", "Hey")

        stats = ctx.get_stats()
        assert stats["active_conversations"] == 2
        assert stats["total_messages"] == 3
        assert stats["max_messages_per_conv"] == 20
        assert stats["timeout_minutes"] == 30
        assert stats["enabled"] is True

    def test_user_stats_with_messages(self) -> None:
        """GIVEN user with messages WHEN get_user_stats THEN returns count and expiry."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Hello")
        ctx.add_message("user1", "#channel", "assistant", "Hi")

        stats = ctx.get_user_stats("user1", "#channel")
        assert stats["message_count"] == 2
        assert stats["max_messages"] == 20
        assert stats["seconds_until_expiry"] > 0
        assert stats["enabled"] is True

    def test_context_thread_safe(self) -> None:
        """GIVEN context WHEN concurrent operations THEN thread-safe."""
        config = ContextConfig(max_messages=1000, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        errors: list[Exception] = []
        lock = threading.Lock()

        def add_messages(user_id: str) -> None:
            try:
                for i in range(100):
                    ctx.add_message(user_id, "#channel", "user", f"Message {i}")
                    ctx.get_messages(user_id, "#channel")
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run 10 threads, each adding 100 messages
        threads = []
        for i in range(10):
            t = threading.Thread(target=add_messages, args=(f"user{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Should complete without errors
        assert len(errors) == 0

        # Each user should have their messages
        stats = ctx.get_stats()
        assert stats["active_conversations"] == 10


class TestChannelContext:
    """Test shared channel context functionality."""

    def test_channel_context_add_and_get(self) -> None:
        """GIVEN channel context WHEN add messages THEN get returns them."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello everyone")
        ctx.add_channel_message("#channel", "bot", "assistant", "Hi Alice!")

        messages = ctx.get_channel_messages("#channel")
        assert len(messages) == 2
        assert messages[0]["nick"] == "alice"
        assert messages[0]["content"] == "Hello everyone"
        assert messages[1]["nick"] == "bot"
        assert messages[1]["content"] == "Hi Alice!"

    def test_channel_context_exclude_nick(self) -> None:
        """GIVEN channel context WHEN exclude nick THEN filters correctly."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Question from Alice")
        ctx.add_channel_message("#channel", "bot", "assistant", "Answer to Alice")
        ctx.add_channel_message("#channel", "bob", "user", "Question from Bob")
        ctx.add_channel_message("#channel", "bot", "assistant", "Answer to Bob")

        # Bob's view should exclude his own messages
        bob_view = ctx.get_channel_messages("#channel", exclude_nick="bob")

        assert len(bob_view) == 3
        nicks = [m["nick"] for m in bob_view]
        assert "bob" not in nicks
        assert "alice" in nicks
        assert "bot" in nicks

    def test_channel_context_exclude_nick_case_insensitive(self) -> None:
        """GIVEN channel context WHEN exclude nick with different case THEN filters."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "Alice", "user", "From Alice")
        ctx.add_channel_message("#channel", "bob", "user", "From Bob")

        # Exclude with different case
        view = ctx.get_channel_messages("#channel", exclude_nick="ALICE")

        assert len(view) == 1
        assert view[0]["nick"] == "bob"

    def test_channel_context_max_messages_limit(self) -> None:
        """GIVEN channel context with limit WHEN exceed THEN oldest removed."""
        config = ContextConfig(
            max_messages=20, timeout_minutes=30, enabled=True, channel_max_messages=3
        )
        ctx = ConversationContext(config)

        # Add 5 messages (exceeds limit of 3)
        for i in range(5):
            ctx.add_channel_message("#channel", f"user{i}", "user", f"Message {i}")

        messages = ctx.get_channel_messages("#channel")
        assert len(messages) == 3
        # Should have messages 2-4, not 0-1
        assert messages[0]["content"] == "Message 2"
        assert messages[2]["content"] == "Message 4"

    def test_channel_context_isolated_by_channel(self) -> None:
        """GIVEN channel context WHEN different channels THEN isolated."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel1", "alice", "user", "In channel 1")
        ctx.add_channel_message("#channel2", "alice", "user", "In channel 2")

        ch1_messages = ctx.get_channel_messages("#channel1")
        ch2_messages = ctx.get_channel_messages("#channel2")

        assert len(ch1_messages) == 1
        assert ch1_messages[0]["content"] == "In channel 1"
        assert len(ch2_messages) == 1
        assert ch2_messages[0]["content"] == "In channel 2"

    def test_channel_context_expiry(self) -> None:
        """GIVEN channel context WHEN timeout expires THEN cleared."""
        config = ContextConfig(
            max_messages=20,
            timeout_minutes=1,
            enabled=True,
        )
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello")

        # Backdate last_activity so it appears expired
        ctx._channel_contexts["#channel"].last_activity -= 120

        messages = ctx.get_channel_messages("#channel")
        assert len(messages) == 0

    def test_channel_context_disabled(self) -> None:
        """GIVEN context disabled WHEN channel operations THEN no storage."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=False)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello")
        messages = ctx.get_channel_messages("#channel")

        assert len(messages) == 0

    def test_channel_context_in_stats(self) -> None:
        """GIVEN channel context WHEN get stats THEN includes channel stats."""
        config = ContextConfig(
            max_messages=20, timeout_minutes=30, enabled=True, channel_max_messages=10
        )
        ctx = ConversationContext(config)

        # Add personal context
        ctx.add_message("user1", "#channel", "user", "Personal message")

        # Add channel context
        ctx.add_channel_message("#channel", "alice", "user", "Channel msg 1")
        ctx.add_channel_message("#channel", "bob", "user", "Channel msg 2")

        stats = ctx.get_stats()
        assert stats["active_channels"] == 1
        assert stats["channel_messages"] == 2
        assert stats["channel_max_messages"] == 10

    def test_channel_context_clear_all(self) -> None:
        """GIVEN channel context WHEN clear all THEN channel context also cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_message("user1", "#channel", "user", "Personal")
        ctx.add_channel_message("#channel", "alice", "user", "Channel")

        ctx.clear_all()

        assert len(ctx.get_messages("user1", "#channel")) == 0
        assert len(ctx.get_channel_messages("#channel")) == 0

    def test_channel_context_returns_copies(self) -> None:
        """GIVEN channel context WHEN get messages THEN returns copies."""
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello")

        messages1 = ctx.get_channel_messages("#channel")
        messages2 = ctx.get_channel_messages("#channel")

        # Should be different list objects
        assert messages1 is not messages2
        # But with same content
        assert messages1 == messages2

        # Modifying one should not affect the other or the stored data
        messages1[0]["content"] = "Modified"
        messages3 = ctx.get_channel_messages("#channel")
        assert messages3[0]["content"] == "Hello"


class TestPersistentContext:
    """Test conversation context with SQLite persistence."""

    def _make_ctx(self, db: LLMDatabase) -> tuple[ConversationContext, LLMDatabase]:
        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=db)
        return ctx, db

    def test_add_message_persists_to_db(self, test_db: LLMDatabase) -> None:
        """GIVEN context with db WHEN add_message THEN conversation is in SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello")

        loaded = db.load_conversations()
        assert len(loaded) == 1
        assert loaded[0][2] == [{"role": "user", "content": "Hello"}]

    def test_add_message_persist_false_skips_db(self, test_db: LLMDatabase) -> None:
        """GIVEN context with db WHEN add_message(persist=False) THEN not in SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello", persist=False)

        loaded = db.load_conversations()
        assert len(loaded) == 0

        # But still in memory
        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1

    def test_clear_deletes_from_db(self, test_db: LLMDatabase) -> None:
        """GIVEN persisted conversation WHEN clear THEN removed from SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello")
        ctx.clear("user1", "#chan")

        assert len(db.load_conversations()) == 0

    def test_clear_all_deletes_from_db(self, test_db: LLMDatabase) -> None:
        """GIVEN persisted conversations WHEN clear_all THEN all removed from SQLite."""
        ctx, db = self._make_ctx(test_db)
        ctx.add_message("user1", "#chan", "user", "Hello")
        ctx.add_message("user2", "#chan", "user", "Hi")
        ctx.clear_all()

        assert len(db.load_conversations()) == 0

    def test_startup_loads_from_db(self, test_db: LLMDatabase) -> None:
        """GIVEN conversations in db WHEN new ConversationContext THEN loaded into memory."""
        test_db.save_conversation(
            "user1",
            "#chan",
            [{"role": "user", "content": "Hello"}],
            time.time(),
        )

        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=test_db)

        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 1
        assert msgs[0]["content"] == "Hello"

    def test_startup_skips_expired(self, test_db: LLMDatabase) -> None:
        """GIVEN expired conversation in db WHEN new ConversationContext THEN not loaded."""
        old_time = time.time() - 7200  # 2 hours ago
        test_db.save_conversation(
            "user1",
            "#chan",
            [{"role": "user", "content": "Hello"}],
            old_time,
        )

        config = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)
        ctx = ConversationContext(config, db=test_db)

        msgs = ctx.get_messages("user1", "#chan")
        assert len(msgs) == 0


class TestPruneDeletesFromDb:
    """Test that pruning expired conversations deletes them from the database."""

    def test_prune_deletes_expired_from_db(
        self, test_db: LLMDatabase, mocker: MockerFixture
    ) -> None:
        """GIVEN persisted conversation WHEN expired and prune runs THEN db has 0 rows."""
        config = ContextConfig(max_messages=20, timeout_minutes=1, enabled=True)
        ctx = ConversationContext(config, db=test_db)

        # Add a message (persisted to db)
        ctx.add_message("user1", "#chan", "user", "Hello")
        assert len(test_db.load_conversations()) == 1

        # Mock time.time to return a value well past the 1-minute timeout
        future = time.time() + 3600  # 1 hour in the future
        mocker.patch("llm.context.time.time", return_value=future)

        # get_stats() calls _prune_expired(force=True), bypassing throttle
        stats = ctx.get_stats()

        assert stats["active_conversations"] == 0
        assert len(test_db.load_conversations()) == 0


class TestChannelContextPrune:
    """Test that pruning clears expired channel contexts."""

    def test_channel_context_pruned_on_expiry(self, mocker: MockerFixture) -> None:
        """GIVEN channel context WHEN expired and prune runs THEN channel contexts cleared."""
        config = ContextConfig(max_messages=20, timeout_minutes=1, enabled=True)
        ctx = ConversationContext(config)

        ctx.add_channel_message("#channel", "alice", "user", "Hello everyone")
        assert len(ctx.get_channel_messages("#channel")) == 1

        # Mock time.time to return a value well past the 1-minute timeout
        future = time.time() + 3600  # 1 hour in the future
        mocker.patch("llm.context.time.time", return_value=future)

        # get_stats() calls _prune_expired(force=True), bypassing throttle
        stats = ctx.get_stats()

        assert stats["active_channels"] == 0
        assert stats["channel_messages"] == 0


class TestContextDbResilience:
    """DB errors are best-effort: persistence/cleanup failures must never crash
    the caller or plugin startup. In-memory state remains the live source of
    truth for the session."""

    _CFG = ContextConfig(max_messages=20, timeout_minutes=30, enabled=True)

    def _mock_db(self, mocker: MockerFixture):
        db = mocker.MagicMock()
        db.load_conversations.return_value = []
        return db

    def test_add_message_survives_db_save_failure(self, mocker: MockerFixture) -> None:
        """A failed save_conversation must not propagate; the message stays in RAM."""
        import sqlite3

        db = self._mock_db(mocker)
        db.save_conversation.side_effect = sqlite3.OperationalError("disk I/O error")
        ctx = ConversationContext(self._CFG, db=db)

        ctx.add_message("user1", "#chan", "user", "Hello")  # must not raise

        assert ctx.get_messages("user1", "#chan") == [{"role": "user", "content": "Hello"}]

    def test_startup_survives_delete_failure_on_expired(self, mocker: MockerFixture) -> None:
        """A delete failure while pruning an expired row at startup must not
        crash plugin initialization."""
        import sqlite3

        db = self._mock_db(mocker)
        db.load_conversations.return_value = [
            ("old", "#chan", [{"role": "user", "content": "hi"}], 0.0)  # last_activity=0 → expired
        ]
        db.delete_conversation.side_effect = sqlite3.OperationalError("database is locked")

        ConversationContext(self._CFG, db=db)  # must not raise

    def test_clear_survives_db_delete_failure(self, mocker: MockerFixture) -> None:
        """A failed delete_conversation in clear() must not propagate; RAM is
        still cleared so the user's forget request is honored for the session."""
        import sqlite3

        db = self._mock_db(mocker)
        ctx = ConversationContext(self._CFG, db=db)
        ctx.add_message("user1", "#chan", "user", "Hello", persist=False)
        db.delete_conversation.side_effect = sqlite3.OperationalError("locked")

        assert ctx.clear("user1", "#chan") is True
        assert ctx.get_messages("user1", "#chan") == []

    def test_prune_survives_db_delete_failure(self, mocker: MockerFixture) -> None:
        """A delete failure while pruning one expired row must not propagate and
        must not block removing it from RAM."""
        import sqlite3

        db = self._mock_db(mocker)
        ctx = ConversationContext(self._CFG, db=db)
        ctx._conversations[("old", "#chan")] = Conversation(
            messages=[{"role": "user", "content": "x"}], last_activity=0.0
        )
        db.delete_conversation.side_effect = sqlite3.OperationalError("locked")

        ctx.add_message("new", "#chan", "user", "hi")  # triggers prune; must not raise

        assert ("old", "#chan") not in ctx._conversations
