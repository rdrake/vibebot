"""Integration tests for LLM plugin.

These tests verify the integration between plugin components and
realistic Limnoria-like scenarios without requiring a full Limnoria runtime.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest
from llm.service import SubjectDossier

from .conftest import assistant_result

if TYPE_CHECKING:
    from pathlib import Path
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestDoPrivmsgIntegration:
    """Test doPrivmsg message tracking integration."""

    @pytest.fixture
    def plugin_for_tracking(self, mock_irc: MagicMock, mocker: MockerFixture) -> tuple:
        """Create plugin configured for message tracking."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry_side_effect = make_registry_side_effect({"privacyOptInOnly": False})

        mocker.patch.object(LLM, "registryValue", side_effect=registry_side_effect)
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry_side_effect)

        return plugin, mock_irc

    def test_doprivmsg_tracks_channel_messages(
        self, plugin_for_tracking: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN tracking enabled WHEN channel message received THEN tracked."""
        plugin, mock_irc = plugin_for_tracking

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "Hello world")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100  # Future time (not playback)
        mock_msg.server_tags = {}

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        # Should have tracked the message
        messages = plugin.context.get_messages("!user1", "#channel")
        assert len(messages) >= 1

    def test_doprivmsg_tracks_action_messages(
        self, plugin_for_tracking: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN ACTION message WHEN received THEN tracked normally."""
        plugin, mock_irc = plugin_for_tracking

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "user1!user@host"
        mock_msg.args = ("#channel", "\x01ACTION does something\x01")
        mock_msg.channel = "#channel"
        mock_msg.nick = "user1"
        mock_msg.time = time.time() + 100
        mock_msg.server_tags = {}

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=True)
        mocker.patch("supybot.ircmsgs.isAction", return_value=True)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        # Should have tracked the message
        messages = plugin.context.get_messages("!user1", "#channel")
        assert len(messages) >= 1


class TestHTTPCallbackIntegration:
    """Test HTTP callback integration with plugin."""

    @pytest.fixture
    def http_callback_with_plugin(self, mocker: MockerFixture) -> tuple:
        """Create HTTP callback with mock plugin."""
        from llm.plugin import LLMHTTPCallback

        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = ""
        callback = LLMHTTPCallback(mock_plugin)
        return callback, mock_plugin

    def test_http_callback_handles_concurrent_requests(
        self, http_callback_with_plugin: tuple, tmp_path, mocker: MockerFixture
    ) -> None:
        """GIVEN multiple concurrent requests WHEN served THEN no race conditions."""
        callback, _ = http_callback_with_plugin
        mocker.patch.object(callback, "_get_web_dir", return_value=str(tmp_path))

        # Create test file
        test_file = tmp_path / "concurrent.txt"
        test_file.write_bytes(b"test content")

        errors = []
        lock = threading.Lock()

        def make_request(request_id: int) -> None:
            try:
                mock_handler = mocker.MagicMock()
                mock_handler.wfile = mocker.MagicMock()

                callback.doGet(mock_handler, "concurrent.txt")

                mock_handler.send_response.assert_called_with(200)
            except Exception as e:
                with lock:
                    errors.append((request_id, e))

        # Launch concurrent requests
        threads = []
        for i in range(20):
            t = threading.Thread(target=make_request, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent requests: {errors}"


class TestRateLimitFullFlow:
    """Integration test for rate limiting on expensive commands.

    Uses a real SQLite database to verify that repeated draw requests
    trigger rate limiting, that unflagging still works independently,
    and that normal requests succeed after the window expires.
    """

    @pytest.fixture
    def plugin_with_real_db(
        self, mock_irc: MagicMock, mocker: MockerFixture, llm_data_dir
    ) -> tuple:
        """Create plugin with real database but mocked LLM service."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry = make_registry_side_effect(
            {
                "drawRateLimitCount": 2,
                "drawRateLimitWindow": 60,
            }
        )

        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin._MetaSynchronized_rlock = threading.RLock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.llm_service.draw_unavailable_reason.return_value = None
        # Subject research finds nobody real, so it books no usage row.
        plugin.llm_service.subject_dossier.return_value = SubjectDossier(text="")

        return plugin, mock_irc

    def test_rate_limit_full_flow(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN enforced rate limits WHEN user exceeds threshold THEN blocked then recovers.

        Full flow:
        1. User makes 2 successful draw requests (at limit)
        2. 3rd request is rate_limited
        3. After window expires, user can draw again
        4. Unflag flow works independently of rate limiting
        """
        from llm.service import AssistantResult, Spend

        plugin, mock_irc = plugin_with_real_db

        # Set up user identity
        mock_irc.state.nickToAccount.return_value = "testuser"

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "draw something")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"
        mock_msg.server_tags = {}

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            spend=Spend(prompt_tokens=5, completion_tokens=0, cost=0.02),
            model="dall-e-3",
        )

        # Step 1: Two successful draws fill the bucket
        for i in range(2):
            mock_irc.reset_mock()
            plugin.draw(mock_irc, mock_msg, [f"prompt {i}"])
            mock_irc.reply.assert_called_once()

        # Step 2: 3rd draw is rate limited
        mock_irc.reset_mock()
        plugin.llm_service.assistant_completion.reset_mock()
        plugin.draw(mock_irc, mock_msg, ["one too many"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]
        plugin.llm_service.assistant_completion.assert_not_called()

        # Verify rate_limited was logged in the real database
        conn = plugin.db._connect()
        usage_rows = conn.execute(
            "SELECT status FROM usage WHERE nick = 'testuser' AND status = 'rate_limited'"
        ).fetchall()
        assert len(usage_rows) >= 1

        # Step 3: Simulate window expiration by clearing buckets
        plugin._rate_buckets.clear()
        mock_irc.reset_mock()
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen2.png",
            grounding_used=False,
            spend=Spend(prompt_tokens=5, completion_tokens=0, cost=0.02),
            model="dall-e-3",
        )
        plugin.draw(mock_irc, mock_msg, ["fresh prompt"])
        mock_irc.reply.assert_called_once()
        assert "http://img.example/gen2.png" in mock_irc.reply.call_args[0][0]


class TestMemoryIntegration:
    """Test memory extraction and retrieval wiring."""

    @pytest.fixture
    def plugin_with_real_db(
        self, mock_irc: MagicMock, mocker: MockerFixture, llm_data_dir: Path
    ) -> tuple:
        """Create plugin with real database."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin._MetaSynchronized_rlock = threading.RLock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.llm_service.draw_unavailable_reason.return_value = None

        return plugin, mock_irc

    def test_ask_passes_memories_to_completion(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN saved memory WHEN ask called THEN completion receives memories kwarg."""
        plugin, mock_irc = plugin_with_real_db

        # Save a memory for the user
        plugin.db.save_memory("testuser", "Likes Python programming", "#test")

        mock_irc.state.nickToAccount.return_value = "testuser"

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "ask hello")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"
        mock_msg.server_tags = {}

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.llm_service.assistant_completion.return_value = assistant_result(
            content="Hello there!",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.detect_images.return_value = []

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # Verify completion was called with memories kwarg
        plugin.llm_service.assistant_completion.assert_called_once()
        call_kwargs = plugin.llm_service.assistant_completion.call_args
        assert call_kwargs.kwargs.get("memories") == ["Likes Python programming"]

    def test_ask_triggers_background_extraction(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN successful ask WHEN response received THEN background extraction scheduled."""
        from llm.plugin import schedule

        plugin, mock_irc = plugin_with_real_db

        mock_irc.state.nickToAccount.return_value = "testuser"

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "ask hello")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"
        mock_msg.server_tags = {}

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.llm_service.assistant_completion.return_value = assistant_result(
            content="Hello there!",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.llm_service.detect_images.return_value = []

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # Verify schedule.addEvent was called with a "llm_memory_" prefixed name
        add_event_calls = schedule.addEvent.call_args_list
        memory_calls = [c for c in add_event_calls if str(c).find("llm_memory_") != -1]
        assert len(memory_calls) == 1
        # Verify the name kwarg starts with llm_memory_
        name_arg = memory_calls[0].kwargs.get("name", memory_calls[0][1].get("name", ""))
        assert name_arg.startswith("llm_memory_")


class TestMemoriesCommand:
    """Test the %memories command for viewing and managing stored memories."""

    @pytest.fixture
    def plugin_with_real_db(
        self, mock_irc: MagicMock, mocker: MockerFixture, llm_data_dir: Path
    ) -> tuple:
        """Create plugin with real database for memories command testing."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin._MetaSynchronized_rlock = threading.RLock()

        # Set up user identity: nick resolves to account "testuser"
        mock_irc.state.nickToAccount.return_value = "testuser"

        return plugin, mock_irc

    @staticmethod
    def _make_msg(mocker: MockerFixture) -> MagicMock:
        """Create a mock IRC message from testuser."""
        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "testuser!user@host"
        mock_msg.args = ("#test", "memories")
        mock_msg.channel = "#test"
        mock_msg.nick = "testuser"
        mock_msg.time = time.time() + 100
        mock_msg.server_tags = {}
        return mock_msg

    def test_memories_list_shows_facts(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN saved memories WHEN memories called with no args THEN reply contains facts."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.db.save_memory("testuser", "Likes Python", "#test")
        plugin.db.save_memory("testuser", "Lives in Canada", "#test")

        plugin.memories(mock_irc, mock_msg, [])

        mock_irc.replies.assert_called_once()
        items = mock_irc.replies.call_args[0][0]
        all_text = " ".join(items)
        assert "Likes Python" in all_text
        assert "Lives in Canada" in all_text

    def test_memories_list_empty(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN no saved memories WHEN memories called THEN reply says no memories."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.memories(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "No memories stored" in reply_text

    def test_memories_delete_removes_fact(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN saved memory WHEN memories delete <id> THEN memory is removed."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        memory_id = plugin.db.save_memory("testuser", "Likes Python", "#test")

        plugin.memories(mock_irc, mock_msg, [f"delete {memory_id}"])

        mock_irc.reply.assert_called_once()
        assert "deleted" in mock_irc.reply.call_args[0][0].lower()

        # Verify it's actually gone
        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 0

    def test_memories_delete_invalid_id(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN invalid id WHEN memories delete abc THEN usage error shown."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.memories(mock_irc, mock_msg, ["delete abc"])

        mock_irc.error.assert_called_once()
        assert "Usage" in mock_irc.error.call_args[0][0]

    def test_memories_delete_nonexistent_id(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN nonexistent id WHEN memories delete 999 THEN error shown."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.memories(mock_irc, mock_msg, ["delete 999"])

        mock_irc.error.assert_called_once()
        assert "no matching" in mock_irc.error.call_args[0][0].lower()

    def test_memories_clear_deletes_all(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN saved memories WHEN memories clear THEN all deleted and count shown."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.db.save_memory("testuser", "Likes Python", "#test")
        plugin.db.save_memory("testuser", "Lives in Canada", "#test")
        plugin.db.save_memory("testuser", "Uses Vim", "#test")

        plugin.memories(mock_irc, mock_msg, ["clear"])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "3" in reply_text
        assert "Cleared" in reply_text

        # Verify all gone
        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 0

    def test_memories_edit_updates_fact(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN saved memory WHEN memories edit <id> <text> THEN fact is updated."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        memory_id = plugin.db.save_memory("testuser", "Likes Python", "#test")

        plugin.memories(mock_irc, mock_msg, [f"edit {memory_id} Loves Rust"])

        mock_irc.reply.assert_called_once()
        assert "updated" in mock_irc.reply.call_args[0][0].lower()

        # Verify the fact was actually changed
        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 1
        assert rows[0].fact == "Loves Rust"

    def test_memories_edit_nonexistent_id(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN nonexistent id WHEN memories edit 999 text THEN error shown."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.memories(mock_irc, mock_msg, ["edit 999 new text"])

        mock_irc.error.assert_called_once()
        assert "not found" in mock_irc.error.call_args[0][0].lower()

    def test_memories_del_shorthand(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN saved memory WHEN memories del <id> THEN memory is removed."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        memory_id = plugin.db.save_memory("testuser", "Likes Python", "#test")

        plugin.memories(mock_irc, mock_msg, [f"del {memory_id}"])

        mock_irc.reply.assert_called_once()
        assert "deleted" in mock_irc.reply.call_args[0][0].lower()
        assert len(plugin.db.get_memories("testuser")) == 0

    def test_memories_owner_views_other_user(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN owner WHEN memories <nick> THEN shows that user's memories."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.db.save_memory("otheruser", "Likes Rust", "#test")
        # Each nick is authenticated to its own account.
        mock_irc.state.nickToAccount.side_effect = lambda nick: nick

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        plugin.memories(mock_irc, mock_msg, ["otheruser"])

        mock_irc.replies.assert_called_once()
        items = mock_irc.replies.call_args[0][0]
        assert "Likes Rust" in " ".join(items)

    def test_memories_non_owner_cannot_view_other_user(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN non-owner WHEN memories <nick> THEN usage error."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        plugin.memories(mock_irc, mock_msg, ["otheruser"])

        mock_irc.error.assert_called_once()
        assert "Usage" in mock_irc.error.call_args[0][0]

    def test_memories_unrecognized_subcommand_uses_irc_error(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN garbage subcommand WHEN memories ... THEN failure goes through irc.error.

        Pins the failure-reply prefix: Limnoria's irc.error renders as "Error:"
        so users can distinguish failures from informational replies.
        """
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.memories(mock_irc, mock_msg, ["frobnicate this and that"])

        mock_irc.error.assert_called_once()
        assert "Usage" in mock_irc.error.call_args[0][0]
        mock_irc.reply.assert_not_called()

    def test_memories_cleanup_own(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN user with memories WHEN memories cleanup THEN summary shown."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")
        plugin.db.save_memory("testuser", "stale fact", "#test")

        plugin.llm_service.cleanup_memories = mocker.MagicMock(
            return_value=CleanupResult(drop=[0], merge=[])
        )

        plugin.memories(mock_irc, mock_msg, ["cleanup"])

        mock_irc.reply.assert_called_once()
        reply = mock_irc.reply.call_args[0][0]
        assert "Before: 3" in reply
        assert "after: 2" in reply

    def test_memories_cleanup_runs_under_executor_permit(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN memories WHEN memories cleanup THEN the LLM call holds a permit."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db
        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")
        running: list[int] = []

        def _cleanup(*_a, **_k):
            running.append(plugin._llm_executor.running())
            return CleanupResult(drop=[], merge=[])

        plugin.llm_service.cleanup_memories = mocker.MagicMock(side_effect=_cleanup)

        plugin.memories(mock_irc, self._make_msg(mocker), ["cleanup"])

        assert running == [1]

    def test_memories_cleanup_blocked_by_preflight(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN a rate-limited caller WHEN memories cleanup THEN no LLM call."""
        plugin, mock_irc = plugin_with_real_db
        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")
        preflight = mocker.patch.object(
            plugin, "_run_preflight", return_value=mocker.MagicMock(blocked=True)
        )
        plugin.llm_service.cleanup_memories = mocker.MagicMock()

        plugin.memories(mock_irc, self._make_msg(mocker), ["cleanup"])

        assert preflight.call_args.args[3] == "ask"
        plugin.llm_service.cleanup_memories.assert_not_called()

    def test_memories_cleanup_other_user_owner(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN owner WHEN memories cleanup <nick> THEN cleanup runs for that user."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.db.save_memory("otheruser", "fact a", "#test")
        plugin.db.save_memory("otheruser", "fact b", "#test")
        # Each nick is authenticated to its own account.
        mock_irc.state.nickToAccount.side_effect = lambda nick: nick

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.llm_service.cleanup_memories = mocker.MagicMock(
            return_value=CleanupResult(drop=[], merge=[])
        )

        plugin.memories(mock_irc, mock_msg, ["cleanup otheruser"])

        mock_irc.reply.assert_called_once()
        reply = mock_irc.reply.call_args[0][0]
        assert "Before: 2" in reply

    def test_account_on_someone_elses_nick_takes_none_of_their_data(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN account alice's personal data, conversation and usage
        WHEN account bob speaks from the nick "alice"
        THEN everything stays under alice and bob's @memories shows none of it.

        Regression: the nick→account migration used to move every row
        stored under the nick "alice" to whichever account appeared on it.
        """
        plugin, mock_irc = plugin_with_real_db
        db = plugin.db
        db.save_memory("alice", "Secret fact", "#test")
        db.save_avatar_persona("alice", "A wandering knight")
        db.save_instruction("alice", "Answer in French")
        db.save_user_timezone("alice", "Europe/Paris")
        db.save_clock_offset("alice", 3600)
        db.save_conversation("alice", "#test", [{"role": "user", "content": "hi"}], time.time())
        db.log_usage("alice", "#test", "ask", "m", 1, 1, 0.5)

        mock_irc.state.nickToAccount.return_value = "bob"
        msg = self._make_msg(mocker)
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        # Every path that used to trigger the migration.
        plugin._resolve_caller(mock_irc, msg)
        assert plugin._resolve_nick_to_identity(mock_irc, "alice") == "bob"
        plugin.memories(mock_irc, msg, [])

        assert "No memories stored" in mock_irc.reply.call_args[0][0]
        assert [m.fact for m in db.get_memories("alice")] == ["Secret fact"]
        assert db.get_memories("bob") == []
        assert db.get_avatar_persona("alice") == "A wandering knight"
        assert db.get_avatar_persona("bob") is None
        assert db.get_instruction("alice") == "Answer in French"
        assert db.get_instruction("bob") is None
        assert db.get_user_timezone("alice") == "Europe/Paris"
        assert db.get_user_timezone("bob") is None
        assert db.get_clock_offset("alice") is not None
        assert db.get_clock_offset("bob") is None
        assert [(n, c) for n, c, _m, _t in db.load_conversations()] == [("alice", "#test")]
        assert db.get_usage_summary_for_nick("alice").total_cost == 0.5
        assert db.get_usage_summary_for_nick("bob").total_requests == 0

    def test_memories_cleanup_other_user_non_owner(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN non-owner WHEN memories cleanup <nick> THEN error."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        plugin.memories(mock_irc, mock_msg, ["cleanup otheruser"])

        mock_irc.error.assert_called_once()

    def test_memories_cleanup_too_few(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN user with < 2 memories WHEN cleanup THEN not enough message."""
        plugin, mock_irc = plugin_with_real_db
        mock_msg = self._make_msg(mocker)

        plugin.db.save_memory("testuser", "only one fact", "#test")

        plugin.memories(mock_irc, mock_msg, ["cleanup"])

        mock_irc.reply.assert_called_once()
        assert "Not enough" in mock_irc.reply.call_args[0][0]


class TestMemoryCleanup:
    """Test background memory cleanup trigger and application."""

    @pytest.fixture
    def plugin_with_real_db(
        self, mock_irc: MagicMock, mocker: MockerFixture, llm_data_dir: Path
    ) -> tuple:
        """Create plugin with real database for cleanup testing."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker, mock_database=False)
        mocker.patch("llm.plugin.schedule.addEvent")

        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin._MetaSynchronized_rlock = threading.RLock()

        mock_irc.state.nickToAccount.return_value = "testuser"

        return plugin, mock_irc

    def test_cleanup_applies_drop(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN cleanup returns drop WHEN applied THEN memories are deleted."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "useful fact", "#test")
        plugin.db.save_memory("testuser", "stale fact", "#test")

        # Memories are newest-first: [0]="stale fact", [1]="useful fact"
        plugin.llm_service.cleanup_memories = mocker.MagicMock(
            return_value=CleanupResult(drop=[0], merge=[])
        )

        plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 1
        assert rows[0].fact == "useful fact"

    def test_cleanup_applies_merge(self, plugin_with_real_db: tuple, mocker: MockerFixture) -> None:
        """GIVEN cleanup returns merge WHEN applied THEN memories are merged."""
        from llm.service import CleanupResult, MergeOp

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "likes Python programming", "#test")
        plugin.db.save_memory("testuser", "enjoys writing Python", "#test")

        plugin.llm_service.cleanup_memories = mocker.MagicMock(
            return_value=CleanupResult(drop=[], merge=[MergeOp([0, 1], "likes Python programming")])
        )

        plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 1
        assert rows[0].fact == "likes Python programming"

    def test_cleanup_applies_multiway_merge(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN cleanup returns 3-way merge WHEN applied THEN memories consolidated."""
        from llm.service import CleanupResult, MergeOp

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "uses Arch Linux", "#test")
        plugin.db.save_memory("testuser", "uses Debian", "#test")
        plugin.db.save_memory("testuser", "uses Fedora", "#test")

        # Memories newest-first: [0]=Fedora, [1]=Debian, [2]=Arch
        plugin.llm_service.cleanup_memories = mocker.MagicMock(
            return_value=CleanupResult(
                drop=[], merge=[MergeOp([0, 1, 2], "uses Linux (Arch, Debian, Fedora)")]
            )
        )

        plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 1
        assert rows[0].fact == "uses Linux (Arch, Debian, Fedora)"

    def test_cleanup_aborts_on_error(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN cleanup returns error WHEN applied THEN no DB changes."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")

        plugin.llm_service.cleanup_memories = mocker.MagicMock(
            return_value=CleanupResult(error="LLM returned garbage")
        )

        summary = plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 2
        assert summary.ok is False
        assert "Cleanup failed" in summary.message

    def test_cleanup_aborts_on_snapshot_mismatch(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN memory count changes during cleanup WHEN applying THEN abort."""
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")

        def cleanup_with_side_effect(*args, **kwargs):
            plugin.db.save_memory("testuser", "new fact during cleanup", "#test")
            return CleanupResult(drop=[1], merge=[])

        plugin.llm_service.cleanup_memories = mocker.MagicMock(side_effect=cleanup_with_side_effect)

        summary = plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        assert len(rows) == 3
        assert "skipped" in summary.message.lower()

    def test_cleanup_aborts_on_count_preserving_mutation(
        self, plugin_with_real_db: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN row IDs change but count is preserved WHEN cleanup applies THEN abort.

        Regression for the count-only race guard: a delete+insert during the
        LLM call leaves len() unchanged but shifts indices, which would
        otherwise let cleanup mis-target rows.
        """
        from llm.service import CleanupResult

        plugin, mock_irc = plugin_with_real_db

        plugin.db.save_memory("testuser", "fact a", "#test")
        plugin.db.save_memory("testuser", "fact b", "#test")
        snapshot_rows = plugin.db.get_memories("testuser")

        # CleanupResult will tell the function to drop snapshot[1] = "fact a".
        # If the abort guard fires correctly, this drop must NOT be applied.
        target_index = 1
        target_fact = snapshot_rows[target_index].fact

        def cleanup_with_swap(*args, **kwargs):
            # Delete the OTHER snapshot row and insert a new one — count
            # preserved, IDs differ.
            other_index = 0 if target_index == 1 else 1
            plugin.db.delete_memory("testuser", snapshot_rows[other_index].id)
            plugin.db.save_memory("testuser", "fact c", "#test")
            return CleanupResult(drop=[target_index], merge=[])

        plugin.llm_service.cleanup_memories = mocker.MagicMock(side_effect=cleanup_with_swap)

        summary = plugin._run_memory_cleanup("testuser", "#test")

        rows = plugin.db.get_memories("testuser")
        facts = {r.fact for r in rows}
        # The abort must prevent the stale drop from running. Target fact
        # should still be present; count stays at 2 (one mutation + abort).
        assert target_fact in facts
        assert len(rows) == 2
        assert "skipped" in summary.message.lower()
