"""Plugin delivery: pending-task scheduling, result delivery/retry, wakeups, safety-poll."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest
from llm.assistant import ToolCallbackResult

from .conftest import make_registry_side_effect, make_reminder_row

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestSafePrivmsg:
    """``_safe_privmsg`` neutralizes IRC command injection on the raw-queue path.

    Worker-thread sends (``_safe_queue`` + raw ``ircmsgs.privmsg``) bypass the
    ``ircutils.safeArgument`` that ``irc.reply`` applies on the chat-loop path.
    Model- or user-derived bodies routed through ``_safe_privmsg`` must have
    CR/LF/NUL neutralized so they cannot smuggle a second IRC command onto the
    wire (the underlying ``IrcMsg`` ``assert`` vanishes under ``python -O``).
    """

    def test_strips_crlf_and_nul_from_body(self, plugin_env) -> None:
        """A body carrying CR/LF/NUL cannot smuggle a second IRC command."""
        plugin, _irc, _msg = plugin_env
        msg = plugin._safe_privmsg("#chan", "ok\r\nQUIT :pwned\x00")
        body = msg.args[1]
        assert "\r" not in body
        assert "\n" not in body
        assert "\x00" not in body
        # Only the trailing CRLF terminator is a real line break on the wire.
        assert "\r\n" not in str(msg)[:-2]

    def test_clean_body_passes_through_unchanged(self, plugin_env) -> None:
        """Clean single-line text is sent verbatim (safeArgument is a no-op)."""
        plugin, _irc, _msg = plugin_env
        msg = plugin._safe_privmsg("#chan", "hello world")
        assert msg.args[1] == "hello world"


class TestSafetyPollGuard:
    def test_overlapping_poll_is_skipped(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._safety_poll_inflight.set()

        plugin._enqueue_safety_poll()
        plugin._llm_executor.submit.assert_not_called()

    def test_flag_clears_after_worker_completes(self, plugin_env, mocker) -> None:
        """Use a real LLMExecutor so add_done_callback fires."""
        plugin, _irc, _msg = plugin_env
        # Stub the worker body so the future completes promptly with a
        # known result. Without this stub, the worker enters the real
        # `_check_pending_tasks` which iterates a MagicMock service
        # return value (TypeError) — the test would still "pass" but
        # only via the exception path, not the success path.
        plugin.llm_service.check_pending_tasks = mocker.MagicMock(return_value=[])

        plugin._enqueue_safety_poll()
        # The future's add_done_callback clears the inflight flag.
        # _enqueue_safety_poll returns None (no future handle), and an
        # Event cannot wait() for a *clear*, so deadline-poll until the
        # callback has cleared it. Exits in ~ms once done; never the
        # full 5s unless the callback is broken.
        deadline = time.monotonic() + 5.0
        while plugin._safety_poll_inflight.is_set() and time.monotonic() < deadline:
            time.sleep(0.005)
        assert not plugin._safety_poll_inflight.is_set()

    def test_flag_clears_on_synchronous_submit_failure(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._llm_executor.submit.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError):
            plugin._enqueue_safety_poll()
        assert not plugin._safety_poll_inflight.is_set()

    def test_closing_short_circuits(self, plugin_env, mocker) -> None:
        plugin, _irc, _msg = plugin_env
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = True

        plugin._enqueue_safety_poll()
        plugin._llm_executor.submit.assert_not_called()


class TestCompletionResultUsageData:
    """Test that result NamedTuples carry usage data for logging."""

    def test_completion_result_carries_usage_data(self) -> None:
        """GIVEN CompletionResult with usage WHEN accessed THEN data available for logging."""
        from llm.service import CompletionResult

        result = CompletionResult(
            content="response",
            grounding_used=False,
            prompt_tokens=150,
            completion_tokens=75,
            cost=0.002,
            model="gemini/flash",
        )
        assert result.prompt_tokens == 150
        assert result.completion_tokens == 75
        assert result.cost == 0.002
        assert result.model == "gemini/flash"

    def test_image_result_carries_usage_data(self) -> None:
        """GIVEN ImageResult with usage WHEN accessed THEN data available for logging."""
        from llm.service import ImageResult

        result = ImageResult(
            content="http://example.com/image.png",
            prompt_tokens=50,
            completion_tokens=0,
            cost=0.04,
            model="vertex/imagen-3",
        )
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 0
        assert result.cost == 0.04
        assert result.model == "vertex/imagen-3"


class TestPendingTaskScheduler:
    """Test pending task scheduler event naming and lifecycle."""

    def test_init_schedules_pending_tasks_event(self, mocker: MockerFixture) -> None:
        """GIVEN plugin init WHEN started THEN schedules llm_pending_tasks event."""
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        mocker.patch("llm.plugin.LLMService")
        mocker.patch("llm.plugin.LLMDatabase")
        mocker.patch("llm.plugin.log")
        mocker.patch("llm.plugin.httpserver.hook")
        mock_add = mocker.patch("llm.plugin.schedule.addPeriodicEvent")
        mocker.patch("llm.plugin.schedule.removeEvent")
        # Also patch addEvent: the status poller (and the daily compaction
        # timer) call it unconditionally from __init__, and a real call
        # would collide with the real supybot schedule between tests.
        mocker.patch("llm.plugin.schedule.addEvent")

        LLM(mock_irc)

        # Check that llm_pending_tasks was scheduled
        event_names = [call[1].get("name", "") for call in mock_add.call_args_list]
        assert "llm_pending_tasks" in event_names

    def test_die_removes_pending_tasks_event(self, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN die called THEN removes llm_pending_tasks event."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._http_callback = None

        mock_remove = mocker.patch("supybot.schedule.removeEvent")
        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)
        plugin.die()

        mock_remove.assert_any_call("llm_pending_tasks")


class TestDeliverPendingResult:
    """Test _deliver_pending_result sends messages to correct targets."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for delivery testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.llm_service.save_code_to_http.return_value = None
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
        return plugin

    def _make_result(self, **overrides):
        """Create a PendingTaskResult with defaults."""
        from llm.service import PendingTaskResult

        defaults = {
            "status": "completed",
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello world",
            "model": "gpt-4",
            "content": "The answer is 42",
            "reason": "",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.01,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_delivers_completed_ask_to_channel(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN completed ask result WHEN delivered THEN sends to channel."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result()
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg = mock_irc.queueMsg.call_args[0][0]
        assert "alice" in str(msg)

    def test_collapses_multiline_completed_content(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN a multi-line completed result WHEN delivered THEN content is
        collapsed to one IRC-safe line (flood guard).

        Deferred delivery bypasses ``_send_long_reply``, so ``_deliver_pending_result``
        must collapse embedded newlines before handing the text to
        ``ircmsgs.privmsg``. A raw ``\n`` in the PRIVMSG body is rejected by
        Limnoria's ``isValidArgument`` (it would put literal newlines on the
        wire and trigger an Excess Flood disconnect), so without the collapse
        ``ircmsgs.privmsg`` raises and nothing is ever queued.
        """
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(content="line one\nline two\nline three")
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        # All three lines survive, joined by the IRC-safe separator...
        assert "alice: line one | line two | line three" in msg_text
        # ...and no raw newline reaches the wire payload.
        assert "\n" not in msg_text.removesuffix("\r\n")

    def test_nul_in_completed_content_still_delivers(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN recovered content carrying a NUL byte WHEN delivered THEN the
        body is neutralized and the message still goes out.

        Regression: this path built ``ircmsgs.privmsg`` raw. Limnoria asserts
        the argument is valid, so a NUL raised, the send was recorded as failed,
        and the task burned all ten delivery retries before landing in
        delivery_failed — the user simply never got their answer. (Under
        ``python -O`` the assertion vanishes instead and the NUL reaches the
        wire.) Routing through ``_safe_privmsg`` neutralizes it either way.

        The fixture stubs ``sanitize_output`` to identity, so this exercises the
        raw-queue guard specifically, not the sanitize-time strip.
        """
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(content="the answer is\x0042")
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        assert "\x00" not in str(mock_irc.queueMsg.call_args[0][0])

    def test_long_completed_content_is_pastebinned_not_truncated(
        self, plugin, mocker: MockerFixture
    ) -> None:
        """A long recovered ask/verse result must be saved to the HTTP server
        and delivered as a teaser + URL, not collapsed into one oversized
        PRIVMSG the server silently truncates.

        Verse timeouts recover under the 'ask' task_type and verse is unbounded
        (PROFILE_VERSE.max_output_tokens is None), so a recovered multi-paragraph
        scene would otherwise overflow the 512-byte IRC line limit and lose most
        of the scene (and the would-be pastebin URL). Mirrors the live
        _send_long_reply teaser+URL behaviour.
        """
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        plugin.llm_service.save_markdown_to_http.return_value = "http://h/scene.html"
        long_scene = "\n\n".join(f"Paragraph {i}: " + "word " * 60 for i in range(6))

        r = self._make_result(content=long_scene)
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        # Delivered as a teaser + pastebin URL...
        assert "http://h/scene.html" in msg_text
        # ...not the full inlined body.
        assert "Paragraph 5" not in msg_text
        plugin.llm_service.save_markdown_to_http.assert_called_once()

    def test_delivers_expired_notification(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN expired result WHEN delivered THEN sends apology."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(status="expired", content="", reason="expired")
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        assert "expired" in msg_text.lower()

    def test_delivers_terminal_failure_notification(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN terminal failure WHEN delivered THEN sends failure message."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        # A key failure never reached the provider, so nothing billed.
        r = self._make_result(
            status="failed_terminal",
            content="",
            reason="API key not configured",
            cost=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        assert "failed" in msg_text.lower()

    def test_delivers_to_pm_target(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN PM result WHEN delivered THEN sends to nick."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(reply_target="alice", is_channel=False)
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()

    def test_books_usage_for_completed_task(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN a completed durable result WHEN delivered THEN its usage is
        booked as success, before the ack deletes the row."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        calls = mocker.MagicMock()
        calls.attach_mock(plugin.db.book_pending_task_usage, "book")
        calls.attach_mock(plugin.db.delete_pending_task, "delete")

        plugin._deliver_pending_result(self._make_result(task_id=7))

        assert [c[0] for c in calls.mock_calls] == ["book", "delete"]
        plugin.db.book_pending_task_usage.assert_called_once_with(
            7, status="success", error_detail=""
        )

    def test_logs_usage_for_billed_terminal_failure(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN a retry that billed tokens but answered nothing WHEN its failure
        is delivered THEN the spend is still booked."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice_account"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(
            status="failed_terminal",
            content="",
            reason="the model returned an empty answer",
            cost=0.02,
            prompt_tokens=100,
            completion_tokens=4096,
            task_id=7,
        )
        plugin._deliver_pending_result(r)

        plugin.db.book_pending_task_usage.assert_called_once_with(
            7, status="error", error_detail="the model returned an empty answer"
        )

    def test_logs_structured_expired_outcome(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN expired deferred result WHEN delivered THEN logs structured operator entry."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(
            status="expired",
            task_type="draw",
            content="",
            reason="Request expired after retry timeout",
        )
        plugin._deliver_pending_result(r)

        # Should log a structured warning for operator visibility
        plugin.log.warning.assert_called_once()
        log_msg = plugin.log.warning.call_args[0][0]
        assert "expired" in log_msg.lower()
        # Should include key fields for grep/monitoring
        assert "draw" in plugin.log.warning.call_args[0][1]
        assert "alice" in plugin.log.warning.call_args[0][2]

    def test_logs_structured_failed_terminal_outcome(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN terminal-failure deferred result WHEN delivered THEN logs structured operator entry."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(
            status="failed_terminal",
            task_type="draw",
            content="",
            reason="API key not configured",
            cost=0.0,
            prompt_tokens=0,
            completion_tokens=0,
        )
        plugin._deliver_pending_result(r)

        # Should log a structured warning for operator visibility
        plugin.log.warning.assert_called_once()
        log_msg = plugin.log.warning.call_args[0][0]
        assert "failed_terminal" in log_msg.lower()
        assert "draw" in plugin.log.warning.call_args[0][1]
        assert "alice" in plugin.log.warning.call_args[0][2]


class TestDeliveryRetry:
    """Test delivery retry with bounded backoff and per-result error isolation."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for delivery testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.llm_service.save_code_to_http.return_value = None
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._next_wakeup_time = None
        plugin._queue_wakeup_lock = threading.Lock()
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
        return plugin

    def _make_result(self, **overrides):
        """Create a PendingTaskResult with defaults including task_id."""
        from llm.service import PendingTaskResult

        defaults = {
            "status": "completed",
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello world",
            "model": "gpt-4",
            "content": "The answer is 42",
            "reason": "",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.01,
            "task_id": 42,
            "delivery_attempt_count": 0,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_poll_snapshots_channels_without_keys(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN the bot's live IrcDict of channels WHEN the worker-thread poll
        reads it THEN it takes an atomic snapshot (not IrcDict.keys(), a
        Python-level loop a concurrent JOIN/PART can break mid-iteration) and
        keeps the channel's original case for the reply_target match."""
        import supybot.world as world_mod
        from supybot.ircutils import IrcDict

        class _NoKeys(IrcDict):
            __slots__ = ()

            def keys(self):
                raise RuntimeError("dictionary changed size during iteration")

        conn = mocker.MagicMock()
        conn.state.channels = _NoKeys({"#Linux": object()})
        mocker.patch.object(world_mod, "ircs", [conn])
        plugin.llm_service.check_pending_tasks.return_value = []
        mocker.patch.object(plugin, "_schedule_queue_wakeup")

        plugin._check_pending_tasks()

        plugin.llm_service.check_pending_tasks.assert_called_once_with({"#Linux"})

    def test_successful_delivery_deletes_task(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN delivery succeeds WHEN queueMsg works THEN task deleted from DB."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)

        plugin.db.delete_pending_task.assert_called_once_with(42)

    def test_delivered_then_closing_still_acks(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN a send succeeds but shutdown begins before the ack THEN the row is still deleted.

        Models the race where ``_safe_queue`` already queued the message (closing
        was False) and ``_llm_executor.closing`` flips True before the post-send
        check. Bailing without deleting would leave the row to re-deliver next
        process lifetime — a duplicate IRC send. A successful send must be acked.
        """
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        # Send succeeds, then shutdown is observed before the ack.
        mocker.patch.object(plugin, "_safe_queue", return_value=True)
        plugin._llm_executor.closing = True

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)

        plugin.db.delete_pending_task.assert_called_once_with(42)
        # Retry-state writes stay suppressed during shutdown.
        plugin.db.update_delivery_attempt.assert_not_called()

    def test_delivered_ack_failure_is_best_effort(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN the send succeeds but delete raises (transient DB lock) THEN no exception escapes.

        The delivery already went out; an ack failure must be swallowed and
        logged (the row re-delivers next tick — at-least-once) instead of
        bubbling up as a misleading 'delivery failed' and never being retried.
        """
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        mocker.patch.object(plugin, "_safe_queue", return_value=True)
        plugin.db.delete_pending_task.side_effect = Exception("database is locked")

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)  # must not raise

        plugin.db.delete_pending_task.assert_called_once_with(42)
        plugin.log.warning.assert_called()

    def test_delivery_failure_retries_with_backoff(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN queueMsg raises WHEN delivering THEN delivery retried with backoff."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("IRC connection lost")
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)

        # Should NOT delete, should update delivery state
        plugin.db.delete_pending_task.assert_not_called()
        plugin.db.update_delivery_attempt.assert_called_once()
        call_args = plugin.db.update_delivery_attempt.call_args
        assert call_args[1]["task_id"] == 42
        assert call_args[1]["delivery_state"] == "retrying"
        assert call_args[1]["delivery_attempt_count"] == 1

    def test_delivery_backoff_formula(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN 3 prior delivery failures WHEN failing THEN next backoff is capped at 120s."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("connection reset")
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        mocker.patch("llm.plugin.time.time", return_value=1000000.0)

        # 3 prior failures means this failure is attempt 4:
        # delay = min(15 * 2^(4-1), 120) = 120
        r = self._make_result(task_id=42, delivery_attempt_count=3)
        plugin._deliver_pending_result(r)

        call_args = plugin.db.update_delivery_attempt.call_args[1]
        assert call_args["delivery_attempt_count"] == 4
        assert call_args["next_attempt_at"] == 1000000.0 + 120

    def test_delivery_exhaustion_marks_failed(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN 10 delivery failures WHEN delivering THEN set delivery_failed, retain row."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("persistent failure")
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        mock_wakeup = mocker.patch.object(plugin, "_schedule_queue_wakeup")

        # Task already at delivery_attempt_count = 9 (this is the 10th attempt)
        r = self._make_result(task_id=42, delivery_attempt_count=9)
        plugin._deliver_pending_result(r)

        plugin.db.update_delivery_attempt.assert_called_once()
        call_args = plugin.db.update_delivery_attempt.call_args[1]
        assert call_args["delivery_attempt_count"] == 10
        assert call_args["delivery_state"] == "delivery_failed"
        # Exhausted rows should not schedule another automatic wakeup.
        mock_wakeup.assert_not_called()

    def test_batch_cascade_isolation(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN batch of 3 results WHEN second delivery fails THEN first and third still delivered."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        call_count = 0

        def flaky_queue(msg):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("IRC send failed")

        mock_irc.queueMsg.side_effect = flaky_queue
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        results = [
            self._make_result(task_id=1, nick="alice"),
            self._make_result(task_id=2, nick="bob"),
            self._make_result(task_id=3, nick="charlie"),
        ]

        # Simulate the loop in _check_pending_tasks
        plugin.llm_service.check_pending_tasks.return_value = results
        plugin._check_pending_tasks()

        # All 3 should be attempted, not just the first
        assert mock_irc.queueMsg.call_count == 3
        # Tasks 1 and 3 should be deleted (delivered successfully)
        delete_calls = plugin.db.delete_pending_task.call_args_list
        assert len(delete_calls) == 2
        deleted_ids = {c[0][0] for c in delete_calls}
        assert deleted_ids == {1, 3}
        # Task 2 should be retried
        plugin.db.update_delivery_attempt.assert_called_once()
        assert plugin.db.update_delivery_attempt.call_args[1]["task_id"] == 2

    def test_ephemeral_results_no_db_operations(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN expired result with no task_id WHEN delivered THEN no DB delete/update."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result(status="expired", task_id=None, content="", reason="expired")
        plugin._deliver_pending_result(r)

        # Should deliver message but not touch DB
        mock_irc.queueMsg.assert_called_once()
        plugin.db.delete_pending_task.assert_not_called()
        plugin.db.update_delivery_attempt.assert_not_called()


class TestScheduleQueueWakeup:
    """Test event-driven queue wakeup scheduling (Phase 2)."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for wakeup testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._next_wakeup_time = None
        plugin._queue_wakeup_lock = threading.Lock()
        return plugin

    def test_check_remove_add_is_serialised(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN another thread mid-reschedule WHEN a second caller arrives THEN
        it waits — check, removeEvent and addEvent run as one unit, so neither
        an earlier wakeup is lost nor addEvent trips Limnoria's duplicate-name
        assert."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        plugin.db.get_next_due_time.return_value = 1060.0

        with plugin._queue_wakeup_lock:
            worker = threading.Thread(target=plugin._schedule_queue_wakeup, daemon=True)
            worker.start()
            worker.join(timeout=0.2)
            assert worker.is_alive()
            mock_schedule.addEvent.assert_not_called()
        worker.join(timeout=5)

        assert not worker.is_alive()
        mock_schedule.addEvent.assert_called_once()

    def test_no_tasks_does_nothing(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN empty queue WHEN _schedule_queue_wakeup called THEN no event scheduled."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        plugin.db.get_next_due_time.return_value = None

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_not_called()
        assert plugin._next_wakeup_time is None

    def test_schedules_at_next_due_time(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN task due at T WHEN _schedule_queue_wakeup called THEN one-shot event at T."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        plugin.db.get_next_due_time.return_value = 1060.0

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_called_once()
        call_args = mock_schedule.addEvent.call_args
        assert call_args[1]["name"] == "llm_queue_wakeup"
        assert call_args[0][1] == 1060.0  # at= parameter
        assert plugin._next_wakeup_time == 1060.0

    def test_replaces_if_earlier(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN scheduled wakeup at T=100 WHEN new due time T=50 THEN reschedule to T=50."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=10.0)
        plugin._next_wakeup_time = 100.0
        plugin.db.get_next_due_time.return_value = 50.0

        plugin._schedule_queue_wakeup()

        mock_schedule.removeEvent.assert_any_call("llm_queue_wakeup")
        mock_schedule.addEvent.assert_called_once()
        assert plugin._next_wakeup_time == 50.0

    def test_keeps_earlier_existing(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN scheduled wakeup at T=50 WHEN new due time T=100 THEN keep T=50."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=10.0)
        plugin._next_wakeup_time = 50.0
        plugin.db.get_next_due_time.return_value = 100.0

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_not_called()
        assert plugin._next_wakeup_time == 50.0

    def test_past_due_schedules_immediately(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN task due in the past WHEN _schedule_queue_wakeup called THEN schedule at now+1."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        plugin.db.get_next_due_time.return_value = 900.0  # in the past

        plugin._schedule_queue_wakeup()

        mock_schedule.addEvent.assert_called_once()
        call_args = mock_schedule.addEvent.call_args
        # Should schedule at now + 1, not in the past
        assert call_args[0][1] == 1001.0

    def test_explicit_at_time_bypasses_db(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN explicit at_time WHEN _schedule_queue_wakeup(at_time=T) THEN uses T, no DB query."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)

        plugin._schedule_queue_wakeup(at_time=1030.0)

        plugin.db.get_next_due_time.assert_not_called()
        mock_schedule.addEvent.assert_called_once()
        assert plugin._next_wakeup_time == 1030.0

    def test_clears_stale_wakeup_in_past(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN existing wakeup already in the past WHEN new due time THEN replace it."""
        mock_schedule = mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        plugin._next_wakeup_time = 900.0  # already past
        plugin.db.get_next_due_time.return_value = 1060.0

        plugin._schedule_queue_wakeup()

        mock_schedule.removeEvent.assert_any_call("llm_queue_wakeup")
        mock_schedule.addEvent.assert_called_once()
        assert plugin._next_wakeup_time == 1060.0


class TestSafetyPollInterval:
    """Test that the safety poll runs at 5-minute intervals (Phase 2)."""

    def test_safety_poll_interval_is_300_seconds(self, mocker: MockerFixture) -> None:
        """GIVEN plugin init WHEN addPeriodicEvent called for pending tasks THEN interval is 300."""
        from llm.plugin import LLM

        mocker.patch("llm.plugin.schedule")
        mocker.patch("llm.plugin.httpserver")
        mocker.patch("llm.plugin.conf")
        mocker.patch("llm.plugin.world")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin.registryValue = mocker.MagicMock(return_value="")
        plugin._http_callback = None
        plugin._reminders = {}
        plugin._reminders_lock = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin._apply_log_level = mocker.MagicMock()
        plugin._next_wakeup_time = None

        # Check that the constant is defined
        assert hasattr(LLM, "_SAFETY_POLL_INTERVAL")
        assert LLM._SAFETY_POLL_INTERVAL == 300


class TestWakeupTriggers:
    """Test that wakeup is triggered from all queue mutation points (Phase 2)."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for wakeup trigger testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.llm_service.save_code_to_http.return_value = None
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._next_wakeup_time = None
        plugin._queue_wakeup_lock = threading.Lock()
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
        return plugin

    def _make_result(self, **overrides):
        """Create a PendingTaskResult with defaults."""
        from llm.service import PendingTaskResult

        defaults = {
            "status": "completed",
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello",
            "model": "gpt-4",
            "content": "answer",
            "reason": "",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.01,
            "task_id": 42,
            "delivery_attempt_count": 0,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_check_pending_tasks_reschedules_after_batch(
        self, plugin, mocker: MockerFixture
    ) -> None:
        """GIVEN batch completes WHEN _check_pending_tasks finishes THEN _schedule_queue_wakeup called."""
        import supybot.world as world_mod

        mocker.patch.object(world_mod, "ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = []
        mock_wakeup = mocker.patch.object(plugin, "_schedule_queue_wakeup")

        plugin._check_pending_tasks()

        mock_wakeup.assert_called_once()

    def test_check_pending_tasks_clears_stale_wakeup(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN active wakeup WHEN _check_pending_tasks runs THEN _next_wakeup_time cleared first."""
        import supybot.world as world_mod

        mocker.patch.object(world_mod, "ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = []
        plugin._next_wakeup_time = 999.0

        # Use real _schedule_queue_wakeup but mock schedule module
        mocker.patch("llm.plugin.schedule")
        plugin.db.get_next_due_time.return_value = None

        plugin._check_pending_tasks()

        # Wakeup time should be cleared since no pending tasks
        assert plugin._next_wakeup_time is None

    def test_delivery_retry_triggers_wakeup(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN delivery fails WHEN _deliver_pending_result retries THEN wakeup scheduled at retry time."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.queueMsg.side_effect = Exception("IRC send failed")
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        mocker.patch("llm.plugin.time.time", return_value=1000.0)
        mock_wakeup = mocker.patch.object(plugin, "_schedule_queue_wakeup")

        r = self._make_result(task_id=42)
        plugin._deliver_pending_result(r)

        # Should schedule wakeup at the retry time (now + backoff)
        mock_wakeup.assert_called_once_with(at_time=1000.0 + 15)

    def test_stash_triggers_wakeup(self, mocker: MockerFixture) -> None:
        """GIVEN a request times out WHEN _stash_timeout succeeds THEN a wakeup
        is scheduled at the first-attempt time (submitted_at + initial backoff)."""
        from llm.service import PENDING_INITIAL_BACKOFF_SECONDS, LLMService

        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.return_value = 3600  # expiry
        mock_plugin.db.save_pending_task.return_value = 1

        service = LLMService.__new__(LLMService)
        service.plugin = mock_plugin
        service.log = mocker.MagicMock()

        now = 1000.0
        result = service._stash_timeout(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt="hello",
            model="gpt-4",
            request_data={"messages": []},
            submitted_at=now,
        )

        assert result is True
        mock_plugin._schedule_queue_wakeup.assert_called_once_with(
            at_time=now + PENDING_INITIAL_BACKOFF_SECONDS
        )


class TestDeliverPendingResultCodeBranch:
    """Test _deliver_pending_result code branch with HTTP URL."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for delivery testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
        return plugin

    def _make_result(self, **overrides):
        """Create a PendingTaskResult with defaults."""
        from llm.service import PendingTaskResult

        defaults = {
            "status": "completed",
            "task_type": "ask",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "hello world",
            "model": "gpt-4",
            "content": "The answer is 42",
            "reason": "",
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "cost": 0.01,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_code_result_with_url_sends_code_is_ready(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN completed code result WHEN save_code_to_http returns URL THEN sends 'code is ready' message."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        plugin.llm_service.save_code_to_http.return_value = "http://example.com/code_abc.html"

        r = self._make_result(
            task_type="code",
            nick="alice",
            content="print('hello')",
            prompt_preview="hello world",
            task_id=1,
        )
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg = mock_irc.queueMsg.call_args[0][0]
        msg_text = str(msg)
        assert "code is ready" in msg_text
        assert "http://example.com/code_abc.html" in msg_text
        assert "alice" in msg_text

    def test_code_result_without_url_sends_content(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN completed code result WHEN save_code_to_http returns None THEN sends raw content."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        plugin.llm_service.save_code_to_http.return_value = None

        r = self._make_result(
            task_type="code",
            nick="alice",
            content="print('hello')",
            prompt_preview="hello world",
        )
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        assert "print('hello')" in msg_text
        assert "code is ready" not in msg_text


class TestDeliverPendingResultAnimateBranch:
    """Test _deliver_pending_result animate branch end to end.

    _format_animate_delivery is unit-tested directly in test_animate.py, but
    nothing there proves _deliver_pending_result calls it with its arguments
    in the right order. The formatter's signature is
    ``(nick, prompt, url, target)``; the call site passes
    ``(nick, prompt_preview, content, target)``. A swap of the two middle
    arguments would still satisfy every direct formatter test (they all pass
    their own explicit prompt/url) while producing a delivered line whose
    *prompt* is budgeted and whose *URL* is echoed as quoted text — the exact
    URL-loss failure mode the formatter exists to prevent. This test drives
    the real delivery path and pins nick, quoted prompt, and URL in their
    correct positions in the sent text.
    """

    @pytest.fixture
    def plugin(self, mocker: MockerFixture):
        """Create a minimal plugin for delivery testing."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock()
        plugin._llm_executor.closing = False
        plugin._irc_send_lock = threading.Lock()
        return plugin

    def _make_result(self, **overrides):
        """Create a PendingTaskResult with defaults."""
        from llm.service import PendingTaskResult

        defaults = {
            "status": "completed",
            "task_type": "animate",
            "nick": "alice",
            "reply_target": "#test",
            "is_channel": True,
            "prompt_preview": "a corgi riding a unicorn",
            "model": "grok-imagine",
            "content": "https://paste.boxlabs.uk/img/vid_6a8830b21af1d.mp4",
            "reason": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost": 0.0,
        }
        defaults.update(overrides)
        return PendingTaskResult(**defaults)

    def test_animate_delivery_carries_nick_prompt_and_url(
        self, plugin, mocker: MockerFixture
    ) -> None:
        """GIVEN a completed animate result WHEN delivered THEN the sent text
        has the nick, the quoted prompt, and the URL in the shape
        _format_animate_delivery produces — not a bare URL, and not the
        prompt/URL positions swapped."""
        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mock_irc.state.nickToAccount.return_value = "alice"
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = self._make_result()
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_called_once()
        msg_text = str(mock_irc.queueMsg.call_args[0][0])
        url = "https://paste.boxlabs.uk/img/vid_6a8830b21af1d.mp4"
        assert f'alice: your video is ready! "a corgi riding a unicorn" → {url}' in msg_text
        # The URL must be the trailing link, never quoted as if it were the
        # prompt.
        assert f'"{url}"' not in msg_text


class TestDeliverPendingResultUnknownStatus:
    """Test _deliver_pending_result with an unknown status."""

    def test_unknown_status_returns_early_no_message(self, mocker: MockerFixture) -> None:
        """GIVEN result with unknown status WHEN _deliver_pending_result called THEN returns early with no IRC message."""
        from llm.plugin import LLM
        from llm.service import PendingTaskResult

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.llm_service = mocker.MagicMock()
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        plugin.db = mocker.MagicMock()
        plugin.log = mocker.MagicMock()

        import supybot.world as world_mod

        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])

        r = PendingTaskResult(
            status="weird",
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            content="some content",
            reason="",
        )
        plugin._deliver_pending_result(r)

        mock_irc.queueMsg.assert_not_called()
        mock_irc.sendMsg.assert_not_called()


class TestPendingTaskFns:
    """Phase 2 follow-up — unified `_pending_task_fns` helper wiring.

    Replaces the older split between `_reminder_fns` and
    `_scheduled_llm_task_fns`; the LLM-facing list/cancel surface now spans
    both reminders and scheduled tasks via a single helper.
    """

    def test_helper_returns_unified_callables_with_owner_identity_bound(
        self, mocker: MockerFixture
    ) -> None:
        """The helper closes over caller/irc/msg and dispatches to the
        right backend by id prefix."""
        from llm.identity import Caller, Identity
        from llm.persistence import ScheduledLlmTaskRow
        from llm.plugin import LLM
        from llm.service import ScheduleLlmTaskResult

        stand_in = mocker.MagicMock()
        stand_in.llm_service = mocker.MagicMock()
        stand_in.llm_service.schedule_llm_task.return_value = ScheduleLlmTaskResult(
            status="ok",
            event_name="llm_task_xyz",
            fire_at=1700000000.0,
            message="Scheduled.",
            note=None,
        )
        stand_in.llm_service.list_scheduled_llm_tasks.return_value = [
            ScheduledLlmTaskRow(
                id=1,
                event_name="llm_task_ev1",
                creator_nick="rdrake",
                account="rdrake_a",
                channel="#t",
                network="afternet",
                wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
                prompt="check the build" * 4,  # >80 chars to verify truncation
                fire_at=1700000000.0,
                created_at=1699999000.0,
                chain_position=1,
                recurrence_seconds=300,
                recurrence_rrule=None,
                watch_mode=False,
            ),
        ]
        stand_in.llm_service.cancel_scheduled_llm_task.return_value = ScheduleLlmTaskResult(
            status="ok",
            event_name="llm_task_ev1",
            fire_at=0.0,
            message="Cancelled.",
            note=None,
        )
        # Reminder side: stub _get_user_reminders + the per-id helpers used
        # internally by cancel_pending_task_fn.
        stand_in._get_user_reminders.return_value = [
            (
                "llm_remind_rdrake_abc123",
                make_reminder_row(
                    event_name="llm_remind_rdrake_abc123",
                    nick="rdrake",
                    channel="#t",
                    message="check build",
                    fire_at=1_800_000_000.0,
                ),
            ),
        ]
        stand_in._remind_set_for_assistant.return_value = ToolCallbackResult(
            True, "I'll remind you."
        )
        stand_in._remind_delete_for_assistant.return_value = ToolCallbackResult(
            True, "Deleted reminder abc123."
        )
        stand_in._remind_clear_for_assistant.return_value = "Cancelled 1 reminder."

        helper = LLM._pending_task_fns.__get__(stand_in, LLM)
        caller = Caller(Identity(raw_nick="rdrake", account="rdrake_a"), "#t")
        irc = mocker.MagicMock()
        msg = mocker.MagicMock()
        fns = helper(caller=caller, irc=irc, msg=msg)

        assert set(fns.keys()) == {
            "set_reminder_fn",
            "schedule_llm_task_fn",
            "list_pending_tasks_fn",
            "cancel_pending_task_fn",
            "cancel_all_pending_tasks_fn",
        }

        # schedule_fn forwards keyword args and binds caller identity.
        out = fns["schedule_llm_task_fn"](when_natural="at 9am", prompt="ping me")
        assert out["status"] == "ok"
        assert out["event_name"] == "llm_task_xyz"
        stand_in.llm_service.schedule_llm_task.assert_called_once_with(
            irc=irc,
            msg=msg,
            creator_nick="rdrake",
            account="rdrake_a",
            channel="#t",
            when_natural="at 9am",
            prompt="ping me",
            reply_target=None,
            user_tz=stand_in._resolve_user_tz.return_value,
        )
        # A user-initiated wall-clock schedule may probe the client clock.
        stand_in._resolve_user_tz.assert_called_once_with(caller.identity, irc=irc)

        # list_pending_tasks_fn merges reminders + scheduled tasks with
        # `kind` discriminators.
        listed = fns["list_pending_tasks_fn"]()
        assert {row["kind"] for row in listed} == {"reminder", "scheduled_task"}
        scheduled = next(r for r in listed if r["kind"] == "scheduled_task")
        assert scheduled["id"] == "llm_task_ev1"
        assert len(scheduled["description"]) <= 80
        assert scheduled["recurrence"] == "every 300s"
        reminder = next(r for r in listed if r["kind"] == "reminder")
        assert reminder["id"] == "abc123"
        assert reminder["description"] == "check build"
        assert reminder["channel"] == "#t"
        assert reminder["when"] == "2027-01-15T08:00:00Z"

        # cancel_pending_task_fn routes by id prefix to the right backend.
        cancelled = fns["cancel_pending_task_fn"]("llm_task_ev1")
        assert cancelled["status"] == "ok"
        assert cancelled["kind"] == "scheduled_task"
        stand_in.llm_service.cancel_scheduled_llm_task.assert_called_once_with(
            event_name="llm_task_ev1",
            creator_nick="rdrake",
            account="rdrake_a",
        )

        cancelled_reminder = fns["cancel_pending_task_fn"]("abc123")
        assert cancelled_reminder["kind"] == "reminder"
        stand_in._remind_delete_for_assistant.assert_called_once()


class TestAgendaWakeup:
    """The one wakeup timer for reminders and scheduled tasks (real SQLite)."""

    @pytest.fixture
    def plugin(self, mock_irc, mocker: MockerFixture, tmp_path):
        from llm.plugin import LLM

        from .conftest import plugin_init_patches

        mocker.patch.object(
            LLM,
            "registryValue",
            side_effect=make_registry_side_effect({"databasePath": str(tmp_path / "llm.db")}),
        )
        plugin_init_patches(mocker, mock_database=False)
        plugin = LLM(mock_irc)
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        return plugin

    @staticmethod
    def _armed(add_event) -> list[float]:
        return [
            c.args[1]
            for c in add_event.call_args_list
            if c.kwargs.get("name") == "llm_agenda_wakeup"
        ]

    def test_nothing_pending_arms_nothing(self, plugin, mocker: MockerFixture) -> None:
        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin._arm_agenda()
        assert self._armed(add_event) == []

    def test_only_ever_moves_earlier(self, plugin, mocker: MockerFixture) -> None:
        """A later entry leaves the armed wakeup alone; an earlier one replaces it."""
        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        now = time.time()
        plugin.db.save_reminder("llm_remind_a", "u", "#t", "a", now + 600)
        plugin._arm_agenda()
        plugin.db.save_reminder("llm_remind_b", "u", "#t", "b", now + 900)
        plugin._arm_agenda()
        plugin.db.save_reminder("llm_remind_c", "u", "#t", "c", now + 300)
        plugin._arm_agenda()
        assert self._armed(add_event) == [now + 600, now + 300]

    def test_check_remove_add_is_serialised(self, plugin, mocker: MockerFixture) -> None:
        """GIVEN another thread mid-arm WHEN a second caller arrives THEN it
        waits, so an earlier wakeup is never lost and addEvent never trips
        Limnoria's duplicate-name assert (the race a5ea4a9 fixed for the queue)."""
        plugin.db.save_reminder("llm_remind_a", "u", "#t", "a", time.time() + 600)
        entered = threading.Event()
        release = threading.Event()

        def slow_add(*_args, **_kwargs):
            entered.set()
            release.wait(5)

        add_event = mocker.patch("llm.plugin.schedule.addEvent", side_effect=slow_add)
        first = threading.Thread(target=plugin._arm_agenda)
        first.start()
        assert entered.wait(5)
        second = threading.Thread(target=plugin._arm_agenda)
        second.start()
        second.join(0.2)
        assert second.is_alive()  # blocked on the lock, not racing addEvent
        release.set()
        first.join(5)
        second.join(5)
        assert add_event.call_count == 1

    def test_a_fire_that_raises_is_still_settled(self, plugin, mocker: MockerFixture) -> None:
        """A one-shot whose delivery blows up on the scheduler thread is
        removed rather than retried every second forever."""
        plugin.db.save_reminder("llm_remind_a", "u", "#t", "a", time.time() - 5)
        mocker.patch.object(plugin, "_fire_reminder", side_effect=RuntimeError("boom"))

        plugin._agenda_tick()

        assert plugin.db.get_reminder("llm_remind_a") is None

    def test_tick_does_nothing_while_closing(self, plugin, mocker: MockerFixture) -> None:
        plugin.db.save_reminder("llm_remind_a", "u", "#t", "a", time.time() - 5)
        fire = mocker.patch.object(plugin, "_fire_reminder")
        plugin._llm_executor.closing = True

        plugin._agenda_tick()

        fire.assert_not_called()
        assert plugin.db.get_reminder("llm_remind_a") is not None


class TestPendingUsageBookedOnce:
    """Delivery is at-least-once; the billed retry's usage row is not."""

    @pytest.fixture
    def env(self, plugin_env, mocker: MockerFixture, tmp_path):
        import json

        import supybot.world as world_mod
        from llm.persistence import LLMDatabase

        plugin, _irc, _msg = plugin_env
        plugin.db = LLMDatabase(str(tmp_path / "pending.db"))
        plugin._schedule_queue_wakeup = mocker.MagicMock()
        mock_irc = mocker.MagicMock()
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch.object(world_mod, "ircs", [mock_irc])
        now = time.time()
        task_id = plugin.db.save_pending_task(
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            request_data="{}",
            submitted_at=now,
            expires_at=now + 600,
            next_attempt_at=now,
        )
        payload = {
            "status": "completed",
            "content": "hi",
            "cost": 0.01,
            "prompt_tokens": 100,
            "completion_tokens": 50,
        }
        plugin.db.update_task_for_delivery(task_id, "ready", json.dumps(payload))
        return plugin, task_id

    @staticmethod
    def _result(task_id: int, attempts: int = 0):
        from llm.service import PendingTaskResult

        return PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            content="hi",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.01,
            task_id=task_id,
            delivery_attempt_count=attempts,
        )

    @staticmethod
    def _rows(plugin) -> list[tuple]:
        return plugin.db._connect().execute("SELECT nick, status, cost FROM usage").fetchall()

    def test_failed_sends_book_once_when_delivery_fails(self, env, mocker) -> None:
        """GIVEN every send fails WHEN retries run out THEN exactly one error row,
        and a further pass over the dead row adds none."""
        plugin, task_id = env
        mocker.patch.object(plugin, "_safe_queue", return_value=False)
        for attempt in range(plugin._DELIVERY_MAX_ATTEMPTS - 1):
            plugin._deliver_pending_result(self._result(task_id, attempt))
            assert self._rows(plugin) == []

        plugin._deliver_pending_result(self._result(task_id, plugin._DELIVERY_MAX_ATTEMPTS - 1))
        plugin._deliver_pending_result(self._result(task_id, plugin._DELIVERY_MAX_ATTEMPTS))

        assert self._rows(plugin) == [("!alice", "error", 0.01)]

    def test_redelivery_after_a_failed_ack_books_once(self, env, mocker) -> None:
        """GIVEN the ack (delete) fails so the row re-delivers THEN one row."""
        plugin, task_id = env
        mocker.patch.object(plugin, "_safe_queue", return_value=True)
        mocker.patch.object(plugin.db, "delete_pending_task", side_effect=RuntimeError("locked"))

        plugin._deliver_pending_result(self._result(task_id))
        plugin._deliver_pending_result(self._result(task_id))

        assert self._rows(plugin) == [("!alice", "success", 0.01)]

    def test_cancelled_task_books_its_spend(self, env) -> None:
        """GIVEN a billed task is cancelled before delivery THEN its spend books."""
        plugin, _task_id = env
        plugin.llm_service.cancel_video.return_value = True
        (row,) = plugin.db.load_pending_tasks("ask")

        plugin._cancel_animate_row(row)

        assert self._rows(plugin) == [("!alice", "error", 0.01)]
