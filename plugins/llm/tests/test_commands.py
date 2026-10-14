"""Tests that call actual plugin command methods (not reimplementations).

These tests exercise the real ask, code, draw, forget, usage,
and remind methods on a properly-initialised LLM plugin instance
with mocked dependencies.

Unlike the _call_ask / _call_code / _call_draw helpers in test_plugin.py
which reimplement command logic, these tests invoke the actual methods so
regressions in the real command code are caught.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from llm.assistant import PENDING_TASK_TOOLS
from llm.context import Role
from llm.persistence import UsageBreakdown, UsageSummary
from llm.plugin import LLM
from llm.service import AssistantResult, CompletionResult, ReminderParseResult

from .conftest import make_registry_side_effect, make_reminder_row

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------


class TestCommandPathPermit:
    """Verify command paths acquire the global LLMExecutor permit."""

    def test_ask_acquires_permit(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="ok",
            grounding_used=False,
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="m",
        )
        spy = mocker.spy(plugin._llm_executor, "permit")
        plugin.ask(mock_irc, mock_msg, ["hello"])
        spy.assert_called_once()


class TestAskCommand:
    """Tests for the real LLM.ask method."""

    def test_ask_routes_through_assistant_request(self, plugin_env):
        """GIVEN ask WHEN executed THEN it uses the shared assistant_request facade."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Hello from unified assistant",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        request_context = plugin.llm_service.assistant_request.call_args.kwargs["request_context"]
        assert request_context.profile == "chat"
        assert request_context.entry_route == "ask"
        mock_irc.reply.assert_called_once_with("Hello from unified assistant", prefixNick=False)

    def test_ask_replies_with_completion_content(self, plugin_env, mocker: MockerFixture):
        """GIVEN a normal prompt WHEN ask is called THEN irc.reply receives the completion content."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hello from AI",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["What", "is", "Python?"])

        mock_irc.reply.assert_called_once_with("Hello from AI", prefixNick=False)

    def test_ask_stores_context_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN a successful completion WHEN ask is called THEN conversation context is stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="response text",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # Context should have both user and assistant messages
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "response text"

    def test_ask_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN completion with cost WHEN ask completes THEN usage is logged in db."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.005,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "ask",
            "gpt-4",
            100,
            50,
            0.005,
            prompt="hello",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_ask_skips_znc_playback(self, plugin_env, mocker: MockerFixture):
        """GIVEN a message older than startup WHEN ask called THEN no reply is sent."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100  # before startup

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()

    def test_ask_prepends_grounding_icon_when_used(self, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used is True WHEN ask completes THEN reply has globe icon prefix."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="searched result",
            grounding_used=True,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["search", "something"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")
        assert "searched result" in reply_text

    def test_ask_does_not_store_context_on_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN completion returns an error WHEN ask completes THEN context is NOT stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # No context should be stored because result has an error
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_ask_skips_context_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN context disabled WHEN ask completes THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="reply",
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="gpt-4",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_ask_sends_action_for_me_response(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with /me WHEN ask called THEN sends IRC action."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me shrugs",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")

        plugin.ask(mock_irc, mock_msg, ["how", "are", "you?"])

        mock_irc.reply.assert_not_called()
        mock_action.assert_called_once_with("#test", "shrugs")
        mock_irc.queueMsg.assert_called_once_with(mock_action.return_value)

    def test_ask_normal_response_uses_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds normally WHEN ask called THEN uses irc.reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="The capital is Paris.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["what", "is", "the", "capital?"])

        mock_irc.reply.assert_called_once_with("The capital is Paris.", prefixNick=False)

    def test_ask_reminder_mutation_with_empty_text_suppresses_reply(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN successful set_reminder + empty post-tool text WHEN ask called THEN no irc.reply, but usage logged."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=20,
            completion_tokens=2,
            cost=0.0002,
            model="gpt-4",
            last_successful_tool="set_reminder",
            final_text_after_tools="",
        )

        plugin.ask(mock_irc, mock_msg, ["remind", "me", "in", "1m"])

        # Reaction is the user-visible ack; no duplicate reply.
        mock_irc.reply.assert_not_called()
        mock_irc.queueMsg.assert_not_called()
        # Suppression isn't an error path.
        mock_irc.error.assert_not_called()
        # Usage still logged so the suppressed path isn't free.
        plugin.db.log_usage.assert_called_once()

    def test_ask_reminder_mutation_with_text_is_still_suppressed(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN successful set_reminder + follow-up text WHEN ask called THEN nothing is sent.

        The reaction is the only ack. Observed 2026-09-25: gemini narrated
        its own silence ("An empty response is submitted to stay quiet...").
        """
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        follow_up = (
            "An empty response is submitted to stay quiet, as the user is "
            "acknowledged via reaction."
        )
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content=follow_up,
            grounding_used=False,
            prompt_tokens=20,
            completion_tokens=12,
            cost=0.0003,
            model="gpt-4",
            last_successful_tool="set_reminder",
            final_text_after_tools=follow_up,
        )

        plugin.ask(mock_irc, mock_msg, ["remind", "me", "in", "1m"])

        mock_irc.reply.assert_not_called()
        mock_irc.error.assert_not_called()

    def test_ask_non_reminder_tool_with_empty_text_does_not_suppress(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN non-mutation tool + empty text WHEN ask called THEN falls through to empty-response error."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=15,
            completion_tokens=0,
            cost=0.0001,
            model="gpt-4",
            last_successful_tool="search_web",
            final_text_after_tools="",
        )

        plugin.ask(mock_irc, mock_msg, ["what", "is", "happening"])

        mock_irc.reply.assert_not_called()
        mock_irc.error.assert_called_once()

    def test_ask_no_tool_called_with_empty_text_falls_through_to_error(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN no tool call + empty text WHEN ask called THEN empty-response error fires."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.0001,
            model="gpt-4",
            last_successful_tool=None,
            final_text_after_tools="",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()
        mock_irc.error.assert_called_once()

    def test_ask_action_stores_context_with_star_prefix(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with /me WHEN ask called THEN context stores * BotNick text."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me thinks about it",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["hmm"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "* testbot thinks about it"

    def test_ask_action_with_grounding_icon(self, plugin_env, mocker: MockerFixture):
        """GIVEN /me response with grounding WHEN ask called THEN action includes globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me looks it up",
            grounding_used=True,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["search", "for", "it"])

        mock_action.assert_called_once_with("#test", "\U0001f310 looks it up")

    def test_ask_bare_me_not_treated_as_action(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with '/me' with no trailing text WHEN ask called THEN uses reply."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="/me",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["test"])

        mock_irc.reply.assert_called_once_with("/me", prefixNick=False)

    def test_ask_star_botname_treated_as_action(self, plugin_env, mocker: MockerFixture):
        """GIVEN LLM responds with '* BotNick ...' WHEN ask called THEN sent as IRC action."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="* testbot nudges a pair of glasses toward you",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")
        plugin.ask(mock_irc, mock_msg, ["test"])

        mock_action.assert_called_once_with("#test", "nudges a pair of glasses toward you")


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------


class TestCodeCommand:
    """Tests for the real LLM.code method."""

    def _make_code_result(self, **overrides):
        """Build a AssistantResult with sensible defaults for code tests."""
        defaults = {
            "content": "Here is your code — http://x/code.html",
            "grounding_used": False,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "cost": 0.001,
            "model": "gpt-4",
        }
        defaults.update(overrides)
        return AssistantResult(**defaults)  # ty: ignore[invalid-argument-type]

    def test_code_routes_through_assistant_request(self, plugin_env, mocker: MockerFixture):
        """GIVEN code WHEN executed THEN it uses assistant_request with code profile."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result()

        plugin.code(mock_irc, mock_msg, ["Python", "hello"])

        plugin.llm_service.assistant_request.assert_called_once()
        ctx = plugin.llm_service.assistant_request.call_args.kwargs["request_context"]
        assert ctx.profile == "code"
        assert ctx.entry_route == "code"

    def test_code_replies_with_planner_content(self, plugin_env, mocker: MockerFixture):
        """GIVEN planner returns summary with URL WHEN code called THEN reply contains it."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="Fibonacci function — http://localhost:8080/llm/code_abc.html",
        )

        plugin.code(mock_irc, mock_msg, ["Python", "hello"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "http://localhost:8080/llm/code_abc.html" in reply_text
        assert "Fibonacci function" in reply_text

    def test_code_stores_context(self, plugin_env, mocker: MockerFixture):
        """GIVEN code command succeeds WHEN executed THEN conversation context is stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="code output",
        )

        plugin.code(mock_irc, mock_msg, ["generate", "something"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_code_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN code completion with cost WHEN code completes THEN usage is logged."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="x = 1",
            prompt_tokens=50,
            completion_tokens=20,
            cost=0.003,
        )

        plugin.code(mock_irc, mock_msg, ["assign"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "code",
            "gpt-4",
            50,
            20,
            0.003,
            prompt="assign",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_code_skips_znc_playback(self, plugin_env, mocker: MockerFixture):
        """GIVEN an old message WHEN code called THEN nothing happens."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100

        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_not_called()

    def test_code_grounding_icon_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used is True WHEN code response returned THEN reply has globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="summary — http://x/c.html",
            grounding_used=True,
        )

        plugin.code(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")

    def test_code_action_with_grounding_icon(self, plugin_env, mocker: MockerFixture):
        """GIVEN /me-style code response with grounding WHEN code called THEN action carries globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result(
            content="/me runs the unit tests",
            grounding_used=True,
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")
        plugin.code(mock_irc, mock_msg, ["run", "tests"])

        mock_action.assert_called_once_with("#test", "\U0001f310 runs the unit tests")

    def test_code_preserves_preflight(self, plugin_env, mocker: MockerFixture):
        """GIVEN code WHEN executed THEN preflight checks still run."""
        plugin, mock_irc, mock_msg = plugin_env
        # Make preflight block the request (e.g. rate limited)
        mocker.patch.object(
            plugin,
            "_run_preflight",
            return_value=mocker.MagicMock(blocked=True),
        )

        plugin.code(mock_irc, mock_msg, ["test"])

        plugin.llm_service.assistant_request.assert_not_called()
        mock_irc.reply.assert_not_called()

    def test_code_user_instruction_layers_on_facade_prompt(self, plugin_env):
        """GIVEN user has instruction WHEN code called THEN planner sees CODE_SYSTEM_PROMPT.

        Regression: a user instruction must not replace the assistant facade
        prompt with the (inner-call) registry codeSystemPrompt — otherwise
        the planner stops calling generate_code and the pastebin breaks. The
        instruction itself now rides as user_instruction (user-role data), not
        prepended to the facade.
        """
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = "You are Captain Picard."
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_code_result()

        plugin.code(mock_irc, mock_msg, ["fibonacci"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        system_prompt = kwargs["system_prompt"]
        # Facade preserved (planner still told to call generate_code)...
        assert "generate_code" in system_prompt
        # ...but the instruction is relocated out of the system prompt.
        assert "Picard" not in system_prompt
        assert kwargs["user_instruction"] == "You are Captain Picard."


# ---------------------------------------------------------------------------
# _send_long_reply (single-line or pastebin teaser)
# ---------------------------------------------------------------------------


class TestPendingTaskGate:
    """``pendingTasksEnabled`` gates the reminder/scheduled tool surface."""

    def test_disabled_excludes_pending_task_tools(self, plugin_env):
        plugin, _mock_irc, _mock_msg = plugin_env
        assert plugin._pending_task_excludes("#test") == PENDING_TASK_TOOLS

    def test_enabled_excludes_nothing(self, plugin_env, mocker: MockerFixture):
        plugin, _mock_irc, _mock_msg = plugin_env
        registry = make_registry_side_effect({"pendingTasksEnabled": True})
        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        assert plugin._pending_task_excludes("#test") == frozenset()

    def test_ask_passes_exclusion_to_assistant_request(self, plugin_env):
        """The chat entry route threads the gate into assistant_request."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="m",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert kwargs["exclude_tools"] == PENDING_TASK_TOOLS


class TestSendLongReply:
    """Tests for _send_long_reply — single-line-or-pastebin-teaser helper.

    The bot never paginates: replies that fit one IRC line go via irc.reply
    as-is; anything multi-line (or that wraps past one wire-line) is saved
    to the HTTP pastebin and the channel sees one teaser+URL line.
    """

    def test_short_text_uses_irc_reply(self, plugin_env):
        """GIVEN one-line short text WHEN sent THEN goes via irc.reply, no pastebin."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin._send_long_reply(mock_irc, mock_msg, "hello world")

        mock_irc.reply.assert_called_once_with("hello world", prefixNick=False)
        plugin.llm_service.save_markdown_to_http.assert_not_called()

    def test_short_text_dropped_when_closing(self, plugin_env):
        """GIVEN the executor is closing WHEN a single-line reply is sent THEN
        it is serialized through _safe_reply and dropped (no raw irc.reply)."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin._llm_executor.shutdown()

        plugin._send_long_reply(mock_irc, mock_msg, "hello world")

        mock_irc.reply.assert_not_called()

    def test_single_logical_line_with_blank_padding_uses_irc_reply(self, plugin_env):
        """Padding blank lines around one real line collapse to a single irc.reply."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin._send_long_reply(mock_irc, mock_msg, "\n\nhello\n\n")

        mock_irc.reply.assert_called_once_with("hello", prefixNick=False)
        plugin.llm_service.save_markdown_to_http.assert_not_called()

    def test_multiline_text_pastebins_and_sends_teaser(self, plugin_env, mocker):
        """GIVEN \\n in text WHEN sent THEN whole text goes to pastebin and one teaser+URL line is replied."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.save_markdown_to_http.return_value = "https://example.com/llm/full.html"
        plugin.llm_service.summarize_for_irc.return_value = "Three short lines about A, B, and C."

        plugin._send_long_reply(mock_irc, mock_msg, "line one\nline two\nline three")

        # The summary doubles as the page <title>, reused at no extra cost.
        plugin.llm_service.save_markdown_to_http.assert_called_once_with(
            "line one\nline two\nline three",
            title="Three short lines about A, B, and C.",
            style="answer",
        )
        mock_irc.reply.assert_called_once_with(
            "Three short lines about A, B, and C. - Full answer: https://example.com/llm/full.html",
            prefixNick=False,
        )

    def test_story_style_threads_through_to_save(self, plugin_env):
        """Verse overflow pastes render with the storybook theme."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.save_markdown_to_http.return_value = "https://e.co/s.html"
        plugin.llm_service.summarize_for_irc.return_value = "A tale."

        plugin._send_long_reply(mock_irc, mock_msg, "beat one\nbeat two", style="story")

        plugin.llm_service.save_markdown_to_http.assert_called_once_with(
            "beat one\nbeat two", title="A tale.", style="story"
        )

    def test_blank_lines_do_not_force_pastebin(self, plugin_env):
        """A single real line padded with blanks stays a one-line irc.reply."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin._send_long_reply(mock_irc, mock_msg, "line one\n\n   \n")

        mock_irc.reply.assert_called_once_with("line one", prefixNick=False)
        plugin.llm_service.save_markdown_to_http.assert_not_called()

    def test_long_wrapped_single_line_pastebins(self, plugin_env, mocker):
        """A single logical line that wraps past one wire-line still pastebins."""
        plugin, mock_irc, mock_msg = plugin_env
        # Force a 100-byte wrap budget. 250 chars of payload wrap into 3
        # wire-lines, but 100 still leaves room for "<teaser> - Full answer: <url>".
        mocker.patch("llm.plugin.conf.get", return_value=100)
        plugin.llm_service.save_markdown_to_http.return_value = "https://e.co/x.html"
        plugin.llm_service.summarize_for_irc.return_value = "Long line."

        long_line = "alpha " * 50  # ~300 chars
        plugin._send_long_reply(mock_irc, mock_msg, long_line)

        plugin.llm_service.save_markdown_to_http.assert_called_once_with(
            long_line, title="Long line.", style="answer"
        )
        mock_irc.reply.assert_called_once_with(
            "Long line. - Full answer: https://e.co/x.html",
            prefixNick=False,
        )

    def test_long_reply_uses_fallback_teaser_when_summary_fails(self, plugin_env, mocker):
        """GIVEN summarize returns None THEN first non-blank line is used as the teaser."""
        plugin, mock_irc, mock_msg = plugin_env
        long_text = "\n".join(
            [
                "### Abbreviated History of Liberia",
                "- 1822: Founded by freed US slaves.",
                "- 1847: Declared independence.",
            ]
        )
        plugin.llm_service.save_markdown_to_http.return_value = "https://example.com/llm/full.html"
        plugin.llm_service.summarize_for_irc.return_value = None

        plugin._send_long_reply(mock_irc, mock_msg, long_text)

        mock_irc.reply.assert_called_once_with(
            "Abbreviated History of Liberia - Full answer: https://example.com/llm/full.html",
            prefixNick=False,
        )

    def test_long_reply_drops_url_when_pastebin_save_fails(self, plugin_env, mocker):
        """GIVEN save returns None THEN reply is a teaser only — no broken URL."""
        plugin, mock_irc, mock_msg = plugin_env
        long_text = "\n".join(f"line {i}" for i in range(1, 5))
        plugin.llm_service.save_markdown_to_http.return_value = None
        plugin.llm_service.summarize_for_irc.return_value = "Four short lines."

        plugin._send_long_reply(mock_irc, mock_msg, long_text)

        mock_irc.reply.assert_called_once_with("Four short lines.", prefixNick=False)

    def test_long_reply_caps_teaser_to_link_budget(self, plugin_env, mocker):
        """GIVEN a tight IRC line budget WHEN linked THEN teaser leaves room for the URL."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.conf.get", return_value=80)
        long_text = "\n".join(f"line {i}" for i in range(1, 8))
        url = "https://example.com/llm/full.html"
        suffix = f" - Full answer: {url}"
        plugin.llm_service.save_markdown_to_http.return_value = url
        plugin.llm_service.summarize_for_irc.return_value = (
            "Liberia has a much longer summary than the available budget allows."
        )

        plugin._send_long_reply(mock_irc, mock_msg, long_text)

        # The summary is generated once at the configured width (default 220) so
        # it can also serve as the page <title>; the IRC teaser is then trimmed to
        # the link budget. The cap is verified by the final_reply assertions below.
        plugin.llm_service.summarize_for_irc.assert_called_once_with(
            long_text, channel="#test", max_chars=220
        )
        final_reply = mock_irc.reply.call_args.args[0]
        assert len(final_reply) <= 80
        assert final_reply.endswith(suffix)


class TestSendLongReplyMultiline:
    """Short overflows go out as one draft/multiline batch instead of a paste.

    Only when the cap is negotiated and the reply wraps to at most
    ``multilineMaxLines`` wire-lines; anything longer still pastebins.
    """

    @pytest.fixture
    def env(self, plugin_env, mocker: MockerFixture):
        from supybot import conf, ircutils

        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.capabilities_ack = {"draft/multiline", "message-tags"}
        mock_irc.state.capabilities_ls = {"draft/multiline": "max-bytes=16384,max-lines=100"}
        mock_irc.isChannel = ircutils.isChannel
        mock_irc.stripChannelPrefix = lambda s: s
        mock_irc.network = "afternet"
        mocker.patch.object(conf.supybot.protocols.irc.experimentalExtensions, "value", True)
        mocker.patch.object(plugin, "_reply_mores_length", return_value=100)
        plugin.llm_service.save_markdown_to_http.return_value = "https://e.co/x.html"
        plugin.llm_service.summarize_for_irc.return_value = "Teaser."
        return plugin, mock_irc, mock_msg

    @staticmethod
    def _batch(mock_irc):
        mock_irc.queueBatch.assert_called_once()
        return mock_irc.queueBatch.call_args.args[0]

    def test_wrapped_line_batches_with_concat(self, env):
        """GIVEN one line wrapping to 2 wire-lines WHEN sent THEN one batch, the
        continuation tagged multiline-concat, and no paste or summary call."""
        plugin, mock_irc, mock_msg = env
        text = "alpha " * 30  # 180 bytes -> 2 chunks at a 100-byte budget

        plugin._send_long_reply(mock_irc, mock_msg, text)

        batch = self._batch(mock_irc)
        assert [m.command for m in batch] == ["BATCH", "PRIVMSG", "PRIVMSG", "BATCH"]
        assert batch[0].args[1:] == ("draft/multiline", "#test")
        name = batch[0].args[0][1:]
        assert batch[-1].args == ("-" + name,)
        assert batch[1].server_tags == {"batch": name}
        assert batch[2].server_tags == {"batch": name, "draft/multiline-concat": None}
        assert all(m.args[0] == "#test" for m in batch[1:3])
        assert "".join(m.args[1] for m in batch[1:3]).split() == text.split()
        mock_irc.reply.assert_not_called()
        plugin.llm_service.save_markdown_to_http.assert_not_called()
        plugin.llm_service.summarize_for_irc.assert_not_called()

    def test_distinct_lines_are_not_concatenated(self, env):
        """Separate logical lines stay separate lines on the receiving end."""
        plugin, mock_irc, mock_msg = env

        plugin._send_long_reply(mock_irc, mock_msg, "first point\n\nsecond point")

        batch = self._batch(mock_irc)
        lines = batch[1:-1]
        assert [m.args[1] for m in lines] == ["first point", "second point"]
        assert all("draft/multiline-concat" not in m.server_tags for m in lines)

    def test_reply_tag_moves_to_batch_opener(self, env):
        """Client tags belong on the BATCH command, never on the lines inside it."""
        plugin, mock_irc, mock_msg = env
        mock_msg.server_tags = {"msgid": "abc123"}

        plugin._send_long_reply(mock_irc, mock_msg, "one\ntwo")

        batch = self._batch(mock_irc)
        assert batch[0].server_tags == {"+draft/reply": "abc123"}
        assert all(set(m.server_tags) <= {"batch", "draft/multiline-concat"} for m in batch[1:])

    def test_over_cap_still_pastebins(self, env):
        """GIVEN 4 wire-lines against a cap of 3 WHEN sent THEN teaser + URL."""
        plugin, mock_irc, mock_msg = env

        plugin._send_long_reply(mock_irc, mock_msg, "a\nb\nc\nd")

        mock_irc.queueBatch.assert_not_called()
        mock_irc.reply.assert_called_once_with(
            "Teaser. - Full answer: https://e.co/x.html", prefixNick=False
        )

    def test_cap_zero_disables(self, env):
        plugin, mock_irc, mock_msg = env
        plugin.registryValue = make_registry_side_effect({"multilineMaxLines": 0})

        plugin._send_long_reply(mock_irc, mock_msg, "one\ntwo")

        mock_irc.queueBatch.assert_not_called()
        plugin.llm_service.save_markdown_to_http.assert_called_once()

    def test_not_negotiated_pastebins(self, env):
        plugin, mock_irc, mock_msg = env
        mock_irc.state.capabilities_ack = {"message-tags"}

        plugin._send_long_reply(mock_irc, mock_msg, "one\ntwo")

        mock_irc.queueBatch.assert_not_called()
        plugin.llm_service.save_markdown_to_http.assert_called_once()

    def test_single_line_still_plain_reply(self, env):
        plugin, mock_irc, mock_msg = env

        plugin._send_long_reply(mock_irc, mock_msg, "hello")

        mock_irc.queueBatch.assert_not_called()
        mock_irc.reply.assert_called_once_with("hello", prefixNick=False)

    def test_dropped_when_closing(self, env):
        plugin, mock_irc, mock_msg = env
        plugin._llm_executor.shutdown()

        plugin._send_long_reply(mock_irc, mock_msg, "one\ntwo")

        mock_irc.queueBatch.assert_not_called()
        plugin.llm_service.save_markdown_to_http.assert_not_called()


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------


class TestDrawCommand:
    """Tests for the real LLM.draw method (thin wrapper over assistant facade)."""

    def _make_draw_result(self, **overrides):
        """Build a AssistantResult with sensible defaults for draw tests."""
        defaults = {
            "content": "Here is your image: http://img.example/gen.png",
            "grounding_used": False,
            "prompt_tokens": 5,
            "completion_tokens": 0,
            "cost": 0.02,
            "model": "dall-e-3",
        }
        defaults.update(overrides)
        return AssistantResult(**defaults)  # ty: ignore[invalid-argument-type]

    def test_draw_routes_through_assistant_request(self, plugin_env, mocker: MockerFixture):
        """@draw calls assistant_request with draw profile."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        plugin.llm_service.assistant_request.assert_called_once()
        ctx = plugin.llm_service.assistant_request.call_args.kwargs["request_context"]
        assert ctx.profile == "draw"
        assert ctx.entry_route == "draw"

    def test_draw_grounds_in_canon_when_referenced(self, plugin_env, mocker: MockerFixture):
        """@draw of canon layers the lore block onto the draw overlay so the
        image depicts the real cast."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()
        mocker.patch.object(
            plugin, "_verse_context_for", return_value="Established characters:\n- Archie: windbag"
        )

        plugin.draw(mock_irc, mock_msg, ["the", "stinky", "lads"])

        sp = plugin.llm_service.assistant_request.call_args.kwargs["system_prompt"]
        assert sp is not None and "Archie: windbag" in sp

    def test_draw_no_grounding_without_canon(self, plugin_env, mocker: MockerFixture):
        """No canon reference → system_prompt stays None (default draw prompt)."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()
        mocker.patch.object(plugin, "_verse_context_for", return_value=None)

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        assert plugin.llm_service.assistant_request.call_args.kwargs["system_prompt"] is None

    def test_canon_grounding_survives_a_profile_with_no_overlay_key(
        self, plugin_env, mocker: MockerFixture
    ):
        """The media profiles set overlay_setting=None, and None is not a name.

        Limnoria's registryValue calls registry.split(name), which raises
        TypeError on None — so handing a profile's None overlay_setting
        straight to registryValue turns every canon-referencing @draw in a
        verse channel into an error reply instead of a picture. The plain
        MagicMock every other test uses swallows that (it accepts None
        happily), so this one enforces the real contract.
        """
        from llm.profile import PROFILE_ANIMATE, PROFILE_DRAW

        plugin, _mock_irc, _mock_msg = plugin_env
        permissive = plugin.registryValue.side_effect

        def strict(name, *args):
            if not isinstance(name, str):
                raise TypeError("expected string or bytes-like object")
            return permissive(name, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=strict)

        for profile_id in (PROFILE_DRAW, PROFILE_ANIMATE):
            overlay = plugin._verse_grounded_overlay(profile_id, "#test", "CANON: the lads")
            assert overlay == "CANON: the lads"

    def test_draw_requires_account(self, plugin_env, mocker: MockerFixture):
        """@draw still requires authenticated account."""
        plugin, mock_irc, mock_msg = plugin_env
        # nickToAccount returns None (default in plugin_env)

        plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        mock_irc.error.assert_called_once()
        plugin.llm_service.assistant_request.assert_not_called()

    def test_draw_passes_recent_context(self, plugin_env, mocker: MockerFixture):
        """@draw fetches history with the max-age window set on the context calls."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.context = mocker.MagicMock()
        plugin.context.get_messages.return_value = [{"role": "user", "content": "hi"}]
        plugin.context.get_channel_messages.return_value = []

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        get_msgs_kwargs = plugin.context.get_messages.call_args.kwargs
        assert get_msgs_kwargs["max_age_seconds"] == 60
        get_ch_kwargs = plugin.context.get_channel_messages.call_args.kwargs
        assert get_ch_kwargs["max_age_seconds"] == 60

        assistant_kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert assistant_kwargs["history"] == [{"role": "user", "content": "hi"}]
        assert assistant_kwargs["memories"] == []

    def test_draw_skips_context_when_max_age_zero(self, plugin_env, mocker: MockerFixture):
        """@draw with drawContextMaxAgeSeconds=0 skips context entirely."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        original = plugin.registryValue.side_effect

        def registry(key: str, *args: object) -> object:
            if key == "drawContextMaxAgeSeconds":
                return 0
            return original(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=registry)
        plugin.context = mocker.MagicMock()

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        plugin.context.get_messages.assert_not_called()
        plugin.context.get_channel_messages.assert_not_called()
        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert kwargs["history"] == []
        assert kwargs["channel_history"] == []

    def test_draw_replies_with_planner_content(self, plugin_env, mocker: MockerFixture):
        """GIVEN planner returns image URL WHEN draw called THEN reply contains it."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Here is your image: http://img.example/gen.png",
        )

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "http://img.example/gen.png" in reply_text

    def test_draw_logs_usage(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw with cost WHEN draw completes THEN usage is logged."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            10,
            0,
            0.04,
            prompt="a cat",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_draw_logs_usage_even_with_zero_cost(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw with zero cost/tokens WHEN draw succeeds THEN usage is still logged."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            0,
            0,
            0.0,
            prompt="test",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_draw_logs_usage_on_content_blocked(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw content blocked WHEN draw completes THEN usage logged with status=content_blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Error: content blocked",
            error="Error: content blocked",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            0,
            0,
            0.0,
            prompt="test",
            status="content_blocked",
            error_detail="Error: content blocked",
            rendered_prompt="",
        )

    def test_draw_logs_usage_on_generic_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw that errors (non-content) WHEN draw completes THEN usage logged with status=error."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Error: timeout exceeded",
            error="Error: timeout exceeded",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.db.log_usage.assert_called_once_with(
            "test_account",
            "#test",
            "draw",
            "dall-e-3",
            0,
            0,
            0.0,
            prompt="test",
            status="error",
            error_detail="Error: timeout exceeded",
            rendered_prompt="",
        )

    def test_draw_skips_znc_playback(self, plugin_env, mocker: MockerFixture):
        """GIVEN old message WHEN draw called THEN no reply."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.time = plugin.startup_time - 100

        plugin.draw(mock_irc, mock_msg, ["sunset"])

        mock_irc.reply.assert_not_called()

    def test_draw_stores_context_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw succeeds WHEN executed THEN personal and channel context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.draw(mock_irc, mock_msg, ["a", "sunset"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "a sunset"
        assert messages[1]["role"] == "assistant"

    def test_draw_does_not_store_context_on_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw returns error WHEN executed THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="Error: something went wrong",
            error="Error: something went wrong",
        )

        plugin.draw(mock_irc, mock_msg, ["bad", "prompt"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 0

    def test_draw_skips_context_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN context disabled WHEN draw succeeds THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result()

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        plugin.draw(mock_irc, mock_msg, ["sunset"])

        messages = plugin.context.get_messages("test_account", "#test")
        assert len(messages) == 0

    def test_draw_grounding_icon_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used is True WHEN draw response returned THEN reply has globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="image — http://x/img.png",
            grounding_used=True,
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310")

    def test_draw_action_with_grounding_icon(self, plugin_env, mocker: MockerFixture):
        """GIVEN /me-style draw response with grounding WHEN draw called THEN action carries globe icon."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._make_draw_result(
            content="/me sketches a sunset",
            grounding_used=True,
        )

        mock_action = mocker.patch("llm.plugin.ircmsgs.action")
        plugin.draw(mock_irc, mock_msg, ["sunset"])

        mock_action.assert_called_once_with("#test", "\U0001f310 sketches a sunset")

    def test_draw_preserves_preflight(self, plugin_env, mocker: MockerFixture):
        """GIVEN draw WHEN preflight blocks THEN assistant_request is not called."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(
            plugin,
            "_run_preflight",
            return_value=mocker.MagicMock(blocked=True),
        )

        plugin.draw(mock_irc, mock_msg, ["test"])

        plugin.llm_service.assistant_request.assert_not_called()
        mock_irc.reply.assert_not_called()


# ---------------------------------------------------------------------------
# _dispatch_assistant_reply cross-command behaviour
# ---------------------------------------------------------------------------


class TestDispatchAssistantReply:
    """Cross-command tests for the three critical behaviours in _dispatch_assistant_reply.

    These tests cover ask, code, and draw together so any regression in the
    shared helper is caught regardless of which entry point triggered it.
    """

    # ------------------------------------------------------------------
    # grounding-icon prefix — all three commands
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("command", ["ask", "code", "draw"])
    def test_grounding_icon_prefixed_consistently(self, command, plugin_env, mocker: MockerFixture):
        """GIVEN grounding_used=True and a normal text reply WHEN each command is called THEN sent line starts with GROUNDING_ICON."""
        plugin, mock_irc, mock_msg = plugin_env

        # draw requires an authenticated account
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.detect_images.return_value = []

        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="some reply",
            grounding_used=True,
            prompt_tokens=5,
            completion_tokens=3,
            cost=0.001,
            model="gpt-4",
        )

        getattr(plugin, command)(mock_irc, mock_msg, ["hello"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310"), (
            f"{command}: expected grounding icon prefix, got {reply_text!r}"
        )
        assert "some reply" in reply_text

    @pytest.mark.parametrize("command", ["ask", "draw"])
    def test_reworded_image_icon_prefixed_consistently(
        self, command, plugin_env, mocker: MockerFixture
    ):
        """GIVEN a delivered image whose prompt the rewriter changed THEN the
        reply carries the reworded icon, the same way grounding does.

        Without it the user gets a picture that may not match what they asked
        for and no hint as to why.
        """
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="https://img.example/cat.png",
            image_reworded=True,
            model="xai/grok-imagine-image",
        )

        getattr(plugin, command)(mock_irc, mock_msg, ["draw a cat"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f501"), (
            f"{command}: expected reworded icon prefix, got {reply_text!r}"
        )
        assert "https://img.example/cat.png" in reply_text

    def test_a_first_try_image_gets_no_icon(self, plugin_env, mocker: MockerFixture):
        """The icon means something only if the ordinary case is bare."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
        )

        plugin.draw(mock_irc, mock_msg, ["a cat"])

        assert mock_irc.reply.call_args[0][0] == "https://img.example/cat.png"

    def test_both_signals_chain_rather_than_choosing(self, plugin_env, mocker: MockerFixture):
        """A grounded answer that also drew a reworded image raised both."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="https://img.example/cat.png",
            grounding_used=True,
            image_reworded=True,
            model="xai/grok-imagine-image",
        )

        plugin.ask(mock_irc, mock_msg, ["what does a cat look like, draw it"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert reply_text.startswith("\U0001f310 \U0001f501")

    # ------------------------------------------------------------------
    # Reminder-mutation suppression (ask only)
    # ------------------------------------------------------------------

    def test_ask_suppresses_empty_followup_after_reminder_mutation(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN last_successful_tool ∈ _REMINDER_MUTATION_TOOLS and final_text_after_tools is empty WHEN ask called THEN no reply or error but usage IS recorded."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=20,
            completion_tokens=2,
            cost=0.0002,
            model="gpt-4",
            last_successful_tool="set_reminder",
            final_text_after_tools="",
        )

        plugin.ask(mock_irc, mock_msg, ["remind", "me", "tomorrow"])

        # No reply and no error — the emoji reaction is the user-visible ack.
        mock_irc.reply.assert_not_called()
        mock_irc.queueMsg.assert_not_called()
        mock_irc.error.assert_not_called()
        # Usage must still be logged (suppression is not a free path).
        plugin.db.log_usage.assert_called_once()

    def test_reminder_mutation_stores_context_note_not_empty_turn(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN a suppressed set_reminder ack WHEN ask called THEN the stored assistant turn is the context note, not an empty string."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            model="gpt-4",
            last_successful_tool="set_reminder",
            final_text_after_tools="",
        )
        add_message = mocker.spy(plugin.context, "add_message")

        plugin.ask(mock_irc, mock_msg, ["remind", "me", "tomorrow"])

        # An empty assistant turn reads back as an unanswered request and the
        # model sets the same reminder again on the next message.
        stored = [
            call.args[3] for call in add_message.call_args_list if call.args[2] == Role.ASSISTANT
        ]
        assert stored == ["[reminder set]"]

    def test_reminder_mutation_note_carries_what_the_tool_reported(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN the set_reminder tool reported what is pending WHEN ask called THEN the stored note repeats it, so a later revision has an antecedent."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            model="gpt-4",
            last_successful_tool="set_reminder",
            last_tool_message="Righto. (now pending: smell cheese in 60s)",
            final_text_after_tools="",
        )
        add_message = mocker.spy(plugin.context, "add_message")

        plugin.ask(mock_irc, mock_msg, ["remind", "me", "to", "smell", "cheese", "in", "60s"])

        stored = [
            call.args[3] for call in add_message.call_args_list if call.args[2] == Role.ASSISTANT
        ]
        assert stored == ["[reminder set: Righto. (now pending: smell cheese in 60s)]"]

    # ------------------------------------------------------------------
    # verse_storybook suppression — async link is the only reply
    # ------------------------------------------------------------------

    def test_verse_storybook_empty_reply_suppressed_not_errored(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN last_successful_tool == verse_storybook and empty final text /
        response WHEN _dispatch_assistant_reply runs THEN it sends nothing and
        does NOT emit the empty-response error (the background job posts the
        illustrated-page link asynchronously)."""
        plugin, mock_irc, mock_msg = plugin_env
        result = AssistantResult(
            content="",
            grounding_used=False,
            prompt_tokens=12,
            completion_tokens=0,
            cost=0.0,
            model="gpt-4",
            last_successful_tool="verse_storybook",
            final_text_after_tools="",
        )

        response, should_log = plugin._dispatch_assistant_reply(
            mock_irc,
            mock_msg,
            result,
            nick="testnick",
            channel="#test",
            response="",
        )

        # No message and crucially NO empty-response error to the channel.
        mock_irc.reply.assert_not_called()
        mock_irc.queueMsg.assert_not_called()
        mock_irc.error.assert_not_called()
        assert should_log is True

    # ------------------------------------------------------------------
    # Action rebinding — stored response carries "* botnick text"
    # ------------------------------------------------------------------

    def test_action_response_stored_with_action_prefix(self, plugin_env, mocker: MockerFixture):
        """GIVEN _extract_action returns a non-empty action text WHEN ask called THEN the rebound response stored in context starts with '* botnick'."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="/me ponders the question",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        mocker.patch("llm.plugin.ircmsgs.action")

        plugin.ask(mock_irc, mock_msg, ["think", "about", "it"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2
        assert messages[1]["role"] == "assistant"
        # The stored form must be the IRC action emote, not the /me prefix.
        assert messages[1]["content"].startswith("* testbot "), (
            f"Expected '* testbot ...', got {messages[1]['content']!r}"
        )
        assert "ponders the question" in messages[1]["content"]

    def test_action_dropped_and_log_skipped_when_closing(self, plugin_env, mocker: MockerFixture):
        """GIVEN the executor is closing WHEN _dispatch_assistant_reply emits an
        action THEN the send is serialized through _safe_queue (dropped) and the
        helper reports should_log=False so shutdown does not store/log."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin._llm_executor.shutdown()
        mocker.patch("llm.plugin.ircmsgs.action", return_value=mocker.sentinel.action_msg)

        result = AssistantResult(
            content="/me waves",
            grounding_used=False,
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="gpt-4",
        )

        response, should_log = plugin._dispatch_assistant_reply(
            mock_irc,
            mock_msg,
            result,
            nick="testnick",
            channel="#test",
            response="/me waves",
        )

        mock_irc.queueMsg.assert_not_called()
        assert should_log is False


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


class TestForgetCommand:
    """Tests for the real LLM.forget method."""

    def test_forget_clears_existing_context(self, plugin_env):
        """GIVEN user has context WHEN forget called THEN context cleared and confirmed."""
        plugin, mock_irc, mock_msg = plugin_env

        # Pre-populate context
        plugin.context.add_message("testnick", "#test", "user", "hi")
        plugin.context.add_message("testnick", "#test", "assistant", "hello")

        plugin.forget(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cleared" in reply_text.lower() or "fresh" in reply_text.lower()

        # Context should be empty
        assert len(plugin.context.get_messages("testnick", "#test")) == 0

    def test_forget_clears_shared_channel_context(self, plugin_env):
        """GIVEN channel context has stale bot text WHEN forget called THEN channel context cleared."""
        plugin, mock_irc, mock_msg = plugin_env

        stale = "Search down again. No nefarious2 release news since 2026-05-02 check."
        plugin.context.add_channel_message("#test", "testnick", "user", "search latest nefarious")
        plugin.context.add_channel_message("#test", "testbot", "assistant", stale)

        plugin.forget(mock_irc, mock_msg, [])

        assert plugin.context.get_channel_messages("#test") == []

    def test_forget_reports_cleared_even_when_empty(self, plugin_env):
        """GIVEN user has no context WHEN forget called THEN still says cleared."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.forget(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cleared" in reply_text.lower()

    def test_forget_defaults_to_current_channel(self, plugin_env):
        """GIVEN user in a channel WHEN forget called without args THEN clears current channel."""
        plugin, mock_irc, mock_msg = plugin_env

        # Add context for the current channel (#test from fixture)
        plugin.context.add_message("testnick", "#test", "user", "hello")
        plugin.context.add_message("testnick", "#test", "assistant", "hi")
        assert len(plugin.context.get_messages("testnick", "#test")) == 2

        # Also add context for another channel (should not be affected)
        plugin.context.add_message("testnick", "#other", "user", "keep me")

        plugin.forget(mock_irc, mock_msg, [])

        # Current channel context should be cleared
        assert len(plugin.context.get_messages("testnick", "#test")) == 0
        # Other channel context should be untouched
        assert len(plugin.context.get_messages("testnick", "#other")) == 1


# ---------------------------------------------------------------------------
# usage
# ---------------------------------------------------------------------------


class TestUsageExecutorField:
    """Verify the global %usage output exposes executor running/queued/max."""

    def test_global_usage_includes_executor(self, plugin_env, mocker: MockerFixture) -> None:
        plugin, irc, msg = plugin_env
        msg.channel = None
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm.") or cap == "admin",
        )
        plugin.db.get_usage_summary.return_value = UsageSummary(
            total_requests=0, total_prompt_tokens=0, total_completion_tokens=0, total_cost=0.0
        )
        plugin.db.get_usage_by_nick.return_value = []
        plugin.db.get_usage_by_channel.return_value = []
        plugin.usage(irc, msg, [])
        replies = " ".join(str(call.args[0]) for call in irc.reply.call_args_list)
        # running/queued/max — at construction time both are 0.
        assert "executor: 0/0/16" in replies

    def test_global_usage_executor_field_under_load(
        self, plugin_env, mocker: MockerFixture
    ) -> None:
        """Field reflects actual executor counters."""
        import threading
        import time

        plugin, irc, msg = plugin_env
        msg.channel = None
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm.") or cap == "admin",
        )
        plugin.db.get_usage_summary.return_value = UsageSummary(
            total_requests=0, total_prompt_tokens=0, total_completion_tokens=0, total_cost=0.0
        )
        plugin.db.get_usage_by_nick.return_value = []
        plugin.db.get_usage_by_channel.return_value = []

        release = threading.Event()
        plugin._llm_executor.submit("hold", release.wait, 5)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and plugin._llm_executor.running() < 1:
            time.sleep(0.02)
        try:
            plugin.usage(irc, msg, [])
            replies = " ".join(str(call.args[0]) for call in irc.reply.call_args_list)
            assert "executor: 1/0/16" in replies
        finally:
            release.set()


class TestUsageCommand:
    """Tests for the real LLM.usage method (dual-mode: channel + PM)."""

    # -- PM mode (global stats, admin only) --

    def test_usage_pm_shows_today_and_month_stats(self, plugin_env, mocker: MockerFixture):
        """GIVEN admin via PM WHEN usage called THEN response includes today and monthly stats."""
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm.") or cap == "admin",
        )
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary.return_value = UsageSummary(
            total_requests=10,
            total_prompt_tokens=1000,
            total_completion_tokens=500,
            total_cost=0.05,
        )
        plugin.db.get_usage_by_nick.return_value = [
            UsageBreakdown(
                name="testnick",
                total_requests=5,
                total_prompt_tokens=500,
                total_completion_tokens=250,
                total_cost=0.03,
            )
        ]
        plugin.db.get_usage_by_channel.return_value = [
            UsageBreakdown(
                name="#test",
                total_requests=10,
                total_prompt_tokens=1000,
                total_completion_tokens=500,
                total_cost=0.05,
            )
        ]

        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Today:" in reply_text
        assert "This month:" in reply_text
        assert "Top users:" in reply_text
        assert "Top channels:" in reply_text
        assert "Context:" in reply_text
        assert "conversations" in reply_text
        # Sent privately
        assert mock_irc.reply.call_args.kwargs.get("private") is True

    def test_usage_pm_with_no_top_users_or_channels(self, plugin_env, mocker: MockerFixture):
        """GIVEN no usage data via PM WHEN usage called THEN response omits top users/channels."""
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm.") or cap == "admin",
        )
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary.return_value = UsageSummary(
            total_requests=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_cost=0.0,
        )
        plugin.db.get_usage_by_nick.return_value = []
        plugin.db.get_usage_by_channel.return_value = []

        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Top users:" not in reply_text
        assert "Top channels:" not in reply_text

    def test_usage_pm_requires_admin(self, plugin_env, mocker: MockerFixture):
        """GIVEN non-admin via PM WHEN usage called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "admin" in error_text.lower()

    # -- Channel mode (personal + channel stats, any user) --

    def test_usage_channel_shows_channel_and_personal_stats(self, plugin_env):
        """GIVEN user in channel WHEN usage called THEN shows channel and personal stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        # msg.channel is already "#test" from fixture
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=45,
            total_prompt_tokens=5000,
            total_completion_tokens=2500,
            total_cost=0.0292,
        )
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=12,
            total_prompt_tokens=1200,
            total_completion_tokens=600,
            total_cost=0.0139,
        )
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=5)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=8)

        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        # Channel stats
        assert "#test this month:" in reply_text
        assert "45 requests" in reply_text
        assert "rank 1/5 channels" in reply_text
        # Personal stats
        assert "You:" in reply_text
        assert "12 requests" in reply_text
        assert "rank 1/8 users" in reply_text
        # Context info
        assert "Context:" in reply_text
        # Not sent privately
        assert mock_irc.reply.call_args.kwargs.get("private") is not True
        assert mock_irc.reply.call_args.kwargs.get("prefixNick") is False

    def test_usage_channel_works_without_admin(self, plugin_env, mocker: MockerFixture):
        """GIVEN non-admin in channel WHEN usage called THEN stats shown (no error)."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=0, total=0)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=0, total=0)

        # No admin capability needed — should still work
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        plugin.usage(mock_irc, mock_msg, [])

        mock_irc.reply.assert_called_once()
        mock_irc.error.assert_not_called()

    def test_usage_channel_zero_data_graceful(self, plugin_env):
        """GIVEN no usage data WHEN usage called in channel THEN shows zeros without rank."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=0, total=0)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=0, total=0)

        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#test this month: $0.0000" in reply_text
        assert "You: $0.0000" in reply_text
        # rank should not appear when rank=0
        assert "rank" not in reply_text
        # Context should show empty when no messages
        assert "Context: empty" in reply_text

    def test_usage_channel_shows_context_message_count(self, plugin_env):
        """GIVEN user with active context WHEN usage called THEN shows message count and expiry."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(10, 500, 250, 0.01)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(5, 250, 125, 0.005)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=3)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        # Add some messages to the user's context
        plugin.context.add_message("testnick", "#test", "user", "Hello")
        plugin.context.add_message("testnick", "#test", "assistant", "Hi there")

        plugin.usage(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Context: 2/" in reply_text
        assert "msgs" in reply_text
        assert "expires in" in reply_text

    # -- Target nick mode --

    def test_usage_strips_irc_status_prefix_from_nick(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick with @ prefix WHEN usage called THEN prefix stripped before lookup."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(7, 800, 400, 0.01)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        plugin.usage(mock_irc, mock_msg, ["@Larry"])

        # Should query for "Larry", not "@Larry"
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "Larry"
        assert "Larry" in mock_irc.reply.call_args[0][0]

    def test_usage_handles_nick_with_brackets(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick with brackets WHEN usage called THEN raw arg parsed correctly."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(3, 300, 150, 0.005)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=4)

        plugin.usage(mock_irc, mock_msg, ["Rubin[F]"])

        # Should query DB with the full bracket nick
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "Rubin[F]"
        assert "Rubin[F]" in mock_irc.reply.call_args[0][0]

    def test_usage_resolves_target_nick_to_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN target nick with NickServ account WHEN usage called THEN queries by account."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        # Target nick "OldNick" resolves to account "RealAccount"
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="RealAccount")
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(5, 500, 250, 0.01)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=2, total=6)

        plugin.usage(mock_irc, mock_msg, ["OldNick"])

        # DB should be queried with the account name
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "RealAccount"
        # But display should still show the original nick typed
        reply_text = mock_irc.reply.call_args[0][0]
        assert "OldNick" in reply_text

    def test_usage_with_nick_in_channel(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick target in channel WHEN usage called THEN shows that nick's channel stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=7,
            total_prompt_tokens=800,
            total_completion_tokens=400,
            total_cost=0.0100,
        )
        plugin.db.get_nick_rank.return_value = UsageRank(rank=3, total=10)

        plugin.usage(mock_irc, mock_msg, ["othernick"])

        mock_irc.reply.assert_called_once()
        reply_text = mock_irc.reply.call_args[0][0]
        assert "othernick" in reply_text
        assert "in #test" in reply_text
        assert "7 requests" in reply_text
        assert "rank 3/10 users" in reply_text
        # Scoped to current channel
        plugin.db.get_usage_summary_for_nick.assert_called_once()
        call_kwargs = plugin.db.get_usage_summary_for_nick.call_args
        assert call_kwargs[0][0] == "othernick"
        assert call_kwargs[1]["channel"] == "#test"

    def test_usage_with_nick_via_pm(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick target via PM WHEN usage called THEN shows that nick's global stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(
            total_requests=20,
            total_prompt_tokens=2000,
            total_completion_tokens=1000,
            total_cost=0.0500,
        )
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        plugin.usage(mock_irc, mock_msg, ["othernick"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "othernick" in reply_text
        assert "in #" not in reply_text  # no channel scope
        assert "20 requests" in reply_text

    # -- Target channel mode --

    def test_usage_with_channel_target(self, plugin_env, mocker: MockerFixture):
        """GIVEN channel target WHEN usage called THEN shows that channel's stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=100,
            total_prompt_tokens=10000,
            total_completion_tokens=5000,
            total_cost=0.1234,
        )
        plugin.db.get_channel_rank.return_value = UsageRank(rank=2, total=8)

        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, ["#other"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#other this month:" in reply_text
        assert "100 requests" in reply_text
        assert "rank 2/8 channels" in reply_text
        plugin.db.get_usage_summary_for_channel.assert_called_once()
        assert plugin.db.get_usage_summary_for_channel.call_args[0][0] == "#other"

    def test_usage_with_channel_target_via_pm(self, plugin_env, mocker: MockerFixture):
        """GIVEN channel target via PM WHEN usage called THEN shows channel stats."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        mock_msg.channel = None  # PM mode
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(
            total_requests=50,
            total_prompt_tokens=5000,
            total_completion_tokens=2500,
            total_cost=0.0750,
        )
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=3)

        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, ["#somechan"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#somechan this month:" in reply_text
        assert "50 requests" in reply_text


# ---------------------------------------------------------------------------
# remind
# ---------------------------------------------------------------------------


class TestRemindSetCommand:
    """Tests for the real LLM.remind method (set subcommand)."""

    def test_remind_schedules_reminder_on_success(self, plugin_env, mocker: MockerFixture):
        """GIVEN valid reminder WHEN remind called THEN reminder is scheduled and persisted."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=1800,
            message="check the build",
            confirmation="Reminder set for 30 minutes from now.",
            note=None,
        )

        mock_add_event = mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remind(mock_irc, mock_msg, ["in 30m check the build"])

        # Should schedule the event
        mock_add_event.assert_called_once()
        # Should persist to database
        plugin.db.save_reminder.assert_called_once()
        # Should reply with confirmation
        reply_text = mock_irc.reply.call_args[0][0]
        assert "Reminder set" in reply_text

    def test_remind_includes_note_in_reply(self, plugin_env, mocker: MockerFixture):
        """GIVEN reminder with timezone note WHEN remind called THEN reply includes note."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="meeting",
            confirmation="Reminder set for 1 hour from now.",
            note="Assuming UTC timezone",
        )

        mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remind(mock_irc, mock_msg, ["in 1h meeting"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Assuming UTC timezone" in reply_text

    def test_remind_handles_clarification(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse returns clarify WHEN remind called THEN asks for clarification."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="clarify",
            confirmation="When should I remind you?",
        )

        plugin.remind(mock_irc, mock_msg, ["something vague"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "When should I remind you?" in reply_text

    def test_remind_rejects_too_short_duration(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration < 10 seconds WHEN remind called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=5,
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["in 5s test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "10 seconds" in error_text

    def test_remind_rejects_too_long_duration(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration > 7 days WHEN remind called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=604801,  # >7 days
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["in 8d test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "7 days" in error_text

    def test_remind_rejects_none_seconds(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse result has seconds=None WHEN remind called THEN error is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=None,
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["test"])

        reply_text = mock_irc.error.call_args[0][0]
        assert "could not determine" in reply_text.lower()

    def test_remind_handles_schedule_failure(self, plugin_env, mocker: MockerFixture):
        """GIVEN schedule.addEvent raises WHEN remind called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message="test",
            confirmation="ok",
        )

        mocker.patch("llm.plugin.schedule.addEvent", side_effect=RuntimeError("scheduler broke"))
        plugin.remind(mock_irc, mock_msg, ["in 1m test"])

        mock_irc.error.assert_called_once()


# ---------------------------------------------------------------------------
# remind list
# ---------------------------------------------------------------------------


class TestRemindListCommand:
    """Tests for the real LLM.remind method (list subcommand)."""

    def test_remind_list_shows_pending_reminders(self, plugin_env):
        """GIVEN user has reminders WHEN remind list called THEN formatted list is shown."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="check build",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="testnick",
                channel="#test",
                message="call Bob",
            )

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#1:" in reply_text or "#2:" in reply_text
        assert "check build" in reply_text
        assert "call Bob" in reply_text

    def test_remind_list_shows_no_pending_message(self, plugin_env):
        """GIVEN user has no reminders WHEN remind list called THEN reports none."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no pending" in reply_text.lower()

    def test_remind_no_args_defaults_to_list(self, plugin_env):
        """GIVEN no arguments WHEN remind called THEN defaults to list."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no pending" in reply_text.lower()

    def test_remind_list_only_shows_own_reminders(self, plugin_env):
        """GIVEN reminders from different users WHEN remind list called THEN only shows own."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="my reminder",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="otheruser",
                channel="#test",
                message="not mine",
            )

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "my reminder" in reply_text
        assert "not mine" not in reply_text


# ---------------------------------------------------------------------------
# remind delete
# ---------------------------------------------------------------------------


class TestRemindDeleteCommand:
    """Tests for the real LLM.remind method (delete subcommand)."""

    def test_remind_delete_cancels_own_reminder(self, plugin_env, mocker: MockerFixture):
        """GIVEN user owns a reminder WHEN remind delete called THEN reminder is cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_42"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = make_reminder_row(
                event_name=event_name,
                nick="testnick",
                channel="#test",
                message="my reminder",
            )

        mock_remove = mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["delete 42"])

        # Should remove from schedule
        mock_remove.assert_called_once_with(event_name)
        # Should remove from internal dict
        assert event_name not in plugin._reminders
        # Should delete from database
        plugin.db.delete_reminder.assert_called_once_with(event_name)
        # Should confirm
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()

    def test_remind_delete_rejects_nonexistent_reminder(self, plugin_env):
        """GIVEN no matching reminder WHEN remind delete called THEN error is reported."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, ["delete 999"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "no matching" in error_text.lower()

    def test_remind_delete_rejects_other_users_reminder(self, plugin_env):
        """GIVEN reminder owned by another user WHEN remind delete called THEN error."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_5"] = make_reminder_row(
                event_name="llm_remind_100_5",
                nick="otheruser",
                channel="#test",
                message="their reminder",
            )

        plugin.remind(mock_irc, mock_msg, ["delete 5"])

        mock_irc.error.assert_called_once()

    def test_remind_delete_handles_missing_schedule_event_gracefully(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN reminder exists but schedule event is gone WHEN remind delete THEN no crash."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_7"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = make_reminder_row(
                event_name=event_name,
                nick="testnick",
                channel="#test",
                message="my reminder",
            )

        mocker.patch("llm.plugin.schedule.removeEvent", side_effect=KeyError("gone"))
        plugin.remind(mock_irc, mock_msg, ["delete 7"])

        # Still confirmed
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()

    def test_remind_del_shorthand_works(self, plugin_env, mocker: MockerFixture):
        """GIVEN user owns a reminder WHEN remind del called THEN reminder is cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        event_name = "llm_remind_100_42"
        with plugin._reminders_lock:
            plugin._reminders[event_name] = make_reminder_row(
                event_name=event_name,
                nick="testnick",
                channel="#test",
                message="my reminder",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["del 42"])

        assert event_name not in plugin._reminders
        reply_text = mock_irc.reply.call_args[0][0]
        assert "cancelled" in reply_text.lower()

    def test_remind_delete_multiple_ids(self, plugin_env, mocker: MockerFixture):
        """GIVEN user owns multiple reminders WHEN remind delete with multiple IDs THEN all cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="first",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="testnick",
                channel="#test",
                message="second",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["delete 1 2"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "2 reminders" in reply_text.lower()


# ---------------------------------------------------------------------------
# remind clear
# ---------------------------------------------------------------------------


class TestRemindClearCommand:
    """Tests for the real LLM.remind method (clear subcommand)."""

    def test_remind_clear_removes_all_own_reminders(self, plugin_env, mocker: MockerFixture):
        """GIVEN user has reminders WHEN remind clear called THEN all are removed."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="first",
            )
            plugin._reminders["llm_remind_100_2"] = make_reminder_row(
                event_name="llm_remind_100_2",
                nick="testnick",
                channel="#test",
                message="second",
            )
            plugin._reminders["llm_remind_100_3"] = make_reminder_row(
                event_name="llm_remind_100_3",
                nick="otheruser",
                channel="#test",
                message="not mine",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["clear"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "2 reminders" in reply_text.lower()
        # Other user's reminder should remain
        assert "llm_remind_100_3" in plugin._reminders

    def test_remind_clear_reports_no_reminders(self, plugin_env):
        """GIVEN user has no reminders WHEN remind clear called THEN reports none."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, ["clear"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no reminders" in reply_text.lower()

    def test_remind_clear_singular_label(self, plugin_env, mocker: MockerFixture):
        """GIVEN user has one reminder WHEN remind clear called THEN uses singular label."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="only one",
            )

        mocker.patch("llm.plugin.schedule.removeEvent")
        plugin.remind(mock_irc, mock_msg, ["clear"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "1 reminder." in reply_text


# ---------------------------------------------------------------------------
# remind: the caller's own scheduled LLM tasks
# ---------------------------------------------------------------------------


def _fake_task(mocker, event_name: str, prompt: str, creator: str = "testnick"):
    """A real ScheduledLlmTaskRow.

    Deliberately not a MagicMock: a mock answers to any attribute, so a typo
    in a field name the command reads would pass silently.
    """
    from llm.persistence import ScheduledLlmTaskRow

    return ScheduledLlmTaskRow(
        id=1,
        event_name=event_name,
        creator_nick=creator,
        account=creator.upper(),
        channel="#test",
        network="testnet",
        wire_msg="",
        prompt=prompt,
        fire_at=time.time() + 3600,
        created_at=time.time(),
        recurrence_seconds=None,
        recurrence_rrule=None,
        chain_position=1,
        watch_mode=False,
    )


class TestRemindUserScheduledTasks:
    """``@remind`` reaches the caller's own scheduled tasks, not just reminders.

    Before this, the three pending-task tools were hidden from the chat
    profile and ``@remind`` read only the reminders dict, so a user could
    create a scheduled task in plain language and had no way to remove it.
    """

    def test_list_includes_own_scheduled_tasks(self, plugin_env, mocker: MockerFixture):
        """GIVEN a scheduled task WHEN remind list THEN it appears with a [task] marker."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.list_scheduled_llm_tasks.return_value = [
            _fake_task(mocker, "llm_task_abc123", "check my open PRs"),
        ]

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#abc123:" in reply_text
        assert "check my open PRs" in reply_text
        assert "[task]" in reply_text
        # The lookup must be scoped to the caller, or list leaks other users'
        # tasks. Without this a hardcoded/empty owner passes every other case.
        plugin.llm_service.list_scheduled_llm_tasks.assert_called_once_with(
            creator_nick="testnick", account=None
        )

    def test_list_truncates_a_long_task_prompt(self, plugin_env, mocker: MockerFixture):
        """GIVEN a long prompt WHEN remind list THEN it is cut at 40 chars."""
        plugin, mock_irc, mock_msg = plugin_env
        prompt = "summarise every open pull request and say which ones are stale"
        plugin.llm_service.list_scheduled_llm_tasks.return_value = [
            _fake_task(mocker, "llm_task_abc123", prompt),
        ]

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert prompt[:40] + "..." in reply_text
        assert "are stale" not in reply_text

    def test_identified_caller_forwards_the_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN an identified caller WHEN remind list/del THEN the account is used.

        Scheduled tasks are account-owned, so the account is the half of the
        identity that decides ownership; every other test here runs
        unauthenticated.
        """
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="TESTACC")
        plugin.llm_service.cancel_scheduled_llm_task.return_value = mocker.MagicMock(status="ok")

        plugin.remind(mock_irc, mock_msg, ["list"])
        plugin.llm_service.list_scheduled_llm_tasks.assert_called_once_with(
            creator_nick="testnick", account="TESTACC"
        )

        plugin.remind(mock_irc, mock_msg, ["del abc123"])
        assert plugin.llm_service.cancel_scheduled_llm_task.call_args.kwargs["account"] == "TESTACC"

    def test_list_shows_reminders_and_tasks_together(self, plugin_env, mocker: MockerFixture):
        """GIVEN both kinds WHEN remind list THEN both are listed in one reply."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="check build",
            )
        plugin.llm_service.list_scheduled_llm_tasks.return_value = [
            _fake_task(mocker, "llm_task_deadbeef", "summarise the news"),
        ]

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "check build" in reply_text
        assert "summarise the news" in reply_text

    def test_list_empty_mentions_scheduled_tasks(self, plugin_env):
        """GIVEN nothing pending WHEN remind list THEN the reply covers both kinds."""
        plugin, mock_irc, mock_msg = plugin_env

        plugin.remind(mock_irc, mock_msg, ["list"])

        reply_text = mock_irc.reply.call_args[0][0].lower()
        assert "no pending reminders" in reply_text
        assert "scheduled task" in reply_text

    def test_delete_cancels_task_by_bare_id(self, plugin_env, mocker: MockerFixture):
        """GIVEN the id shown by list WHEN remind del THEN the task is cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.cancel_scheduled_llm_task.return_value = mocker.MagicMock(status="ok")

        plugin.remind(mock_irc, mock_msg, ["del abc123"])

        plugin.llm_service.cancel_scheduled_llm_task.assert_called_once()
        kwargs = plugin.llm_service.cancel_scheduled_llm_task.call_args.kwargs
        assert kwargs["event_name"] == "llm_task_abc123"
        assert kwargs["creator_nick"] == "testnick"
        assert "1 scheduled task" in mock_irc.reply.call_args[0][0]

    def test_delete_accepts_full_event_name(self, plugin_env, mocker: MockerFixture):
        """GIVEN the full llm_task_ name WHEN remind del THEN it is not double-prefixed."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.cancel_scheduled_llm_task.return_value = mocker.MagicMock(status="ok")

        plugin.remind(mock_irc, mock_msg, ["del llm_task_abc123"])

        kwargs = plugin.llm_service.cancel_scheduled_llm_task.call_args.kwargs
        assert kwargs["event_name"] == "llm_task_abc123"

    def test_delete_reports_someone_elses_task_as_no_match(self, plugin_env, mocker: MockerFixture):
        """GIVEN the service refuses on ownership WHEN remind del THEN the user gets an error."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.cancel_scheduled_llm_task.return_value = mocker.MagicMock(
            status="error", message="belongs to someone else"
        )

        plugin.remind(mock_irc, mock_msg, ["del abc123"])

        mock_irc.error.assert_called_once()
        assert "no matching" in mock_irc.error.call_args[0][0].lower()

    def test_delete_mixed_reminder_and_task(self, plugin_env, mocker: MockerFixture):
        """GIVEN one of each WHEN remind del with both ids THEN both are cancelled."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_7"] = make_reminder_row(
                event_name="llm_remind_100_7",
                nick="testnick",
                channel="#test",
                message="mine",
            )
        plugin.llm_service.cancel_scheduled_llm_task.return_value = mocker.MagicMock(status="ok")
        mocker.patch("llm.plugin.schedule.removeEvent")

        plugin.remind(mock_irc, mock_msg, ["del 7 abc123"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "1 reminder and 1 scheduled task" in reply_text

    def test_clear_also_cancels_scheduled_tasks(self, plugin_env, mocker: MockerFixture):
        """GIVEN a reminder and two tasks WHEN remind clear THEN all three go."""
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="mine",
            )
        plugin.llm_service.list_scheduled_llm_tasks.return_value = [
            _fake_task(mocker, "llm_task_aaa", "one"),
            _fake_task(mocker, "llm_task_bbb", "two"),
        ]
        plugin.llm_service.cancel_scheduled_llm_task.return_value = mocker.MagicMock(status="ok")
        mocker.patch("llm.plugin.schedule.removeEvent")

        plugin.remind(mock_irc, mock_msg, ["clear"])

        assert plugin.llm_service.cancel_scheduled_llm_task.call_count == 2
        cancelled = {
            c.kwargs["event_name"]
            for c in plugin.llm_service.cancel_scheduled_llm_task.call_args_list
        }
        assert cancelled == {"llm_task_aaa", "llm_task_bbb"}
        assert "1 reminder and 2 scheduled tasks" in mock_irc.reply.call_args[0][0]

    def test_remind_requires_llm_ask(self, plugin_env, mocker: MockerFixture):
        """GIVEN a caller without llm.ask WHEN remind clear THEN nothing is cancelled.

        `@remind` reaches persisted, account-owned rows, so it is gated like
        `@ask`, `@code` and `@forget`. Locks the gate in: drop the
        checkCapability converter and this goes red.
        """
        plugin, mock_irc, mock_msg = plugin_env
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_100_1"] = make_reminder_row(
                event_name="llm_remind_100_1",
                nick="testnick",
                channel="#test",
                message="mine",
            )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        mocker.patch("llm.plugin.schedule.removeEvent")

        plugin.remind(mock_irc, mock_msg, ["clear"])

        assert "llm_remind_100_1" in plugin._reminders
        plugin.db.delete_reminder.assert_not_called()
        plugin.llm_service.cancel_scheduled_llm_task.assert_not_called()

    def test_clear_reports_nothing_when_every_cancel_fails(self, plugin_env, mocker: MockerFixture):
        """GIVEN tasks that vanish mid-clear THEN the bot does not claim success.

        The listed tasks can fire between the snapshot and the cancel loop, in
        which case the service refuses every one and nothing is cleared.
        """
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.list_scheduled_llm_tasks.return_value = [
            _fake_task(mocker, "llm_task_aaa", "one"),
        ]
        plugin.llm_service.cancel_scheduled_llm_task.return_value = mocker.MagicMock(
            status="error", message="No scheduled task with id llm_task_aaa."
        )

        plugin.remind(mock_irc, mock_msg, ["clear"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "Cleared ." not in reply_text
        assert "nothing left to clear" in reply_text.lower()

    def test_clear_counts_only_tasks_the_service_cancelled(self, plugin_env, mocker: MockerFixture):
        """GIVEN one cancel fails WHEN remind clear THEN the count reflects reality."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.list_scheduled_llm_tasks.return_value = [
            _fake_task(mocker, "llm_task_aaa", "one"),
            _fake_task(mocker, "llm_task_bbb", "two"),
        ]
        plugin.llm_service.cancel_scheduled_llm_task.side_effect = [
            mocker.MagicMock(status="ok"),
            mocker.MagicMock(status="error", message="gone"),
        ]

        plugin.remind(mock_irc, mock_msg, ["clear"])

        assert "1 scheduled task" in mock_irc.reply.call_args[0][0]


# ---------------------------------------------------------------------------


class TestRemindAdminCommand:
    """Tests for the owner-only ``remind admin`` subcommand."""

    def _seed(self, plugin) -> None:
        with plugin._reminders_lock:
            plugin._reminders["llm_remind_aaaa_a1"] = make_reminder_row(
                event_name="llm_remind_aaaa_a1",
                nick="targetnick",
                channel="#test",
                message="target one",
            )
            plugin._reminders["llm_remind_aaaa_a2"] = make_reminder_row(
                event_name="llm_remind_aaaa_a2",
                nick="targetnick",
                channel="#test",
                message="target two",
                account="TargetAccount",
            )
            plugin._reminders["llm_remind_aaaa_a3"] = make_reminder_row(
                event_name="llm_remind_aaaa_a3",
                nick="someoneelse",
                channel="#test",
                message="not target",
            )

    def test_admin_requires_owner(self, plugin_env, mocker: MockerFixture):
        """Non-owner invoking remind admin gets an error and no state changes."""
        plugin, mock_irc, mock_msg = plugin_env
        self._seed(plugin)
        # Holds llm.ask (so the command's own capability gate passes) but not
        # owner — otherwise the wrap-level gate denies first and this stops
        # testing the admin check.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap, **kw: cap == "llm.ask",
        )

        plugin.remind(mock_irc, mock_msg, ["admin clear targetnick"])

        mock_irc.error.assert_called_once()
        assert "owner" in mock_irc.error.call_args[0][0].lower()
        assert "llm_remind_aaaa_a1" in plugin._reminders
        assert "llm_remind_aaaa_a2" in plugin._reminders

    def test_admin_list_shows_target_reminders(self, plugin_env, mocker: MockerFixture):
        """Owner running remind admin list <nick> sees that user's reminders."""
        plugin, mock_irc, mock_msg = plugin_env
        self._seed(plugin)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch.object(plugin.db, "load_scheduled_llm_tasks_for_target", return_value=[])

        plugin.remind(mock_irc, mock_msg, ["admin list targetnick"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "target one" in reply_text
        assert "target two" in reply_text
        assert "not target" not in reply_text

    def test_admin_list_matches_account(self, plugin_env, mocker: MockerFixture):
        """Account name resolves the same rows as the nick."""
        plugin, mock_irc, mock_msg = plugin_env
        self._seed(plugin)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch.object(plugin.db, "load_scheduled_llm_tasks_for_target", return_value=[])

        plugin.remind(mock_irc, mock_msg, ["admin list targetaccount"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "target two" in reply_text

    def test_admin_list_no_reminders(self, plugin_env, mocker: MockerFixture):
        """Reports no reminders when target has none."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch.object(plugin.db, "load_scheduled_llm_tasks_for_target", return_value=[])

        plugin.remind(mock_irc, mock_msg, ["admin list ghost"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "no pending reminders" in reply_text.lower()

    def test_admin_list_includes_scheduled_tasks(self, plugin_env, mocker: MockerFixture):
        """Owner list shows scheduled_llm_tasks alongside reminders."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        fake_task = mocker.MagicMock(
            event_name="llm_task_abc",
            prompt="be annoying every 20s",
            creator_nick="targetnick",
            account=None,
        )
        mocker.patch.object(
            plugin.db, "load_scheduled_llm_tasks_for_target", return_value=[fake_task]
        )

        plugin.remind(mock_irc, mock_msg, ["admin list targetnick"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "task:llm_task_abc" in reply_text
        assert "be annoying" in reply_text

    def test_admin_clear_removes_only_target_reminders(self, plugin_env, mocker: MockerFixture):
        """Clearing for a target leaves other users' reminders intact."""
        plugin, mock_irc, mock_msg = plugin_env
        self._seed(plugin)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch.object(plugin.db, "load_scheduled_llm_tasks_for_target", return_value=[])

        plugin.remind(mock_irc, mock_msg, ["admin clear targetnick"])

        assert "llm_remind_aaaa_a1" not in plugin._reminders
        assert "llm_remind_aaaa_a2" not in plugin._reminders
        assert "llm_remind_aaaa_a3" in plugin._reminders

    def test_admin_clear_cancels_scheduled_tasks_too(self, plugin_env, mocker: MockerFixture):
        """admin clear nukes scheduled_llm_tasks rows alongside reminders."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.schedule.removeEvent")
        fake_task = mocker.MagicMock(event_name="llm_task_xyz", account=None)
        mocker.patch.object(
            plugin.db, "load_scheduled_llm_tasks_for_target", return_value=[fake_task]
        )
        delete_mock = mocker.patch.object(plugin.db, "delete_scheduled_llm_task")

        plugin.remind(mock_irc, mock_msg, ["admin clear fc42"])

        delete_mock.assert_called_once_with("llm_task_xyz")
        reply_text = mock_irc.reply.call_args[0][0]
        assert "1 entry" in reply_text

    def test_admin_clear_no_reminders(self, plugin_env, mocker: MockerFixture):
        """Reports nothing to clear when target has none."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch.object(plugin.db, "load_scheduled_llm_tasks_for_target", return_value=[])

        plugin.remind(mock_irc, mock_msg, ["admin clear ghost"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "nothing to clear" in reply_text.lower()

    def test_admin_del_removes_specific_reminder(self, plugin_env, mocker: MockerFixture):
        """Owner deletes one reminder of a target user by ID."""
        plugin, mock_irc, mock_msg = plugin_env
        self._seed(plugin)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.schedule.removeEvent")
        mocker.patch.object(plugin.db, "get_scheduled_llm_task", return_value=None)

        plugin.remind(mock_irc, mock_msg, ["admin del targetnick a1"])

        assert "llm_remind_aaaa_a1" not in plugin._reminders
        assert "llm_remind_aaaa_a2" in plugin._reminders
        assert "llm_remind_aaaa_a3" in plugin._reminders

    def test_admin_del_scheduled_task_by_event_name(self, plugin_env, mocker: MockerFixture):
        """Owner deletes a scheduled_llm_task by full event_name."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.schedule.removeEvent")
        fake_row = mocker.MagicMock(creator_nick="fc42", account=None)
        mocker.patch.object(plugin.db, "get_scheduled_llm_task", return_value=fake_row)
        delete_mock = mocker.patch.object(plugin.db, "delete_scheduled_llm_task")

        plugin.remind(mock_irc, mock_msg, ["admin del fc42 llm_task_abc"])

        delete_mock.assert_called_once_with("llm_task_abc")

    def test_admin_del_unknown_id_errors(self, plugin_env, mocker: MockerFixture):
        """Unknown ID produces an error and leaves reminders untouched."""
        plugin, mock_irc, mock_msg = plugin_env
        self._seed(plugin)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch.object(plugin.db, "get_scheduled_llm_task", return_value=None)

        plugin.remind(mock_irc, mock_msg, ["admin del targetnick zzzz"])

        mock_irc.error.assert_called_once()
        assert "llm_remind_aaaa_a1" in plugin._reminders

    def test_admin_usage_error_when_args_missing(self, plugin_env, mocker: MockerFixture):
        """Bare ``remind admin`` returns a usage error."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        plugin.remind(mock_irc, mock_msg, ["admin"])

        mock_irc.error.assert_called_once()
        assert "usage" in mock_irc.error.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# Account-based identity
# ---------------------------------------------------------------------------


class TestAccountBasedIdentity:
    """Tests for NickServ account-based identity resolution across commands."""

    @pytest.fixture
    def account_env(self, plugin_env, mocker: MockerFixture):
        """Extend plugin_env so the calling user has a NickServ account."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="MyAccount")
        return plugin, mock_irc, mock_msg

    # -- Usage logging under account --

    def test_ask_logs_usage_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN ask completes THEN usage logged under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "MyAccount",
            "#test",
            "ask",
            "gpt-4",
            10,
            5,
            0.001,
            prompt="hello",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_code_logs_usage_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN code completes THEN usage logged under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="x = 1",
            prompt_tokens=50,
            completion_tokens=20,
            cost=0.003,
            model="gpt-4",
        )

        plugin.code(mock_irc, mock_msg, ["assign"])

        plugin.db.log_usage.assert_called_once_with(
            "MyAccount",
            "#test",
            "code",
            "gpt-4",
            50,
            20,
            0.003,
            prompt="assign",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_draw_logs_usage_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN draw completes THEN usage logged under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.04,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["a", "cat"])

        plugin.db.log_usage.assert_called_once_with(
            "MyAccount",
            "#test",
            "draw",
            "dall-e-3",
            10,
            0,
            0.04,
            prompt="a cat",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    # -- Context storage under account --

    def test_ask_stores_context_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN ask completes THEN context stored under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="response",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        # Context keyed by account, not nick
        messages = plugin.context.get_messages("MyAccount", "#test")
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

        # No context under the raw nick
        assert len(plugin.context.get_messages("testnick", "#test")) == 0

    def test_draw_stores_context_under_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with NickServ account WHEN draw completes THEN context stored under account."""
        plugin, mock_irc, mock_msg = account_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.draw(mock_irc, mock_msg, ["sunset"])

        messages = plugin.context.get_messages("MyAccount", "#test")
        assert len(messages) == 2
        assert len(plugin.context.get_messages("testnick", "#test")) == 0

    # -- Usage query resolves calling user to account --

    def test_usage_channel_resolves_caller_to_account(self, account_env, mocker: MockerFixture):
        """GIVEN user with account WHEN usage called in channel THEN queries by account."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = account_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(10, 1000, 500, 0.01)
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(5, 500, 250, 0.005)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=1, total=3)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=1, total=5)

        plugin.usage(mock_irc, mock_msg, [])

        # Personal stats should query by account name
        plugin.db.get_usage_summary_for_nick.assert_called_once()
        assert plugin.db.get_usage_summary_for_nick.call_args[0][0] == "MyAccount"

    # -- Fallback when no account --

    def test_ask_falls_back_to_nick_when_no_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN user without NickServ account WHEN ask completes THEN usage logged under nick."""
        plugin, mock_irc, mock_msg = plugin_env
        # nickToAccount returns None (default in plugin_env fixture)
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "ask",
            "gpt-4",
            10,
            5,
            0.001,
            prompt="hello",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_ask_falls_back_to_nick_on_keyerror(self, plugin_env, mocker: MockerFixture):
        """GIVEN nickToAccount raises KeyError WHEN ask completes THEN usage logged under nick."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(side_effect=KeyError("unknown"))
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="ok",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        plugin.db.log_usage.assert_called_once_with(
            "testnick",
            "#test",
            "ask",
            "gpt-4",
            10,
            5,
            0.001,
            prompt="hello",
            status="success",
            error_detail="",
            rendered_prompt="",
        )

    def test_context_stored_under_nick_when_no_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN user without account WHEN ask completes THEN context stored under nick."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="response",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 2

    def test_forget_clears_context_under_account(self, account_env):
        """GIVEN user with account has context WHEN forget called THEN account context cleared."""
        plugin, mock_irc, mock_msg = account_env

        # Pre-populate context under account name
        plugin.context.add_message("MyAccount", "#test", "user", "hi")
        plugin.context.add_message("MyAccount", "#test", "assistant", "hello")

        plugin.forget(mock_irc, mock_msg, [])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "cleared" in reply_text.lower() or "fresh" in reply_text.lower()
        assert len(plugin.context.get_messages("MyAccount", "#test")) == 0

    # -- Lazy nick→account migration --

    def test_migrate_called_when_nick_differs_from_account(self, account_env):
        """GIVEN nick != account WHEN identity resolved THEN migrate_nick called."""
        plugin, mock_irc, _ = account_env
        plugin.db.migrate_nick.return_value = 3

        identity = plugin._resolve_nick_to_identity(mock_irc, "testnick")

        assert identity == "MyAccount"
        plugin.db.migrate_nick.assert_called_once_with("testnick", "MyAccount")

    def test_migrate_not_called_when_nick_matches_account(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick == account (case-insensitive) WHEN resolved THEN no migration."""
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value="testnick")

        identity = plugin._resolve_nick_to_identity(mock_irc, "testnick")

        assert identity == "testnick"
        plugin.db.migrate_nick.assert_not_called()

    def test_migrate_called_only_once_per_nick(self, account_env):
        """GIVEN nick already migrated WHEN resolved again THEN no second DB call."""
        plugin, mock_irc, _ = account_env
        plugin.db.migrate_nick.return_value = 0

        plugin._resolve_nick_to_identity(mock_irc, "testnick")
        plugin._resolve_nick_to_identity(mock_irc, "testnick")

        plugin.db.migrate_nick.assert_called_once()

    def test_migrate_not_called_when_no_account(self, plugin_env):
        """GIVEN nickToAccount returns None WHEN resolved THEN no migration."""
        plugin, mock_irc, _ = plugin_env

        identity = plugin._resolve_nick_to_identity(mock_irc, "testnick")

        assert identity == "testnick"
        plugin.db.migrate_nick.assert_not_called()


# ---------------------------------------------------------------------------
# Additional coverage: edge cases for usage, remind, code
# ---------------------------------------------------------------------------


class TestUsageEdgeCases:
    """Additional edge-case tests for usage command flows."""

    def test_usage_for_nick_with_zero_rank(self, plugin_env, mocker: MockerFixture):
        """GIVEN nick target with no usage WHEN usage called THEN rank is omitted."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_nick.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_nick_rank.return_value = UsageRank(rank=0, total=5)

        plugin.usage(mock_irc, mock_msg, ["somenick"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "somenick" in reply_text
        assert "rank" not in reply_text

    def test_usage_for_channel_with_zero_rank(self, plugin_env, mocker: MockerFixture):
        """GIVEN channel target with no usage WHEN usage called THEN rank is omitted."""
        from llm.persistence import UsageRank

        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_usage_summary_for_channel.return_value = UsageSummary(0, 0, 0, 0.0)
        plugin.db.get_channel_rank.return_value = UsageRank(rank=0, total=0)

        mocker.patch("llm.plugin.ircutils.isChannel", return_value=True)
        plugin.usage(mock_irc, mock_msg, ["#empty"])

        reply_text = mock_irc.reply.call_args[0][0]
        assert "#empty this month:" in reply_text
        assert "rank" not in reply_text


class TestRemindEdgeCases:
    """Additional edge-case tests for remind command."""

    def test_remind_rejects_negative_seconds_via_min_check(self, plugin_env, mocker: MockerFixture):
        """GIVEN duration < 0 WHEN remind called THEN caught by the <10s check."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=-100,
            message="test",
            confirmation="ok",
        )

        plugin.remind(mock_irc, mock_msg, ["yesterday test"])

        mock_irc.error.assert_called_once()
        error_text = mock_irc.error.call_args[0][0]
        assert "10 seconds" in error_text

    def test_remind_uses_input_text_when_no_message(self, plugin_env, mocker: MockerFixture):
        """GIVEN parse result with no message WHEN remind called THEN uses original text."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message=None,  # No message extracted
            confirmation="Reminder set for 1 minute.",
        )

        mocker.patch("llm.plugin.schedule.addEvent")
        plugin.remind(mock_irc, mock_msg, ["in 1m something"])

        # The save_reminder call should use the original text as fallback
        save_call = plugin.db.save_reminder.call_args
        assert save_call[0][3] == "in 1m something"  # message arg = original text


class TestReloadRemindersEdgeCases:
    """Tests for _reload_reminders error handling."""

    def test_reload_reminders_handles_schedule_failure(self, plugin_env, mocker: MockerFixture):
        """GIVEN future reminder WHEN schedule.addEvent fails THEN reminder cleaned from DB."""
        import time as time_module

        from llm.persistence import ReminderRow

        plugin, mock_irc, mock_msg = plugin_env
        future_time = time_module.time() + 3600
        created = time_module.time()
        reminder = ReminderRow(
            id=1,
            event_name="llm_remind_broken_1",
            nick="testuser",
            channel="#test",
            message="test",
            action_prompt="",
            account=None,
            fire_at=future_time,
            created_at=created,
            chain_position=1,
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        )

        plugin.db.load_pending_reminders.return_value = [reminder]

        mocker.patch("llm.plugin.schedule.addEvent", side_effect=RuntimeError("scheduler broke"))
        plugin._reload_reminders(mock_irc)

        # Failed reminder should be deleted from DB
        plugin.db.delete_reminder.assert_called_with("llm_remind_broken_1")
        # Should NOT be stored in _reminders dict
        assert "llm_remind_broken_1" not in plugin._reminders


class TestCodeEdgeCases:
    """Additional edge-case tests for code command."""

    def test_code_skips_context_when_disabled(self, plugin_env, mocker: MockerFixture):
        """GIVEN context disabled WHEN code called THEN no context stored."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="print('hi')",
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"contextEnabled": False})
        )

        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()
        messages = plugin.context.get_messages("testnick", "#test")
        assert len(messages) == 0

    def test_code_empty_response_triggers_error(self, plugin_env, mocker: MockerFixture):
        """GIVEN planner returns empty content WHEN code called THEN irc.error is sent."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.0,
            model="gpt-4",
        )

        plugin.code(mock_irc, mock_msg, ["generate"])

        mock_irc.error.assert_called_once()
        mock_irc.reply.assert_not_called()


# ---------------------------------------------------------------------------
# Rate limiting: draw respects per-command rate limits
# ---------------------------------------------------------------------------


class TestResolveTier:
    """Tests for _resolve_tier user classification."""

    @pytest.mark.parametrize(
        ("granted_caps", "account", "expected"),
        [
            # Capabilities are checked most-to-least privileged and short-circuit
            # before the account lookup, so account is irrelevant for cap-based tiers.
            ({"owner"}, None, "owner"),
            ({"admin", "trusted"}, None, "admin"),  # admin implies trusted
            ({"trusted"}, None, "trusted"),
            (set(), "some_account", "registered"),  # identified, no caps
            (set(), None, "unregistered"),  # unidentified, no caps
        ],
    )
    def test_resolve_tier(self, plugin_env, mocker: MockerFixture, granted_caps, account, expected):
        """GIVEN a user's capabilities/account WHEN _resolve_tier THEN the
        most-privileged matching tier is returned."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = account
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap in granted_caps,
        )
        assert plugin._resolve_tier(mock_irc, mock_msg) == expected


class TestRateLimitIntegration:
    """Test that commands respect per-command, per-tier rate limits."""

    def test_draw_rate_limited_when_enforced(self, plugin_env, mocker: MockerFixture):
        """GIVEN rate limit exceeded and enforced WHEN draw called THEN blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"

        # Override to enforce rate limits with low threshold
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "enforceRateLimits": True,
                "drawRateLimitCount": 2,
                "drawRateLimitWindow": 60,
            }.get(key, "")
        )

        # Fill the bucket
        now = time.time()
        plugin._record_rate_limit_hit("draw", "test_account", now - 5)
        plugin._record_rate_limit_hit("draw", "test_account", now - 2)

        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]
        plugin.llm_service.assistant_request.assert_not_called()

    def test_draw_over_threshold_logs_shadow_when_not_enforced(
        self, plugin_env, mocker: MockerFixture
    ):
        """GIVEN enforce=False and over limit WHEN draw called THEN request runs and shadow log is emitted."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "enforceRateLimits": False,
                "drawRateLimitCount": 1,
                "drawRateLimitWindow": 60,
                "drawContextMaxAgeSeconds": 0,
            }.get(key, "")
        )
        plugin._record_rate_limit_hit("draw", "test_account", time.time() - 2)

        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_not_called()
        plugin.llm_service.assistant_request.assert_called_once()
        assert any(
            "rate_limit_shadow" in c.args[0] for c in plugin.log.info.call_args_list if c.args
        )

    def test_ask_succeeds_under_limit(self, plugin_env, mocker: MockerFixture):
        """GIVEN user under ask rate limit WHEN ask called THEN request succeeds."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="hello",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )

        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()

    def test_code_succeeds_under_limit(self, plugin_env, mocker: MockerFixture):
        """GIVEN user under code rate limit WHEN code called THEN request succeeds."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="print('hi') — http://x/code.html",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )

        plugin.code(mock_irc, mock_msg, ["hello"])

        mock_irc.reply.assert_called_once()

    def test_owner_exempt_from_rate_limits(self, plugin_env, mocker: MockerFixture):
        """GIVEN owner user over limit WHEN draw called THEN not blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "owner_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "drawRateLimitCount": 1,
                    "drawRateLimitWindow": 60,
                }
            )
        )
        # Fill bucket way past limit
        now = time.time()
        for _ in range(10):
            plugin._record_rate_limit_hit("draw", "owner_account", now - 1)

        # Mock: user is owner
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: True,  # owner has all caps
        )
        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_not_called()
        plugin.llm_service.assistant_request.assert_called_once()

    def test_trusted_gets_relaxed_limits(self, plugin_env, mocker: MockerFixture):
        """GIVEN trusted user within trusted limit but over registered limit WHEN draw called THEN allowed."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "trusted_account"
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = AssistantResult(
            content="Here is your image: http://img.example/gen.png",
            grounding_used=False,
            prompt_tokens=5,
            completion_tokens=0,
            cost=0.02,
            model="dall-e-3",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "drawRateLimitCount": 2,  # registered: 2
                    "drawRateLimitWindow": 60,
                    "drawTrustedRateLimitCount": 10,  # trusted: 10
                    "drawTrustedRateLimitWindow": 60,
                }
            )
        )
        # 5 hits — over registered limit (2), under trusted limit (10)
        now = time.time()
        for _ in range(5):
            plugin._record_rate_limit_hit("draw", "trusted_account", now - 1)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "trusted" or cap.startswith("llm."),
        )
        plugin.draw(mock_irc, mock_msg, ["test prompt"])

        mock_irc.error.assert_not_called()
        plugin.llm_service.assistant_request.assert_called_once()

    def test_ask_rate_limited_for_unregistered(self, plugin_env, mocker: MockerFixture):
        """GIVEN unregistered user over unreg limit WHEN ask called THEN blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = None

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "askUnregRateLimitCount": 2,
                    "askUnregRateLimitWindow": 60,
                }
            )
        )
        now = time.time()
        # Use nick as bucket key for unregistered users
        nick = "testnick"
        for _ in range(3):
            plugin._record_rate_limit_hit("ask", nick, now - 1)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.error.assert_called_once()
        assert "Rate limit" in mock_irc.error.call_args[0][0]

    def test_zero_count_disables_rate_limit(self, plugin_env, mocker: MockerFixture):
        """GIVEN trusted tier with count=0 WHEN ask called many times THEN never blocked."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "trusted_account"
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="hello",
            prompt_tokens=5,
            completion_tokens=10,
            cost=0.001,
            model="gpt-4",
        )

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect(
                {
                    "enforceRateLimits": True,
                    "askTrustedRateLimitCount": 0,  # 0 = disabled
                    "askTrustedRateLimitWindow": 60,
                }
            )
        )
        now = time.time()
        for _ in range(100):
            plugin._record_rate_limit_hit("ask", "trusted_account", now - 1)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "trusted" or cap.startswith("llm."),
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])

        mock_irc.error.assert_not_called()
        mock_irc.reply.assert_called_once()


# ---------------------------------------------------------------------------
# instruct
# ---------------------------------------------------------------------------


class TestInstructCommand:
    """Tests for the %instruct command."""

    def test_instruct_sets_instruction(self, plugin_env):
        """GIVEN text WHEN instruct called THEN saves to DB and confirms."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.instruct(mock_irc, mock_msg, ["You are Captain Picard."])
        plugin.db.save_instruction.assert_called_once_with("testnick", "You are Captain Picard.")
        mock_irc.reply.assert_called_once()
        assert "set" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_no_args_shows_current(self, plugin_env):
        """GIVEN no text and existing instruction WHEN instruct called THEN shows it."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = "You are Picard."
        plugin.instruct(mock_irc, mock_msg, [])
        mock_irc.reply.assert_called_once()
        assert "Picard" in mock_irc.reply.call_args.args[0]

    def test_instruct_no_args_no_instruction(self, plugin_env):
        """GIVEN no text and no instruction WHEN instruct called THEN says none set."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = None
        plugin.instruct(mock_irc, mock_msg, [])
        mock_irc.reply.assert_called_once()
        assert "no instruction" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_clear_removes(self, plugin_env):
        """GIVEN 'clear' WHEN instruct called THEN deletes and confirms."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.delete_instruction.return_value = True
        plugin.instruct(mock_irc, mock_msg, ["clear"])
        plugin.db.delete_instruction.assert_called_once_with("testnick")
        mock_irc.reply.assert_called_once()
        assert "cleared" in mock_irc.reply.call_args.args[0].lower()

    def test_instruct_clear_when_none_set(self, plugin_env):
        """GIVEN 'clear' with no instruction WHEN instruct called THEN says none set."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.delete_instruction.return_value = False
        plugin.instruct(mock_irc, mock_msg, ["clear"])
        assert "no instruction" in mock_irc.reply.call_args.args[0].lower()


# ---------------------------------------------------------------------------
# ask + instruct integration
# ---------------------------------------------------------------------------


class TestAskWithInstruction:
    """Tests for %ask using user instructions from %instruct."""

    def test_ask_passes_user_instruction_as_data_not_system(self, plugin_env):
        """GIVEN user has instruction WHEN ask called THEN it is forwarded as
        user_instruction (user-role data) and NOT baked into the system prompt."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = "You are Captain Picard."
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Make it so.",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])
        call_kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert call_kwargs["user_instruction"] == "You are Captain Picard."
        assert "Picard" not in (call_kwargs.get("system_prompt") or "")

    def test_ask_no_instruction_passes_personality_overlay(self, plugin_env):
        """GIVEN no instruction WHEN ask called THEN the channel personality
        (assistantSystemPrompt) is still forwarded as the overlay so the
        operator's persona survives — the structural framework is layered in
        by ``assistant_completion`` from the route_profile."""
        plugin, mock_irc, mock_msg = plugin_env
        plugin.db.get_instruction.return_value = None
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.completion.return_value = CompletionResult(
            content="Hello!",
            grounding_used=False,
            prompt_tokens=10,
            completion_tokens=5,
            cost=0.001,
            model="gpt-4",
        )
        plugin.ask(mock_irc, mock_msg, ["hello"])
        call_kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        # assistantSystemPrompt is "You are helpful." in the test registry —
        # the overlay must reach the lower layer regardless of user instruction.
        assert call_kwargs.get("system_prompt") == "You are helpful."


# ---------------------------------------------------------------------------
# _check_pending_tasks
# ---------------------------------------------------------------------------


class TestCheckPendingTasks:
    """Tests for the _check_pending_tasks polling loop."""

    def test_delivery_exception_does_not_propagate(self, mock_irc, mocker: MockerFixture) -> None:
        """GIVEN _deliver_pending_result raises WHEN _check_pending_tasks runs THEN no exception propagates and error is logged."""
        from llm.service import PendingTaskResult

        from .conftest import plugin_init_patches

        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker)
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)

        # Provide a deliverable result
        mock_result = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="testnick",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            content="Hello from AI",
            task_id=42,
        )
        mocker.patch("llm.plugin.world.ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = [mock_result]
        mocker.patch.object(
            plugin, "_deliver_pending_result", side_effect=RuntimeError("delivery failed")
        )

        # Should NOT raise
        plugin._check_pending_tasks()

        plugin.log.error.assert_called()
        # Verify the error message mentions the task_id
        error_args = plugin.log.error.call_args[0]
        assert "42" in str(error_args)

    def test_delivery_exception_does_not_block_other_results(
        self, mock_irc, mocker: MockerFixture
    ) -> None:
        """GIVEN two results and first delivery raises WHEN _check_pending_tasks runs THEN second result is still delivered."""
        from llm.service import PendingTaskResult

        from .conftest import plugin_init_patches

        registry = make_registry_side_effect()
        mocker.patch.object(LLM, "registryValue", side_effect=registry)
        plugin_init_patches(mocker)
        mocker.patch("llm.plugin.schedule.addEvent")
        plugin = LLM(mock_irc)
        plugin.registryValue = mocker.MagicMock(side_effect=registry)

        result_a = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="alice",
            reply_target="#test",
            is_channel=True,
            prompt_preview="hello",
            model="gpt-4",
            content="Reply A",
            task_id=1,
        )
        result_b = PendingTaskResult(
            status="completed",
            task_type="ask",
            nick="bob",
            reply_target="#test",
            is_channel=True,
            prompt_preview="world",
            model="gpt-4",
            content="Reply B",
            task_id=2,
        )

        mocker.patch("llm.plugin.world.ircs", [])
        plugin.llm_service.check_pending_tasks.return_value = [result_a, result_b]

        call_count = 0

        def deliver_side_effect(r: PendingTaskResult) -> None:
            nonlocal call_count
            call_count += 1
            if r.task_id == 1:
                raise RuntimeError("delivery failed for first")

        mocker.patch.object(plugin, "_deliver_pending_result", side_effect=deliver_side_effect)

        plugin._check_pending_tasks()

        # Both results were attempted
        assert call_count == 2


# ---------------------------------------------------------------------------
# verse_storybook
# ---------------------------------------------------------------------------


class TestVerseStorybook:
    """Cover the gated verse_storybook tool: spec parity + handler gates."""

    def _enable_storybook(self, plugin, mocker, *, enabled=True):
        """Swap registryValue so verseStorybookEnabled is on and the
        storybook tuning keys have sane numeric defaults."""
        base = make_registry_side_effect()

        def side_effect(key, *args):
            if key == "verseStorybookEnabled":
                return enabled
            if key == "verseStorybookMaxPerTurn":
                return 1
            if key == "verseStorybookCooldownSeconds":
                return 300
            return base(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=side_effect)

    # --- spec parity -------------------------------------------------------

    def test_spec_excludes_storybook_when_flag_off(self):
        from llm.verse.avatar import make_verse_tool_specs

        specs = make_verse_tool_specs(max_actors=8, storybook=False)
        names = {s["function"]["name"] for s in specs}
        assert "verse_storybook" not in names

    def test_spec_includes_storybook_when_flag_on(self):
        from llm.verse.avatar import make_verse_tool_specs

        specs = make_verse_tool_specs(max_actors=8, storybook=True)
        names = {s["function"]["name"] for s in specs}
        assert "verse_storybook" in names

    def test_route_spec_tracks_flag(self, plugin_env, mocker):
        """_verse_route_for advertises verse_storybook iff the flag is on."""
        plugin, mock_irc, mock_msg = plugin_env

        store = mocker.MagicMock()
        store.find_avatar_by_account.return_value = None
        store.find_avatar_by_nick.return_value = 7
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        mocker.patch.object(plugin.db, "get_avatar_persona", return_value="a sly fox")
        mocker.patch("llm.plugin.build_verse_system_prompt", return_value="SCENE")
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.is_ooc", return_value=False)

        # Flag off → no storybook spec. force_roleplay=True enters the roleplay
        # route explicitly (a bare mention no longer routes); this test is about
        # tool specs, not the trigger.
        base = make_registry_side_effect({"verseEnabled": True, "verseTriggerRegex": "."})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        route = plugin._verse_route_for("#test", "testnick", None, "hi", force_roleplay=True)
        assert route is not None
        names_off = {t["function"]["name"] for t in route.tools}
        assert "verse_storybook" not in names_off

        # Flag on → storybook spec present.
        def side_effect(key, *args):
            if key == "verseStorybookEnabled":
                return True
            return base(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=side_effect)
        route = plugin._verse_route_for("#test", "testnick", None, "hi", force_roleplay=True)
        names_on = {t["function"]["name"] for t in route.tools}
        assert "verse_storybook" in names_on

    def test_bare_mention_does_not_enter_roleplay(self, plugin_env, mocker):
        """Canon-layer split: a bare trigger no longer arms roleplay mode; only
        an explicit @verse (force_roleplay=True) routes into it."""
        plugin, mock_irc, mock_msg = plugin_env
        store = mocker.MagicMock()
        store.find_avatar_by_account.return_value = None
        store.find_avatar_by_nick.return_value = 7
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        mocker.patch.object(plugin.db, "get_avatar_persona", return_value="a sly fox")
        mocker.patch("llm.plugin.build_verse_system_prompt", return_value="SCENE")
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        mocker.patch("llm.plugin.is_ooc", return_value=False)
        base = make_registry_side_effect({"verseEnabled": True, "verseTriggerRegex": "."})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        # Trigger would match, but without the explicit signal → chat path.
        assert plugin._verse_route_for("#test", "testnick", None, "hi") is None
        # Explicit @verse opt-in still enters roleplay.
        assert (
            plugin._verse_route_for("#test", "testnick", None, "hi", force_roleplay=True)
            is not None
        )

    def test_verse_context_for_injects_canon_on_trigger(self, plugin_env, mocker, tmp_path):
        """Retrieval side: a canon reference yields a facts block for the chat turn."""
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        archie = store.add_entity("npc", "Archie", "the windbag")
        store.apply_direct(
            op="set_pinned",
            payload={"entity_id": archie, "pinned": True},
            source="operator",
            provenance="t",
        )
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        base = make_registry_side_effect(
            {"verseEnabled": True, "verseTriggerRegex": "", "verseRosterMaxChars": 2000}
        )
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="testnick", account=None)
        block = plugin._verse_context_for(pf, "what does Archie think?")
        assert block is not None
        assert "Archie: the windbag" in block
        assert "You are" not in block  # facts only, no persona takeover

    def test_verse_context_for_none_when_no_reference(self, plugin_env, mocker, tmp_path):
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        store.add_entity("npc", "Archie", "windbag")  # not pinned, not mentioned
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        base = make_registry_side_effect(
            {"verseEnabled": True, "verseTriggerRegex": "", "verseRosterMaxChars": 2000}
        )
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="testnick", account=None)
        assert plugin._verse_context_for(pf, "hello there") is None

    def test_verse_context_for_chat_appends_flavour(self, plugin_env, mocker, tmp_path):
        """for_chat=True adds the world's-voice nudge (livelier answers); the
        default (the @draw grounding path) leaves it off."""
        from llm.plugin import _VERSE_CHAT_FLAVOUR_NUDGE
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        archie = store.add_entity("npc", "Archie", "the windbag")
        store.apply_direct(
            op="set_pinned",
            payload={"entity_id": archie, "pinned": True},
            source="operator",
            provenance="t",
        )
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        base = make_registry_side_effect(
            {"verseEnabled": True, "verseTriggerRegex": "", "verseRosterMaxChars": 2000}
        )
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="testnick", account=None)

        chat = plugin._verse_context_for(pf, "who is Archie?", for_chat=True)
        assert _VERSE_CHAT_FLAVOUR_NUDGE in chat
        draw = plugin._verse_context_for(pf, "who is Archie?")
        assert _VERSE_CHAT_FLAVOUR_NUDGE not in draw

    def test_duplicate_dispatch_suppressed_within_window(self, plugin_env, mocker):
        """First identical line passes; an immediate echo is dropped; a fresh
        line and a different channel both pass."""
        import llm.plugin as plugmod

        plugin, mock_irc, mock_msg = plugin_env
        t = [1000.0]
        mocker.patch.object(plugmod.time, "time", side_effect=lambda: t[0])

        assert plugin._is_duplicate_dispatch("#c", "vibebot story about the lads") is False
        # Same text, same channel, a beat later → duplicate.
        t[0] += 2.0
        assert plugin._is_duplicate_dispatch("#c", "vibebot story about the lads") is True
        # Different channel is independent.
        assert plugin._is_duplicate_dispatch("#other", "vibebot story about the lads") is False
        # A different line passes.
        assert plugin._is_duplicate_dispatch("#c", "vibebot draw the lads") is False
        # After the window lapses the same line is fresh again.
        t[0] += plugmod._DISPATCH_DEDUP_WINDOW + 1.0
        assert plugin._is_duplicate_dispatch("#c", "vibebot story about the lads") is False

    # --- Slice 2: chat-path canon recording --------------------------------

    def test_chat_record_handler_none_when_flag_off(self, plugin_env, mocker, tmp_path):
        """Default (verseChatRecordEnabled off) → no live handler; chat-path
        verse_record stays denied."""
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        av = store.add_entity("avatar", "Alice")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        mocker.patch.object(plugin, "_find_caller_avatar", return_value=av)
        base = make_registry_side_effect({"verseEnabled": True, "verseChatRecordEnabled": False})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="alice", account=None)
        assert plugin._verse_chat_record_handler(pf) is None

    def test_chat_record_handler_none_without_avatar(self, plugin_env, mocker, tmp_path):
        """Flag on but caller has no avatar → still None (only opted-in avatars
        write canon)."""
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        mocker.patch.object(plugin, "_find_caller_avatar", return_value=None)
        base = make_registry_side_effect({"verseEnabled": True, "verseChatRecordEnabled": True})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="bob", account=None)
        assert plugin._verse_chat_record_handler(pf) is None

    def test_chat_record_handler_live_records_to_canon(self, plugin_env, mocker, tmp_path):
        """Flag on + opted-in avatar → live handler that actually writes an
        event to the store (canon accrues from ordinary chat)."""
        import json

        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        av = store.add_entity("avatar", "Alice")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        mocker.patch.object(plugin, "_find_caller_avatar", return_value=av)
        base = make_registry_side_effect(
            {
                "verseEnabled": True,
                "verseChatRecordEnabled": True,
                "verseAutoEntityMaxNamesPerCall": 8,
            }
        )
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="alice", account=None)
        handler = plugin._verse_chat_record_handler(pf)
        assert handler is not None
        res = handler({"summary": "Alice found the golden key"})
        assert json.loads(res.content)["status"] == "ok"
        assert any("golden key" in e.summary for e in store.recent_events(limit=10))

    def test_verse_context_appends_record_nudge_when_enabled(self, plugin_env, mocker, tmp_path):
        """With recording on + an avatar, the canon block carries the
        verse_record nudge; off, it does not."""
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        archie = store.add_entity("npc", "Archie", "the windbag")
        store.apply_direct(
            op="set_pinned",
            payload={"entity_id": archie, "pinned": True},
            source="operator",
            provenance="t",
        )
        av = store.add_entity("avatar", "Alice")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        mocker.patch.object(plugin, "_find_caller_avatar", return_value=av)

        def _regs(enabled):
            return make_registry_side_effect(
                {
                    "verseEnabled": True,
                    "verseTriggerRegex": "",
                    "verseRosterMaxChars": 2000,
                    "verseChatRecordEnabled": enabled,
                }
            )

        pf = mocker.MagicMock(channel="#test", nick="alice", account=None)

        plugin.registryValue = mocker.MagicMock(side_effect=_regs(True))
        on = plugin._verse_context_for(pf, "what does Archie think?")
        assert on is not None and "verse_record" in on

        plugin.registryValue = mocker.MagicMock(side_effect=_regs(False))
        off = plugin._verse_context_for(pf, "what does Archie think?")
        assert off is not None and "verse_record" not in off

    # --- Slice 3: sticky @rp roleplay --------------------------------------

    def test_roleplay_sticky_on_off(self, plugin_env, mocker):
        plugin, _i, _m = plugin_env
        base = make_registry_side_effect({"verseRoleplayStickyTtlSeconds": 900})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="alice", account=None)
        assert plugin._roleplay_sticky_active(pf) is False
        plugin._roleplay_sticky_set(pf, True)
        assert plugin._roleplay_sticky_active(pf) is True
        plugin._roleplay_sticky_set(pf, False)
        assert plugin._roleplay_sticky_active(pf) is False

    def test_roleplay_sticky_expires_and_evicts(self, plugin_env, mocker):
        import time as _t

        plugin, _i, _m = plugin_env
        base = make_registry_side_effect({"verseRoleplayStickyTtlSeconds": 900})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="alice", account=None)
        plugin._roleplay_sticky_set(pf, True)
        plugin._roleplay_sticky[("#test", "alice")] = _t.time() - 1  # force lapse
        assert plugin._roleplay_sticky_active(pf) is False
        assert ("#test", "alice") not in plugin._roleplay_sticky

    def test_roleplay_sticky_never_expires_when_ttl_zero(self, plugin_env, mocker):
        import math

        plugin, _i, _m = plugin_env
        base = make_registry_side_effect({"verseRoleplayStickyTtlSeconds": 0})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf = mocker.MagicMock(channel="#test", nick="alice", account=None)
        plugin._roleplay_sticky_set(pf, True)
        assert plugin._roleplay_sticky[("#test", "alice")] == math.inf
        assert plugin._roleplay_sticky_active(pf) is True

    def test_roleplay_sticky_keyed_by_account_over_nick(self, plugin_env, mocker):
        """Account identity wins so a nick change doesn't drop the session."""
        plugin, _i, _m = plugin_env
        base = make_registry_side_effect({"verseRoleplayStickyTtlSeconds": 900})
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        pf_on = mocker.MagicMock(channel="#test", nick="alice", account="acct1")
        plugin._roleplay_sticky_set(pf_on, True)
        pf_renamed = mocker.MagicMock(channel="#test", nick="alice_away", account="acct1")
        assert plugin._roleplay_sticky_active(pf_renamed) is True

    def test_sticky_promotes_ambient_but_not_explicit(self, plugin_env, mocker):
        """Sticky roleplay promotes ambient (nick-addressed) turns to roleplay,
        but leaves explicit @ask untouched."""
        plugin, mock_irc, mock_msg = plugin_env
        base = make_registry_side_effect(
            {"verseEnabled": True, "verseRoleplayStickyTtlSeconds": 900}
        )
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        rf = mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_verse_context_for", return_value=None)
        mocker.patch.object(plugin, "_verse_chat_record_handler", return_value=None)
        mocker.patch.object(plugin, "_ask_impl")
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account=None)
        plugin._roleplay_sticky_set(pf, True)

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "hello", pf, entry_route="addressed"
        )
        assert rf.call_args.kwargs["force_roleplay"] is True

        plugin._dispatch_with_verse_routing(mock_irc, mock_msg, "hello", pf, entry_route="ask")
        assert rf.call_args.kwargs["force_roleplay"] is False

    # --- intent-routed ambient stories ------------------------------------

    def test_ambient_verse_intent_classifier(self):
        from llm.plugin import LLM

        f = LLM._ambient_verse_intent
        assert f("draw the stinky lads") == "draw"
        assert f("sketch of the lads") == "draw"
        assert f("illustrate the lads' saga") == "illustrate"  # illustrated tale != single pic
        assert f("a comic about the lads") == "illustrate"
        assert f("draw the story with pictures") == "illustrate"  # illustrate wins over draw
        # Everything else → story, questions included: a canon mention is always
        # a story cue, we never guess that some phrasing wants a short answer.
        assert f("who is Diarrhoea Dan?") == "story"
        assert f("what are the stinky lads") == "story"
        assert f("") == "story"
        assert f("what have the stinky lads done at school today") == "story"
        assert f("the stinky lads stormed the chippy") == "story"
        assert f("what do the lads think of the headmaster?") == "story"

    def test_looks_like_question_classifier(self):
        from llm.plugin import LLM

        f = LLM._looks_like_question
        assert f("who got rid of school milk entirely") is True
        assert f("what have the stinky lads done today") is True
        assert f("how does photosynthesis work") is True
        assert f("is the sky blue") is True
        assert f("tell me about it?") is True  # trailing ? still counts
        # Not questions.
        assert f("the stinky lads stormed the chippy") is False
        assert f("go mental") is False
        assert f("") is False

    def test_factual_question_gets_straight_overlay(self, plugin_env, mocker):
        """A real-world question with no canon reference → chat path with the
        factual overlay swapped in for the tall-tale channel overlay."""
        from llm.plugin import _FACTUAL_CHAT_OVERLAY

        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_roleplay_sticky_active", return_value=False)
        mocker.patch.object(plugin, "_ambient_inline_story", return_value=False)
        mocker.patch.object(plugin, "_verse_context_for", return_value=None)  # no canon
        mocker.patch.object(plugin, "_verse_chat_record_handler", return_value=None)
        ask = mocker.patch.object(plugin, "_ask_impl")
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "who got rid of school milk entirely", pf, entry_route="addressed"
        )
        ask.assert_called_once()
        assert ask.call_args.kwargs["overlay_override"] == _FACTUAL_CHAT_OVERLAY

    def test_canon_question_keeps_inworld_overlay(self, plugin_env, mocker):
        """A question that DOES pull canon keeps the in-world overlay (no swap)."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_roleplay_sticky_active", return_value=False)
        mocker.patch.object(plugin, "_ambient_inline_story", return_value=False)
        mocker.patch.object(plugin, "_verse_context_for", return_value="CANON: the lads")
        mocker.patch.object(plugin, "_verse_chat_record_handler", return_value=None)
        ask = mocker.patch.object(plugin, "_ask_impl")
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "who is the headmaster", pf, entry_route="addressed"
        )
        ask.assert_called_once()
        assert ask.call_args.kwargs["overlay_override"] is None

    def test_non_question_keeps_inworld_overlay(self, plugin_env, mocker):
        """A non-question with no canon still keeps the tall-tale overlay — the
        factual swap is only for genuine questions."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_roleplay_sticky_active", return_value=False)
        mocker.patch.object(plugin, "_ambient_inline_story", return_value=False)
        mocker.patch.object(plugin, "_verse_context_for", return_value=None)
        mocker.patch.object(plugin, "_verse_chat_record_handler", return_value=None)
        ask = mocker.patch.object(plugin, "_ask_impl")
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "go mental about dragons", pf, entry_route="addressed"
        )
        ask.assert_called_once()
        assert ask.call_args.kwargs["overlay_override"] is None

    def test_ambient_storybook_brief_gates(self, plugin_env, mocker, tmp_path):
        """Only an explicit 'illustrate' request → brief; a plain narrative /
        recount / question / draw / no-account / flag-off → None (a plain
        narrative is an inline prose tale, not an image storybook)."""
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        base = make_registry_side_effect(
            {
                "verseEnabled": True,
                "verseStorybookEnabled": True,
                "verseTriggerRegex": r"\bstinky lads\b",
                "verseStorybookCooldownSeconds": 0,
            }
        )
        plugin.registryValue = mocker.MagicMock(side_effect=base)
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        # Explicit "illustrate" → the image storybook.
        illustrate = "illustrate the stinky lads' saga"
        assert plugin._ambient_storybook_brief(mock_msg, pf, illustrate) == illustrate
        # A plain narrative or recount is a prose-first INLINE tale, not here.
        assert (
            plugin._ambient_storybook_brief(mock_msg, pf, "the stinky lads stormed the chippy")
            is None
        )
        assert (
            plugin._ambient_storybook_brief(mock_msg, pf, "what have the stinky lads done today")
            is None
        )
        # An identity lookup and a single-picture draw stay off the story path.
        assert plugin._ambient_storybook_brief(mock_msg, pf, "who are the stinky lads?") is None
        assert plugin._ambient_storybook_brief(mock_msg, pf, "draw the stinky lads") is None
        assert (
            plugin._ambient_storybook_brief(mock_msg, pf, "illustrate a fox") is None
        )  # not canon

        pf_anon = mocker.MagicMock(channel="#test", nick="alice", account=None)
        assert plugin._ambient_storybook_brief(mock_msg, pf_anon, illustrate) is None  # no account

    def test_ambient_inline_story_gates(self, plugin_env, mocker, tmp_path):
        """Any canon mention from an avatar-holder → True (inline prose tale),
        questions included; draw/illustrate, non-canon, no-avatar, flag-off →
        False."""
        from llm.verse.store import VerseStore

        plugin, mock_irc, mock_msg = plugin_env
        store = VerseStore(tmp_path, "#test")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        mocker.patch.object(plugin, "_find_caller_avatar", return_value=7)
        mocker.patch.object(plugin, "_verse_triggered", return_value=True)
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        assert plugin._ambient_inline_story(pf, "the stinky lads stormed the chippy") is True
        assert plugin._ambient_inline_story(pf, "what have the lads done today") is True
        # A question is a story too — no short-answer guessing.
        assert plugin._ambient_inline_story(pf, "who is Dan?") is True
        # Only the explicit picture asks keep their own paths.
        assert plugin._ambient_inline_story(pf, "draw the lads") is False
        assert plugin._ambient_inline_story(pf, "illustrate the lads") is False
        # No avatar → grounded chat instead of prose.
        plugin._find_caller_avatar.return_value = None
        assert plugin._ambient_inline_story(pf, "the lads stormed the chippy") is False
        plugin._find_caller_avatar.return_value = 7
        # Canon not referenced → not a verse story.
        plugin._verse_triggered.return_value = False
        assert plugin._ambient_inline_story(pf, "the lads stormed the chippy") is False

    def test_ambient_narrative_mention_enters_inline_prose(self, plugin_env, mocker):
        """A plain narrative mention promotes to a one-shot verse-prose route
        (force_roleplay=True) — NOT the image storybook job."""
        plugin, mock_irc, mock_msg = plugin_env
        rf = mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_roleplay_sticky_active", return_value=False)
        mocker.patch.object(plugin, "_ambient_inline_story", return_value=True)
        mocker.patch.object(plugin, "_verse_context_for", return_value=None)
        mocker.patch.object(plugin, "_verse_chat_record_handler", return_value=None)
        job = mocker.patch.object(plugin, "_submit_storybook_job")
        mocker.patch.object(plugin, "_ask_impl")
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "the lads stormed the chippy", pf, entry_route="addressed"
        )
        assert rf.call_args.kwargs["force_roleplay"] is True
        job.assert_not_called()

    def test_ambient_illustrate_mention_fires_storybook(self, plugin_env, mocker):
        """An explicit 'illustrate' mention still fires the image storybook job."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_roleplay_sticky_active", return_value=False)
        mocker.patch.object(plugin, "_ambient_inline_story", return_value=False)
        mocker.patch.object(
            plugin, "_ambient_storybook_brief", return_value="illustrate the lads' saga"
        )
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value="")
        job = mocker.patch.object(plugin, "_submit_storybook_job")
        ask = mocker.patch.object(plugin, "_ask_impl")
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "illustrate the lads' saga", pf, entry_route="addressed"
        )
        job.assert_called_once()
        assert job.call_args.kwargs["brief"] == "illustrate the lads' saga"
        ask.assert_not_called()

    def test_ambient_question_falls_through_to_chat(self, plugin_env, mocker):
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_roleplay_sticky_active", return_value=False)
        mocker.patch.object(plugin, "_ambient_inline_story", return_value=False)
        mocker.patch.object(plugin, "_ambient_storybook_brief", return_value=None)
        mocker.patch.object(plugin, "_verse_context_for", return_value=None)
        mocker.patch.object(plugin, "_verse_chat_record_handler", return_value=None)
        job = mocker.patch.object(plugin, "_submit_storybook_job")
        ask = mocker.patch.object(plugin, "_ask_impl")
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "who is Dan?", pf, entry_route="addressed"
        )
        job.assert_not_called()
        ask.assert_called_once()

    def test_explicit_ask_does_not_autofire_story(self, plugin_env, mocker):
        """entry_route='ask' is not ambient → the auto-story gate is skipped
        entirely (brief never computed), honouring the explicit command."""
        plugin, mock_irc, mock_msg = plugin_env
        mocker.patch.object(plugin, "_verse_route_for", return_value=None)
        mocker.patch.object(plugin, "_roleplay_sticky_active", return_value=False)
        sb = mocker.patch.object(plugin, "_ambient_storybook_brief")
        mocker.patch.object(plugin, "_verse_context_for", return_value=None)
        mocker.patch.object(plugin, "_verse_chat_record_handler", return_value=None)
        job = mocker.patch.object(plugin, "_submit_storybook_job")
        ask = mocker.patch.object(plugin, "_ask_impl")
        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )
        mock_msg.prefix = "alice!u@h"
        pf = mocker.MagicMock(channel="#test", nick="alice", account="acct")

        plugin._dispatch_with_verse_routing(
            mock_irc, mock_msg, "the lads stormed the chippy", pf, entry_route="ask"
        )
        sb.assert_not_called()
        job.assert_not_called()
        ask.assert_called_once()

    # --- handler gates -----------------------------------------------------

    def _build(self, plugin, mock_irc, mock_msg, mocker, *, account="acct"):
        return plugin._storybook_handler(
            irc=mock_irc,
            msg=mock_msg,
            channel="#test",
            account=account,
            nick="testnick",
            persona="a sly fox",
        )

    def test_no_account_returns_error(self, plugin_env, mocker):
        import json

        plugin, mock_irc, mock_msg = plugin_env
        self._enable_storybook(plugin, mocker)
        submit = mocker.patch.object(plugin._llm_executor, "submit")
        handler = self._build(plugin, mock_irc, mock_msg, mocker, account=None)

        out = json.loads(handler({"brief": "a tale"}).content)
        assert out["status"] == "error"
        plugin.llm_service.generate_storybook.assert_not_called()
        submit.assert_not_called()

    def test_missing_draw_capability_returns_error(self, plugin_env, mocker):
        import json

        plugin, mock_irc, mock_msg = plugin_env
        self._enable_storybook(plugin, mocker)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        submit = mocker.patch.object(plugin._llm_executor, "submit")
        handler = self._build(plugin, mock_irc, mock_msg, mocker)

        out = json.loads(handler({"brief": "a tale"}).content)
        assert out["status"] == "error"
        plugin.llm_service.generate_storybook.assert_not_called()
        submit.assert_not_called()

    def test_cooldown_active_returns_error(self, plugin_env, mocker):
        import json

        plugin, mock_irc, mock_msg = plugin_env
        self._enable_storybook(plugin, mocker)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        submit = mocker.patch.object(plugin._llm_executor, "submit")
        # Pre-load the cooldown bucket so the account is already limited.
        plugin._rate_buckets["verse_storybook:acct"] = __import__("collections").deque(
            [time.time()], maxlen=1
        )
        handler = self._build(plugin, mock_irc, mock_msg, mocker)

        out = json.loads(handler({"brief": "a tale"}).content)
        assert out["status"] == "error"
        plugin.llm_service.generate_storybook.assert_not_called()
        submit.assert_not_called()

    def test_per_turn_cap_blocks_second_call(self, plugin_env, mocker):
        import json

        plugin, mock_irc, mock_msg = plugin_env
        self._enable_storybook(plugin, mocker)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        submit = mocker.patch.object(plugin._llm_executor, "submit")
        handler = self._build(plugin, mock_irc, mock_msg, mocker)

        first = json.loads(handler({"brief": "first"}).content)
        assert first["status"] == "ok"
        second = json.loads(handler({"brief": "second"}).content)
        assert second["status"] == "error"
        # Only the first call scheduled a job.
        assert submit.call_count == 1

    def test_happy_path_returns_ok_and_schedules_job(self, plugin_env, mocker):
        import json

        plugin, mock_irc, mock_msg = plugin_env
        self._enable_storybook(plugin, mocker)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        submit = mocker.patch.object(plugin._llm_executor, "submit")
        handler = self._build(plugin, mock_irc, mock_msg, mocker)

        out = json.loads(handler({"brief": "a grand tale"}).content)
        assert out["status"] == "ok"
        # The note must steer the model away from announcing a pending link.
        assert "not announce" in out["note"].lower()
        # Job scheduled exactly once on the executor.
        assert submit.call_count == 1
        assert submit.call_args[0][0] == "verse_storybook"
        # generate_storybook is NOT invoked synchronously in the handler.
        plugin.llm_service.generate_storybook.assert_not_called()

    def test_job_delivers_url_to_channel(self, plugin_env, mocker):
        """The scheduled job posts the storybook URL via the world.ircs path."""
        from llm.service import StorybookResult

        plugin, mock_irc, mock_msg = plugin_env
        self._enable_storybook(plugin, mocker)
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.llm_service.generate_storybook.return_value = StorybookResult(
            url="http://h/x.md", title="A Tale", image_count=2, dropped=0
        )
        # Run the executor job inline so we can observe delivery.
        submit = mocker.patch.object(
            plugin._llm_executor,
            "submit",
            side_effect=lambda label, fn, *a: fn(*a),
        )
        # Make world.ircs yield our mock_irc which is in the channel.
        mock_irc.state.channels = {"#test": mocker.MagicMock()}
        mocker.patch("llm.plugin.world.ircs", [mock_irc])
        safe_queue = mocker.patch.object(plugin, "_safe_queue", return_value=True)

        handler = self._build(plugin, mock_irc, mock_msg, mocker)
        handler({"brief": "a grand tale"})

        assert submit.call_count == 1
        plugin.llm_service.generate_storybook.assert_called_once()
        # Delivered the URL to the channel.
        assert safe_queue.called


class TestStoryCommand:
    """The @story command: standalone illustrated-storybook generation outside
    verse mode. Gated like @draw (authenticated + llm.draw) plus the shared
    storybook cooldown; fires the same background render+deliver job."""

    def test_story_fires_storybook_job(self, plugin_env, mocker):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "acct"
        plugin.db.get_avatar_persona.return_value = ""
        job = mocker.patch.object(plugin, "_submit_storybook_job")

        plugin.story(mock_irc, mock_msg, ["illustrated", "tale", "of", "lads"])

        job.assert_called_once()
        assert job.call_args.kwargs["brief"] == "illustrated tale of lads"
        assert job.call_args.kwargs["channel"] == "#test"

    def test_story_requires_account(self, plugin_env, mocker):
        plugin, mock_irc, mock_msg = plugin_env
        # nickToAccount returns None (default) -> not authenticated.
        job = mocker.patch.object(plugin, "_submit_storybook_job")

        plugin.story(mock_irc, mock_msg, ["a", "tale"])

        mock_irc.error.assert_called_once()
        job.assert_not_called()

    def test_story_cooldown_blocks(self, plugin_env, mocker):
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "acct"
        mocker.patch.object(plugin, "_storybook_cooldown_active", return_value=True)
        err = mocker.patch.object(plugin, "_safe_error")
        job = mocker.patch.object(plugin, "_submit_storybook_job")

        plugin.story(mock_irc, mock_msg, ["a", "tale"])

        err.assert_called_once()
        job.assert_not_called()

    def test_storybook_cooldown_reserve_then_block(self, plugin_env, mocker):
        plugin, _mock_irc, _mock_msg = plugin_env
        # First call reserves the slot (not limited); the second within the
        # window is blocked. A different account is independent. cooldown<=0 or
        # a missing account disables limiting entirely.
        assert plugin._storybook_cooldown_active("acct", 300) is False
        assert plugin._storybook_cooldown_active("acct", 300) is True
        assert plugin._storybook_cooldown_active("other", 300) is False
        assert plugin._storybook_cooldown_active("acct", 0) is False
        assert plugin._storybook_cooldown_active(None, 300) is False


class TestRateLimitPeek:
    """Scheduled fires peek the bucket instead of consuming interactive slots."""

    def test_record_false_does_not_consume_bucket(self, plugin_env):
        plugin, _irc, _msg = plugin_env

        blocked = plugin._check_rate_limit(
            None, "ask", "acct", "", "", "", tier="registered", silent=True, record=False
        )

        assert blocked is False
        with plugin._rate_buckets_lock:
            assert len(plugin._rate_buckets.get("ask:acct", ())) == 0

    def test_record_true_still_consumes(self, plugin_env):
        plugin, _irc, _msg = plugin_env

        plugin._check_rate_limit(None, "ask", "acct", "", "", "", tier="registered", silent=True)

        with plugin._rate_buckets_lock:
            assert len(plugin._rate_buckets.get("ask:acct", ())) == 1
