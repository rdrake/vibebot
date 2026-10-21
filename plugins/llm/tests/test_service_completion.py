"""Service completion: search/url grounding, error classification, retry, xAI cache key, assistant facade."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from llm import provider
from llm.identity import Caller, Identity
from llm.service import (
    AssistantRequestContext,
    AssistantResult,
    CompletionResult,
    LLMService,
    Spend,
)

from .conftest import make_completion_response

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


def _search_reply(
    text: str = "result text",
    *,
    grounded: bool = True,
    input_tokens: int = 11,
    output_tokens: int = 22,
    cost: float = 0.0,
) -> provider.Reply:
    """A provider.search reply."""
    return provider.Reply(
        text=text,
        usage=provider.Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        cost=cost,
        grounded=grounded,
    )


class TestGroundedCompletion:
    """search_completion + url_completion run provider.search and map its Reply."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.plugin = make_service(searchModel="xai/grok-4.3", timeout=30)

    def test_search_sends_query_to_search_model(self) -> None:
        search = self.mocker.patch("llm.service.provider.search", return_value=_search_reply())

        self.service.search_completion("what is grok", channel="#dev")

        search.assert_called_once()
        assert search.call_args.args == ("xai/grok-4.3", "what is grok")
        assert search.call_args.kwargs["timeout"] == 30

    def test_url_prompt_names_the_url(self) -> None:
        search = self.mocker.patch("llm.service.provider.search", return_value=_search_reply())

        self.service.url_completion("https://example.com", channel="#dev")

        search.assert_called_once()
        assert "https://example.com" in search.call_args.args[1]

    @pytest.mark.parametrize("grounded", [True, False])
    def test_grounding_used_mirrors_the_reply(self, grounded: bool) -> None:
        self.mocker.patch(
            "llm.service.provider.search", return_value=_search_reply(grounded=grounded)
        )
        result = self.service.search_completion("q", channel="#dev")
        assert result.grounding_used is grounded

    def test_usage_and_cost_ride_back(self) -> None:
        self.mocker.patch(
            "llm.service.provider.search",
            return_value=_search_reply("hi", input_tokens=42, output_tokens=7, cost=0.004),
        )
        result = self.service.search_completion("q", channel="#dev")
        assert result.content == "hi"
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 7
        assert result.cost == pytest.approx(0.004)

    def test_returns_error_tool_result_on_exception(self) -> None:
        self.mocker.patch("llm.service.provider.search", side_effect=RuntimeError("upstream 500"))
        result = self.service.search_completion("q", channel="#dev")
        assert "Search failed" in result.content
        result_url = self.service.url_completion("https://example.com", channel="#dev")
        assert "URL fetch failed" in result_url.content

    @pytest.mark.parametrize("kind", ["search", "url"])
    def test_cache_key_uses_the_grounded_lane(self, kind: str) -> None:
        """Search and URL fetch share the ``grounded`` cache lane so the short
        search prefix doesn't compete with assistant_step_* on the same server."""
        search = self.mocker.patch("llm.service.provider.search", return_value=_search_reply())
        if kind == "search":
            self.service.search_completion("q", channel="#dev")
        else:
            self.service.url_completion("https://example.com", channel="#dev")
        assert search.call_args.kwargs["cache_key"] == "chan:#dev:grounded"


class TestCompletion:
    """completion(): result shape, errors, typing indicators, prompt override."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            assistantModel="anthropic/claude-haiku-5-5",
            assistantSystemPrompt="You are helpful.",
            timeout=30,
            maxPromptLength=10000,
            commandPrefixes=["."],
        )

    def test_completion_returns_completion_result(self) -> None:
        """GIVEN successful completion WHEN completing THEN returns CompletionResult."""
        self.mocker.patch(
            "llm.service.provider.chat", return_value=make_completion_response("Test response")
        )
        result = self.service.completion("test", command="ask")

        assert isinstance(result, CompletionResult)
        assert result.content == "Test response"
        assert result.grounding_used is False

    def test_completion_books_reply_usage_and_cost(self) -> None:
        """GIVEN a reply with usage and cost WHEN completing THEN both reach the result."""
        self.mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response(
                "ok", prompt_tokens=100, completion_tokens=50, cost=0.003
            ),
        )
        result = self.service.completion("test", command="ask")

        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.cost == pytest.approx(0.003)
        assert result.model == "anthropic/claude-haiku-5-5"

    def test_completion_error_returns_completion_result_with_error(self) -> None:
        """GIVEN completion error WHEN completing THEN returns CompletionResult with error."""
        self.mocker.patch(
            "llm.service.provider.chat",
            side_effect=Exception("Test error"),
        )
        result = self.service.completion("test", command="ask")

        assert isinstance(result, CompletionResult)
        assert "Error" in result.content
        assert result.grounding_used is False

    def test_completion_sends_typing_indicators(self) -> None:
        """GIVEN irc context WHEN completion called THEN sends typing indicators."""
        irc = self.mocker.Mock()
        irc.state = self.mocker.Mock()
        irc.state.capabilities_ack = {"message-tags"}
        irc.queueMsg = self.mocker.Mock()

        msg = self.mocker.Mock()
        msg.args = ("#test",)
        msg.prefix = "user!user@host"

        self.mocker.patch(
            "llm.service.provider.chat", return_value=make_completion_response("Response")
        )
        self.service.completion("test", command="ask", irc=irc, msg=msg)

        # Should have called queueMsg twice - active and done
        assert irc.queueMsg.call_count == 2
        first_msg = irc.queueMsg.call_args_list[0][0][0]
        assert first_msg.server_tags == {"+typing": "active"}
        second_msg = irc.queueMsg.call_args_list[1][0][0]
        assert second_msg.server_tags == {"+typing": "done"}

    def test_completion_sends_done_on_error(self) -> None:
        """GIVEN error during completion WHEN irc context THEN still sends done indicator."""
        irc = self.mocker.Mock()
        irc.state = self.mocker.Mock()
        irc.state.capabilities_ack = {"message-tags"}
        irc.queueMsg = self.mocker.Mock()

        msg = self.mocker.Mock()
        msg.args = ("#test",)
        msg.prefix = "user!user@host"

        self.mocker.patch("llm.service.provider.chat", side_effect=Exception("API error"))
        result = self.service.completion("test", command="ask", irc=irc, msg=msg)

        assert "Error" in result.content
        # Should still send typing=done in finally block
        assert irc.queueMsg.call_count == 2
        second_msg = irc.queueMsg.call_args_list[1][0][0]
        assert second_msg.server_tags == {"+typing": "done"}

    def test_completion_uses_system_prompt_override(self) -> None:
        """GIVEN system_prompt kwarg WHEN completion THEN uses override instead of registry."""
        messages_sent: list[dict] = []

        def mock_completion(**kwargs: dict) -> Mock:
            messages_sent.extend(kwargs.get("messages", []))
            return make_completion_response("Make it so.")

        self.mocker.patch("llm.service.provider.chat", side_effect=mock_completion)
        result = self.service.completion(
            "What are your orders?",
            command="ask",
            system_prompt="You are Captain Picard.",
        )

        assert result.content == "Make it so."
        assert messages_sent[0]["role"] == "system"
        assert "Captain Picard" in messages_sent[0]["content"]
        assert "You are helpful" not in messages_sent[0]["content"]


class TestUsageExtraction:
    """Usage fields on the result dataclasses."""

    def test_completion_result_has_usage_fields(self) -> None:
        """GIVEN CompletionResult WHEN created with usage THEN fields accessible."""
        result = CompletionResult(
            content="hello",
            grounding_used=False,
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
            model="anthropic/claude-haiku-5-5",
        )
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.cost == 0.001
        assert result.model == "anthropic/claude-haiku-5-5"

    def test_completion_result_usage_defaults(self) -> None:
        """GIVEN CompletionResult WHEN created without usage THEN defaults to zero."""
        result = CompletionResult(content="hello")
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.cost == 0.0
        assert result.model == ""

    def test_image_result_has_usage_fields(self) -> None:
        """GIVEN ImageResult WHEN created with usage THEN fields accessible."""
        from llm.service import ImageResult

        result = ImageResult(
            content="http://example.com/img.png",
            prompt_tokens=10,
            completion_tokens=0,
            cost=0.02,
            model="xai/grok-imagine-image",
        )
        assert result.content == "http://example.com/img.png"
        assert result.cost == 0.02

    def test_image_result_defaults(self) -> None:
        """GIVEN ImageResult WHEN created with just content THEN defaults to zero."""
        from llm.service import ImageResult

        result = ImageResult(content="error message")
        assert result.prompt_tokens == 0
        assert result.cost == 0.0
        assert result.model == ""


class TestErrorClassification:
    """Test _is_terminal_error classification."""

    def test_auth_error_is_terminal(self) -> None:
        """GIVEN AuthenticationError WHEN classified THEN terminal."""
        err = provider.AuthenticationError("bad key")
        assert LLMService._is_terminal_error(err) is True

    def test_timeout_is_transient(self) -> None:
        """GIVEN RequestTimeoutError WHEN classified THEN not terminal."""
        err = provider.RequestTimeoutError("timeout")
        assert LLMService._is_terminal_error(err) is False

    def test_rate_limit_is_transient(self) -> None:
        """GIVEN RateLimitError WHEN classified THEN not terminal."""
        err = provider.RateLimitError("rate limited")
        assert LLMService._is_terminal_error(err) is False

    def test_content_policy_violation_is_terminal(self) -> None:
        """GIVEN ContentPolicyViolationError WHEN classified THEN terminal.

        Reclassifying this as transient would retry a guardrail block ~10x
        with backoff and re-bill each attempt.
        """
        err = provider.ContentPolicyViolationError("blocked")
        assert LLMService._is_terminal_error(err) is True

    def test_bad_request_is_terminal(self) -> None:
        """GIVEN BadRequestError WHEN classified THEN terminal (a malformed
        request will never succeed on retry)."""
        err = provider.BadRequestError("bad request")
        assert LLMService._is_terminal_error(err) is True


class TestComputeBackoff:
    """Canonical executable spec for ``_compute_backoff``.

    Bounded-above, bounded-below, monotone-non-decreasing, and
    initial-floor properties are covered by
    ``test_service_helpers_properties.py``. The single example below
    pins one numeric step so a regression that broke the formula but
    kept the bounds (e.g. swapping ``2**n`` for ``n``) would still
    fail here.
    """

    def test_first_retry_backoff(self) -> None:
        """GIVEN attempt 1 WHEN computing backoff THEN 60 seconds."""
        assert LLMService._compute_backoff(1) == 60


class TestCompletionValidation:
    """Tests for completion() early validation error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.make_service = make_service
        self.service, self.mock_plugin = make_service()

    def test_completion_invalid_prompt(self) -> None:
        """GIVEN empty prompt WHEN completion called THEN returns error in result content."""
        result = self.service.completion("", command="ask")

        assert result.error is not None
        assert "Error" in result.content

    def test_completion_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN no key for the model's provider WHEN completion called THEN returns error.

        assistantModel is anthropic/, so ANTHROPIC_API_KEY is the variable
        this guard reads.
        """
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        service, _ = self.make_service(assistantModel="anthropic/claude-haiku-5-5")

        result = service.completion("Hello world", command="ask")

        assert result.error is not None
        assert "ANTHROPIC_API_KEY" in result.content


class TestXaiCacheKey:
    """xAI prompt-cache routing key, passed to provider.chat as ``cache_key``."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()
        self.chat = mocker.patch(
            "llm.service.provider.chat", return_value=make_completion_response("ok")
        )

    def test_key_set_for_xai_with_channel(self) -> None:
        self.service._timed_completion(
            "assistant_step_1",
            model="xai/grok-4.3",
            messages=[{"role": "user", "content": "hi"}],
            channel="#dev",
        )
        assert self.chat.call_args.kwargs["cache_key"] == "chan:#dev:main"

    def test_key_lane_per_op(self) -> None:
        """Different ops route to different cache lanes on the same channel.

        This is the eviction-prevention split: ``extract_memories`` and
        ``ask_helper`` no longer share a server with ``assistant_step_*``,
        so their short distinct prefixes can't kick the long main prefix
        out of the per-server cache between turns.
        """
        lanes: dict[str, str] = {}
        for op in ("assistant_step_1", "ask_helper", "extract_memories", "prompt_rewrite"):
            self.service._timed_completion(
                op,
                model="xai/grok-4.3",
                messages=[{"role": "user", "content": "hi"}],
                channel="#dev",
            )
            lanes[op] = self.chat.call_args.kwargs["cache_key"]
        assert lanes == {
            "assistant_step_1": "chan:#dev:main",
            "ask_helper": "chan:#dev:helper",
            "extract_memories": "chan:#dev:memory",
            "prompt_rewrite": "chan:#dev:rewrite",
        }

    def test_key_omitted_for_non_xai(self) -> None:
        self.service._timed_completion(
            "op",
            model="anthropic/claude-haiku-5-5",
            messages=[{"role": "user", "content": "hi"}],
            channel="#dev",
        )
        assert self.chat.call_args.kwargs["cache_key"] is None

    def test_key_omitted_for_xai_without_channel(self) -> None:
        self.service._timed_completion(
            "op",
            model="xai/grok-4.3",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert self.chat.call_args.kwargs["cache_key"] is None


class TestAssistantResultGroundingUsed:
    """Tests for AssistantResult.grounding_used field."""

    def test_grounding_used_defaults_to_false(self) -> None:
        """GIVEN AssistantResult with no grounding_used WHEN accessed THEN defaults to False."""
        result = AssistantResult(content="hello", spend=Spend.ZERO)
        assert result.grounding_used is False

    def test_grounding_used_can_be_set_true(self) -> None:
        """GIVEN AssistantResult with grounding_used=True WHEN accessed THEN is True."""
        result = AssistantResult(content="hello", spend=Spend.ZERO, grounding_used=True)
        assert result.grounding_used is True

    def test_grounding_used_coexists_with_other_fields(self) -> None:
        """GIVEN AssistantResult with all fields WHEN accessed THEN all fields correct."""

        result = AssistantResult(
            content="response",
            spend=Spend(prompt_tokens=100, completion_tokens=50, cost=0.01),
            model="anthropic/claude-haiku-5-5",
            grounding_used=True,
            error=None,
        )
        assert result.content == "response"
        assert result.grounding_used is True
        assert result.prompt_tokens == 100
        assert result.model == "anthropic/claude-haiku-5-5"
        assert result.error is None


# ---------------------------------------------------------------------------
# Search / URL completion
# ---------------------------------------------------------------------------


class TestSearchCompletion:
    """Tests for LLMService.search_completion()."""

    @pytest.fixture()
    def service(self, make_service) -> LLMService:
        service, self.plugin = make_service(
            assistantModel="anthropic/claude-haiku-5-5",
            searchModel="xai/grok-4.3",
        )
        return service

    def test_returns_tool_result_with_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """search_completion returns ToolResult with response content."""
        from llm.assistant import ToolResult

        mocker.patch("llm.service.provider.search", return_value=_search_reply("Search answer"))

        result = service.search_completion("what is Python?", channel="#test")

        assert isinstance(result, ToolResult)
        assert result.content == "Search answer"

    def test_uses_search_model_when_configured(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """search_completion prefers searchModel over assistantModel."""
        # Override to provide a searchModel
        service.plugin = mocker.Mock()
        service.plugin.registryValue = mocker.Mock(
            side_effect=lambda key, *a: {
                "searchModel": "xai/grok-4-1-fast-reasoning",
                "assistantModel": "anthropic/claude-haiku-5-5",
                "timeout": 30,
            }.get(key, "")
        )

        search = mocker.patch("llm.service.provider.search", return_value=_search_reply())

        service.search_completion("test", channel="#test")

        assert search.call_args.args[0] == "xai/grok-4-1-fast-reasoning"

    def test_returns_error_on_exception(self, service: LLMService, mocker: MockerFixture) -> None:
        """search_completion returns error ToolResult on failure."""
        from llm.assistant import ToolResult

        mocker.patch(
            "llm.service.provider.search",
            side_effect=RuntimeError("API down"),
        )

        result = service.search_completion("test", channel="#test")

        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed
        # Internal exception text must NOT leak into tool payload
        assert "API down" not in parsed["error"]
        assert parsed["error"] == "Search failed."


class TestUrlCompletion:
    """Tests for LLMService.url_completion()."""

    @pytest.fixture()
    def service(self, make_service) -> LLMService:
        service, self.plugin = make_service(
            assistantModel="anthropic/claude-haiku-5-5",
            searchModel="xai/grok-4.3",
        )
        return service

    def test_returns_summary(self, service: LLMService, mocker: MockerFixture) -> None:
        """url_completion returns page summary as ToolResult."""
        from llm.assistant import ToolResult

        mocker.patch("llm.service.provider.search", return_value=_search_reply("Page summary here"))

        result = service.url_completion("https://example.com/article", channel="#test")

        assert isinstance(result, ToolResult)
        assert result.content == "Page summary here"

    def test_rejects_unsafe_url(self, service: LLMService, mocker: MockerFixture) -> None:
        """url_completion returns error for private IPs."""
        from llm.assistant import ToolResult

        search = mocker.patch("llm.service.provider.search")

        result = service.url_completion("http://192.168.1.1/admin", channel="#test")

        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed
        assert "not allowed" in parsed["error"].lower()
        search.assert_not_called()

    def test_returns_error_on_exception(self, service: LLMService, mocker: MockerFixture) -> None:
        """url_completion returns error ToolResult on failure."""
        from llm.assistant import ToolResult

        mocker.patch(
            "llm.service.provider.search",
            side_effect=RuntimeError("connection refused"),
        )

        result = service.url_completion("https://example.com", channel="#test")

        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed
        # Internal exception text must NOT leak into tool payload
        assert "connection refused" not in parsed["error"]
        assert parsed["error"] == "URL fetch failed."


# =============================================================================
# TestAssistantRequestFacade — Task 9
# =============================================================================


class TestCompletionKwargs:
    """AssistantRequestContext.completion_kwargs: what a request decides about
    the assistant_completion call."""

    def _make_ctx(
        self,
        *,
        entry_route: str = "ask",
        profile: str = "chat",
        nick: str = "user",
        account: str | None = None,
        channel: str | None = "#test",
        is_owner: bool = False,
        capabilities: frozenset[str] = frozenset({"llm.ask"}),
    ) -> AssistantRequestContext:
        return AssistantRequestContext(
            entry_route=entry_route,
            profile=profile,
            caller=Caller(Identity(nick, account), channel),
            is_owner=is_owner,
            capabilities=capabilities,
        )

    @pytest.mark.parametrize("profile", ["chat", "code", "verse", "draw", "remind_action"])
    def test_profile_becomes_route_profile(self, profile: str) -> None:
        """The profile picks the structural framework downstream; the
        personality overlay (system_prompt) is not this value's to set."""
        kw = self._make_ctx(profile=profile, entry_route=profile).completion_kwargs()
        assert kw["route_profile"] == profile
        assert kw["entry_route"] == profile
        assert "system_prompt" not in kw

    def test_forwards_context_fields(self) -> None:
        """The caller's storage key, reply target and account, plus is_owner and
        capabilities, are forwarded."""
        kw = self._make_ctx(
            nick="alice",
            channel="#dev",
            account="alice_acct",
            is_owner=True,
            capabilities=frozenset({"llm.ask", "llm.code"}),
        ).completion_kwargs()
        assert kw["nick"] == "alice_acct"
        assert kw["channel"] == "#dev"
        assert kw["account"] == "alice_acct"
        assert kw["is_owner"] is True
        assert kw["capabilities"] == frozenset({"llm.ask", "llm.code"})

    def test_private_message_passes_the_senders_nick(self) -> None:
        """Private messages (channel=None) pass the sender's nick, the reply
        target, to assistant_completion: never the bot's own nick."""
        kw = self._make_ctx(nick="user", channel=None).completion_kwargs()
        assert kw["channel"] == "user"

    def test_every_key_is_an_assistant_completion_parameter(self) -> None:
        import inspect

        params = inspect.signature(LLMService.assistant_completion).parameters
        assert set(self._make_ctx().completion_kwargs()) <= set(params)


# =============================================================================
# Coverage-fill tests for service.py uncovered branches
# =============================================================================


class TestRetryCompletion:
    """Tests for _retry_completion api-key selection and validation."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def _make_task(self, *, task_type: str = "ask", model: str = "m"):
        from llm.persistence import PendingTaskRow

        return PendingTaskRow(
            id=1,
            task_type=task_type,
            nick="alice",
            reply_target="#test",
            is_channel=1,
            prompt_preview="hi",
            model=model,
            request_data='{"messages": [{"role": "user", "content": "hi"}]}',
            submitted_at=100.0,
            expires_at=200.0,
            attempt_count=0,
            next_attempt_at=100.0,
            claimed_until=0,
            last_error="",
            delivery_state="pending",
            result_payload="",
            last_delivery_error="",
            delivery_attempt_count=0,
            origin_request_id="",
            account=None,
        )

    def test_retry_completion_malformed_messages(self) -> None:
        """request_data without a list 'messages' returns failed_terminal."""
        task = self._make_task()
        result = self.service._retry_completion(task, {"messages": "oops"})
        assert result.status == "failed_terminal"
        assert "Malformed" in result.reason

    def test_retry_completion_key_follows_the_persisted_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A retry resolves its key from task.model, not from the task type.

        The old ladder paired a 'code' task with codeApiKey regardless of which
        provider ``task.model`` names. Here the stashed model is xAI, so
        XAI_API_KEY is what has to be missing for the retry to fail terminally.
        """
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        task = self._make_task(task_type="code", model="xai/grok-4.3")
        result = self.service._retry_completion(task, {"messages": [{"role": "u", "content": "x"}]})
        assert result.status == "failed_terminal"
        assert "XAI_API_KEY" in result.reason

    @pytest.mark.parametrize("content", [None, "", "  \n "])
    def test_retry_completion_empty_content_fails_terminally(
        self, mocker: MockerFixture, content: str | None
    ) -> None:
        """GIVEN the retried model returns no text (thinking ate the budget)
        WHEN the retry finishes THEN it fails terminally instead of delivering
        a bare ``nick: ``, and the tokens it billed still ride along."""
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response(content, completion_tokens=4096),
        )
        task = self._make_task()
        result = self.service._retry_completion(task, {"messages": [{"role": "u", "content": "x"}]})
        assert result.status == "failed_terminal"
        assert result.reason
        assert result.completion_tokens == 4096

    def test_retry_completion_unmanaged_model_is_not_a_hard_failure(
        self, mocker: MockerFixture
    ) -> None:
        """A row stashed with a model we supply no key for still gets retried.

        ``task.model`` is a persisted column, so rows queued before a deploy can
        carry an empty or unrecognised model. Failing those terminally would
        silently kill in-flight @ask/@code/@draw recoveries across a restart;
        the call proceeds and the provider module reports whatever the real problem is.
        """
        fallback = mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("recovered"),
        )
        task = self._make_task(model="")
        result = self.service._retry_completion(task, {"messages": [{"role": "u", "content": "x"}]})
        assert result.status == "completed"
        assert fallback.called

    def test_retry_completion_reapplies_stashed_caps(self, mocker: MockerFixture) -> None:
        """The profile caps stashed at timeout ride the retry, as on the
        foreground call."""
        chat = mocker.patch(
            "llm.service.provider.chat", return_value=make_completion_response("ok")
        )
        task = self._make_task(model="xai/grok-4.3")
        self.service._retry_completion(
            task,
            {
                "messages": [{"role": "u", "content": "x"}],
                "max_tokens": 512,
            },
        )
        assert chat.call_args.kwargs["max_tokens"] == 512


def test_assistant_result_was_verse_defaults_false():
    assert AssistantResult(content="x", spend=Spend.ZERO).was_verse is False


def test_assistant_result_was_verse_settable():
    assert AssistantResult(content="x", spend=Spend.ZERO, was_verse=True).was_verse is True
