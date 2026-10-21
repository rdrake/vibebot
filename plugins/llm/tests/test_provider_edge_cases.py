"""Tests for API provider edge cases.

These tests verify how LLMService handles provider error conditions, partial
replies and key checks: the user-facing message for each provider exception,
key sanitization, and the pre-flight key guard.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from llm import provider
from llm.service import LLMService

from .conftest import FAKE_PROVIDER_KEYS, make_completion_response

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestProviderSpecificErrors:
    """Test handling of provider-specific error conditions."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with default config."""
        service, _ = make_service()
        return service

    def test_handles_timeout_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN timeout error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.RequestTimeoutError(
                "Request timed out", model="xai/grok-4.3", provider="xai"
            ),
        )
        result = service.completion("test", command="ask")

        assert "timed out" in result.content.lower()
        assert result.error is not None

    def test_handles_rate_limit_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN rate limit error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.RateLimitError(
                "Rate limit exceeded", model="xai/grok-4.3", provider="xai"
            ),
        )
        result = service.completion("test", command="ask")

        assert "rate limit" in result.content.lower()
        assert "Error" in result.content

    def test_handles_authentication_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN auth error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.AuthenticationError(
                "Invalid API key", model="xai/grok-4.3", provider="xai"
            ),
        )
        result = service.completion("test", command="ask")

        assert "api key" in result.content.lower() or "invalid" in result.content.lower()
        assert "Error" in result.content

    def test_handles_content_policy_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN content policy error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.ContentPolicyViolationError(
                "Content violates policy", model="xai/grok-4.3", provider="xai"
            ),
        )
        result = service.completion("test", command="ask")
        assert any(word in result.content.lower() for word in ("safety", "policy"))
        assert "Error" in result.content

    def test_refusal_text_never_reaches_the_channel(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a provider refusal WHEN completing THEN the provider's wording stays in the log."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.RefusalError(
                "Content violates usage guidelines. Team: abc, category: harassment",
                model="xai/grok-4.3",
                provider="xai",
            ),
        )
        result = service.completion("test", command="ask")
        assert result.error is not None
        assert "harassment" not in result.content
        assert "usage guidelines" not in result.content

    def test_handles_generic_api_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN generic API error WHEN completing THEN returns sanitized message."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=provider.ProviderError(
                "API error with key sk-secret123456789",
                model="xai/grok-4.3",
                provider="xai",
                status_code=500,
            ),
        )
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        # API key should be sanitized
        assert "sk-secret123456789" not in result.content

    def test_handles_unknown_exception(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN unknown exception WHEN completing THEN returns generic message."""
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=RuntimeError("Unexpected internal error"),
        )
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "Unexpected internal error" not in result.content  # Don't leak internals


class TestImageGenerationErrors:
    """Test error handling in image generation."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with draw config."""
        service, _ = make_service(imageModel="xai/grok-imagine-image", drawAutoRewriteMax=0)
        return service

    def test_handles_no_picture(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN the provider returns no picture WHEN generating image THEN returns content filter message."""
        mocker.patch("llm.service.provider.draw", return_value=provider.Picture(data=None))
        result = service.image_generation("test prompt")

        assert "No image generated" in result.content
        assert "content safety" in result.content.lower()

    def test_handles_timeout_during_generation(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN timeout WHEN generating image THEN returns timeout message."""
        mocker.patch(
            "llm.service.provider.draw",
            side_effect=provider.RequestTimeoutError(
                "Generation timed out", model="xai/grok-imagine-image", provider="xai"
            ),
        )
        result = service.image_generation("test prompt")

        assert "timed out" in result.content.lower()

    def test_sends_typing_done_on_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN error during generation WHEN with irc context THEN sends done indicator."""
        mock_irc = mocker.Mock()
        mock_irc.state = mocker.Mock()
        mock_irc.state.capabilities_ack = {"message-tags"}
        mock_irc.queueMsg = mocker.Mock()

        mock_msg = mocker.Mock()
        mock_msg.args = ("#test",)
        mock_msg.prefix = "testnick!user@host"

        mocker.patch(
            "llm.service.provider.draw",
            side_effect=Exception("API error"),
        )
        service.image_generation("test", irc=mock_irc, msg=mock_msg)

        # Should have sent typing indicators (active + done)
        assert mock_irc.queueMsg.call_count == 2
        # Last call should be typing=done
        last_msg = mock_irc.queueMsg.call_args_list[-1][0][0]
        assert last_msg.server_tags == {"+typing": "done"}


class TestPartialResponseHandling:
    """Test handling of partial or malformed responses."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:
        """Create service with default config."""
        service, _ = make_service()
        return service

    def test_handles_empty_content_response(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response with empty content WHEN completing THEN handles gracefully."""
        mocker.patch("llm.service.provider.chat", return_value=make_completion_response(""))
        result = service.completion("test", command="ask")

        # Should return empty string, not crash
        assert result.content == ""

    def test_handles_whitespace_only_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response with only whitespace WHEN completing THEN returns as-is."""
        mocker.patch(
            "llm.service.provider.chat", return_value=make_completion_response("   \n\t  ")
        )
        result = service.completion("test", command="ask")

        # Should return the whitespace (sanitized)
        assert result.content.strip() == ""


class TestAnthropicSpecificBehaviors:
    """Test Anthropic-specific behaviors."""

    @pytest.fixture
    def anthropic_service(self, make_service) -> LLMService:
        """Create service configured for Anthropic."""
        service, _ = make_service(assistantModel="anthropic/claude-haiku-5-5")
        return service

    @pytest.mark.parametrize(
        ("thinking_blocks", "expected"),
        [
            (({"type": "thinking", "thinking": "", "signature": "sig"},), True),
            ((), False),
        ],
    )
    def test_tool_turn_carries_thinking_blocks_back(
        self,
        anthropic_service: LLMService,
        mocker: MockerFixture,
        thinking_blocks: tuple,
        expected: bool,
    ) -> None:
        """GIVEN a tool-call turn with thinking blocks WHEN the loop continues
        THEN the assistant turn sent back carries them; without any, the key
        is absent so other providers never see it."""
        from llm.assistant import ToolResult

        from .conftest import make_tool_call

        first = make_completion_response(
            None, tool_calls=[make_tool_call("probe", {})], thinking=thinking_blocks
        )
        responses = [first, make_completion_response("done")]
        calls: list[dict] = []

        def fake_completion(**kwargs: object) -> object:
            calls.append(kwargs)  # type: ignore[arg-type]
            return responses[len(calls) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)
        anthropic_service.assistant_completion(
            prompt="probe it",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask"}),
            account="rdrake",
            extra_tools=[{"type": "function", "function": {"name": "probe", "parameters": {}}}],
            extra_handlers={"probe": lambda _args: ToolResult(content="ok")},
        )

        assert len(calls) == 2
        turn = next(m for m in calls[1]["messages"] if m.get("tool_calls"))
        assert ("thinking_blocks" in turn) is expected
        if expected:
            assert turn["thinking_blocks"] == list(thinking_blocks)


class TestAPIKeyHandling:
    """Test API key handling across different scenarios."""

    def test_missing_ask_key_returns_error(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key for the ask model's provider WHEN completing THEN returns error."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        service, _ = make_service(assistantModel="anthropic/claude-haiku-5-5")
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "ANTHROPIC_API_KEY" in result.content

    def test_empty_ask_key_returns_error(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an empty key for the ask model's provider THEN returns error.

        An empty value must be treated exactly like an unset one, otherwise ""
        reaches the provider and the failure surfaces as a provider auth error.
        """
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")
        service, _ = make_service(assistantModel="anthropic/claude-haiku-5-5")
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "ANTHROPIC_API_KEY" in result.content

    @pytest.mark.parametrize(
        "error_class",
        [
            provider.ProviderError,
            provider.RateLimitError,
            provider.AuthenticationError,
            provider.BadRequestError,
            provider.RefusalError,
            RuntimeError,
        ],
    )
    def test_api_key_sanitized_in_errors(
        self,
        make_service,
        mocker: MockerFixture,
        monkeypatch: pytest.MonkeyPatch,
        error_class: type[Exception],
    ) -> None:
        """GIVEN an error containing the key WHEN handling THEN key sanitized.

        `_sanitize` is environment-backed, so the key lives in the model's
        provider variable (ANTHROPIC_API_KEY), not the registry.
        """
        fake_key = "sk-ant-" + "x" * 25
        monkeypatch.setenv("ANTHROPIC_API_KEY", fake_key)
        service, _ = make_service(assistantModel="anthropic/claude-haiku-5-5")
        log = mocker.patch.object(service, "log")

        mocker.patch(
            "llm.service.provider.chat",
            side_effect=error_class(f"Error with key {fake_key}"),
        )
        result = service.completion("test", command="ask")

        # Key should not appear in the result or in anything logged
        assert fake_key not in str(result)
        assert fake_key not in str(log.mock_calls)


class TestSearchToolDate:
    """Anthropic calls with search_web carry today's date in its description."""

    def _capture(self, make_service, mocker: MockerFixture) -> tuple:
        service, _ = make_service()
        seen: dict[str, object] = {}
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=lambda **kwargs: seen.update(kwargs) or make_completion_response(),
        )
        return service, seen

    @staticmethod
    def _search_tools() -> list[dict]:
        return [
            {"type": "function", "function": {"name": "search_web", "description": "Search."}},
            {"type": "function", "function": {"name": "fetch_url", "description": "Fetch."}},
        ]

    def test_anthropic_search_tool_carries_the_date(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """GIVEN an anthropic call with search_web WHEN completing THEN its
        description ends with today's date, other tools and the caller's
        schemas are untouched."""
        service, seen = self._capture(make_service, mocker)
        tools = self._search_tools()

        service._timed_completion(
            "ask", model="anthropic/claude-haiku-5-5", messages=[], channel=None, tools=tools
        )

        sent = {t["function"]["name"]: t["function"]["description"] for t in seen["tools"]}
        assert re.fullmatch(r"Search\. The current date is \d{4}-\d{2}-\d{2}\.", sent["search_web"])
        assert sent["fetch_url"] == "Fetch."
        assert tools == self._search_tools()

    def test_other_providers_get_no_date(self, make_service, mocker: MockerFixture) -> None:
        service, seen = self._capture(make_service, mocker)

        service._timed_completion(
            "ask", model="xai/grok-4.3", messages=[], channel=None, tools=self._search_tools()
        )

        assert seen["tools"][0]["function"]["description"] == "Search."


class TestMissingKeyError:
    """The pre-flight key check names the model's provider variable."""

    def test_missing_key_error_names_provider_and_variable(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key WHEN building the error THEN it names both."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service()
        message = service._missing_key_error("xai/grok-4.3")
        assert message is not None
        assert "xai" in message
        assert "XAI_API_KEY" in message

    @pytest.mark.parametrize("model", ["vertex_ai/imagen-4.0-generate-001", "", "junk-model"])
    def test_unmanaged_models_have_no_error(self, make_service, model: str) -> None:
        """GIVEN an unmanaged model WHEN building the error THEN None.

        Unmanaged is not a failure here: the provider module reports it.
        """
        service, _ = make_service()
        assert service._missing_key_error(model) is None

    def test_error_never_contains_a_key(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN the provider's variable is unset WHEN building the error
        THEN a real message is produced and no configured key value appears in it.

        The previous version of this test set XAI_API_KEY (already the
        autouse ``_isolate_provider_env`` default), so
        ``_missing_key_error("xai/grok-4.3")`` returned None, ``or ""``
        produced ``""``, and ``secret not in ""`` held for any
        implementation — including one that interpolated the key. Unsetting
        the variable forces the message-producing branch to actually run,
        and checking every other provider's configured value (still present
        in the environment) catches an implementation that dumps configured
        secrets into the message rather than just naming the missing one.
        """
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service()
        message = service._missing_key_error("xai/grok-4.3")
        assert message is not None
        for secret in FAKE_PROVIDER_KEYS.values():
            assert secret not in message


class TestGuardsUseTheEffectiveModel:
    """A pre-flight key check must read the model the call will actually send.

    ``model_override`` can name a different provider than the channel's registry
    model. Checking the registry value alone produces the wrong verdict in both
    directions: a false block when the override's provider is configured and the
    registry model's is not, and a missed error the other way round. The key
    itself is always right — it comes from ``_timed_completion`` — so the damage
    is entirely in the guard, which is why nothing else in the suite notices.
    """

    def test_completion_override_provider_is_what_is_checked(
        self, make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an override on an unconfigured provider THEN the error names it."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        # The registry model's provider IS configured — only the override's is not.
        service, _ = make_service(assistantModel="anthropic/claude-haiku-5-5")
        completion = mocker.patch("llm.service.provider.chat")

        result = service.completion("hi", command="ask", model_override="xai/grok-4.3")

        assert result.error is not None
        assert "XAI_API_KEY" in result.content
        completion.assert_not_called()

    def test_completion_override_is_not_blocked_by_the_registry_model(
        self, make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a configured override THEN an unconfigured registry model cannot block it."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service(assistantModel="xai/grok-4.3")
        completion = mocker.patch(
            "llm.service.provider.chat", return_value=make_completion_response()
        )

        result = service.completion(
            "hi", command="ask", model_override="anthropic/claude-haiku-5-5"
        )

        assert result.error is None
        assert completion.call_args.kwargs["model"] == "anthropic/claude-haiku-5-5"

    def test_assistant_override_provider_is_what_is_checked(
        self, make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an override on an unconfigured provider THEN assistant_completion errors."""
        from llm.profile import PROFILE_CHAT

        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service(assistantModel="anthropic/claude-haiku-5-5")
        completion = mocker.patch("llm.service.provider.chat")

        result = service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
            model_override="xai/grok-4.3",
        )

        assert result.error is not None
        assert "XAI_API_KEY" in result.content
        completion.assert_not_called()

    def test_assistant_override_is_not_blocked_by_the_registry_model(
        self, make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a configured override THEN an unconfigured registry model cannot block it."""
        from llm.profile import PROFILE_CHAT

        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service(assistantModel="xai/grok-4.3")
        completion = mocker.patch(
            "llm.service.provider.chat", return_value=make_completion_response()
        )

        result = service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
            model_override="anthropic/claude-haiku-5-5",
        )

        assert result.error is None
        assert completion.call_args.kwargs["model"] == "anthropic/claude-haiku-5-5"


class TestStripSelfLabel:
    """Models continue the ``nick: text`` channel transcript and label their reply."""

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            (
                "vibebot: Your bollocks are two satsumas, eck.",
                "Your bollocks are two satsumas, eck.",
            ),
            ("VibeBot, fine, I'll do it.", "fine, I'll do it."),
            ("<vibebot> sure thing", "sure thing"),
            ("vibebot I'm an IRC bot.", "I'm an IRC bot."),
            ("vibebot is just a bot with opinions.", "vibebot is just a bot with opinions."),
            ("rdrake: your code sucks.", "rdrake: your code sucks."),
            ("Ask vibebot: it knows.", "Ask vibebot: it knows."),
            ("vibebot:", "vibebot:"),
        ],
    )
    def test_strips_only_a_leading_self_label(self, reply: str, expected: str) -> None:
        from llm.service import _strip_self_label

        assert _strip_self_label(reply, "vibebot") == expected

    def test_final_reply_is_unlabelled(self, make_service, mocker) -> None:
        """GIVEN the model labels its reply WHEN the turn ends THEN the label is gone."""
        service, _ = make_service()
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("vibebot: Eck, you absolute weapon."),
        )

        result = service.assistant_completion(
            prompt="call eck a weapon",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
        )

        assert result.content == "Eck, you absolute weapon."
