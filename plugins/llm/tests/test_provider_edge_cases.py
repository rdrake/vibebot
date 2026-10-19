"""Tests for API provider edge cases.

These tests verify handling of provider-specific behaviors, error conditions,
and edge cases that may occur with different LLM providers.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import litellm
import pytest
from llm.service import LLMService

from .conftest import FAKE_PROVIDER_KEYS, make_completion_response

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def _stub_response():
    """A litellm-shaped chat-completion response, for kwarg-capture stubs."""
    return make_completion_response()


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
            "llm.service.litellm.completion",
            side_effect=litellm.Timeout(
                message="Request timed out", model="gpt-4", llm_provider="openai"
            ),
        )
        result = service.completion("test", command="ask")

        assert "timed out" in result.content.lower()
        assert result.error is not None

    def test_handles_rate_limit_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN rate limit error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm.RateLimitError(
                message="Rate limit exceeded",
                model="gpt-4",
                llm_provider="openai",
            ),
        )
        result = service.completion("test", command="ask")

        assert "rate limit" in result.content.lower()
        assert "Error" in result.content

    def test_handles_authentication_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN auth error WHEN completing THEN returns user-friendly message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm.AuthenticationError(
                message="Invalid API key",
                model="gpt-4",
                llm_provider="openai",
            ),
        )
        result = service.completion("test", command="ask")

        assert "api key" in result.content.lower() or "invalid" in result.content.lower()
        assert "Error" in result.content

    @pytest.mark.parametrize(
        ("error_class", "error_kwargs", "expected_words"),
        [
            (
                litellm.ContentPolicyViolationError,
                {"message": "Content violates policy", "model": "gpt-4", "llm_provider": "openai"},
                ["safety", "policy"],
            ),
            (
                litellm.BadRequestError,
                {
                    "message": "moderation_blocked: content was flagged",
                    "model": "gpt-4",
                    "llm_provider": "openai",
                },
                ["safety", "policy"],
            ),
            (
                litellm.BadRequestError,
                {
                    "message": "Blocked by safety system filters",
                    "model": "gpt-4",
                    "llm_provider": "openai",
                },
                ["safety", "policy"],
            ),
        ],
        ids=[
            "content_policy_violation",
            "bad_request_moderation_blocked",
            "bad_request_safety_system",
        ],
    )
    def test_handles_content_safety_errors(
        self,
        service: LLMService,
        mocker: MockerFixture,
        error_class,
        error_kwargs,
        expected_words,
    ) -> None:
        """GIVEN content safety error WHEN completing THEN returns user-friendly message."""
        mocker.patch("llm.service.litellm.completion", side_effect=error_class(**error_kwargs))
        result = service.completion("test", command="ask")
        assert any(word in result.content.lower() for word in expected_words)
        assert "Error" in result.content

    def test_handles_generic_api_error(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN generic API error WHEN completing THEN returns sanitized message."""
        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=litellm.APIError(
                message="API error with key sk-secret123456789",
                model="gpt-4",
                llm_provider="openai",
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
            "llm.service.litellm.completion",
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
        service, _ = make_service(imageModel="dall-e-3", drawAutoRewriteMax=0)
        return service

    def test_handles_empty_response_data(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN empty response data WHEN generating image THEN returns content filter message."""
        mock_response = mocker.Mock()
        mock_response.data = []

        mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = service.image_generation("test prompt")

        assert "No image generated" in result.content
        assert "content safety" in result.content.lower()

    def test_handles_none_url_and_b64(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN response with neither URL nor base64 WHEN generating THEN returns error."""
        mock_response = mocker.Mock()
        mock_response.data = [mocker.Mock(url=None, b64_json=None)]

        mocker.patch("llm.service.litellm.image_generation", return_value=mock_response)
        result = service.image_generation("test prompt")

        assert "No image generated" in result.content

    def test_handles_timeout_during_generation(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN timeout WHEN generating image THEN returns timeout message."""
        mocker.patch(
            "llm.service.litellm.image_generation",
            side_effect=litellm.Timeout(
                message="Generation timed out", model="dall-e-3", llm_provider="openai"
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
            "llm.service.litellm.image_generation",
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
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = ""
        mock_response.choices[0].message.tool_calls = None

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.completion("test", command="ask")

        # Should return empty string, not crash
        assert result.content == ""

    def test_handles_none_content_response(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response with None content WHEN completing THEN handles gracefully."""
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = None

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.completion("test", command="ask")

        # Should normalize None to empty string
        assert result.content == ""

    def test_handles_no_choices_response(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN response with no choices WHEN completing THEN handles gracefully."""
        mock_response = mocker.Mock()
        mock_response.choices = []

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        # Should raise IndexError which gets caught
        result = service.completion("test", command="ask")

        assert "Error" in result.content

    def test_handles_whitespace_only_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response with only whitespace WHEN completing THEN returns as-is."""
        mock_response = mocker.Mock()
        mock_response.choices = [mocker.Mock()]
        mock_response.choices[0].message = mocker.Mock()
        mock_response.choices[0].message.content = "   \n\t  "
        mock_response.choices[0].message.tool_calls = None

        mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = service.completion("test", command="ask")

        # Should return the whitespace (sanitized)
        assert result.content.strip() == ""


class TestGeminiSpecificBehaviors:
    """Test Gemini-specific features and behaviors."""

    @pytest.fixture
    def gemini_service(self, make_service) -> LLMService:
        """Create service configured for Gemini."""
        service, _ = make_service(assistantModel="gemini/gemini-2.0-flash")
        return service

    def test_gemini_tools_included_for_2x_models(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini 2.x model WHEN getting tools THEN returns search tools."""
        tools = gemini_service._get_gemini_tools("gemini/gemini-2.0-flash")
        assert tools is not None
        assert len(tools) == 2
        assert {"googleSearch": {}} in tools
        assert {"urlContext": {}} in tools

    def test_gemini_tools_included_for_25_models(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini 2.5 model WHEN getting tools THEN returns search tools."""
        tools = gemini_service._get_gemini_tools("gemini/gemini-2.5-pro")
        assert tools is not None
        assert len(tools) == 2

    def test_gemini_tools_not_included_for_15_models(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini 1.5 model WHEN getting tools THEN returns None."""
        tools = gemini_service._get_gemini_tools("gemini/gemini-1.5-flash")
        assert tools is None

    def test_gemini_tools_not_included_for_imagen(self, gemini_service: LLMService) -> None:
        """GIVEN Imagen model WHEN getting tools THEN returns None."""
        tools = gemini_service._get_gemini_tools("vertex_ai/imagen-4.0-generate-001")
        assert tools is None

    def test_gemini_safety_settings_applied(self, gemini_service: LLMService) -> None:
        """GIVEN Gemini model WHEN getting safety settings THEN all categories set."""
        settings = gemini_service._get_safety_settings()

        expected_categories = [
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
            "HARM_CATEGORY_CIVIC_INTEGRITY",
        ]

        assert len(settings) == len(expected_categories)
        for setting in settings:
            assert setting["category"] in expected_categories
            assert setting["threshold"] == "BLOCK_NONE"

    def test_completion_passes_gemini_tools(
        self, gemini_service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN Gemini model WHEN completing THEN tools passed to API."""
        call_kwargs = {}

        def capture_call(**kwargs):
            call_kwargs.update(kwargs)
            response = mocker.Mock()
            response.choices = [mocker.Mock()]
            response.choices[0].message = mocker.Mock()
            response.choices[0].message.content = "Response"
            return response

        mocker.patch("llm.service.litellm.completion", side_effect=capture_call)
        gemini_service.completion("test", command="ask")

        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert "safety_settings" in call_kwargs
        assert call_kwargs["safety_settings"] is not None


class TestOpenAISpecificBehaviors:
    """Test OpenAI-specific behaviors."""

    @pytest.fixture
    def openai_service(self, make_service) -> LLMService:
        """Create service configured for OpenAI."""
        service, _ = make_service()
        return service

    def test_no_gemini_tools_for_openai(self, openai_service: LLMService) -> None:
        """GIVEN OpenAI model WHEN getting tools THEN returns None."""
        tools = openai_service._get_gemini_tools("gpt-4")
        assert tools is None

    def test_no_safety_settings_for_openai(
        self, openai_service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN OpenAI model WHEN completing THEN no safety_settings passed."""
        call_kwargs = {}

        def capture_call(**kwargs):
            call_kwargs.update(kwargs)
            response = mocker.Mock()
            response.choices = [mocker.Mock()]
            response.choices[0].message = mocker.Mock()
            response.choices[0].message.content = "Response"
            return response

        mocker.patch("llm.service.litellm.completion", side_effect=capture_call)
        openai_service.completion("test", command="ask")

        assert call_kwargs.get("safety_settings") is None


class TestAnthropicSpecificBehaviors:
    """Test Anthropic-specific behaviors."""

    @pytest.fixture
    def anthropic_service(self, make_service) -> LLMService:
        """Create service configured for Anthropic."""
        service, _ = make_service(assistantModel="anthropic/claude-3-opus")
        return service

    def test_no_gemini_tools_for_anthropic(self, anthropic_service: LLMService) -> None:
        """GIVEN Anthropic model WHEN getting tools THEN returns None."""
        tools = anthropic_service._get_gemini_tools("anthropic/claude-3-opus")
        assert tools is None

    @pytest.mark.parametrize(
        ("thinking_blocks", "expected"),
        [
            ([{"type": "thinking", "thinking": "", "signature": "sig"}], True),
            (None, False),
        ],
    )
    def test_tool_turn_carries_thinking_blocks_back(
        self,
        anthropic_service: LLMService,
        mocker: MockerFixture,
        thinking_blocks: list | None,
        expected: bool,
    ) -> None:
        """GIVEN a tool-call turn with thinking blocks WHEN the loop continues
        THEN the assistant turn sent back carries them; without any, the key
        is absent so OpenAI-compatible providers never see it."""
        from llm.assistant import ToolResult

        from .conftest import make_tool_call

        first = make_completion_response(None, tool_calls=[make_tool_call("probe", {})])
        first.choices[0].message.thinking_blocks = thinking_blocks
        responses = [first, make_completion_response("done")]
        calls: list[dict] = []

        def fake_completion(**kwargs: object) -> object:
            calls.append(kwargs)  # type: ignore[arg-type]
            return responses[len(calls) - 1]

        mocker.patch("llm.service.litellm.completion", side_effect=fake_completion)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
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
            assert turn["thinking_blocks"] == thinking_blocks


class TestAPIKeyHandling:
    """Test API key handling across different scenarios."""

    def test_missing_ask_key_returns_error(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key for the ask model's provider WHEN completing THEN returns error.

        The default ``assistantModel`` is TEST_MODEL ("gpt-4"), whose provider
        is openai, so unsetting OPENAI_API_KEY is what makes the key missing.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service, _ = make_service()
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "OPENAI_API_KEY" in result.content

    def test_empty_ask_key_returns_error(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an empty key for the ask model's provider THEN returns error.

        An empty value must be treated exactly like an unset one, otherwise ""
        reaches litellm and the failure surfaces as a provider auth error.
        """
        monkeypatch.setenv("OPENAI_API_KEY", "")
        service, _ = make_service()
        result = service.completion("test", command="ask")

        assert "Error" in result.content
        assert "OPENAI_API_KEY" in result.content

    def test_api_key_sanitized_in_errors(
        self, make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN API error containing key WHEN handling THEN key sanitized.

        TEST_MODEL is "gpt-4", whose provider is openai, so `_sanitize` (now
        environment-backed) needs the key in OPENAI_API_KEY, not the registry.
        """
        fake_key = "sk-" + "x" * 25  # noqa: S105
        monkeypatch.setenv("OPENAI_API_KEY", fake_key)
        service, _ = make_service()

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception(f"Error with key {fake_key}"),
        )
        result = service.completion("test", command="ask")

        # Key should not appear in result
        assert fake_key not in str(result)


class TestAnthropicPromptCaching:
    """Anthropic caches only marked prefixes; the boundary marks them."""

    def _capture(self, make_service, monkeypatch: pytest.MonkeyPatch) -> tuple:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-for-tests-0000")
        service, _ = make_service()
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        return service, seen

    def test_anthropic_call_marks_system_and_tail(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an anthropic model WHEN completing THEN the first system message
        carries a breakpoint, automatic caching is on, and the caller's list is
        untouched (it is reused across tool-loop steps)."""
        service, seen = self._capture(make_service, monkeypatch)
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "late note"},
        ]

        service._timed_completion(
            "ask", model="anthropic/claude-haiku-5-5", messages=messages, channel=None
        )

        sent = seen["messages"]
        assert sent[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in sent[2]
        assert seen["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in messages[0]

    def test_list_content_marks_the_last_block(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """LiteLLM reads the marker per block when system content is a list."""
        service, seen = self._capture(make_service, monkeypatch)
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            },
            {"role": "user", "content": "hi"},
        ]

        service._timed_completion(
            "ask", model="anthropic/claude-haiku-5-5", messages=messages, channel=None
        )

        blocks = seen["messages"][0]["content"]
        assert "cache_control" not in blocks[0]
        assert blocks[1]["cache_control"] == {"type": "ephemeral"}

    def test_other_providers_are_not_marked(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an xai model WHEN completing THEN no Anthropic cache fields."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, seen = self._capture(make_service, monkeypatch)
        messages = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "hi"}]

        service._timed_completion("ask", model="xai/grok-4.3", messages=messages, channel=None)

        assert "cache_control" not in seen
        assert "cache_control" not in seen["messages"][0]

    @staticmethod
    def _search_tools() -> list[dict]:
        return [
            {"type": "function", "function": {"name": "search_web", "description": "Search."}},
            {"type": "function", "function": {"name": "fetch_url", "description": "Fetch."}},
        ]

    def test_anthropic_search_tool_carries_the_date(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an anthropic call with search_web WHEN completing THEN its
        description ends with today's date, other tools and the caller's
        schemas are untouched."""
        service, seen = self._capture(make_service, monkeypatch)
        tools = self._search_tools()

        service._timed_completion(
            "ask", model="anthropic/claude-haiku-5-5", messages=[], channel=None, tools=tools
        )

        sent = {t["function"]["name"]: t["function"]["description"] for t in seen["tools"]}
        assert re.fullmatch(r"Search\. The current date is \d{4}-\d{2}-\d{2}\.", sent["search_web"])
        assert sent["fetch_url"] == "Fetch."
        assert tools == self._search_tools()

    def test_other_providers_get_no_date(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, seen = self._capture(make_service, monkeypatch)

        service._timed_completion(
            "ask", model="xai/grok-4.3", messages=[], channel=None, tools=self._search_tools()
        )

        assert seen["tools"][0]["function"]["description"] == "Search."


class TestBoundaryKeyResolution:
    """The key litellm receives is the one the model's provider variable holds."""

    def test_xai_model_gets_xai_key(self, make_service, monkeypatch: pytest.MonkeyPatch) -> None:
        """GIVEN an xai model WHEN completing THEN XAI_API_KEY reaches litellm."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        service, _ = make_service(assistantModel="xai/grok-4.3")
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion("ask", model="xai/grok-4.3", messages=[], channel=None)
        assert seen["api_key"] == "xai-fake-key-for-tests-0000"

    def test_gemini_model_gets_gemini_key(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a gemini model WHEN completing THEN the gemini key, not the xai one."""
        monkeypatch.setenv("XAI_API_KEY", "xai-fake-key-for-tests-0000")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-fake-key-for-tests-0000")
        service, _ = make_service()
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion(
            "ask", model="gemini/gemini-3-flash-preview", messages=[], channel=None
        )
        assert seen["api_key"] == "AIza-fake-key-for-tests-0000"

    def test_unmanaged_provider_passes_none(
        self, make_service, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN a vertex_ai model WHEN completing THEN api_key is None.

        None is what makes LiteLLM fall back to ADC. Sending "" or a mapped
        key here would break service-account auth.
        """
        service, _ = make_service()
        seen: dict[str, object] = {}
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or _stub_response(),
        )
        service._timed_completion(
            "ask", model="vertex_ai/gemini-2.0-flash", messages=[], channel=None
        )
        assert seen["api_key"] is None

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

        Unmanaged is not a failure — it is delegation to LiteLLM.
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
        service, _ = make_service(assistantModel="gemini/gemini-flash-latest")
        completion = mocker.patch("llm.service.litellm.completion")

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
        completion = mocker.patch("llm.service.litellm.completion", return_value=_stub_response())
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.completion(
            "hi", command="ask", model_override="gemini/gemini-3-flash-preview"
        )

        assert result.error is None
        kwargs = completion.call_args.kwargs
        assert kwargs["model"] == "gemini/gemini-3-flash-preview"
        assert kwargs["api_key"] == FAKE_PROVIDER_KEYS["GEMINI_API_KEY"]

    def test_assistant_override_provider_is_what_is_checked(
        self, make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN an override on an unconfigured provider THEN assistant_completion errors."""
        from llm.profile import PROFILE_CHAT

        monkeypatch.delenv("XAI_API_KEY", raising=False)
        service, _ = make_service(assistantModel="gemini/gemini-flash-latest")
        completion = mocker.patch("llm.service.litellm.completion")

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
        completion = mocker.patch("llm.service.litellm.completion", return_value=_stub_response())
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.assistant_completion(
            prompt="hi",
            nick="tester",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=PROFILE_CHAT,
            model_override="gemini/gemini-3-flash-preview",
        )

        assert result.error is None
        kwargs = completion.call_args.kwargs
        assert kwargs["model"] == "gemini/gemini-3-flash-preview"
        assert kwargs["api_key"] == FAKE_PROVIDER_KEYS["GEMINI_API_KEY"]


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
            "llm.service.litellm.completion",
            return_value=make_completion_response("vibebot: Eck, you absolute weapon."),
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        result = service.assistant_completion(
            prompt="call eck a weapon",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="vibebot",
        )

        assert result.content == "Eck, you absolute weapon."
