"""Service completion: provider routing, grounding, retry, responses API, search/url, assistant facade."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.identity import Caller, Identity
from llm.service import (
    AssistantRequestContext,
    AssistantResult,
    CompletionResult,
    LLMService,
    Spend,
)

from .conftest import FAKE_PROVIDER_KEYS, make_completion_response

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


class TestResolveGroundingKwargs:
    """Provider-aware grounding kwargs for search/url completion."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service) -> None:
        self.service, _ = make_service()

    @pytest.mark.parametrize(
        ("model", "kind"),
        [
            ("gemini/gemini-2.5-flash", "search"),
            ("gemini/gemini-2.5-flash", "url"),
            ("vertex_ai/gemini-2.5-pro", "search"),
            ("vertex_ai_beta/gemini-2.5-pro", "url"),
        ],
    )
    def test_gemini_provider_registers_both_grounding_tools(self, model: str, kind: str) -> None:
        # Gemini supports both googleSearch and urlContext on the same
        # request; registering both lets the model pivot between
        # searching the web and fetching a specific URL within one turn.
        kwargs = self.service._resolve_grounding_kwargs(model, kind)
        assert kwargs == {"tools": [{"googleSearch": {}}, {"urlContext": {}}]}

    @pytest.mark.parametrize("kind", ["search", "url"])
    def test_xai_provider_drops_tools_chat_completions_path(self, kind: str) -> None:
        # xAI grounding now goes through the Responses API in
        # ``_xai_responses_call``; this kwargs path is for Chat
        # Completions only, so it must hand back an empty tools list.
        kwargs = self.service._resolve_grounding_kwargs("xai/grok-4.3", kind)
        assert kwargs == {"tools": []}
        assert "extra_body" not in kwargs

    @pytest.mark.parametrize(
        "model",
        ["openai/gpt-4", "anthropic/claude-3-sonnet", "gpt-4o-mini", "ollama/llama3"],
    )
    @pytest.mark.parametrize("kind", ["search", "url"])
    def test_other_providers_drop_tools_for_plain_completion(self, model: str, kind: str) -> None:
        kwargs = self.service._resolve_grounding_kwargs(model, kind)
        assert kwargs == {"tools": []}

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown grounding kind"):
            self.service._resolve_grounding_kwargs("gemini/gemini-2.5-flash", "wat")


class TestSearchCompletionProviderRouting:
    """search_completion + url_completion route grounding kwargs by provider."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.service, self.plugin = make_service()
        self._completion_mock = mocker.patch.object(self.service, "_completion_with_tool_fallback")
        # Minimal response object — search_completion only reads choices and
        # passes the response to _check_grounding_used / _extract_usage.
        response = mocker.MagicMock()
        response.choices[0].message.content = "result"
        self._completion_mock.return_value = response
        mocker.patch.object(self.service, "_check_grounding_used", return_value=False)
        mocker.patch.object(self.service, "_extract_usage", return_value=(10, 20, 0.001))

    def _captured_kwargs(self):
        return self._completion_mock.call_args.kwargs["optional_kwargs"]

    def test_gemini_search_keeps_google_search_tool(self) -> None:
        self.plugin.registryValue.side_effect = lambda k, ch=None: (
            "gemini/gemini-2.5-flash" if k == "searchModel" else 30 if k == "timeout" else ""
        )
        self.service.search_completion("hi", channel="#t")
        kwargs = self._captured_kwargs()
        # Both grounding tools ride together so Gemini can pivot from
        # searching to fetching a referenced URL within one turn.
        assert kwargs["tools"] == [{"googleSearch": {}}, {"urlContext": {}}]

    def test_xai_search_skips_chat_completions(self, mocker: MockerFixture) -> None:
        # xAI must not hit the Chat Completions path at all — search goes
        # through ``_xai_responses_call`` (Responses API).
        responses_mock = mocker.patch.object(self.service, "_xai_responses_call")
        responses_mock.return_value = mocker.MagicMock()
        self.plugin.registryValue.side_effect = lambda k, ch=None: (
            "xai/grok-4.3" if k == "searchModel" else 30 if k == "timeout" else ""
        )
        self.service.search_completion("hi", channel="#t")
        self._completion_mock.assert_not_called()
        responses_mock.assert_called_once()
        call_kwargs = responses_mock.call_args.kwargs
        assert call_kwargs["model"] == "xai/grok-4.3"
        # No api_key is threaded any more — _xai_responses_call resolves it
        # from the model it is about to send.
        assert "api_key" not in call_kwargs
        assert call_kwargs["timeout"] == 30
        assert call_kwargs["kind"] == "search"
        assert responses_mock.call_args.args[0] == "hi"

    def test_gemini_url_uses_url_context(self) -> None:
        self.plugin.registryValue.side_effect = lambda k, ch=None: (
            "gemini/gemini-2.5-flash" if k == "searchModel" else 30 if k == "timeout" else ""
        )
        self.service.url_completion("https://example.com", channel="#t")
        kwargs = self._captured_kwargs()
        # Both grounding tools ride together so Gemini can pivot from
        # fetching the URL to searching for related context within one turn.
        assert kwargs["tools"] == [{"googleSearch": {}}, {"urlContext": {}}]

    def test_xai_url_skips_chat_completions(self, mocker: MockerFixture) -> None:
        # Same dispatch story for URL fetch — xAI uses Responses API
        # web_search instead of the Chat Completions urlContext path.
        responses_mock = mocker.patch.object(self.service, "_xai_responses_call")
        responses_mock.return_value = mocker.MagicMock()
        self.plugin.registryValue.side_effect = lambda k, ch=None: (
            "xai/grok-4.3" if k == "searchModel" else 30 if k == "timeout" else ""
        )
        self.service.url_completion("https://example.com", channel="#t")
        self._completion_mock.assert_not_called()
        responses_mock.assert_called_once()
        call_kwargs = responses_mock.call_args.kwargs
        assert call_kwargs["kind"] == "url"
        assert "https://example.com" in responses_mock.call_args.args[0]

    def test_gemini_url_none_content_coerced_to_empty_string(self) -> None:
        # Regression: Gemini urlContext can run the fetch but return
        # content=None. A null tool-result content propagates into the
        # follow-up completion and xAI's strict deserializer rejects it
        # with "missing field `content`" (prod 422s 2026-06-01). The
        # ToolResult must carry "" rather than None.
        self._completion_mock.return_value.choices[0].message.content = None
        self.plugin.registryValue.side_effect = lambda k, ch=None: (
            "gemini/gemini-2.5-flash" if k == "searchModel" else 30 if k == "timeout" else ""
        )
        result = self.service.url_completion("https://example.com", channel="#t")
        assert result.content == ""
        assert result.content is not None


class TestXAIResponsesCall:
    """xAI Responses API path: web_search tool, citations, usage shape."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.service, _ = make_service()
        self.mocker = mocker

    def _make_response(
        self,
        text: str,
        *,
        annotations: list | None = None,
        with_search_call: bool = False,
        input_tokens: int = 11,
        output_tokens: int = 22,
    ):
        output: list[dict] = []
        if with_search_call:
            output.append({"type": "web_search_call", "id": "ws_1"})
        output.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": text, "annotations": annotations or []}
                ],
            }
        )
        response = self.mocker.MagicMock()
        response.output_text = text
        response.output = output
        response.usage = self.mocker.Mock(
            input_tokens=input_tokens, output_tokens=output_tokens, cost=None
        )
        return response

    def test_sends_web_search_tool_to_responses_api(self) -> None:
        responses = self.mocker.patch(
            "llm.service.litellm.responses",
            return_value=self._make_response("ok", with_search_call=True),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        self.service._xai_responses_call(
            "what is grok", model="xai/grok-4.3", timeout=30, kind="search"
        )

        responses.assert_called_once()
        kwargs = responses.call_args.kwargs
        assert kwargs["model"] == "xai/grok-4.3"
        assert kwargs["input"] == "what is grok"
        assert kwargs["tools"] == [{"type": "web_search"}]
        # xai/ model -> XAI_API_KEY. The Responses path resolves the key from
        # the model, so a gemini key can never ride an xAI request.
        assert kwargs["api_key"] == FAKE_PROVIDER_KEYS["XAI_API_KEY"]
        assert kwargs["timeout"] == 30

    def test_grounding_true_when_web_search_call_in_output(self) -> None:
        self.mocker.patch(
            "llm.service.litellm.responses",
            return_value=self._make_response("ok", with_search_call=True),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service._xai_responses_call(
            "q", model="xai/grok-4.3", timeout=30, kind="search"
        )
        assert result.grounding_used is True

    def test_grounding_true_when_annotations_present(self) -> None:
        self.mocker.patch(
            "llm.service.litellm.responses",
            return_value=self._make_response(
                "ok",
                annotations=[{"type": "url_citation", "url": "https://example.com", "title": "Ex"}],
            ),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service._xai_responses_call(
            "q", model="xai/grok-4.3", timeout=30, kind="search"
        )
        assert result.grounding_used is True

    def test_grounding_false_when_no_search_signal(self) -> None:
        self.mocker.patch(
            "llm.service.litellm.responses",
            return_value=self._make_response("ok"),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service._xai_responses_call(
            "q", model="xai/grok-4.3", timeout=30, kind="search"
        )
        assert result.grounding_used is False

    def test_extracts_responses_api_token_usage(self) -> None:
        # Responses API names tokens input_tokens/output_tokens — make sure
        # the helper maps those to the chat-style prompt/completion fields
        # (extract_usage would silently zero them otherwise).
        self.mocker.patch(
            "llm.service.litellm.responses",
            return_value=self._make_response(
                "hi", input_tokens=42, output_tokens=7, with_search_call=True
            ),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        result = self.service._xai_responses_call(
            "q", model="xai/grok-4.3", timeout=30, kind="search"
        )
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 7

    def test_returns_error_tool_result_on_exception(self) -> None:
        self.mocker.patch("llm.service.litellm.responses", side_effect=RuntimeError("upstream 500"))
        result = self.service._xai_responses_call(
            "q", model="xai/grok-4.3", timeout=30, kind="search"
        )
        assert "Search failed" in result.content
        result_url = self.service._xai_responses_call(
            "q", model="xai/grok-4.3", timeout=30, kind="url"
        )
        assert "URL fetch failed" in result_url.content

    def test_responses_uses_prompt_cache_key_in_extra_body(self) -> None:
        """xAI Responses API expects prompt_cache_key as a body field, not as a header."""
        responses = self.mocker.patch(
            "llm.service.litellm.responses",
            return_value=self._make_response("ok", with_search_call=True),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        self.service._xai_responses_call(
            "q",
            model="xai/grok-4.3",
            timeout=30,
            kind="search",
            channel="#dev",
        )

        kwargs = responses.call_args.kwargs
        # ``kind="search"`` maps to the ``grounded`` cache lane so the
        # short search prefix doesn't compete with assistant_step_* on the
        # same server.
        assert kwargs.get("extra_body") == {"prompt_cache_key": "chan:#dev:grounded"}
        assert "extra_headers" not in kwargs

    def test_responses_omits_cache_key_without_channel(self) -> None:
        """Without a channel context, no prompt_cache_key is attached."""
        responses = self.mocker.patch(
            "llm.service.litellm.responses",
            return_value=self._make_response("ok", with_search_call=True),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        self.service._xai_responses_call("q", model="xai/grok-4.3", timeout=30, kind="search")

        kwargs = responses.call_args.kwargs
        assert "extra_body" not in kwargs
        assert "extra_headers" not in kwargs


class TestGroundingDetection:
    """Tests for _check_grounding_used and CompletionResult."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service(
            assistantModel="gemini/gemini-2.0-flash",
            assistantSystemPrompt="You are helpful.",
            timeout=30,
            maxPromptLength=10000,
            commandPrefixes=["."],
        )

    def test_check_grounding_used_returns_false_for_no_metadata(self) -> None:
        """GIVEN response with no grounding metadata WHEN checking THEN returns False."""
        mock_response = self.mocker.Mock(spec=["choices"])
        mock_choice = self.mocker.Mock(spec=["message"])
        mock_message = self.mocker.Mock(spec=["tool_calls"])
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = self.service._check_grounding_used(mock_response)
        assert result is False

    def test_check_grounding_used_returns_true_for_grounding_metadata(self) -> None:
        """GIVEN response with grounding_metadata WHEN checking THEN returns True."""
        mock_response = self.mocker.Mock(spec=["choices"])
        mock_choice = self.mocker.Mock(spec=["message", "grounding_metadata"])
        mock_choice.grounding_metadata = {"search_queries": ["test"]}
        mock_message = self.mocker.Mock(spec=["tool_calls"])
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = self.service._check_grounding_used(mock_response)
        assert result is True

    def test_check_grounding_used_returns_true_for_google_search_tool_call(self) -> None:
        """GIVEN response with googleSearch tool call WHEN checking THEN returns True."""
        mock_tool_call = self.mocker.Mock()
        mock_tool_call.function = self.mocker.Mock()
        mock_tool_call.function.name = "googleSearch"

        mock_response = self.mocker.Mock(spec=["choices"])
        mock_choice = self.mocker.Mock(spec=["message"])
        mock_message = self.mocker.Mock(spec=["tool_calls"])
        mock_message.tool_calls = [mock_tool_call]
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = self.service._check_grounding_used(mock_response)
        assert result is True

    def test_check_grounding_used_handles_missing_attributes(self) -> None:
        """GIVEN response with missing attributes WHEN checking THEN handles gracefully."""
        mock_response = self.mocker.Mock(spec=[])  # Empty spec means no attributes

        result = self.service._check_grounding_used(mock_response)
        assert result is False

    def test_check_grounding_used_returns_true_for_vertex_ai_grounding_metadata(self) -> None:
        """GIVEN response with vertex_ai_grounding_metadata in _hidden_params WHEN checking THEN returns True."""
        mock_response = self.mocker.Mock(spec=["choices", "_hidden_params"])
        mock_response._hidden_params = {
            "vertex_ai_grounding_metadata": {"web_search_queries": ["test"]}
        }
        mock_choice = self.mocker.Mock(spec=["message"])
        mock_message = self.mocker.Mock(spec=["tool_calls"])
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = self.service._check_grounding_used(mock_response)
        assert result is True

    def test_check_grounding_used_returns_false_for_empty_grounding_metadata(self) -> None:
        """GIVEN response with empty vertex_ai_grounding_metadata WHEN checking THEN returns False.

        LiteLLM may set the grounding metadata key to None or empty dict when
        grounding tools are available but weren't actually used.
        """
        mock_response = self.mocker.Mock(spec=["choices", "_hidden_params"])
        # Key exists but value is None - grounding available but not used
        mock_response._hidden_params = {"vertex_ai_grounding_metadata": None}
        mock_choice = self.mocker.Mock(spec=["message"])
        mock_message = self.mocker.Mock(spec=["tool_calls"])
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = self.service._check_grounding_used(mock_response)
        assert result is False

    def test_check_grounding_used_returns_false_for_empty_dict_metadata(self) -> None:
        """GIVEN response with empty dict grounding_metadata WHEN checking THEN returns False."""
        mock_response = self.mocker.Mock(spec=["choices", "_hidden_params"])
        # Key exists but value is empty dict
        mock_response._hidden_params = {"vertex_ai_grounding_metadata": {}}
        mock_choice = self.mocker.Mock(spec=["message"])
        mock_message = self.mocker.Mock(spec=["tool_calls"])
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        result = self.service._check_grounding_used(mock_response)
        assert result is False

    def test_completion_returns_completion_result(self) -> None:
        """GIVEN successful completion WHEN completing THEN returns CompletionResult."""
        mock_response = self.mocker.Mock(spec=["choices"])
        mock_choice = self.mocker.Mock(spec=["message"])
        mock_message = self.mocker.Mock(spec=["content", "tool_calls"])
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        self.mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = self.service.completion("test", command="ask")

        assert isinstance(result, CompletionResult)
        assert result.content == "Test response"
        assert result.grounding_used is False

    def test_completion_passes_trace_metadata_to_litellm(self) -> None:
        """GIVEN request_id set WHEN completion THEN litellm receives metadata with trace_id."""
        from llm.tracing import request_id

        captured_kwargs: dict = {}

        def mock_completion(**kwargs: dict) -> Mock:
            captured_kwargs.update(kwargs)
            return make_completion_response("Response")

        token = request_id.set("test1234")
        try:
            self.mocker.patch("llm.service.litellm.completion", side_effect=mock_completion)
            self.service.completion("Hello", command="ask")
        finally:
            request_id.reset(token)

        assert captured_kwargs.get("metadata") == {"trace_id": "test1234"}

    def test_completion_returns_grounding_used_true_when_grounded(self) -> None:
        """GIVEN completion with grounding WHEN completing THEN grounding_used is True."""
        mock_response = self.mocker.Mock(spec=["choices"])
        mock_choice = self.mocker.Mock(spec=["message", "grounding_metadata"])
        mock_choice.grounding_metadata = {"web_search_queries": ["test"]}
        mock_message = self.mocker.Mock(spec=["content", "tool_calls"])
        mock_message.content = "Grounded response"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        self.mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        result = self.service.completion("test", command="ask")

        assert result.grounding_used is True

    def test_completion_error_returns_completion_result_with_error(self) -> None:
        """GIVEN completion error WHEN completing THEN returns CompletionResult with error."""
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception("Test error"),
        )
        result = self.service.completion("test", command="ask")

        assert isinstance(result, CompletionResult)
        assert "Error" in result.content
        assert result.grounding_used is False

    def test_completion_sends_typing_indicators(self) -> None:
        """GIVEN irc context WHEN completion called THEN sends typing indicators."""
        mock_response = self.mocker.Mock(spec=["choices"])
        mock_choice = self.mocker.Mock(spec=["message"])
        mock_message = self.mocker.Mock(spec=["content", "tool_calls"])
        mock_message.content = "Response"
        mock_message.tool_calls = None
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        irc = self.mocker.Mock()
        irc.state = self.mocker.Mock()
        irc.state.capabilities_ack = {"message-tags"}
        irc.queueMsg = self.mocker.Mock()

        msg = self.mocker.Mock()
        msg.args = ("#test",)
        msg.prefix = "user!user@host"

        self.mocker.patch("llm.service.litellm.completion", return_value=mock_response)
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

        self.mocker.patch("llm.service.litellm.completion", side_effect=Exception("API error"))
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

        self.mocker.patch("llm.service.litellm.completion", side_effect=mock_completion)
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
    """Tests for extracting usage data from LiteLLM responses."""

    @pytest.fixture()
    def service(self, make_service) -> LLMService:
        """Create an LLMService with mock plugin."""
        service, _ = make_service()
        return service

    def test_completion_result_has_usage_fields(self) -> None:
        """GIVEN CompletionResult WHEN created with usage THEN fields accessible."""
        result = CompletionResult(
            content="hello",
            grounding_used=False,
            prompt_tokens=100,
            completion_tokens=50,
            cost=0.001,
            model="gemini/flash",
        )
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.cost == 0.001
        assert result.model == "gemini/flash"

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
            model="vertex_ai/imagen",
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

    def test_extract_usage_from_response(self, service: LLMService, mocker: MockerFixture) -> None:
        """GIVEN response with usage WHEN extracted THEN returns tokens and cost."""
        response = mocker.Mock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50

        mocker.patch("llm.service.litellm.completion_cost", return_value=0.003)
        prompt, completion, cost = service._extract_usage(response, "model")

        assert prompt == 100
        assert completion == 50
        assert cost == 0.003

    def test_extract_usage_handles_missing_usage(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN response without usage WHEN extracted THEN returns zeros."""
        response = mocker.Mock(spec=[])  # No attributes

        mocker.patch("llm.service.litellm.completion_cost", side_effect=Exception("no cost"))
        prompt, completion, cost = service._extract_usage(response, "model")

        assert prompt == 0
        assert completion == 0
        assert cost == 0.0

    def test_extract_usage_skips_completion_cost_for_known_image_models(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """GIVEN a known image model THEN completion_cost is never called.

        ``grok-imagine-image`` is absent from LiteLLM's cost map, so calling
        ``completion_cost`` raises and spammed a full traceback on every image.
        Cost is supplied by the caller's IMAGE_COST_PER_IMAGE fallback, so the
        always-failing call must be skipped entirely (no exception, no log).
        """
        response = mocker.Mock(spec=[])  # No usage attrs
        cost_spy = mocker.patch("llm.service.litellm.completion_cost")
        warn_spy = mocker.patch.object(service.log, "warning")

        prompt, completion, cost = service._extract_usage(response, "xai/grok-imagine-image")

        cost_spy.assert_not_called()
        warn_spy.assert_not_called()
        # _extract_usage leaves cost at 0.0; the caller applies the fallback.
        assert (prompt, completion, cost) == (0, 0, 0.0)


class TestErrorClassification:
    """Test _is_terminal_error classification."""

    def test_auth_error_is_terminal(self) -> None:
        """GIVEN AuthenticationError WHEN classified THEN terminal."""
        import litellm as litellm_module

        err = litellm_module.AuthenticationError(
            message="bad key", llm_provider="openai", model="gpt-4"
        )
        assert LLMService._is_terminal_error(err) is True

    def test_timeout_is_transient(self) -> None:
        """GIVEN Timeout WHEN classified THEN not terminal."""
        import litellm as litellm_module

        err = litellm_module.Timeout(message="timeout", model="gpt-4", llm_provider="openai")
        assert LLMService._is_terminal_error(err) is False

    def test_rate_limit_is_transient(self) -> None:
        """GIVEN RateLimitError WHEN classified THEN not terminal."""
        import litellm as litellm_module

        err = litellm_module.RateLimitError(
            message="rate limited", llm_provider="openai", model="gpt-4"
        )
        assert LLMService._is_terminal_error(err) is False

    def test_content_policy_violation_is_terminal(self) -> None:
        """GIVEN ContentPolicyViolationError WHEN classified THEN terminal.

        Reclassifying this as transient would retry a guardrail block ~10x
        with backoff and re-bill each attempt.
        """
        import litellm as litellm_module

        err = litellm_module.ContentPolicyViolationError(
            message="blocked", model="gpt-4", llm_provider="openai"
        )
        assert LLMService._is_terminal_error(err) is True

    def test_bad_request_is_terminal(self) -> None:
        """GIVEN BadRequestError WHEN classified THEN terminal (a malformed
        request will never succeed on retry)."""
        import litellm as litellm_module

        err = litellm_module.BadRequestError(
            message="bad request", model="gpt-4", llm_provider="openai"
        )
        assert LLMService._is_terminal_error(err) is True

    def test_not_found_is_terminal(self) -> None:
        """GIVEN NotFoundError (e.g. unknown model) WHEN classified THEN terminal."""
        import litellm as litellm_module

        err = litellm_module.NotFoundError(
            message="not found", model="gpt-4", llm_provider="openai"
        )
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


class TestServerHeaderLogging:
    """Tests for server header extraction in error paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()
        # Replace the real logger's debug method with a mock so we can inspect calls
        self.mock_debug = mocker.patch.object(self.service.log, "debug")

    def test_completion_error_logs_server_headers(self) -> None:
        """GIVEN completion raises with response headers WHEN error caught THEN headers logged at DEBUG."""
        import httpx
        import litellm

        exc = litellm.APIError(
            status_code=500,
            message="server error",
            llm_provider="xai",
            model="xai/grok-2",
        )
        exc.response = httpx.Response(500, headers={"x-request-id": "srv-abc", "cf-ray": "ray-123"})

        self.mocker.patch("llm.service.litellm.completion", side_effect=exc)

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%ask hello")

        result = self.service.completion("hello", command="ask", msg=mock_msg)
        assert result.error is not None

        # Verify debug log was called with header info
        debug_calls = [str(c) for c in self.mock_debug.call_args_list]
        header_logged = any("x-request-id" in c for c in debug_calls)
        assert header_logged, f"Expected server headers in debug log, got: {debug_calls}"

    def test_image_generation_error_logs_server_headers(self) -> None:
        """GIVEN image_generation raises with response headers WHEN error caught THEN headers logged."""
        import httpx
        import litellm

        exc = litellm.BadRequestError(
            message="moderation_blocked",
            model="xai/grok-imagine-image",
            llm_provider="xai",
            response=httpx.Response(400),
        )
        # Set response after construction to preserve headers (the openai SDK
        # constructor strips headers from the response object).
        exc.response = httpx.Response(400, headers={"x-request-id": "img-xyz", "cf-ray": "ray-456"})

        self.mocker.patch("llm.service.litellm.image_generation", side_effect=exc)

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%draw cat")

        result = self.service.image_generation("a cat", msg=mock_msg)
        assert result.error is not None

        debug_calls = [str(c) for c in self.mock_debug.call_args_list]
        header_logged = any("x-request-id" in c for c in debug_calls)
        assert header_logged, f"Expected server headers in debug log, got: {debug_calls}"

    def test_no_crash_when_exception_has_no_headers(self) -> None:
        """GIVEN exception without response headers WHEN error caught THEN no crash."""
        self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=Exception("generic error"),
        )

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%ask hello")

        result = self.service.completion("hello", command="ask", msg=mock_msg)
        assert result.error is not None  # completed without crash

    def test_completion_success_logs_server_headers(self) -> None:
        """GIVEN successful completion with _response_headers WHEN returned THEN headers logged."""
        mock_response = self.mocker.Mock()
        mock_response.choices = [self.mocker.Mock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response._response_headers = {
            "x-request-id": "success-abc",
            "content-type": "application/json",
        }
        mock_response.usage = self.mocker.Mock(prompt_tokens=10, completion_tokens=5)
        mock_response.id = "test-id"

        self.mocker.patch("llm.service.litellm.completion", return_value=mock_response)
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        mock_msg = self.mocker.Mock()
        mock_msg.nick = "testuser"
        mock_msg.args = ("#test", "%ask hello")

        result = self.service.completion("hello", command="ask", msg=mock_msg)
        assert result.error is None

        debug_calls = [str(c) for c in self.service.log.debug.call_args_list]
        header_logged = any("x-request-id" in c for c in debug_calls)
        assert header_logged, f"Expected server headers in debug log, got: {debug_calls}"


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

        The default assistantModel is TEST_MODEL ("gpt-4") -> openai, so
        OPENAI_API_KEY is the variable this guard reads.
        """
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        service, _ = self.make_service()

        result = service.completion("Hello world", command="ask")

        assert result.error is not None
        assert "OPENAI_API_KEY" in result.content


class TestCompletionWithToolFallback:
    """Tests for _completion_with_tool_fallback Gemini tool retry logic."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_retries_without_tools_on_invalid_argument(self) -> None:
        """GIVEN Gemini INVALID_ARGUMENT error with tools WHEN calling THEN retries without tools."""
        import litellm

        success_response = self.mocker.Mock()

        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[
                litellm.BadRequestError(
                    message="INVALID_ARGUMENT: tools are not supported",
                    model="gemini/gemini-2.5-flash",
                    llm_provider="gemini",
                ),
                success_response,
            ],
        )

        result = self.service._completion_with_tool_fallback(
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "hello"}],
            timeout=30,
            optional_kwargs={"tools": [{"type": "function"}], "safety_settings": "low"},
        )

        assert result is success_response
        assert mock_completion.call_count == 2

        # Second call should NOT include "tools" but should keep other kwargs
        second_call_kwargs = mock_completion.call_args_list[1][1]
        assert "tools" not in second_call_kwargs
        assert second_call_kwargs["safety_settings"] == "low"


class TestXaiConvIdHeader:
    """xAI prompt-cache sticky-routing header (`x-grok-conv-id`)."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_header_set_for_xai_with_channel(self) -> None:
        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion", return_value=self.mocker.Mock()
        )
        self.service._timed_completion(
            "assistant_step_1",
            model="xai/grok-4.3",
            messages=[{"role": "user", "content": "hi"}],
            channel="#dev",
        )
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["extra_headers"] == {"x-grok-conv-id": "chan:#dev:main"}

    def test_header_lane_per_op(self) -> None:
        """Different ops route to different cache lanes on the same channel.

        This is the eviction-prevention split: ``extract_memories`` and
        ``ask_helper`` no longer share a server with ``assistant_step_*``,
        so their short distinct prefixes can't kick the long main prefix
        out of the per-server cache between turns.
        """
        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion", return_value=self.mocker.Mock()
        )
        lanes: dict[str, str] = {}
        for op in ("assistant_step_1", "ask_helper", "extract_memories", "prompt_rewrite"):
            self.service._timed_completion(
                op,
                model="xai/grok-4.3",
                messages=[{"role": "user", "content": "hi"}],
                channel="#dev",
            )
            lanes[op] = mock_completion.call_args.kwargs["extra_headers"]["x-grok-conv-id"]
        assert lanes == {
            "assistant_step_1": "chan:#dev:main",
            "ask_helper": "chan:#dev:helper",
            "extract_memories": "chan:#dev:memory",
            "prompt_rewrite": "chan:#dev:rewrite",
        }

    def test_header_omitted_for_non_xai(self) -> None:
        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion", return_value=self.mocker.Mock()
        )
        self.service._timed_completion(
            "op",
            model="gemini/gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
            channel="#dev",
        )
        assert "extra_headers" not in mock_completion.call_args.kwargs

    def test_header_omitted_for_xai_without_channel(self) -> None:
        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion", return_value=self.mocker.Mock()
        )
        self.service._timed_completion(
            "op",
            model="xai/grok-4.3",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert "extra_headers" not in mock_completion.call_args.kwargs

    def test_header_merged_with_caller_extra_headers(self) -> None:
        mock_completion = self.mocker.patch(
            "llm.service.litellm.completion", return_value=self.mocker.Mock()
        )
        self.service._timed_completion(
            "assistant_step_1",
            model="xai/grok-4.3",
            messages=[{"role": "user", "content": "hi"}],
            channel="#dev",
            extra_headers={"x-trace-id": "abc"},
        )
        kwargs = mock_completion.call_args.kwargs
        assert kwargs["extra_headers"] == {
            "x-trace-id": "abc",
            "x-grok-conv-id": "chan:#dev:main",
        }


class TestExtractUsage:
    """Tests for _extract_usage error handling."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

    def test_extract_usage_handles_type_error_on_usage(self) -> None:
        """GIVEN response where accessing usage raises TypeError WHEN extracting THEN returns zeros with cost."""
        mock_response = self.mocker.Mock()
        mock_response.usage = property(lambda self: (_ for _ in ()).throw(TypeError))
        # Make getattr(response, "usage", None) raise TypeError
        type(mock_response).usage = self.mocker.PropertyMock(side_effect=TypeError)

        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.01)

        prompt_tokens, completion_tokens, cost = self.service._extract_usage(mock_response, "gpt-4")

        assert prompt_tokens == 0
        assert completion_tokens == 0
        assert cost == 0.01


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
            model="gpt-4",
            grounding_used=True,
            error=None,
        )
        assert result.content == "response"
        assert result.grounding_used is True
        assert result.prompt_tokens == 100
        assert result.model == "gpt-4"
        assert result.error is None


# ---------------------------------------------------------------------------
# Search / URL completion helpers
# ---------------------------------------------------------------------------


def _make_litellm_response(mocker, content="result text", grounding=False):  # noqa: ARG001
    """Build a minimal mock that looks like a litellm completion response."""
    return make_completion_response(content, grounding=grounding)


class TestSearchCompletion:
    """Tests for LLMService.search_completion()."""

    @pytest.fixture()
    def service(self, make_service) -> LLMService:
        service, self.plugin = make_service(
            assistantModel="gemini/gemini-2.5-flash",
            searchModel="",
        )
        return service

    def test_returns_tool_result_with_content(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """search_completion returns ToolResult with response content."""
        from llm.assistant import ToolResult

        resp = _make_litellm_response(mocker, content="Search answer")
        mocker.patch("llm.service.litellm.completion", return_value=resp)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

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
                "searchModel": "gemini/gemini-2.5-pro",
                "assistantModel": "gemini/gemini-2.5-flash",
                "timeout": 30,
            }.get(key, "")
        )

        resp = _make_litellm_response(mocker)
        mock_completion = mocker.patch("llm.service.litellm.completion", return_value=resp)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.search_completion("test", channel="#test")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "gemini/gemini-2.5-pro"

    def test_falls_back_to_ask_model(self, service: LLMService, mocker: MockerFixture) -> None:
        """search_completion uses assistantModel when searchModel is empty."""
        resp = _make_litellm_response(mocker)
        mock_completion = mocker.patch("llm.service.litellm.completion", return_value=resp)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.search_completion("test", channel="#test")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "gemini/gemini-2.5-flash"

    def test_passes_google_search_tool(self, service: LLMService, mocker: MockerFixture) -> None:
        """search_completion passes both googleSearch and urlContext to Gemini.

        Both grounding tools ride on the same call so the model can pivot
        from a web search to fetching a specific URL within one turn.
        """
        resp = _make_litellm_response(mocker)
        mock_completion = mocker.patch("llm.service.litellm.completion", return_value=resp)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.search_completion("test query", channel="#test")

        call_kwargs = mock_completion.call_args
        all_kwargs = call_kwargs.kwargs
        assert "tools" in all_kwargs
        assert all_kwargs["tools"] == [{"googleSearch": {}}, {"urlContext": {}}]

    def test_returns_error_on_exception(self, service: LLMService, mocker: MockerFixture) -> None:
        """search_completion returns error ToolResult on failure."""
        import json

        from llm.assistant import ToolResult

        mocker.patch(
            "llm.service.litellm.completion",
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
            assistantModel="gemini/gemini-2.5-flash",
            searchModel="",
        )
        return service

    def test_returns_summary(self, service: LLMService, mocker: MockerFixture) -> None:
        """url_completion returns page summary as ToolResult."""
        from llm.assistant import ToolResult

        resp = _make_litellm_response(mocker, content="Page summary here")
        mocker.patch("llm.service.litellm.completion", return_value=resp)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.001)

        result = service.url_completion("https://example.com/article", channel="#test")

        assert isinstance(result, ToolResult)
        assert result.content == "Page summary here"

    def test_rejects_unsafe_url(self, service: LLMService, mocker: MockerFixture) -> None:
        """url_completion returns error for private IPs."""
        import json

        from llm.assistant import ToolResult

        result = service.url_completion("http://192.168.1.1/admin", channel="#test")

        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed
        assert "not allowed" in parsed["error"].lower()

    def test_passes_url_context_tool(self, service: LLMService, mocker: MockerFixture) -> None:
        """url_completion passes both urlContext and googleSearch to Gemini.

        Both grounding tools ride on the same call so the model can pivot
        from fetching a URL to searching the web within one turn.
        """
        resp = _make_litellm_response(mocker)
        mock_completion = mocker.patch("llm.service.litellm.completion", return_value=resp)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.url_completion("https://example.com/page", channel="#test")

        call_kwargs = mock_completion.call_args
        all_kwargs = call_kwargs.kwargs
        assert "tools" in all_kwargs
        assert all_kwargs["tools"] == [{"googleSearch": {}}, {"urlContext": {}}]

    def test_returns_error_on_exception(self, service: LLMService, mocker: MockerFixture) -> None:
        """url_completion returns error ToolResult on failure."""
        import json

        from llm.assistant import ToolResult

        mocker.patch(
            "llm.service.litellm.completion",
            side_effect=RuntimeError("connection refused"),
        )

        result = service.url_completion("https://example.com", channel="#test")

        assert isinstance(result, ToolResult)
        parsed = json.loads(result.content)
        assert "error" in parsed
        # Internal exception text must NOT leak into tool payload
        assert "connection refused" not in parsed["error"]
        assert parsed["error"] == "URL fetch failed."


def test_search_and_url_completion_use_same_provider_kwargs_base(
    make_service, mocker: MockerFixture
) -> None:
    """search_completion and url_completion produce identical optional_kwargs key sets."""
    service, plugin = make_service(
        assistantModel="gemini/gemini-2.5-flash",
        searchModel="",
    )
    captured: list[dict] = []

    def fake_call(*, model, messages, timeout, optional_kwargs, op="completion", channel=None):
        captured.append(optional_kwargs)
        return make_completion_response("result", prompt_tokens=5, completion_tokens=10)

    mocker.patch.object(service, "_completion_with_tool_fallback", side_effect=fake_call)
    mocker.patch.object(service, "_is_xai_model", return_value=False)

    service.search_completion("ping", channel="#c")
    service.url_completion("https://example.com", channel="#c")

    assert len(captured) == 2
    assert set(captured[0].keys()) == set(captured[1].keys())


# =============================================================================
# TestAssistantRequestFacade — Task 9
# =============================================================================


class TestAssistantRequestFacade:
    """Tests for the assistant_request planner facade."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, self.mock_plugin = make_service()

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

    def test_chat_profile_forwards_route_profile(self) -> None:
        """Chat profile forwards route_profile=chat. ``assistant_completion``
        selects CHAT_SYSTEM_PROMPT from the route_profile, not from
        system_prompt — the latter is reserved for personality overlays."""
        ctx = self._make_ctx(profile="chat")
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="answer", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "hello",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
        )

        call_kwargs = self.service.assistant_completion.call_args.kwargs
        assert call_kwargs["route_profile"] == "chat"
        assert call_kwargs["system_prompt"] is None

    def test_code_profile_forwards_route_profile(self) -> None:
        """Code profile forwards route_profile=code so the planner picks
        CODE_SYSTEM_PROMPT as the structural framework."""
        ctx = self._make_ctx(
            entry_route="code",
            profile="code",
            capabilities=frozenset({"llm.code"}),
        )
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="code link", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "write fibonacci",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
        )

        call_kwargs = self.service.assistant_completion.call_args.kwargs
        assert call_kwargs["route_profile"] == "code"
        assert call_kwargs["system_prompt"] is None

    def test_verse_profile_forwards_route_profile(self) -> None:
        """Verse profile forwards route_profile=verse so the planner treats it
        as an unknown profile (falls back to CHAT_SYSTEM_PROMPT framework) while
        still bypassing the token cap (PROFILE_VERSE not in profile_max_output)."""
        ctx = self._make_ctx(profile="verse")
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="verse answer", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "tell me a long story",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
        )

        call_kwargs = self.service.assistant_completion.call_args.kwargs
        assert call_kwargs["route_profile"] == "verse"
        assert call_kwargs["system_prompt"] is None

    def test_draw_profile_forwards_route_profile(self) -> None:
        """Draw profile forwards route_profile=draw so the planner picks
        DRAW_SYSTEM_PROMPT as the structural framework."""
        ctx = self._make_ctx(
            entry_route="draw",
            profile="draw",
            capabilities=frozenset({"llm.draw"}),
        )
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="image link", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "draw a cat",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
        )

        call_kwargs = self.service.assistant_completion.call_args.kwargs
        assert call_kwargs["route_profile"] == "draw"
        assert call_kwargs["system_prompt"] is None

    def test_returns_meta_result(self) -> None:
        """assistant_request returns AssistantResult with all fields preserved."""
        ctx = self._make_ctx()
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="answer", spend=Spend.ZERO, grounding_used=True),
        )

        result = self.service.assistant_request(
            "hello",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
        )

        assert isinstance(result, AssistantResult)
        assert result.grounding_used is True

    def test_passes_callables_through(self) -> None:
        """Callable handlers are forwarded to assistant_completion."""
        ctx = self._make_ctx()
        search_fn = self.mocker.Mock()
        draw_fn = self.mocker.Mock()
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="answer", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "hello",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
            search_fn=search_fn,
            draw_fn=draw_fn,
        )

        call_kwargs = self.service.assistant_completion.call_args.kwargs
        assert call_kwargs["search_fn"] is search_fn
        assert call_kwargs["draw_fn"] is draw_fn

    def test_explicit_system_prompt_overrides_profile(self) -> None:
        """An explicit system_prompt kwarg takes priority over the profile default."""
        ctx = self._make_ctx(profile="chat")
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="ok", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "hello",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
            system_prompt="Custom prompt override",
        )

        assert (
            self.service.assistant_completion.call_args.kwargs["system_prompt"]
            == "Custom prompt override"
        )

    def test_forwards_context_fields(self) -> None:
        """The caller's storage key, reply target and account, plus is_owner and
        capabilities, are forwarded."""
        ctx = self._make_ctx(
            nick="alice",
            channel="#dev",
            account="alice_acct",
            is_owner=True,
            capabilities=frozenset({"llm.ask", "llm.code"}),
        )
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="ok", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "hello",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="TestBot",
        )

        kw = self.service.assistant_completion.call_args.kwargs
        assert kw["nick"] == "alice_acct"
        assert kw["channel"] == "#dev"
        assert kw["account"] == "alice_acct"
        assert kw["is_owner"] is True
        assert kw["capabilities"] == frozenset({"llm.ask", "llm.code"})
        assert kw["bot_nick"] == "TestBot"

    def test_private_message_passes_the_senders_nick(self) -> None:
        """Private messages (channel=None) pass the sender's nick, the reply
        target, to assistant_completion: never the bot's own nick."""
        ctx = self._make_ctx(nick="user", channel=None)
        self.service.assistant_completion = self.mocker.Mock(
            return_value=AssistantResult(content="ok", spend=Spend.ZERO),
        )

        self.service.assistant_request(
            "hello",
            request_context=ctx,
            db=self.mocker.Mock(),
            context=self.mocker.Mock(),
            bot_nick="Bot",
        )

        assert self.service.assistant_completion.call_args.kwargs["channel"] == "user"


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
        mocker.patch.object(
            self.service,
            "_completion_with_tool_fallback",
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
        the call proceeds and LiteLLM reports whatever the real problem is.
        """
        fallback = mocker.patch.object(
            self.service,
            "_completion_with_tool_fallback",
            return_value=make_completion_response("recovered"),
        )
        task = self._make_task(model="")
        result = self.service._retry_completion(task, {"messages": [{"role": "u", "content": "x"}]})
        assert result.status == "completed"
        assert fallback.called


class TestResponsesApiTextAndUsage:
    """Coverage for _responses_text fallback walk and _extract_responses_usage paths."""

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        self.mocker = mocker
        self.service, _ = make_service()

    def test_responses_text_walks_output_when_output_text_missing(self) -> None:
        """Falls back to walking output items when response.output_text is empty."""
        response = self.mocker.MagicMock()
        response.output_text = ""
        response.output = [
            {"type": "web_search_call", "id": "x"},  # non-message item is skipped
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello "},
                    {"type": "output_text", "text": "world"},
                    {"type": "ignored", "text": "nope"},
                ],
            },
        ]
        text = self.service._responses_text(response)
        assert text == "hello world"

    def test_responses_text_walks_output_with_object_parts(self) -> None:
        """Object-shaped output parts (non-dict) are also extracted."""

        class _Part:
            def __init__(self, type_: str, text: str) -> None:
                self.type = type_
                self.text = text

        class _Item:
            def __init__(self, type_: str, content: list) -> None:
                self.type = type_
                self.content = content

        response = self.mocker.MagicMock()
        response.output_text = None
        response.output = [
            _Item("message", [_Part("output_text", "obj-form")]),
        ]
        assert self.service._responses_text(response) == "obj-form"

    def test_responses_grounding_skips_non_message_non_search_items(self) -> None:
        """An item type that is neither 'search' nor 'message' is skipped."""
        response = self.mocker.MagicMock()
        response.output = [
            {"type": "reasoning", "id": "r1"},  # neither search nor message
        ]
        assert self.service._check_responses_grounding(response) is False

    def test_responses_grounding_handles_attribute_error(self) -> None:
        """An output access that raises AttributeError returns False, not raise."""

        class _Bad:
            @property
            def output(self):
                raise AttributeError("no output")

        assert self.service._check_responses_grounding(_Bad()) is False

    def test_extract_responses_usage_uses_usage_cost_when_present(self) -> None:
        """When usage.cost is truthy, the helper avoids calling completion_cost."""
        response = self.mocker.MagicMock()
        response.usage = self.mocker.Mock(
            input_tokens=10, output_tokens=5, cost=0.05, input_tokens_details=None
        )
        completion_cost = self.mocker.patch("llm.service.litellm.completion_cost")
        prompt_tokens, completion_tokens, cached_tokens, cost = (
            self.service._extract_responses_usage(response, "xai/grok-4.3")
        )
        assert (prompt_tokens, completion_tokens, cached_tokens) == (10, 5, 0)
        assert cost == pytest.approx(0.05)
        completion_cost.assert_not_called()

    def test_extract_responses_usage_swallows_completion_cost_failure(self) -> None:
        """A litellm.completion_cost exception is swallowed; cost defaults to 0.0."""
        response = self.mocker.MagicMock()
        response.usage = self.mocker.Mock(
            input_tokens=3, output_tokens=2, cost=None, input_tokens_details=None
        )
        self.mocker.patch("llm.service.litellm.completion_cost", side_effect=RuntimeError("boom"))
        prompt_tokens, completion_tokens, cached_tokens, cost = (
            self.service._extract_responses_usage(response, "xai/grok-4.3")
        )
        assert (prompt_tokens, completion_tokens, cached_tokens) == (3, 2, 0)
        assert cost == 0.0

    def test_extract_responses_usage_reads_cached_tokens_from_input_details(self) -> None:
        """Responses API exposes cache reads via usage.input_tokens_details.cached_tokens."""
        response = self.mocker.MagicMock()
        response.usage = self.mocker.Mock(
            input_tokens=200,
            output_tokens=12,
            cost=0.01,
            input_tokens_details=self.mocker.Mock(cached_tokens=180),
        )
        self.mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
        prompt_tokens, completion_tokens, cached_tokens, cost = (
            self.service._extract_responses_usage(response, "xai/grok-4.3")
        )
        assert prompt_tokens == 200
        assert completion_tokens == 12
        assert cached_tokens == 180
        assert cost == pytest.approx(0.01)


def test_assistant_result_was_verse_defaults_false():
    assert AssistantResult(content="x", spend=Spend.ZERO).was_verse is False


def test_assistant_result_was_verse_settable():
    assert AssistantResult(content="x", spend=Spend.ZERO, was_verse=True).was_verse is True
