"""Tests for provider.py, at its interface: chat, search, draw and their errors.

The request bodies are checked against tests/fixtures/provider_wire/, which
holds what litellm 1.103.0 sent for the same calls (recorded offline
2026-10-21). Where this module deliberately sends something else, the test
says so and edits the fixture to match, so every divergence is written down.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import urllib.error
from email.message import Message
from pathlib import Path
from typing import Any

import pytest
from llm import provider

WIRE = Path(__file__).parent / "fixtures" / "provider_wire"

HAIKU = "anthropic/claude-haiku-5-5"
GROK = "xai/grok-4-1-fast-non-reasoning"

SYSTEM = [{"role": "system", "content": "You are vibebot."}]
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
THINK = [{"type": "thinking", "thinking": "", "signature": "sig-abc"}]
TOOL_TURN = [
    {"role": "user", "content": "weather in oshawa?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "search_web", "arguments": '{"query":"oshawa weather"}'},
            },
            {
                "id": "toolu_2",
                "type": "function",
                "function": {"name": "search_web", "arguments": '{"query":"oshawa forecast"}'},
            },
        ],
        "thinking_blocks": THINK,
    },
    {"role": "tool", "tool_call_id": "toolu_1", "content": "5C cloudy"},
    {"role": "tool", "tool_call_id": "toolu_2", "content": "rain tomorrow"},
]


def wire(name: str) -> dict[str, Any]:
    return json.loads((WIRE / f"{name}-1.json").read_text())


class FakePost:
    """Stands in for provider._post: records each request, returns ``reply``."""

    def __init__(self, reply: dict[str, Any] | None = None) -> None:
        self.reply = reply if reply is not None else {}
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, headers, body, *, timeout, model, provider):  # noqa: ANN001
        self.calls.append({"url": url, "headers": headers, "body": copy.deepcopy(body)})
        return self.reply

    @property
    def body(self) -> dict[str, Any]:
        return self.calls[-1]["body"]


@pytest.fixture
def post(monkeypatch: pytest.MonkeyPatch) -> FakePost:
    fake = FakePost()
    monkeypatch.setattr(provider, "_post", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-0123456789")
    monkeypatch.setenv("XAI_API_KEY", "xai-test-key-0123456789abcdef")
    return fake


def _without_custom_type(body: dict[str, Any]) -> dict[str, Any]:
    # litellm stamps "type": "custom" on every tool; the API's default.
    for tool in body.get("tools", []):
        tool.pop("type", None)
    return body


# ---------------------------------------------------------------------------
# Anthropic requests
# ---------------------------------------------------------------------------


class TestAnthropicRequest:
    def test_tool_loop_step_matches_litellm(self, post: FakePost) -> None:
        provider.chat(
            HAIKU,
            SYSTEM + TOOL_TURN,
            timeout=30,
            tools=TOOLS,
            tool_choice="required",
            max_tokens=400,
            temperature=0.9,
        )
        expected = _without_custom_type(wire("haiku_assistant_step")["body"])
        # litellm filled the empty assistant text with a placeholder sentence
        # the model then read. A tool-call-only turn needs no text block.
        assistant = expected["messages"][1]["content"]
        expected["messages"][1]["content"] = [b for b in assistant if b.get("type") != "text"]
        assert post.body == expected
        assert post.calls[-1]["url"] == "https://api.anthropic.com/v1/messages"
        assert post.calls[-1]["headers"]["anthropic-version"] == "2023-06-01"

    def test_temperature_never_reaches_anthropic(self, post: FakePost) -> None:
        provider.chat(
            HAIKU, SYSTEM + [{"role": "user", "content": "hi"}], timeout=30, temperature=0.9
        )
        assert "temperature" not in post.body

    def test_forced_tool_choice(self, post: FakePost) -> None:
        provider.chat(
            HAIKU,
            SYSTEM + [{"role": "user", "content": "search cats"}],
            timeout=30,
            tools=TOOLS,
            tool_choice={"type": "function", "function": {"name": "search_web"}},
        )
        expected = _without_custom_type(wire("haiku_forced_tool")["body"])
        expected["max_tokens"] = provider.DEFAULT_MAX_TOKENS  # litellm: the 128K ceiling
        assert post.body == expected

    @pytest.mark.parametrize("name", ["haiku_no_max_tokens", "haiku_vision"])
    def test_default_max_tokens(self, post: FakePost, name: str) -> None:
        messages = {
            "haiku_no_max_tokens": SYSTEM + [{"role": "user", "content": "hi"}],
            "haiku_vision": SYSTEM
            + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://bot.hextalk.org/favicon.ico"},
                        },
                    ],
                }
            ],
        }[name]
        provider.chat(HAIKU, messages, timeout=30)
        expected = wire(name)["body"]
        expected["max_tokens"] = provider.DEFAULT_MAX_TOKENS
        # Always cached here; litellm only marked what the caller marked.
        expected["system"][0]["cache_control"] = {"type": "ephemeral"}
        expected["cache_control"] = {"type": "ephemeral"}
        assert post.body == expected

    def test_tool_choice_none(self, post: FakePost) -> None:
        provider.chat(
            HAIKU,
            SYSTEM + [{"role": "user", "content": "pick"}],
            timeout=30,
            tools=TOOLS,
            tool_choice="none",
        )
        assert post.body["tool_choice"] == wire("haiku_tool_choice_none")["body"]["tool_choice"]

    def test_json_schema_uses_output_config(self, post: FakePost) -> None:
        schema = wire("haiku_json_schema")["body"]["output_format"]["schema"]
        provider.chat(
            HAIKU, SYSTEM + [{"role": "user", "content": "extract"}], timeout=15, json_reply=schema
        )
        # litellm sent the deprecated output_format under an old beta header.
        assert post.body["output_config"] == {"format": {"type": "json_schema", "schema": schema}}
        assert "output_format" not in post.body
        assert "anthropic-beta" not in post.calls[-1]["headers"]

    def test_json_object_has_no_anthropic_switch(self, post: FakePost) -> None:
        provider.chat(
            HAIKU, SYSTEM + [{"role": "user", "content": "x"}], timeout=30, json_reply=True
        )
        assert "output_config" not in post.body

    def test_effort(self, post: FakePost) -> None:
        provider.chat(HAIKU, SYSTEM + [{"role": "user", "content": "x"}], timeout=30, effort="low")
        assert post.body["output_config"] == {"effort": "low"}

    def test_system_messages_hoisted_and_first_one_cached(self, post: FakePost) -> None:
        provider.chat(
            HAIKU,
            [
                {"role": "system", "content": "A"},
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "I see the context."},
                {"role": "system", "content": "B late"},
                {"role": "user", "content": "u2"},
            ],
            timeout=30,
        )
        expected = wire("multi_system")["body"]
        assert [b["text"] for b in post.body["system"]] == [b["text"] for b in expected["system"]]
        assert post.body["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in post.body["system"][1]
        assert post.body["messages"] == expected["messages"]

    def test_data_url_image_goes_as_base64(self, post: FakePost) -> None:
        provider.chat(
            HAIKU,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ],
                }
            ],
            timeout=30,
        )
        assert post.body["messages"][0]["content"] == [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"},
            }
        ]

    def test_missing_key_is_auth_failure(
        self, post: FakePost, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY")
        with pytest.raises(provider.AuthenticationError, match="ANTHROPIC_API_KEY"):
            provider.chat(HAIKU, [{"role": "user", "content": "x"}], timeout=30)
        assert post.calls == []

    def test_unknown_provider(self, post: FakePost) -> None:
        with pytest.raises(provider.BadRequestError):
            provider.chat("gemini/gemini-3-flash", [{"role": "user", "content": "x"}], timeout=30)


# ---------------------------------------------------------------------------
# Anthropic replies
# ---------------------------------------------------------------------------


ANTHROPIC_TOOL_REPLY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "sig-xyz"},
        {"type": "text", "text": "Looking."},
        {"type": "tool_use", "id": "toolu_9", "name": "search_web", "input": {"query": "cats"}},
    ],
    "stop_reason": "tool_use",
    "usage": {
        "input_tokens": 100,
        "cache_creation_input_tokens": 1000,
        "cache_read_input_tokens": 4000,
        "output_tokens": 50,
    },
}


class TestAnthropicReply:
    def test_reply_fields(self, post: FakePost) -> None:
        post.reply = ANTHROPIC_TOOL_REPLY
        reply = provider.chat(HAIKU, [{"role": "user", "content": "x"}], timeout=30, tools=TOOLS)
        assert reply.text == "Looking."
        assert reply.finish_reason == "tool_calls"
        assert reply.tool_calls == (
            provider.ToolCall(id="toolu_9", name="search_web", arguments='{"query": "cats"}'),
        )
        assert reply.usage == provider.Usage(
            input_tokens=5100, output_tokens=50, cached_tokens=4000
        )
        # 100×0.10 + 1000×0.125 + 4000×0.01 + 50×0.50, per million
        assert reply.cost == pytest.approx((10 + 125 + 40 + 25) / 1e6)
        assert reply.id == "msg_1"

    def test_turn_round_trips_thinking_verbatim(self, post: FakePost) -> None:
        post.reply = ANTHROPIC_TOOL_REPLY
        messages: list[dict[str, Any]] = [{"role": "user", "content": "x"}]
        reply = provider.chat(HAIKU, messages, timeout=30, tools=TOOLS)
        messages += [reply.turn(), {"role": "tool", "tool_call_id": "toolu_9", "content": "meow"}]
        provider.chat(HAIKU, messages, timeout=30, tools=TOOLS)
        assistant, results = post.body["messages"][1], post.body["messages"][2]
        assert assistant == {"role": "assistant", "content": ANTHROPIC_TOOL_REPLY["content"]}
        assert results["content"] == [
            {"type": "tool_result", "tool_use_id": "toolu_9", "content": "meow"}
        ]

    def test_turn_drops_thinking_for_xai(self, post: FakePost) -> None:
        post.reply = ANTHROPIC_TOOL_REPLY
        reply = provider.chat(HAIKU, [{"role": "user", "content": "x"}], timeout=30)
        post.reply = {"choices": [{"message": {"content": "ok"}}]}
        provider.chat(GROK, [{"role": "user", "content": "x"}, reply.turn()], timeout=30)
        assert "thinking_blocks" not in post.body["messages"][1]

    @pytest.mark.parametrize(
        ("stop_reason", "finish"),
        [("end_turn", "stop"), ("max_tokens", "length"), ("refusal", "refusal"), (None, "stop")],
    )
    def test_finish_reason(self, post: FakePost, stop_reason: str | None, finish: str) -> None:
        post.reply = {"content": [], "stop_reason": stop_reason}
        assert (
            provider.chat(HAIKU, [{"role": "user", "content": "x"}], timeout=30).finish_reason
            == finish
        )


# ---------------------------------------------------------------------------
# xAI chat
# ---------------------------------------------------------------------------


class TestXaiChat:
    def test_tool_loop_step_matches_litellm(self, post: FakePost) -> None:
        provider.chat(
            GROK,
            SYSTEM + TOOL_TURN,
            timeout=30,
            tools=TOOLS,
            tool_choice="required",
            max_tokens=400,
            temperature=0.9,
            cache_key="abc",
        )
        captured = wire("grok_completion")
        expected = captured["body"]
        # litellm passed Anthropic's thinking blocks to xAI, which ignored them.
        del expected["messages"][2]["thinking_blocks"]
        assert post.body == expected
        assert post.calls[-1]["headers"]["x-grok-conv-id"] == captured["headers"]["x-grok-conv-id"]
        assert post.calls[-1]["url"] == captured["url"]

    def test_effort(self, post: FakePost) -> None:
        provider.chat(
            "xai/grok-4.3",
            SYSTEM + [{"role": "user", "content": "announce"}],
            timeout=30,
            max_tokens=800,
            effort="low",
        )
        assert post.body == wire("grok_reasoning_effort")["body"]

    def test_json_reply(self, post: FakePost) -> None:
        provider.chat(GROK, [{"role": "user", "content": "x"}], timeout=30, json_reply=True)
        assert post.body["response_format"] == {"type": "json_object"}
        provider.chat(
            GROK, [{"role": "user", "content": "x"}], timeout=30, json_reply={"type": "object"}
        )
        assert post.body["response_format"]["json_schema"]["schema"] == {"type": "object"}

    def test_reply_and_ticks_cost(self, post: FakePost) -> None:
        post.reply = {
            "id": "c1",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {"id": "call_1", "function": {"name": "search_web", "arguments": "{}"}}
                        ],
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 800},
                "cost_in_usd_ticks": 5_000_000,
            },
        }
        reply = provider.chat(GROK, [{"role": "user", "content": "x"}], timeout=30)
        assert reply.text == ""
        assert reply.tool_calls == (provider.ToolCall("call_1", "search_web", "{}"),)
        assert reply.usage == provider.Usage(1000, 10, 800)
        assert reply.cost == pytest.approx(0.0005)

    def test_table_cost_without_ticks(self, post: FakePost) -> None:
        post.reply = {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 10,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens_details": {"reasoning_tokens": 90},
            },
        }
        reply = provider.chat(
            "xai/grok-4-1-fast-reasoning", [{"role": "user", "content": "x"}], timeout=30
        )
        # 200 fresh × 1.25 + 800 cached × 0.20 + 100 output × 2.50, per million
        assert reply.cost == pytest.approx((250 + 160 + 250) / 1e6)

    def test_unpriced_model_books_zero_and_warns_once(
        self, post: FakePost, caplog: pytest.LogCaptureFixture
    ) -> None:
        post.reply = {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5}}
        provider._unpriced.discard("xai/grok-unknown")
        for _ in range(2):
            assert (
                provider.chat(
                    "xai/grok-unknown", [{"role": "user", "content": "x"}], timeout=30
                ).cost
                == 0
            )
        assert sum("grok-unknown" in r.message for r in caplog.records) == 1


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_request(self, post: FakePost) -> None:
        provider.search("xai/grok-4.3", "news today", timeout=60, cache_key="abc")
        captured = wire("grok_responses")
        assert post.calls[-1]["url"] == captured["url"]
        assert post.body["tools"] == captured["body"]["tools"]
        assert post.body["prompt_cache_key"] == "abc"
        assert post.body["input"] == "news today"

    def test_reply(self, post: FakePost) -> None:
        post.reply = {
            "output": [
                {"type": "web_search_call", "status": "completed"},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "It rained.", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 50, "output_tokens": 5, "cost_in_usd_ticks": 1_000_000},
        }
        reply = provider.search("xai/grok-4.3", "q", timeout=60)
        assert (reply.text, reply.grounded) == ("It rained.", True)
        assert reply.usage == provider.Usage(50, 5, 0)
        assert reply.cost == pytest.approx(0.0001)

    def test_ungrounded(self, post: FakePost) -> None:
        post.reply = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "x"}]}]
        }
        assert provider.search("xai/grok-4.3", "q", timeout=60).grounded is False

    def test_needs_xai(self, post: FakePost) -> None:
        with pytest.raises(provider.BadRequestError):
            provider.search(HAIKU, "q", timeout=60)


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------


class TestDraw:
    def test_xai_request(self, post: FakePost) -> None:
        provider.draw(
            "xai/grok-imagine-image", "a cat", timeout=60, aspect_ratio="9:16", hires=True
        )
        expected = wire("grok_image")["body"]
        expected["response_format"] = "b64_json"  # bytes back, no second download
        assert post.body == expected
        assert post.calls[-1]["url"] == "https://api.x.ai/v1/images/generations"

    def test_xai_edit(self, post: FakePost) -> None:
        provider.draw("xai/grok-imagine-image", "add a hat", timeout=60, edit_url="https://h/x.png")
        assert post.calls[-1]["url"] == "https://api.x.ai/v1/images/edits"
        assert post.body["image"] == {"url": "https://h/x.png"}

    def test_b64_picture_and_ticks(self, post: FakePost) -> None:
        post.reply = {
            "data": [{"b64_json": base64.b64encode(b"PNG").decode()}],
            "usage": {"cost_in_usd_ticks": 200_000_000},
        }
        picture = provider.draw("xai/grok-imagine-image", "a cat", timeout=60)
        assert picture.data == b"PNG"
        assert picture.cost == pytest.approx(0.02)

    def test_priced_from_table_without_ticks(self, post: FakePost) -> None:
        post.reply = {"data": [{"b64_json": base64.b64encode(b"PNG").decode()}]}
        assert provider.draw("xai/grok-imagine-image-quality", "a cat", timeout=60).cost == 0.05

    def test_needs_xai(self, post: FakePost) -> None:
        with pytest.raises(provider.BadRequestError):
            provider.draw("openai//work/models/x", "a cat", timeout=60)

    def test_empty_data_is_no_picture(self, post: FakePost) -> None:
        post.reply = {"data": []}
        picture = provider.draw("xai/grok-imagine-image", "a cat", timeout=60)
        assert picture.data is None


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------


class TestErrors:
    @pytest.mark.parametrize(
        ("status", "body", "cls"),
        [
            (
                400,
                '{"code":"imagine:content-moderated","error":"Generated image rejected by content moderation."}',
                provider.ContentPolicyViolationError,
            ),
            (
                400,
                '{"error":{"message":"moderation_blocked"}}',
                provider.ContentPolicyViolationError,
            ),
            (
                403,
                '{"code":"permission-denied","error":"Content violates usage guidelines."}',
                provider.RefusalError,
            ),
            (
                403,
                '{"code":"permission-denied","error":"no access to model"}',
                provider.AuthenticationError,
            ),
            (401, "{}", provider.AuthenticationError),
            (404, "{}", provider.NotFoundError),
            (408, "{}", provider.RequestTimeoutError),
            (429, "{}", provider.RateLimitError),
            (400, '{"error":"bad aspect ratio"}', provider.BadRequestError),
            (422, "{}", provider.BadRequestError),
            (500, "{}", provider.ProviderError),
            (529, '{"type":"overloaded_error"}', provider.ProviderError),
        ],
    )
    def test_classification(self, status: int, body: str, cls: type) -> None:
        error = provider._http_error(status, body, "xai/m", "xai")
        assert type(error) is cls
        assert error.status_code == status
        assert body[:50] in str(error)

    def test_blocked_and_refused_are_terminal_bad_requests(self) -> None:
        assert issubclass(provider.ContentPolicyViolationError, provider.BadRequestError)
        assert issubclass(provider.RefusalError, provider.BadRequestError)
        assert not issubclass(provider.RefusalError, provider.ContentPolicyViolationError)

    def test_billed(self) -> None:
        billed = provider._http_error(
            400,
            '{"code":"imagine:content-moderated","usage":{"cost_in_usd_ticks":200000000}}',
            "m",
            "xai",
        )
        free = provider._http_error(400, '{"code":"imagine:content-moderated"}', "m", "xai")
        assert (billed.billed, free.billed) == (True, False)


class TestTransport:
    def _urlopen(self, monkeypatch: pytest.MonkeyPatch, effect: Any) -> None:
        def fake(request, timeout):  # noqa: ANN001
            if isinstance(effect, BaseException):
                raise effect
            return io.BytesIO(effect)

        monkeypatch.setattr(provider.urllib.request, "urlopen", fake)

    def _post(self) -> dict[str, Any]:
        return provider._post(
            "https://api.x.ai/v1/x", {}, {}, timeout=5, model="xai/m", provider="xai"
        )

    def test_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._urlopen(monkeypatch, b'{"ok": 1}')
        assert self._post() == {"ok": 1}

    def test_http_error_body_classified(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._urlopen(
            monkeypatch,
            urllib.error.HTTPError(
                "u", 429, "slow", Message(), io.BytesIO(b'{"error":"slow down"}')
            ),
        )
        with pytest.raises(provider.RateLimitError, match="slow down"):
            self._post()

    @pytest.mark.parametrize(
        "effect",
        [
            TimeoutError("read"),
            urllib.error.URLError(TimeoutError("connect")),
        ],
    )
    def test_timeouts(self, monkeypatch: pytest.MonkeyPatch, effect: BaseException) -> None:
        self._urlopen(monkeypatch, effect)
        with pytest.raises(provider.RequestTimeoutError):
            self._post()

    def test_unreachable_is_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._urlopen(monkeypatch, urllib.error.URLError("refused"))
        with pytest.raises(provider.ProviderError) as info:
            self._post()
        assert type(info.value) is provider.ProviderError

    @pytest.mark.parametrize("raw", [b"<html>", b"[1, 2]"])
    def test_non_object_reply(self, monkeypatch: pytest.MonkeyPatch, raw: bytes) -> None:
        self._urlopen(monkeypatch, raw)
        with pytest.raises(provider.ProviderError):
            self._post()
