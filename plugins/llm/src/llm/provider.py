"""Calls to model providers: Anthropic and xAI.

Everything a provider does differently stays inside this module. That covers
the wire format, which parameters each provider accepts, prompt-cache routing,
prices, and what an error body means. The rest of the plugin speaks one message
format (OpenAI chat, plus ``thinking_blocks`` on assistant turns) and gets one
:class:`Reply` back whoever answered.

Stdlib HTTP on purpose. litellm was compromised on PyPI twice in 2026 (1.82.7/8,
1.107.0). A provider SDK would bring pydantic and an HTTP stack for what is
four JSON POSTs. The request bodies here were checked against what litellm sent
for the same calls (tests/fixtures/provider_wire/).

Model strings are ``provider/name``:

- ``anthropic/<name>``: Messages API.
- ``xai/<name>``: OpenAI-compatible chat and image endpoints, plus the Responses
  API for web search.
"""

from __future__ import annotations

import base64
import binascii
import http.client
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from . import apikeys

_log = logging.getLogger("supybot.plugins.LLM.provider")

ANTHROPIC_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"
XAI_URL = "https://api.x.ai/v1"
USER_AGENT = "vibebot"

# Anthropic requires max_tokens. litellm filled in the model's 128K ceiling;
# nothing an IRC bot says comes close to this one.
DEFAULT_MAX_TOKENS = 16000

# xAI reports cost as integer ticks. A tick of 1e-10 USD is inferred, not
# documented: 200000000 ticks on a refused draw matched the $0.02 list price
# (measured 2026-08-15).
XAI_TICK_USD = 1e-10


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0


_GROK_43 = Price(input=1.25, output=2.50, cache_read=0.20)

# Every chat model prod runs. Missing a model is not a rounding error: its spend
# books at $0 (logged once), which is how grok-4-1-fast-reasoning went
# unpriced under litellm.
PRICES: dict[str, Price] = {
    # Rates for prompts up to 100K tokens; nothing here sends more.
    "anthropic/claude-haiku-5-5": Price(
        input=0.10, output=0.50, cache_read=0.01, cache_write=0.125
    ),
    "xai/grok-4.3": _GROK_43,
    # Both fast aliases are served by grok-4.3 (owner, 2026-10-14), so they
    # are priced at its rates.
    "xai/grok-4-1-fast-reasoning": _GROK_43,
    "xai/grok-4-1-fast-non-reasoning": _GROK_43,
}

# Per image. xAI's own ticks win when present; this is the fallback, and the
# price of a billed refusal, whose ticks are not trusted as a unit.
IMAGE_PRICES: dict[str, float] = {
    "xai/grok-imagine-image": 0.02,
}

_unpriced: set[str] = set()


def _warn_unpriced(model: str) -> None:
    if model not in _unpriced:
        _unpriced.add(model)
        _log.warning("model %s has no price in provider.py; its spend is booked as $0.00", model)


def image_price(model: str) -> float:
    """List price of one image from ``model``, or 0.0 with a one-time warning."""
    price = IMAGE_PRICES.get(model)
    if price is None:
        _warn_unpriced(model)
        return 0.0
    return price


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # JSON text, as the model wrote it


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0  # the whole prompt, cached part included
    output_tokens: int = 0
    cached_tokens: int = 0  # prompt tokens read from the provider's cache


@dataclass(frozen=True)
class Reply:
    """One model turn.

    ``finish_reason`` is ``stop``, ``length``, ``tool_calls`` or ``refusal``
    (Anthropic's safety classifier, which declines with HTTP 200).
    """

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    grounded: bool = False  # search only: the web was actually consulted
    thinking: tuple[dict[str, Any], ...] = ()  # opaque, returned verbatim

    def turn(self) -> dict[str, Any]:
        """This reply as an assistant message to append to the conversation.

        Carries the thinking blocks back: Anthropic wants a turn's thinking
        returned with its tool calls, unchanged, or the next step reasons
        from scratch. Other providers never see them.
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in self.tool_calls
            ]
        if self.thinking:
            message["thinking_blocks"] = list(self.thinking)
        return message


@dataclass(frozen=True)
class Picture:
    """A generated image. ``data`` is None when the provider returned no
    picture, which xAI uses to mean its filter blocked it."""

    data: bytes | None = None
    cost: float = 0.0


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(Exception):
    """A provider call failed. Transient unless a subclass says otherwise.

    ``billed`` is True when the provider's error body carried usage. xAI
    charges for a draw its output filter rejected, and says so this way.
    """

    def __init__(self, message: str = "", *, billed: bool = False) -> None:
        super().__init__(message)
        self.billed = billed


class RequestTimeoutError(ProviderError):
    """No answer in time. The request may still have run."""


class RateLimitError(ProviderError):
    pass


class AuthenticationError(ProviderError):
    """Missing, invalid or unauthorised key. Terminal."""


class BadRequestError(ProviderError):
    """The provider rejected the request itself (any 4xx not classed
    otherwise). Terminal."""


class ContentPolicyViolationError(BadRequestError):
    """The provider's moderation rejected the content. For images it rejects
    the generated picture, and a reworded prompt often gets through."""


class RefusalError(BadRequestError):
    """The model declined on policy grounds.

    Kept apart from :class:`ContentPolicyViolationError` because they drive different
    things: a refusal picks the user-facing line and the chat fallback model,
    a block arms the image rewrite loop. A prompt that tripped a hard safety
    check is exactly the one that loop must not go back and reword.
    """


# Matched on prose because xAI's permission-denied code is also what a key
# with no access to a model gets. Only markers seen from our two providers are
# listed: xAI's image filter, and xAI chat's 403 refusal.
_BLOCKED_MARKERS = ("content-moderated", "content moderation")
_REFUSED_MARKERS = ("violates usage guidelines", "content violates")


def _billed(body: str) -> bool:
    try:
        parsed = json.loads(body)
    except ValueError:
        return False
    return isinstance(parsed, dict) and bool(parsed.get("usage"))


def _http_error(status: int, body: str, provider: str) -> ProviderError:
    """The typed error for a non-2xx response.

    Refusal is checked before block: a chat refusal that also mentions
    moderation must reach the fallback model, not the image rewrite loop.
    """
    lowered = body.lower()
    cls: type[ProviderError]
    if 400 <= status < 500 and any(m in lowered for m in _REFUSED_MARKERS):
        cls = RefusalError
    elif 400 <= status < 500 and any(m in lowered for m in _BLOCKED_MARKERS):
        cls = ContentPolicyViolationError
    elif status == 429:
        cls = RateLimitError
    elif status in (401, 403):
        cls = AuthenticationError
    elif 400 <= status < 500:
        cls = BadRequestError
    else:
        cls = ProviderError
    return cls(f"{provider} HTTP {status}: {body[:1000]}", billed=_billed(body))


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects. urllib would follow a POST's redirect as a GET and
    carry the API key header to whatever host it names."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        raise urllib.error.HTTPError(
            req.full_url, code, f"refused redirect to {newurl}", headers, fp
        )


_opener = urllib.request.build_opener(_NoRedirect)


def _error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", "replace")
    except (OSError, http.client.HTTPException, AttributeError):
        return ""


def _post(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    *,
    timeout: float,
    provider: str,
) -> dict[str, Any]:
    """POST ``body`` as JSON and return the decoded JSON reply, or raise a
    :class:`ProviderError`. Tests replace this one function."""
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT, **headers},
        method="POST",
    )
    try:
        with _opener.open(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise _http_error(exc.code, _error_body(exc), provider) from exc
    except TimeoutError as exc:
        raise RequestTimeoutError(f"{provider} timed out after {timeout}s") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise RequestTimeoutError(f"{provider} timed out after {timeout}s") from exc
        raise ProviderError(f"{provider} unreachable: {exc.reason}") from exc
    except (OSError, http.client.HTTPException) as exc:
        # A reset or truncated read after the connection opened.
        raise ProviderError(f"{provider} connection failed: {exc!r}") from exc
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ProviderError(f"{provider} returned non-JSON: {raw[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise ProviderError(f"{provider} returned {type(parsed).__name__}")
    return parsed


def _split(model: str) -> tuple[str, str]:
    provider, sep, name = model.partition("/")
    if not sep or not name:
        raise BadRequestError(f"model {model!r} is not provider/name")
    return provider.lower(), name


def _key(model: str, provider: str) -> str:
    key = apikeys.api_key_for(model)
    if not key:
        raise AuthenticationError(f"no API key for {provider} (set {apikeys.env_var_for(model)})")
    return key


def _xai_headers(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


def chat(
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout: float,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    max_tokens: int | None = None,
    json_reply: bool | dict[str, Any] = False,
    cache_key: str | None = None,
) -> Reply:
    """One chat completion.

    ``messages`` and ``tools`` are in OpenAI chat format; ``tool_choice`` is
    ``"auto"``, ``"required"``, ``"none"`` or ``{"type": "function",
    "function": {"name": ...}}``. ``json_reply`` is True for "answer with a
    JSON object" or a JSON Schema the answer must match. ``cache_key`` pins
    xAI's prompt-cache routing; Anthropic caching is automatic.
    """
    provider, name = _split(model)
    if provider == "anthropic":
        body = _anthropic_request(
            name,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            json_reply=json_reply,
        )
        headers = {"x-api-key": _key(model, provider), "anthropic-version": ANTHROPIC_VERSION}
        data = _post(
            f"{ANTHROPIC_URL}/messages",
            headers,
            body,
            timeout=timeout,
            provider=provider,
        )
        return _anthropic_reply(model, data)
    if provider == "xai":
        body = _xai_request(
            name,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
            json_reply=json_reply,
        )
        if cache_key:
            # Sticky routing to the server holding this prefix's cache;
            # without it the load balancer scattered requests and
            # cached_tokens stayed 0.
            body["prompt_cache_key"] = cache_key
        data = _post(
            f"{XAI_URL}/chat/completions",
            _xai_headers(_key(model, provider)),
            body,
            timeout=timeout,
            provider=provider,
        )
        return _xai_reply(model, data)
    raise BadRequestError(f"no chat provider {provider!r} (use anthropic/ or xai/)")


def _text_of(content: Any) -> str:
    """Plain text of an OpenAI message ``content`` (string or parts list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text") or ""
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _anthropic_user_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict[str, Any]] = []
    for part in content if isinstance(content, list) else []:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and part.get("text"):
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _tool_input(arguments: Any) -> dict[str, Any]:
    try:
        parsed = json.loads(arguments or "{}")
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _anthropic_request(
    name: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    max_tokens: int | None,
    json_reply: bool | dict[str, Any],
) -> dict[str, Any]:
    """OpenAI-format messages as a Messages API request body.

    System messages are hoisted, in order, to ``system``. A cache breakpoint
    goes on the end of the first one (tool schemas plus the stable prompt,
    identical for every call in a channel), and the top-level ``cache_control``
    caches the growing tail a tool loop's later steps reuse. Measured on
    claude-haiku-5-5: a 4.3K-token prefix read back 4,225 tokens on the second
    call, $0.00054 → $0.000055.
    """
    system: list[dict[str, Any]] = []
    first_system_end = 0
    turns: list[dict[str, Any]] = []

    def add(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"].extend(blocks)
        else:
            turns.append({"role": role, "content": blocks})

    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "system":
            text = _text_of(content)
            if text:
                system.append({"type": "text", "text": text})
                if not first_system_end:
                    first_system_end = len(system)
        elif role == "tool":
            add(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id", ""),
                        "content": _text_of(content),
                    }
                ],
            )
        elif role == "assistant":
            blocks = [dict(block) for block in message.get("thinking_blocks") or []]
            text = _text_of(content)
            if text:
                # An empty text block is a 400. litellm filled the gap with
                # "[System: Empty message content sanitised…]", which the model
                # then read; a tool-call-only turn needs no text at all.
                blocks.append({"type": "text", "text": text})
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": _tool_input(function.get("arguments")),
                    }
                )
            add("assistant", blocks)
        else:
            add("user", _anthropic_user_blocks(content))

    if first_system_end:
        system[first_system_end - 1]["cache_control"] = {"type": "ephemeral"}

    body: dict[str, Any] = {
        "model": name,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        "messages": turns,
        "cache_control": {"type": "ephemeral"},
    }
    if system:
        body["system"] = system
    if tools:
        body["tools"] = [_anthropic_tool(tool) for tool in tools]
    choice = _anthropic_tool_choice(tool_choice)
    if choice:
        body["tool_choice"] = choice
    if isinstance(json_reply, dict):
        body["output_config"] = {"format": {"type": "json_schema", "schema": json_reply}}
    # json_reply=True has no Anthropic switch; the prompt asks for JSON.
    return body


def _anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or {}
    spec: dict[str, Any] = {
        "name": function.get("name", ""),
        "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
    }
    if function.get("description"):
        spec["description"] = function["description"]
    return spec


def _anthropic_tool_choice(choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if choice is None:
        return None
    if isinstance(choice, dict):
        name = (choice.get("function") or {}).get("name")
        return {"type": "tool", "name": name} if name else None
    # Forced choice ("any", "tool") still works on Haiku 5.5; Opus 5.5,
    # Sonnet 5.5 and Fable 5.1 reject it with a 400.
    return {"required": {"type": "any"}, "none": {"type": "none"}}.get(choice)


_ANTHROPIC_FINISH = {
    "end_turn": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "refusal": "refusal",
}


def _anthropic_reply(model: str, data: dict[str, Any]) -> Reply:
    texts: list[str] = []
    calls: list[ToolCall] = []
    thinking: list[dict[str, Any]] = []
    for block in data.get("content") or []:
        kind = block.get("type")
        if kind == "text":
            texts.append(block.get("text") or "")
        elif kind == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=json.dumps(block.get("input") or {}),
                )
            )
        elif kind in ("thinking", "redacted_thinking"):
            thinking.append(block)
    raw = data.get("usage") or {}
    fresh = int(raw.get("input_tokens") or 0)
    written = int(raw.get("cache_creation_input_tokens") or 0)
    read = int(raw.get("cache_read_input_tokens") or 0)
    output = int(raw.get("output_tokens") or 0)
    price = PRICES.get(model)
    if price is None:
        _warn_unpriced(model)
        cost = 0.0
    else:
        cost = (
            fresh * price.input
            + written * price.cache_write
            + read * price.cache_read
            + output * price.output
        ) / 1e6
    return Reply(
        text="".join(texts),
        tool_calls=tuple(calls),
        finish_reason=_ANTHROPIC_FINISH.get(data.get("stop_reason") or "", "stop"),
        usage=Usage(input_tokens=fresh + written + read, output_tokens=output, cached_tokens=read),
        cost=cost,
        thinking=tuple(thinking),
    )


def _xai_request(
    name: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    max_tokens: int | None,
    json_reply: bool | dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": name,
        "messages": [
            {key: value for key, value in message.items() if key != "thinking_blocks"}
            for message in messages
        ],
    }
    if tools:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if max_tokens:
        body["max_tokens"] = max_tokens
    if isinstance(json_reply, dict):
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "strict": True, "schema": json_reply},
        }
    elif json_reply:
        body["response_format"] = {"type": "json_object"}
    return body


def _xai_cost(model: str, usage: dict[str, Any], fresh: int, cached: int, output: int) -> float:
    ticks = usage.get("cost_in_usd_ticks")
    if isinstance(ticks, (int, float)) and ticks > 0:
        return ticks * XAI_TICK_USD
    price = PRICES.get(model)
    if price is None:
        _warn_unpriced(model)
        return 0.0
    return (fresh * price.input + cached * price.cache_read + output * price.output) / 1e6


def _xai_reply(model: str, data: dict[str, Any]) -> Reply:
    choices = data.get("choices") or [{}]
    choice = choices[0] or {}
    message = choice.get("message") or {}
    calls = tuple(
        ToolCall(
            id=call.get("id", ""),
            name=(call.get("function") or {}).get("name", ""),
            arguments=(call.get("function") or {}).get("arguments") or "{}",
        )
        for call in message.get("tool_calls") or []
    )
    usage = data.get("usage") or {}
    prompt = int(usage.get("prompt_tokens") or 0)
    cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    # Reasoning is billed as output and is not in completion_tokens; litellm
    # folded it in, and usage rows and @usage expect it there.
    reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    output = int(usage.get("completion_tokens") or 0) + reasoning
    return Reply(
        text=message.get("content") or "",
        tool_calls=calls,
        finish_reason=choice.get("finish_reason") or "stop",
        usage=Usage(input_tokens=prompt, output_tokens=output, cached_tokens=cached),
        cost=_xai_cost(model, usage, prompt - cached, cached, output),
    )


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def search(model: str, prompt: str, *, timeout: float, cache_key: str | None = None) -> Reply:
    """Answer ``prompt`` with live web search: xAI's Responses API and its
    ``web_search`` tool, the only place xAI offers search now.
    ``Reply.grounded`` says whether the search actually ran."""
    provider, name = _split(model)
    if provider != "xai":
        raise BadRequestError(f"web search needs an xai/ model, not {model!r}")
    body: dict[str, Any] = {"model": name, "input": prompt, "tools": [{"type": "web_search"}]}
    if cache_key:
        body["prompt_cache_key"] = cache_key
    data = _post(
        f"{XAI_URL}/responses",
        _xai_headers(_key(model, provider)),
        body,
        timeout=timeout,
        provider=provider,
    )
    return _responses_reply(model, data)


def _responses_reply(model: str, data: dict[str, Any]) -> Reply:
    texts: list[str] = []
    grounded = False
    for item in data.get("output") or []:
        kind = item.get("type") or ""
        # web_search_call and friends: the tool ran.
        if "search" in kind.lower():
            grounded = True
        if kind != "message":
            continue
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                texts.append(part.get("text") or "")
                if part.get("annotations"):
                    grounded = True  # it cited a result
    usage = data.get("usage") or {}
    prompt = int(usage.get("input_tokens") or 0)
    cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)
    return Reply(
        text="".join(texts),
        usage=Usage(input_tokens=prompt, output_tokens=output, cached_tokens=cached),
        cost=_xai_cost(model, usage, prompt - cached, cached, output),
        grounded=grounded,
    )


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------


def draw(
    model: str,
    prompt: str,
    *,
    timeout: float,
    aspect_ratio: str | None = None,
    hires: bool = False,
    edit_url: str | None = None,
) -> Picture:
    """Generate one image with an ``xai/`` model, or edit the image at
    ``edit_url``. ``hires`` asks for high quality at 2K."""
    provider, name = _split(model)
    if provider != "xai":
        raise BadRequestError(f"images need an xai/ model, not {model!r}")
    body: dict[str, Any] = {"model": name, "prompt": prompt, "response_format": "b64_json"}
    if edit_url:
        body["image"] = {"url": edit_url}
    else:
        body["n"] = 1
    if aspect_ratio:
        body["aspect_ratio"] = aspect_ratio
    if hires:
        body["quality"] = "high"
        body["resolution"] = "2k"
    data = _post(
        f"{XAI_URL}/images/{'edits' if edit_url else 'generations'}",
        _xai_headers(_key(model, provider)),
        body,
        timeout=timeout,
        provider=provider,
    )
    ticks = (data.get("usage") or {}).get("cost_in_usd_ticks")
    cost = (
        ticks * XAI_TICK_USD
        if isinstance(ticks, (int, float)) and ticks > 0
        else image_price(model)
    )
    items = data.get("data") or []
    b64 = items[0].get("b64_json") if items and isinstance(items[0], dict) else None
    try:
        image = base64.b64decode(b64) if isinstance(b64, str) else None
    except (binascii.Error, ValueError):
        image = None
    return Picture(data=image or None, cost=cost)
