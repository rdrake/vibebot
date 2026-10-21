"""make_meme inside the assistant loop: minted like an image, delivered like one.

The fabricated-image guard treats any own-host URL the turn did not mint as
invented and forces generate_image. A meme URL comes from make_meme, so it
must count as minted, and a successful meme short-circuits the same way a
successful draw does — the URL is the deliverable, not a sentence about it.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from llm.assistant import ToolResult
from llm.service import LLMService

from .conftest import make_completion_response, make_tool_call

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

HOSTED = "https://paste.boxlabs.uk/img/img_6a6e571adf508.png"
MEME_SCHEMA = {"type": "function", "function": {"name": "make_meme", "parameters": {}}}


def _ok_handler(_args: dict) -> ToolResult:
    return ToolResult(content=json.dumps({"status": "ok", "message": HOSTED}))


@pytest.fixture
def service(make_service) -> LLMService:  # type: ignore[no-untyped-def]
    svc, _plugin = make_service(
        assistantModel="xai/grok-4.3", httpUrlBase="https://irc.rdrake.org/llm"
    )
    return svc


def _run(service: LLMService, mocker: MockerFixture, responses: list, handler):
    calls: list[dict] = []

    def fake_completion(**kwargs: object) -> object:
        calls.append(kwargs)  # type: ignore[arg-type]
        return responses[len(calls) - 1]

    mocker.patch("llm.service.provider.chat", side_effect=fake_completion)
    result = service.assistant_completion(
        prompt="make a drake meme: a / b",
        nick="rdrake",
        channel="#afternet",
        db=mocker.MagicMock(),
        context=mocker.MagicMock(),
        bot_nick="VibeBot",
        capabilities=frozenset({"llm.ask", "llm.draw"}),
        account="rdrake",
        extra_tools=[MEME_SCHEMA],
        extra_handlers={"make_meme": handler},
    )
    return result, calls


def test_successful_meme_short_circuits_to_the_url(service, mocker) -> None:
    responses = [
        make_completion_response(
            None,
            tool_calls=[make_tool_call("make_meme", {"template": "drake", "lines": ["a", "b"]})],
        ),
    ]
    handler = _ok_handler

    result, calls = _run(service, mocker, responses, handler)

    assert result.content == HOSTED
    assert len(calls) == 1
    assert result.last_successful_tool == "make_meme"


def test_meme_url_is_not_treated_as_fabricated(service, mocker) -> None:
    """A second step that repeats the meme link must not trigger a forced draw."""
    responses = [
        make_completion_response(
            None,
            tool_calls=[
                make_tool_call("make_meme", {"template": "drake", "lines": ["a", "b"]}),
                make_tool_call("search_web", {"query": "x"}),
            ],
        ),
        make_completion_response(f"Here: {HOSTED}"),
    ]
    handler = _ok_handler
    mocker.patch.object(service, "_web_search_tool", create=True)

    result, calls = _run(service, mocker, responses, handler)

    assert HOSTED in (result.content or "")
    assert all(
        c.get("tool_choice") != {"type": "function", "function": {"name": "generate_image"}}
        for c in calls
    )


def test_meme_error_is_relayed_not_redrawn(service, mocker) -> None:
    """Unknown template → the model repeats the suggestions; an invented link is replaced by the real error."""
    error = "No meme template called 'loss'. Try: drake (Drakeposting)"
    responses = [
        make_completion_response(
            None, tool_calls=[make_tool_call("make_meme", {"template": "loss", "lines": ["a"]})]
        ),
        make_completion_response("https://paste.boxlabs.uk/img/img_deadbeef.png"),
    ]
    handler = lambda _args: ToolResult(content=json.dumps({"error": error}))  # noqa: E731

    result, calls = _run(service, mocker, responses, handler)

    assert len(calls) == 2
    assert "img_deadbeef" not in (result.content or "")
    assert "loss" in (result.content or "")
