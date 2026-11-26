"""The chat assistant's advisor: a stronger model Haiku may consult once a turn.

Contract (docs/guide/operator/configuration.md, LLM.advisorModel, and the
advisor tool docs): chat turns only, one consult per turn, a step that may
consult gets a longer timeout, a consult that pauses mid-request is resumed
with its turn sent back unchanged, and a pairing the API would reject is
never sent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from llm import provider
from llm.assistant import ToolResult
from llm.profile import PROFILE_VERSE
from llm.service import ADVISOR_TIMEOUT

from .conftest import make_completion_response, make_tool_call

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

HAIKU = "anthropic/claude-haiku-5-5"
OPUS = "anthropic/claude-opus-5-5"
SONNET = "anthropic/claude-sonnet-5-5"

PENDING = {"type": "server_tool_use", "id": "srvtoolu_1", "name": "advisor", "input": {}}
ADVICE = {
    "type": "advisor_tool_result",
    "tool_use_id": "srvtoolu_1",
    "content": {"type": "advisor_redacted_result", "encrypted_content": "EqQB..."},
}


def consulted(text: str | None = None, **kwargs: Any) -> provider.Reply:
    base = make_completion_response(text, **kwargs)
    return provider.Reply(
        text=base.text,
        tool_calls=base.tool_calls,
        usage=base.usage,
        blocks=(PENDING, ADVICE),
        advisor_called=True,
    )


def paused() -> provider.Reply:
    return provider.Reply(finish_reason="pause", blocks=(PENDING,))


@pytest.fixture
def run(make_service, mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Run one turn against scripted replies; return the result and each call's kwargs."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-0123456789")

    def _run(
        replies: list[provider.Reply], call: dict[str, Any] | None = None, **overrides: object
    ) -> tuple[Any, list[dict[str, Any]], Any]:
        svc, _plugin = make_service(
            **{"assistantModel": HAIKU, "advisorModel": OPUS, "timeout": 30, **overrides}
        )
        svc.log = mocker.MagicMock()
        calls: list[dict[str, Any]] = []

        def fake_chat(**kwargs: Any) -> provider.Reply:
            calls.append({**kwargs, "messages": list(kwargs["messages"])})
            return replies[min(len(calls), len(replies)) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_chat)
        result = svc.assistant_completion(
            prompt="plan my week",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            **(call or {}),
        )
        return result, calls, svc.log

    return _run


def test_chat_step_offers_the_advisor_with_the_longer_timeout(run) -> None:  # type: ignore[no-untyped-def]
    result, calls, _log = run([make_completion_response("done")])

    # Offered but not consulted: the reply carries no ⚡.
    assert not result.advisor_used
    assert calls[0]["advisor"] == OPUS
    assert calls[0]["timeout"] == ADVISOR_TIMEOUT


def test_after_a_consult_later_steps_neither_offer_it_nor_wait_longer(run) -> None:  # type: ignore[no-untyped-def]
    search = consulted(tool_calls=[make_tool_call("search_web", {"query": "week"})])
    result, calls, _log = run(
        [search, make_completion_response("here's the plan")],
        call={"search_fn": lambda _q: ToolResult(content='{"results": "x"}')},
    )

    assert result.content == "here's the plan"
    assert result.advisor_used
    assert calls[0]["advisor"] == OPUS
    assert "advisor" not in calls[1]
    assert calls[1]["timeout"] == 30
    # The encrypted advice rides back to the executor on the next step.
    assert calls[1]["messages"][-2]["anthropic_content"] == [PENDING, ADVICE]


def test_a_paused_consult_is_resumed_with_its_turn_and_the_advisor(run) -> None:  # type: ignore[no-untyped-def]
    result, calls, _log = run([paused(), consulted("the plan")])

    assert result.content == "the plan"
    assert [c["advisor"] for c in calls] == [OPUS, OPUS]
    assert calls[1]["messages"][-1] == {
        "role": "assistant",
        "content": "",
        "anthropic_content": [PENDING],
    }


def test_verse_turns_never_consult(run) -> None:  # type: ignore[no-untyped-def]
    _result, calls, _log = run(
        [make_completion_response("the forest stirs")],
        call={"route_profile": PROFILE_VERSE},
    )

    assert "advisor" not in calls[0]
    assert calls[0]["timeout"] == 30


def test_an_advisor_weaker_than_the_executor_is_skipped_with_a_warning(run) -> None:  # type: ignore[no-untyped-def]
    _result, calls, log = run(
        [make_completion_response("ok")], assistantModel=OPUS, advisorModel=SONNET
    )

    assert "advisor" not in calls[0]
    warnings = [c.args[0] % c.args[1:] for c in log.warning.call_args_list]
    assert f"advisorModel {SONNET} cannot advise {OPUS}; consulting nobody" in warnings


def test_an_xai_executor_skips_the_advisor_quietly(run) -> None:  # type: ignore[no-untyped-def]
    _result, calls, log = run(
        [make_completion_response("ok")], assistantModel="xai/grok-4-1-fast-non-reasoning"
    )

    assert "advisor" not in calls[0]
    assert not any("advisorModel" in c.args[0] for c in log.warning.call_args_list)


def test_empty_advisor_model_turns_it_off(run) -> None:  # type: ignore[no-untyped-def]
    _result, calls, _log = run([make_completion_response("ok")], advisorModel="")

    assert "advisor" not in calls[0]
    assert calls[0]["timeout"] == 30
