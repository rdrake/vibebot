"""A chat message about reminders must reach a pending-task tool.

On 2026-09-25 "vibebot cancel my reminder" got "Slate wiped clean" with
tool_calls=0, and the follow-up "list my reminders" got "You don't have any"
with tool_calls=0 too. The reminder fired four minutes later. Prompt rules
already said ALWAYS list first; gemini skipped them. Step 0 now has to call a
tool, and the model still picks which one.
"""

from __future__ import annotations

from typing import Any

import pytest
from llm.assistant import ToolCallbackResult

from .conftest import make_completion_response


def _run(service: Any, mocker: Any, prompt: str, profile: str = "chat") -> dict[str, Any]:
    """One assistant_completion, returning step 0's completion kwargs."""
    completion = mocker.patch(
        "llm.service.provider.chat",
        side_effect=[make_completion_response("ok")],
    )
    service.assistant_completion(
        prompt=prompt,
        nick="rdrake",
        channel="#linux",
        db=mocker.MagicMock(),
        context=mocker.MagicMock(),
        bot_nick="vibebot",
        route_profile=profile,
        account="rdrake",
        set_reminder_fn=lambda _t: ToolCallbackResult(True, "set"),
        list_pending_tasks_fn=lambda: [],
        cancel_pending_task_fn=lambda _i: {"status": "ok"},
        cancel_all_pending_tasks_fn=lambda: {"status": "ok"},
    )
    return completion.call_args_list[0].kwargs


@pytest.fixture
def service(make_service):  # type: ignore[no-untyped-def]
    svc, _plugin = make_service(assistantModel="gpt-4")
    return svc


@pytest.mark.parametrize(
    "prompt",
    [
        "cancel my reminder",
        "now list my reminders",
        "what are my reminders",
        "remind me to rename my git repo master branches to main in 5 minutes",
    ],
)
def test_reminder_talk_on_chat_requires_a_tool(service, mocker, prompt: str) -> None:
    assert _run(service, mocker, prompt)["tool_choice"] == "required"


def test_other_chat_is_not_forced(service, mocker) -> None:
    assert "tool_choice" not in _run(service, mocker, "salary people aren't in the unit")


def test_reminder_fire_is_not_forced(service, mocker) -> None:
    """The fire frame says "reminder" on every fire; it must not force a tool."""
    kwargs = _run(
        service,
        mocker,
        "(rdrake's reminder, set earlier, is firing now.)\n\nrename branches",
        profile="remind_action",
    )
    assert "tool_choice" not in kwargs
