"""The plugin → service seam, with only LiteLLM mocked.

Every other plugin test replaces LLMService (``plugin_env``) and every service
test calls ``assistant_completion`` directly, so nothing checked that what the
service returns is what the plugin books. Two leaks lived exactly there: an
echo after the retry carried its spend in an empty reply the plugin then
declined to book, and a stashed timeout's placeholder was booked as a success
and written to history as if it were an answer.

These run ``@ask`` through a real ``LLMService`` and a real ``LLMDatabase``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import litellm
import pytest
from llm.persistence import LLMDatabase
from llm.service import LLMService

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture

COST_PER_CALL = 0.001


@pytest.fixture
def seam_env(plugin_env, mocker: MockerFixture, tmp_path):
    """plugin_env with the real service and a real database swapped in."""
    plugin, irc, msg = plugin_env
    plugin.db = LLMDatabase(str(tmp_path / "seam.db"))
    service = LLMService(plugin)
    plugin.llm_service = service
    mocker.patch("llm.service.litellm.completion_cost", return_value=COST_PER_CALL)
    try:
        yield plugin, irc, msg
    finally:
        service.typing.stop()


def _text_response(mocker: MockerFixture, content: str) -> MagicMock:
    resp = mocker.MagicMock()
    resp.choices[0].message.content = content
    resp.choices[0].message.tool_calls = None
    resp.choices[0].finish_reason = "stop"
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 10
    return resp


def _usage_rows(plugin) -> list[tuple]:
    return (
        plugin.db._connect()
        .execute("SELECT command, status, cost, prompt_tokens, error_detail FROM usage")
        .fetchall()
    )


def _history(plugin) -> list[dict[str, str]]:
    # The unauthenticated caller's storage key (Identity.key).
    return plugin.context.get_messages("!testnick", "#test")


def test_delivered_reply_is_booked_and_stored(seam_env, mocker: MockerFixture) -> None:
    plugin, irc, msg = seam_env
    mocker.patch(
        "llm.service.litellm.completion",
        return_value=_text_response(mocker, "The pitch is waterlogged."),
    )

    plugin.ask(irc, msg, ["is the match on"])

    [(command, status, cost, prompt_tokens, _detail)] = _usage_rows(plugin)
    assert (command, status) == ("ask", "success")
    assert cost == pytest.approx(COST_PER_CALL)
    assert prompt_tokens == 100
    assert [m["content"] for m in _history(plugin)] == [
        "is the match on",
        "The pitch is waterlogged.",
    ]


def test_echo_after_retry_is_booked_as_error_and_not_stored(
    seam_env, mocker: MockerFixture
) -> None:
    """GIVEN the model echoes the prompt through its retry WHEN nothing is
    delivered THEN every billed call still lands in one ``error`` row."""
    plugin, irc, msg = seam_env
    completion = mocker.patch(
        "llm.service.litellm.completion",
        side_effect=lambda **_kw: _text_response(mocker, "Finish the story."),
    )

    plugin.ask(irc, msg, ["finish the story"])

    [(_command, status, cost, _tokens, detail)] = _usage_rows(plugin)
    assert status == "error"
    assert "echo" in detail.lower()
    assert completion.call_count >= 2
    assert cost == pytest.approx(COST_PER_CALL * completion.call_count)
    assert _history(plugin) == []


def test_stashed_timeout_is_booked_as_timeout_and_not_stored(
    seam_env, mocker: MockerFixture
) -> None:
    """GIVEN the call times out and the retry is stashed THEN the user sees the
    placeholder, the row says ``timeout``, and history stays empty."""
    plugin, irc, msg = seam_env
    mocker.patch(
        "llm.service.litellm.completion",
        side_effect=litellm.Timeout(message="timed out", model="gpt-4", llm_provider="openai"),
    )

    plugin.ask(irc, msg, ["write me an epic"])

    assert plugin.db._connect().execute("SELECT COUNT(*) FROM pending_tasks").fetchone()[0] == 1
    [(_command, status, _cost, _tokens, _detail)] = _usage_rows(plugin)
    assert status == "timeout"
    assert _history(plugin) == []
