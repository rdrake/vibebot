"""A feature that cannot work is not advertised.

A model offered a tool that cannot work calls it and then narrates the failure,
so the chat surface drops generate_image whenever @draw would be refused: when
the image model's provider has no key.
"""

from __future__ import annotations

import pytest

from .conftest import make_completion_response


def _chat_tool_names(make_service, mocker, **overrides) -> set[str]:
    service, _plugin = make_service(assistantModel="gpt-4", **overrides)
    completion = mocker.patch(
        "llm.service.provider.chat", return_value=make_completion_response("hi")
    )
    service.assistant_completion(
        prompt="hello",
        nick="testuser",
        channel="#test",
        db=mocker.MagicMock(),
        context=mocker.MagicMock(),
        bot_nick="VibeBot",
        route_profile="chat",
        account="tester",
    )
    tools = completion.call_args_list[0].kwargs.get("tools") or []
    return {t["function"]["name"] for t in tools}


class TestDrawSwitch:
    def test_on_and_keyed_advertises_generate_image(self, make_service, mocker) -> None:
        assert "generate_image" in _chat_tool_names(make_service, mocker)

    def test_missing_image_key_hides_generate_image(
        self, make_service, mocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = _chat_tool_names(make_service, mocker)
        assert "generate_image" in names
        monkeypatch.delenv("XAI_API_KEY")
        names = _chat_tool_names(make_service, mocker)
        assert "generate_image" not in names


class TestCommandsRefuse:
    def test_draw_command_replies_with_the_reason(self, plugin_env) -> None:
        plugin, irc, msg = plugin_env
        plugin.llm_service.draw_unavailable_reason.return_value = "Drawing is off."
        plugin.draw(irc, msg, ["a", "sunset"])
        irc.reply.assert_called_once_with("Drawing is off.")
        plugin.llm_service.assistant_completion.assert_not_called()
