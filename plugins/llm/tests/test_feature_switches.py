"""drawEnabled / animateEnabled: a feature that is off or unconfigured is not advertised.

A model offered a tool that cannot work calls it and then narrates the failure,
so the chat surface drops generate_image and generate_video whenever the
matching command would be refused.
"""

from __future__ import annotations

import pytest

from .conftest import make_completion_response


def _chat_tool_names(make_service, mocker, **overrides) -> set[str]:
    service, _plugin = make_service(assistantModel="gpt-4", **overrides)
    completion = mocker.patch(
        "llm.service.litellm.completion", return_value=make_completion_response("hi")
    )
    mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)
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


@pytest.fixture
def video_box(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    monkeypatch.setenv("ANIMATE_API_KEY", "animate-key-for-tests-0000")
    return {"animateApiUrl": "http://video-box:14205"}


class TestDrawSwitch:
    def test_on_and_keyed_advertises_generate_image(self, make_service, mocker) -> None:
        assert "generate_image" in _chat_tool_names(make_service, mocker)

    def test_off_hides_generate_image(self, make_service, mocker) -> None:
        names = _chat_tool_names(make_service, mocker, drawEnabled=False)
        assert "generate_image" not in names

    def test_missing_image_key_hides_generate_image(
        self, make_service, mocker, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = _chat_tool_names(make_service, mocker, imageModel="xai/grok-imagine-image")
        assert "generate_image" in names
        monkeypatch.delenv("XAI_API_KEY")
        names = _chat_tool_names(make_service, mocker, imageModel="xai/grok-imagine-image")
        assert "generate_image" not in names

    def test_reason_names_the_cause(self, make_service) -> None:
        service, _ = make_service(drawEnabled=False)
        assert service.draw_unavailable_reason("#test") == "Drawing is turned off in this channel."
        service, _ = make_service()
        assert service.draw_unavailable_reason("#test") is None


class TestAnimateSwitch:
    def test_configured_advertises_generate_video(self, make_service, mocker, video_box) -> None:
        assert "generate_video" in _chat_tool_names(make_service, mocker, **video_box)

    def test_off_hides_generate_video(self, make_service, mocker, video_box) -> None:
        names = _chat_tool_names(make_service, mocker, animateEnabled=False, **video_box)
        assert "generate_video" not in names

    def test_off_refuses_the_submission(self, make_service, video_box) -> None:
        service, _ = make_service(animateEnabled=False, **video_box)
        result = service.video_generation("a clip", channel="#test")
        assert result.error == "Video is turned off in this channel."


class TestCommandsRefuse:
    def test_draw_command_replies_with_the_reason(self, plugin_env) -> None:
        plugin, irc, msg = plugin_env
        plugin.llm_service.draw_unavailable_reason.return_value = "Drawing is off."
        plugin.draw(irc, msg, ["a", "sunset"])
        irc.reply.assert_called_once_with("Drawing is off.")
        plugin.llm_service.assistant_completion.assert_not_called()

    def test_animate_command_replies_when_off(self, plugin_env) -> None:
        plugin, irc, msg = plugin_env
        plugin.llm_service.video_available.return_value = False
        plugin.llm_service.animate_available.return_value = True
        plugin.animate(irc, msg, ["a", "clip"])
        irc.reply.assert_called_once_with("Video is turned off in this channel.")
        plugin.llm_service.assistant_completion.assert_not_called()
