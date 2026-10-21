"""Tests for IRC etiquette compliance in LLM plugin.

These tests verify that the bot follows IRC etiquette principles:
- Concise, non-flooding responses
- No markdown formatting in ask responses
- Code output via HTTP links (not pasted)
- Rate limiting to prevent spam
- Appropriate tone and formatting
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from unittest.mock import Mock

    from pytest_mock import MockerFixture


class TestSystemPromptEtiquette:
    """Tests that the SHIPPED default system prompts encode IRC etiquette.

    GIVEN the system prompts are the primary mechanism for instructing the
    LLM to behave appropriately for IRC.

    These read the registered defaults from config.py (the values operators
    actually ship with), NOT a prompt the test injected. A regression that
    rewrites the default prompt and drops the IRC / plain-text / brevity
    guidance is therefore caught here.
    """

    def test_assistant_default_prompt_frames_replies_as_irc(self) -> None:
        """GIVEN shipped assistantSystemPrompt default WHEN read THEN frames replies as IRC chat."""
        import llm.config  # noqa: F401 — import side effect registers the value
        import supybot.conf as conf

        prompt = conf.supybot.plugins.LLM.assistantSystemPrompt().lower()

        assert "irc" in prompt

    def test_assistant_default_prompt_forbids_markdown(self) -> None:
        """GIVEN shipped assistantSystemPrompt default WHEN read THEN instructs plain text / no markdown."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        prompt = conf.supybot.plugins.LLM.assistantSystemPrompt().lower()

        assert "markdown" in prompt
        assert "plain text" in prompt

    def test_assistant_default_prompt_instructs_brevity(self) -> None:
        """GIVEN shipped assistantSystemPrompt default WHEN read THEN instructs tight / short replies."""
        import llm.config  # noqa: F401
        import supybot.conf as conf

        prompt = conf.supybot.plugins.LLM.assistantSystemPrompt().lower()

        assert "tight" in prompt or "one line" in prompt

    def test_code_default_prompt_delivers_code_via_url_not_chat(self) -> None:
        """GIVEN shipped codeSystemPrompt default WHEN read THEN code is delivered via URL, not pasted.

        The code *reply* itself is still plain-text IRC ("no markdown"); only
        the linked code page is rendered. This pins that contract — earlier
        tests wrongly asserted the reply "allows markdown".
        """
        import llm.config  # noqa: F401
        import supybot.conf as conf

        prompt = conf.supybot.plugins.LLM.codeSystemPrompt().lower()

        assert "url" in prompt
        assert "do not paste" in prompt
        assert "no markdown" in prompt


class TestResponseLengthHandling:
    """Tests for response length handling to prevent IRC flooding."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path, make_service) -> None:
        """Set up test fixtures."""
        self.tmp_path = tmp_path
        self.service, self.mock_plugin = make_service(
            httpRoot=str(tmp_path),
            httpUrlBase="https://example.com/llm",
            fileCleanupAge=24,
            fileCleanupMax=1000,
        )

    def test_code_save_returns_http_url(self) -> None:
        """GIVEN code content WHEN saved THEN returns HTTP URL."""
        code = "def hello():\n    print('Hello, World!')\n"
        url = self.service.save_code_to_http(code)

        assert url is not None
        assert url.startswith("https://example.com/llm/")
        assert url.endswith(".html")

    def test_code_url_contains_code_prefix(self) -> None:
        """GIVEN code content WHEN saved THEN URL contains 'code' prefix."""
        code = "print('test')"
        url = self.service.save_code_to_http(code)

        assert "code_" in url

    def test_line_count_calculation(self) -> None:
        """GIVEN multi-line text WHEN counted THEN accurate line count."""
        code = "line1\nline2\nline3\nline4\nline5"
        line_count = code.count("\n")

        # 5 lines = 4 newlines (last line doesn't end with newline)
        assert line_count == 4

    def test_image_responses_are_urls(self, mocker: MockerFixture) -> None:
        """GIVEN image generated WHEN response returned THEN is URL not data."""
        from llm import provider

        self.mock_plugin.registryValue = mocker.Mock(
            side_effect=lambda key, channel=None: {
                "imageModel": "xai/grok-imagine-image",
                "timeout": 30,
                "maxPromptLength": 10000,
            }.get(key)
        )

        mocker.patch("llm.service.provider.draw", return_value=provider.Picture(data=b"png"))
        mocker.patch.object(
            self.service,
            "_save_image_bytes",
            return_value="https://example.com/llm/img_local.png",
        )
        result = self.service.image_generation("a cat")

        assert result.content.startswith("http")
        assert "base64" not in result.content.lower()


class TestResponseAppropriateness:
    """Tests for context being passed as user message (not system prompt).

    Context is now passed as a user message to mitigate prompt injection
    attacks via channel topics. These tests verify the context message
    contains the expected information.
    """

    @pytest.fixture(autouse=True)
    def setup(self, make_service, mocker: MockerFixture) -> None:
        """Set up test fixtures."""
        self._mocker = mocker
        self.service, self.mock_plugin = make_service()
        self.mock_plugin.startup_time = time.time() - 3600

    def _make_mock_irc(
        self,
        network: str = "TestNet",
        nick: str = "testbot",
        channels: dict | None = None,
    ) -> Mock:
        """Create a mock IRC object."""
        irc = self._mocker.Mock()
        irc.network = network
        irc.nick = nick
        irc.state = self._mocker.Mock()
        irc.state.channels = channels or {}
        irc.state.nickToAccount = self._mocker.Mock(return_value=None)
        return irc

    def _make_mock_msg(
        self,
        channel: str = "#test",
        nick: str = "user",
    ) -> Mock:
        """Create a mock IRC message object."""
        msg = self._mocker.Mock()
        msg.args = (channel, "some message text")
        msg.prefix = f"{nick}!username@hostname.example.com"
        return msg

    def _make_mock_channel_state(
        self,
        users: set | None = None,
        topic: str | None = None,
        ops: set | None = None,
        halfops: set | None = None,
        voices: set | None = None,
    ) -> Mock:
        """Create a mock channel state object."""
        ch_state = self._mocker.Mock()
        ch_state.users = users or {"user1", "user2", "testbot"}
        ch_state.modes = {"n": None, "t": None}
        ch_state.topic = topic
        ch_state.ops = ops or set()
        ch_state.halfops = halfops or set()
        ch_state.voices = voices or set()
        ch_state.isOp = self._mocker.Mock(return_value=False)
        ch_state.isHalfop = self._mocker.Mock(return_value=False)
        ch_state.isVoice = self._mocker.Mock(return_value=False)
        return ch_state

    def test_channel_context_in_user_message(self) -> None:
        """GIVEN channel message WHEN context built THEN channel in user message."""
        ch_state = self._make_mock_channel_state()
        irc = self._make_mock_irc(channels={"#tech": ch_state})
        msg = self._make_mock_msg(channel="#tech")

        result = self.service._build_context_message(irc, msg)

        assert result is not None
        assert result["role"] == "user"
        assert "#tech" in result["content"]
        assert "Channel:" in result["content"]

    def test_pm_context_excludes_channel(self) -> None:
        """GIVEN private message WHEN context built THEN no channel info."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg(channel="testbot", nick="someuser")

        result = self.service._build_context_message(irc, msg)

        assert result is not None
        # PM has no channel info (target is bot's nick, not a channel)
        assert "Channel:" not in result["content"]

    def test_channel_topic_in_user_message_raw(self) -> None:
        """GIVEN channel with topic WHEN topic message built THEN topic
        passed raw in its own user message. Topic lives outside the
        cacheable prefix (system + context + ack) so topic edits don't
        reset xAI's automatic prompt cache; it's no longer appended to
        the context message."""
        ch_state = self._make_mock_channel_state(topic="Python programming help")
        irc = self._make_mock_irc(channels={"#python": ch_state})
        msg = self._make_mock_msg(channel="#python")

        ctx = self.service._build_context_message(irc, msg)
        topic_msg = self.service._build_topic_message(irc, msg)

        # Topic no longer lives in the context message.
        assert ctx is not None
        assert "Topic" not in ctx["content"]
        # The dedicated topic message carries it raw, no XML framing.
        assert topic_msg is not None
        assert topic_msg["role"] == "user"
        assert "Python programming help" in topic_msg["content"]
        assert "<channel_topic" not in topic_msg["content"]

    def test_speaking_with_in_speaker_message(self) -> None:
        """GIVEN message from user WHEN speaker message built THEN nick included."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg(nick="someuser")

        # Speaker info lives in _build_speaker_message, kept out of the
        # cacheable context prefix.
        ctx = self.service._build_context_message(irc, msg)
        assert ctx is not None
        assert "Speaking with" not in ctx["content"]

        speaker = self.service._build_speaker_message(irc, msg)
        assert speaker is not None
        assert "Speaking with: someuser" in speaker["content"]

    def test_date_included(self) -> None:
        """GIVEN any message WHEN context built THEN date included."""
        irc = self._make_mock_irc()
        msg = self._make_mock_msg()

        result = self.service._build_context_message(irc, msg)

        assert "Date:" in result["content"]
