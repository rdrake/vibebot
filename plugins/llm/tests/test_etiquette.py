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
