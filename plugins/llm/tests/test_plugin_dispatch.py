"""Plugin dispatch: doPrivmsg, inFilter, command registry, channel join."""

from __future__ import annotations

import collections
import inspect
import threading
import time
from typing import TYPE_CHECKING

import pytest
from llm.identity import Caller, Identity
from llm.plugin import PreflightResult

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestDoPrivmsg:
    """Test plugin doPrivmsg for channel message tracking."""

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture):
        """Create plugin with mocked dependencies."""
        import supybot.conf as supy_conf
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.nick = "usernick"
        mock_msg.args = ("#channel", "hello world")
        mock_msg.time = time.time() + 100  # Future time (not ZNC playback)
        mock_msg.channel = "#channel"
        mock_msg.server_tags = {}  # default: no IRCv3 account-tag

        # Configure command prefix so `@cmd` short-circuits in doPrivmsg.
        chars_value = supy_conf.supybot.reply.whenAddressedBy.chars
        original_chars = chars_value()
        chars_value.setValue("@")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        # Every key truthy except privacyOptInOnly, so chatter is tracked.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: key != "privacyOptInOnly"
        )
        plugin.context = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._route_addressed_to_assistant = mocker.MagicMock()
        # Loom caches (PR 2). Tests that exercise the loom hook set _loom
        # explicitly; default state here is "loom not wired".
        plugin._loom = None
        plugin._loom_bridge = None
        plugin._loom_channel_cache = None
        plugin._loom_network_cache = None
        plugin._loom_bot_nicks_cache = ()
        # Bot-loop guard state (normally built in __init__, which these
        # bare-instance fixtures skip). Empty flags = every speaker reads
        # as a person, so the guard never fires in dispatch tests.
        plugin._bot_flags = {}
        plugin._bot_probed_at = {}
        plugin._bot_reply_counts = {}
        plugin._bot_loop_lock = threading.Lock()
        plugin._own_msgids = collections.OrderedDict()
        plugin._own_msgids_lock = threading.Lock()

        try:
            yield plugin, mock_irc, mock_msg
        finally:
            chars_value.setValue(original_chars)

    def test_doprivmsg_routes_private_messages_to_assistant(self, plugin_with_mocks: tuple) -> None:
        """GIVEN private message WHEN doPrivmsg called THEN routed to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("botname", "remove the memories about RMS")
        mock_msg.channel = None  # Private message — no channel context

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "remove the memories about RMS"
        )
        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_old_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN ZNC playback message WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.time = time.time() - 100  # Old message

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_drops_server_prefixed_messages(self, plugin_with_mocks: tuple) -> None:
        """GIVEN server-prefixed PRIVMSG WHEN doPrivmsg called THEN dropped, not routed.

        Downstream code calls ircutils.nickFromHostmask which asserts
        user-hostmask form. Without this gate, services-originated PMs
        would crash _run_preflight (seen as AssertionError on prod).
        """
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.prefix = "luna.AfterNET.Org"  # bare server prefix
        mock_msg.args = ("botname", "some text")
        mock_msg.channel = None

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_when_tracking_disabled(self, plugin_with_mocks: tuple) -> None:
        """GIVEN tracking disabled WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        # Every key truthy, privacyOptInOnly included.
        def registry_side_effect(key, *args):
            return True

        plugin.registryValue.side_effect = registry_side_effect

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_bot_own_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN message from bot itself WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.prefix = "botname!user@host"  # Same as bot nick

        mocker.patch("supybot.ircutils.strEqual", return_value=True)
        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_skips_ctcp_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN CTCP message WHEN doPrivmsg called THEN does not track."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=True)
        mocker.patch("supybot.ircmsgs.isAction", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_tracks_action_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN ACTION message WHEN doPrivmsg called THEN tracks message."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=True)
        mocker.patch("supybot.ircmsgs.isAction", return_value=True)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin.context.add_message.assert_called_once()

    def test_doprivmsg_tracks_normal_messages(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN normal message WHEN doPrivmsg called THEN tracks message with channel config."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("supybot.ircmsgs.isCtcp", return_value=False)
        mocker.patch("supybot.ircutils.strEqual", return_value=False)
        plugin.doPrivmsg(mock_irc, mock_msg)

        # add_message called with channel-specific config kwarg
        call_args = plugin.context.add_message.call_args
        assert call_args[0] == ("!usernick", "#channel", "user", "hello world")
        assert "config" in call_args[1]
        plugin._route_addressed_to_assistant.assert_not_called()

    def test_doprivmsg_routes_nick_addressed_channel_message(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick-addressed channel message WHEN doPrivmsg THEN routes to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname: remove the memories about RMS")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "remove the memories about RMS"
        )
        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_routes_nick_addressed_with_comma_separator(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick-addressed with comma WHEN doPrivmsg THEN routes to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname, draw a cat")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "draw a cat"
        )

    def test_doprivmsg_routes_nick_addressed_with_whitespace_separator(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick-addressed with whitespace WHEN doPrivmsg THEN routes to assistant."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname what time is it?")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "what time is it?"
        )

    def test_doprivmsg_does_not_route_when_nick_prefix_is_part_of_word(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN nick is prefix of a longer word WHEN doPrivmsg THEN treats as plain chatter."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botnamesomething")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        # Falls through to channel chatter tracking
        plugin.context.add_message.assert_called_once()

    def test_doprivmsg_does_not_route_explicit_command_prefix(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN @-prefixed command WHEN doPrivmsg THEN skips (Limnoria handles)."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "@search foo")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        plugin.context.add_message.assert_not_called()

    def test_doprivmsg_does_not_route_bare_nick_mention(self, plugin_with_mocks: tuple) -> None:
        """GIVEN message with nick alone WHEN doPrivmsg THEN treats as chatter."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "botname")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()

    def test_doprivmsg_routes_reply_to_bot_line(self, plugin_with_mocks: tuple) -> None:
        """GIVEN +draft/reply to a msgid the bot sent WHEN doPrivmsg THEN routes, no nick needed."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        plugin._own_msgids[(mock_irc.network, "bot-1")] = None
        mock_msg.args = ("#channel", "and what about tomorrow?")
        mock_msg.server_tags = {"+draft/reply": "bot-1"}

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "and what about tomorrow?"
        )
        plugin.context.add_message.assert_not_called()

    @pytest.mark.parametrize("signal", ["host", "flag"])
    def test_doprivmsg_reply_from_a_bot_is_chatter(
        self, plugin_with_mocks: tuple, signal: str
    ) -> None:
        """GIVEN a bot's +draft/reply to the bot's line WHEN doPrivmsg THEN plain chatter.

        Ender answers every link with a reply carrying the page title; routing
        that reply made the two bots talk to each other.
        """
        plugin, mock_irc, mock_msg = plugin_with_mocks
        plugin._own_msgids[(mock_irc.network, "bot-1")] = None
        mock_msg.args = ("#channel", "Title: Example Domain (at example.com)")
        mock_msg.server_tags = {"+draft/reply": "bot-1"}
        if signal == "host":
            mock_msg.prefix = "Ender!ender@ender.Bot.AfterNET.Org"
            plugin.registryValue.side_effect = lambda name, *_a, **_k: (
                ["Bot.AfterNET.Org"]
                if name == "botLoopHostSuffixes"
                else name != "privacyOptInOnly"
            )
        else:
            mock_msg.prefix = "Ender!ender@ender.Users.AfterNET.Org"
            plugin._bot_flags[plugin._bot_flag_key(mock_irc, mock_msg.nick)] = True

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        plugin.context.add_message.assert_called_once()

    def test_doprivmsg_reply_to_someone_else_is_chatter(self, plugin_with_mocks: tuple) -> None:
        """GIVEN +draft/reply to a msgid the bot never sent WHEN doPrivmsg THEN plain chatter."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        plugin._own_msgids[(mock_irc.network, "bot-1")] = None
        mock_msg.args = ("#channel", "lol same")
        mock_msg.server_tags = {"+draft/reply": "human-7"}

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        plugin.context.add_message.assert_called_once()


class TestOwnMsgidTracking:
    """inFilter learns the bot's own msgids from echo-message."""

    @pytest.fixture
    def plugin_and_irc(self, mocker: MockerFixture):
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin._own_msgids = collections.OrderedDict()
        plugin._own_msgids_lock = threading.Lock()
        plugin._last_msgids = collections.OrderedDict()
        plugin._last_msgids_lock = threading.Lock()
        plugin._last_msgids_dirty = {}
        irc = mocker.MagicMock()
        irc.nick = "botname"
        irc.network = "afternet"
        return plugin, irc

    def _msg(self, command: str, args: tuple, *, sender: str, msgid: str) -> object:
        from supybot import ircmsgs

        return ircmsgs.IrcMsg(
            prefix=f"{sender}!u@h",
            command=command,
            args=args,
            server_tags={"msgid": msgid},
        )

    def _reply(self, msgid: str) -> object:
        from supybot import ircmsgs

        return ircmsgs.IrcMsg(
            prefix="user!u@h",
            command="PRIVMSG",
            args=("#chan", "why?"),
            server_tags={"+draft/reply": msgid},
        )

    def test_echoed_privmsg_and_batch_are_remembered(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        plugin.inFilter(irc, self._msg("PRIVMSG", ("#chan", "hi"), sender="botname", msgid="a"))
        plugin.inFilter(
            irc,
            self._msg("BATCH", ("+r1", "draft/multiline", "#chan"), sender="botname", msgid="b"),
        )

        assert plugin._is_reply_to_own_line(irc, self._reply("a"))
        assert plugin._is_reply_to_own_line(irc, self._reply("b"))

    def test_other_speakers_msgids_are_ignored(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        plugin.inFilter(irc, self._msg("PRIVMSG", ("#chan", "hi"), sender="user", msgid="c"))

        assert not plugin._is_reply_to_own_line(irc, self._reply("c"))

    def test_memory_is_bounded(self, plugin_and_irc: tuple, mocker: MockerFixture) -> None:
        plugin, irc = plugin_and_irc
        mocker.patch("llm.plugin._OWN_MSGID_CAP", 2)
        for mid in ("x", "y", "z"):
            plugin.inFilter(irc, self._msg("PRIVMSG", ("#chan", "hi"), sender="botname", msgid=mid))

        assert not plugin._is_reply_to_own_line(irc, self._reply("x"))
        assert plugin._is_reply_to_own_line(irc, self._reply("z"))


class TestInFilterDispatchGate:
    """inFilter must suppress Limnoria's command dispatcher for non-prefix
    addressed messages by tagging msg.addressed=''."""

    @pytest.fixture
    def plugin_and_irc(self, mocker: MockerFixture):
        import supybot.conf as supy_conf
        from llm.plugin import LLM

        chars_value = supy_conf.supybot.reply.whenAddressedBy.chars
        original_chars = chars_value()
        chars_value.setValue("@")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        irc = mocker.MagicMock()
        irc.nick = "botname"
        try:
            yield plugin, irc
        finally:
            chars_value.setValue(original_chars)

    def _msg(self, target: str, text: str, *, sender: str = "user") -> object:
        from supybot import ircmsgs

        return ircmsgs.IrcMsg(prefix=f"{sender}!u@h", command="PRIVMSG", args=(target, text))

    def test_nick_addressed_channel_msg_is_marked_unaddressed(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("#chan", "botname: remove that thing")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") == ""

    def test_unprefixed_pm_is_marked_unaddressed(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("botname", "remove that thing")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") == ""

    def test_at_prefixed_command_is_not_marked(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("#chan", "@search foo")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") is None

    def test_at_prefixed_pm_is_not_marked(self, plugin_and_irc: tuple) -> None:
        plugin, irc = plugin_and_irc
        msg = self._msg("botname", "@later add foo bar")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") is None

    def test_plain_channel_chatter_is_marked(self, plugin_and_irc: tuple) -> None:
        # Even for non-addressed channel chatter, tagging with '' is a no-op
        # for dispatch (it would already be unaddressed). The tag itself is
        # harmless because doPrivmsg routes only when text actually starts
        # with our nick.
        plugin, irc = plugin_and_irc
        msg = self._msg("#chan", "just chatting")

        result = plugin.inFilter(irc, msg)

        assert result.tagged("addressed") == ""


class TestInFilter:
    """Test inFilter sanitisation of control characters and unbalanced brackets."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture) -> object:
        """Create a bare LLM instance for inFilter tests."""
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        return LLM.__new__(LLM)

    @pytest.fixture
    def irc(self, mocker: MockerFixture) -> object:
        """Mock irc with a nick so the dispatch-gate path can run."""
        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        return mock_irc

    @staticmethod
    def _privmsg(text: str, channel: str = "#test") -> object:
        """Build a minimal PRIVMSG, bypassing Limnoria's argument validation.

        Uses the raw-string constructor so we can inject control
        characters that the keyword constructor would reject.
        """
        import supybot.ircmsgs as ircmsgs

        return ircmsgs.IrcMsg(s=f":n!u@h PRIVMSG {channel} :{text}\r\n")

    def test_normal_text_passes_through(self, plugin: object, irc: object) -> None:
        """GIVEN plain text WHEN inFilter THEN message unchanged."""
        msg = self._privmsg("hello world")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "hello world"

    def test_strips_esc_byte(self, plugin: object, irc: object) -> None:
        """GIVEN text with ESC byte WHEN inFilter THEN ESC removed."""
        msg = self._privmsg("before\x1bafter")
        result = plugin.inFilter(irc, msg)
        assert "\x1b" not in result.args[1]
        assert result.args[1] == "beforeafter"

    def test_ansi_escape_sequence_with_bracket(self, plugin: object, irc: object) -> None:
        """GIVEN ANSI escape \\x1b[6n WHEN inFilter THEN does not crash tokenizer."""
        from supybot import callbacks

        msg = self._privmsg("\x1b[6n cursor position check")
        result = plugin.inFilter(irc, msg)
        # Should not raise SyntaxError
        callbacks.tokenize(result.args[1])

    def test_unbalanced_open_bracket_escaped(self, plugin: object, irc: object) -> None:
        """GIVEN unmatched [ WHEN inFilter THEN brackets replaced with full-width."""
        msg = self._privmsg("explain array[0")
        result = plugin.inFilter(irc, msg)
        assert "[" not in result.args[1]
        assert "\uff3b" in result.args[1]

    def test_balanced_brackets_preserved(self, plugin: object, irc: object) -> None:
        """GIVEN matched brackets WHEN inFilter THEN original brackets kept."""
        msg = self._privmsg("run [echo hello]")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "run [echo hello]"

    def test_non_privmsg_passes_through(self, plugin: object, irc: object) -> None:
        """GIVEN non-PRIVMSG WHEN inFilter THEN returned unchanged."""
        import supybot.ircmsgs as ircmsgs

        msg = ircmsgs.join("#test")
        result = plugin.inFilter(irc, msg)
        assert result is msg

    def test_strips_null_bytes(self, plugin: object, irc: object) -> None:
        """GIVEN text with null bytes WHEN inFilter THEN nulls removed."""
        msg = self._privmsg("hello\x00world")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "helloworld"

    def test_preserves_tabs(self, plugin: object, irc: object) -> None:
        """GIVEN text with tab WHEN inFilter THEN preserved."""
        msg = self._privmsg("col1\tcol2")
        result = plugin.inFilter(irc, msg)
        assert result.args[1] == "col1\tcol2"

    def test_original_crash_message(self, plugin: object, irc: object) -> None:
        r"""GIVEN the exact message that caused the crash WHEN inFilter THEN tokenizable."""
        from supybot import callbacks

        text = (
            r"do this but don't fuck it up suggests sending \x1b[6n"
            " to see if the terminal force-injects its cursor position"
            " into his input buffer."
        )
        msg = self._privmsg(text)
        result = plugin.inFilter(irc, msg)
        # Must not raise SyntaxError
        callbacks.tokenize(result.args[1])


class TestInvalidCommand:
    """Test invalidCommand fallback to ask."""

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture) -> tuple:
        """Create plugin with mocked dependencies for invalidCommand tests."""
        import threading

        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.args = ("#channel", "hello there")
        mock_msg.time = time.time() + 100  # Future time (not ZNC playback)
        mock_msg.channel = "#channel"

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        plugin.ask = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin.context = mocker.MagicMock()
        # Every key truthy except privacyOptInOnly, so chatter is tracked.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: key != "privacyOptInOnly"
        )
        # Limnoria's MetaSynchronized requires this lock for synchronized methods
        plugin._MetaSynchronized_rlock = threading.RLock()
        # _dispatch_addressed_async's worker checks _llm_executor.closing
        # before doing any work; closing=False so the dispatch proceeds.
        plugin._llm_executor = mocker.MagicMock(closing=False)
        # Stub out verse routing — TestInvalidCommand asserts only that the
        # chat path delegates to _ask_impl. Verse routing is covered by
        # TestVerseRouting / TestAskCommand fixtures.
        plugin._verse_route_for = mocker.MagicMock(return_value=None)
        # Likewise stub the chat-path canon retrieval + record handler + sticky
        # roleplay check (each covered by its own tests; __init__ is a no-op here
        # so the sticky lock/dict don't exist).
        plugin._verse_context_for = mocker.MagicMock(return_value=None)
        plugin._verse_chat_record_handler = mocker.MagicMock(return_value=None)
        plugin._roleplay_sticky_active = mocker.MagicMock(return_value=False)
        plugin._ambient_inline_story = mocker.MagicMock(return_value=False)
        plugin._ambient_storybook_brief = mocker.MagicMock(return_value=None)
        # Dedup guard reads an in-memory dict/lock that the no-op __init__ skips.
        plugin._is_duplicate_dispatch = mocker.MagicMock(return_value=False)

        return plugin, mock_irc, mock_msg

    def test_invalid_command_empty_tokens_returns_early(self, plugin_with_mocks: tuple) -> None:
        """GIVEN empty tokens WHEN invalidCommand called THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        plugin.invalidCommand(mock_irc, mock_msg, [])

        plugin.ask.assert_not_called()

    def test_invalid_command_no_capability_returns_early(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN user without llm.ask capability WHEN invalidCommand THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_not_called()

    def test_invalid_command_old_message_returns_early(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN ZNC playback message WHEN invalidCommand THEN returns early."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.time = time.time() - 100  # Old message

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin.ask.assert_not_called()

    def test_invalid_command_delegates_to_ask(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN valid tokens WHEN invalidCommand THEN delegates to _ask_impl."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._run_preflight = mocker.MagicMock(
            return_value=PreflightResult(False, Caller(Identity("testuser", None), "#channel"))
        )
        plugin._ask_impl = mocker.MagicMock()
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin._ask_impl.assert_called_once()
        plugin._run_preflight.assert_called_once()

    def test_invalid_command_routes_through_verse_dispatch(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """When a verse-enabled channel has an avatar for this user, an
        unprefixed `vibebot, …` message must reach _ask_impl with
        profile_override=PROFILE_VERSE — otherwise the chat profile fires
        and verse_record never gets a chance to run.

        Regression: invalidCommand previously called _ask_impl directly,
        bypassing _verse_route_for. The whole verse subsystem only kicked in
        for the explicit @ask command, so unprefixed messages in a verse
        channel produced narration without canon. Fixed by routing every
        addressed-text entry point through _dispatch_with_verse_routing."""
        from llm.plugin import VerseRoute
        from llm.profile import PROFILE_VERSE

        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._run_preflight = mocker.MagicMock(
            return_value=PreflightResult(False, Caller(Identity("rdrake", None), "#afternet"))
        )
        fake_route = VerseRoute(
            avatar_id=1,
            system_prompt="verse system prompt",
            tools=[{"type": "function"}],
            store=mocker.MagicMock(),
        )
        plugin._verse_route_for = mocker.MagicMock(return_value=fake_route)
        plugin._ask_impl = mocker.MagicMock()

        plugin.invalidCommand(mock_irc, mock_msg, ["diarrhoea", "dan", "did", "X"])

        plugin._verse_route_for.assert_called_once()
        plugin._ask_impl.assert_called_once()
        kwargs = plugin._ask_impl.call_args.kwargs
        assert kwargs["profile_override"] == PROFILE_VERSE
        assert kwargs["verse_route"] is fake_route
        assert kwargs["system_prompt_override"] == "verse system prompt"

    def test_dispatch_strips_ooc_wrapper_before_chat_path(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN an OOC-wrapped message on a verse channel that falls to the
        chat path WHEN dispatched THEN _ask_impl receives the unwrapped text,
        not the literal ((parentheses))."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        # verseEnabled=True, but _verse_route_for returns None (OOC bypass).
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else 8
        )
        plugin._verse_route_for = mocker.MagicMock(return_value=None)
        plugin._ask_impl = mocker.MagicMock()
        preflight = PreflightResult(False, Caller(Identity("forest", None), "#afternet"))

        plugin._dispatch_with_verse_routing(
            mock_irc,
            mock_msg,
            "((what model are you running?))",
            preflight,
            entry_route="addressed",
        )

        plugin._ask_impl.assert_called_once()
        # Positional args: (irc, msg, text, preflight).
        assert plugin._ask_impl.call_args.args[2] == "what model are you running?"

    def test_dispatch_keeps_ooc_parens_when_verse_disabled(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN a ((wrapped)) message on a non-verse channel WHEN dispatched
        THEN the parentheses are left intact — ((...)) only means OOC on a
        verse-enabled channel, elsewhere it is ordinary text."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: False if key == "verseEnabled" else 8
        )
        plugin._verse_route_for = mocker.MagicMock(return_value=None)
        plugin._ask_impl = mocker.MagicMock()
        preflight = PreflightResult(False, Caller(Identity("forest", None), "#linux"))

        plugin._dispatch_with_verse_routing(
            mock_irc,
            mock_msg,
            "((array[0]))",
            preflight,
            entry_route="addressed",
        )

        plugin._ask_impl.assert_called_once()
        assert plugin._ask_impl.call_args.args[2] == "((array[0]))"

    def test_dispatch_strips_slash_marker_before_chat_path(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN a leading-// OOC message on a verse channel that falls to the
        chat path WHEN dispatched THEN _ask_impl receives the text with the //
        marker removed, not the literal '// ...' prefix."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        # verseEnabled=True, but _verse_route_for returns None (OOC bypass).
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else 8
        )
        plugin._verse_route_for = mocker.MagicMock(return_value=None)
        plugin._ask_impl = mocker.MagicMock()
        preflight = PreflightResult(False, Caller(Identity("forest", None), "#afternet"))

        plugin._dispatch_with_verse_routing(
            mock_irc,
            mock_msg,
            "// what model are you running?",
            preflight,
            entry_route="addressed",
        )

        plugin._ask_impl.assert_called_once()
        # Positional args: (irc, msg, text, preflight).
        assert plugin._ask_impl.call_args.args[2] == "what model are you running?"

    def test_ooc_message_does_not_fire_ambient_storybook(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN an OOC-marked illustrate request on a verse channel WHEN
        dispatched THEN no storybook is submitted — the // marker means "skip
        the verse engine", so it must suppress the image path too, not just
        roleplay.

        Regression: the caller stripped the marker and then handed the cleaned
        text to _ambient_storybook_brief, so `// illustrate the lads` spent a
        whole illustrated storybook despite the opt-out."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else 8
        )
        plugin._verse_route_for = mocker.MagicMock(return_value=None)
        plugin._ask_impl = mocker.MagicMock()
        plugin._ambient_storybook_brief = mocker.MagicMock(return_value="brief")
        plugin._submit_storybook_job = mocker.MagicMock()
        preflight = PreflightResult(False, Caller(Identity("forest", "forest"), "#afternet"))

        plugin._dispatch_with_verse_routing(
            mock_irc,
            mock_msg,
            "// illustrate the stinky lads",
            preflight,
            entry_route="addressed",
        )

        plugin._submit_storybook_job.assert_not_called()
        plugin._ambient_storybook_brief.assert_not_called()
        # Still answered, with the marker stripped.
        plugin._ask_impl.assert_called_once()
        assert plugin._ask_impl.call_args.args[2] == "illustrate the stinky lads"

    def test_ooc_message_does_not_fire_ambient_inline_story(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN an OOC-marked canon mention WHEN dispatched THEN the ambient
        inline-story promotion is skipped entirely — the opt-out short-circuits
        before the verse store lookup, not after it."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else 8
        )
        plugin._verse_route_for = mocker.MagicMock(return_value=None)
        plugin._ask_impl = mocker.MagicMock()
        plugin._ambient_inline_story = mocker.MagicMock(return_value=True)
        preflight = PreflightResult(False, Caller(Identity("forest", "forest"), "#afternet"))

        plugin._dispatch_with_verse_routing(
            mock_irc,
            mock_msg,
            "((what have the stinky lads been up to))",
            preflight,
            entry_route="addressed",
        )

        plugin._ambient_inline_story.assert_not_called()
        assert plugin._verse_route_for.call_args.kwargs["force_roleplay"] is False

    def test_invalid_command_does_not_call_meta(
        self, plugin_with_mocks: tuple, mocker: MockerFixture
    ) -> None:
        """GIVEN valid tokens WHEN invalidCommand THEN does not call _run_meta."""
        plugin, mock_irc, mock_msg = plugin_with_mocks

        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
        plugin._run_preflight = mocker.MagicMock(
            return_value=PreflightResult(False, Caller(Identity("testuser", None), "#channel"))
        )
        plugin._run_meta = mocker.MagicMock()
        plugin._ask_impl = mocker.MagicMock()
        plugin.invalidCommand(mock_irc, mock_msg, ["hello", "there"])

        plugin._run_meta.assert_not_called()


class TestCommandRegistry:
    """Tests for the command metadata registry."""

    def test_registry_contains_all_commands(self) -> None:
        """GIVEN command registry WHEN checked THEN contains all user-facing commands."""
        from llm.plugin import COMMAND_REGISTRY

        names = {cmd.name for cmd in COMMAND_REGISTRY}
        expected = {
            "ask",
            "code",
            "draw",
            "meme",
            "animate",
            "renders",
            "story",
            "forget",
            "memories",
            "instruct",
            "avatar",
            "remind",
            "tz",
            "usage",
            "verseopt",
            "verse",
            "rp",
            "look",
            "who",
            "canon",
            "versedump",
            "versepurge",
            "versecompact",
            "versedit",
            "channels",
            "names",
            "model",
        }
        assert names == expected

    def test_registry_entries_have_required_fields(self) -> None:
        """GIVEN command registry WHEN checked THEN all entries have name, args, description."""
        from llm.plugin import COMMAND_REGISTRY

        for cmd in COMMAND_REGISTRY:
            assert cmd.name, "name is required"
            assert cmd.description, "description is required"
            assert cmd.category in ("generation", "memory", "utility")

    def test_registry_entries_have_examples(self) -> None:
        """GIVEN command registry WHEN checked THEN all entries have at least one example."""
        from llm.plugin import COMMAND_REGISTRY

        for cmd in COMMAND_REGISTRY:
            assert cmd.examples, f"{cmd.name} needs at least one example"


class TestGetPluginHelp:
    """Tests for getPluginHelp() generation from COMMAND_REGISTRY."""

    def test_get_plugin_help_lists_all_commands(self, mocker: MockerFixture) -> None:
        """GIVEN plugin WHEN getPluginHelp called THEN lists all registered commands."""
        from llm.plugin import COMMAND_REGISTRY, LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.registryValue = mocker.MagicMock(return_value="https://example.com/help")

        help_text = plugin.getPluginHelp()
        for cmd in COMMAND_REGISTRY:
            assert cmd.name in help_text, f"{cmd.name} missing from help"


class TestCommandRegistryCompleteness:
    """Drift-prevention: ensures registry stays in sync with actual commands."""

    def test_all_wrapped_commands_in_registry(self) -> None:
        """GIVEN plugin class WHEN checking command methods THEN all are in registry.

        This test prevents adding a new command to plugin.py without updating
        the command registry. It uses the same introspection as Limnoria's
        isCommandMethod() to find all commands.
        """
        from llm.plugin import COMMAND_REGISTRY, LLM
        from supybot.callbacks import canonicalName

        registry_names = {cmd.name for cmd in COMMAND_REGISTRY}
        registry_names |= {alias for cmd in COMMAND_REGISTRY for alias in cmd.aliases}
        command_args = ["self", "irc", "msg", "args"]

        for name in dir(LLM):
            if name.startswith("_"):
                continue
            if name != canonicalName(name):
                continue  # filters getPluginHelp, invalidCommand, inFilter, etc.
            obj = getattr(LLM, name, None)
            if not inspect.isfunction(obj):
                continue
            if inspect.getargs(obj.__code__)[0] == command_args:
                assert name in registry_names, (
                    f"Command '{name}' is registered with Limnoria but missing from "
                    f"COMMAND_REGISTRY. Add it to keep help in sync."
                )


class TestPatchedDoJoin:
    """The plugin patches supybot.irclib.Irc.doJoin to skip slow auto-queries."""

    def _self_join(self, mocker: MockerFixture, channel="#test", nick="testbot"):
        msg = mocker.MagicMock()
        msg.nick = nick
        msg.args = (channel,)
        return msg

    def test_mode_b_never_queued(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        for call in mock_irc.queueMsg.call_args_list:
            sent = call.args[0]
            if getattr(sent, "command", "") == "MODE" and "+b" in getattr(sent, "args", ()):
                pytest.fail(f"MODE +b should never be queued: {sent}")

    def test_who_skipped_when_both_caps_and_flag_enabled(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" not in commands

    def test_who_kept_when_account_tag_missing(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_who_kept_when_extended_join_missing(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_who_kept_when_flag_disabled(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        # Override the registry default for this test.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: False if key == "skipAutoWhoOnJoin" else ""
        )
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        commands = [c.args[0].command for c in mock_irc.queueMsg.call_args_list]
        assert "WHO" in commands

    def test_channel_mode_always_queued(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        mock_irc.queueMsg = mocker.MagicMock()
        msg = self._self_join(mocker)

        from supybot.irclib import Irc

        Irc.doJoin(mock_irc, msg)

        mode_calls = [
            c.args[0]
            for c in mock_irc.queueMsg.call_args_list
            if getattr(c.args[0], "command", "") == "MODE"
        ]
        # Plain MODE <channel> has args=(channel,) — length 1.
        assert any(len(getattr(m, "args", ())) == 1 for m in mode_calls)


class TestPluginDoJoinPendingChannels:
    """Plugin's own doJoin must not add to _pending_channels when WHO is skipped."""

    def test_pending_added_when_who_will_fire(self, plugin_env, mocker: MockerFixture):
        plugin, mock_irc, _ = plugin_env
        mock_irc.nick = "testbot"
        mock_irc.state.capabilities_ack = set()  # no caps → WHO fires
        plugin._pending_channels.clear()
        msg = mocker.MagicMock()
        msg.nick = "testbot"
        msg.args = ("#test",)

        plugin.doJoin(mock_irc, msg)

        assert "#test" in plugin._pending_channels

    def test_pending_NOT_added_when_who_will_be_skipped(  # noqa: N802
        self, plugin_env, mocker: MockerFixture
    ):
        plugin, mock_irc, _ = plugin_env
        mock_irc.nick = "testbot"
        mock_irc.state.capabilities_ack = {"account-tag", "extended-join"}
        plugin._pending_channels.clear()
        msg = mocker.MagicMock()
        msg.nick = "testbot"
        msg.args = ("#test",)

        plugin.doJoin(mock_irc, msg)

        assert "#test" not in plugin._pending_channels, (
            "When WHO is skipped, do315 won't fire — the bot must not add to "
            "_pending_channels or startup notification will never send."
        )


class TestWrapCapabilityGate:
    """Deny-path coverage for the wrap(checkCapability) gate on ask/code/draw.

    The conftest ``plugin_env`` fixture grants every ``llm.*`` capability, so
    the happy-path command tests only ever exercise the *allow* side of the
    wrap dispatcher. These tests revoke one capability and assert the command
    body never runs — guarding the ``("checkCapability", "llm.<cmd>")`` entry
    on each wrapped command, which a mutation dropping it would otherwise leave
    unprotected (an auth bypass).
    """

    @pytest.mark.parametrize(
        ("command", "capability", "args"),
        [
            ("ask", "llm.ask", ["hello"]),
            ("code", "llm.code", ["Python", "hello"]),
            ("draw", "llm.draw", ["a", "sunset"]),
        ],
    )
    def test_wrapped_command_denied_without_capability(
        self, plugin_env, mocker: MockerFixture, command: str, capability: str, args: list[str]
    ):
        """GIVEN a user lacking the command capability WHEN the command is
        invoked through the wrap dispatcher THEN the body never runs (no LLM
        dispatch) and the capability error is surfaced."""
        plugin, mock_irc, mock_msg = plugin_env
        # A valid account is irrelevant: the wrap capability gate fires first.
        mock_irc.state.nickToAccount.return_value = "test_account"

        # Revoke only this command's capability; keep the rest granted so the
        # gate under test is the sole reason for rejection.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap, denied=capability: (
                cap != denied and cap.startswith("llm.")
            ),
        )

        getattr(plugin, command)(mock_irc, mock_msg, args)

        # Rejected: the LLM dispatch path never ran.
        plugin.llm_service.assistant_request.assert_not_called()
        # The wrap checkCapability converter surfaced the denial.
        assert mock_irc.errorNoCapability.called


class TestNickAtEndAddressing:
    """A trailing nick mention addresses the bot, including inside a CTCP ACTION.

    Regression: ``* Hemingray prods vibebot`` produced no reply — the nick
    matcher was prefix-only, and inFilter's control-character scrub ate the
    CTCP ``\\x01`` delimiters so the payload reached context as the literal
    string ``ACTION prods vibebot``.
    """

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture):
        import supybot.conf as supy_conf
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.nick = "usernick"
        mock_msg.command = "PRIVMSG"
        mock_msg.args = ("#channel", "hello world")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#channel"
        mock_msg.server_tags = {}

        chars_value = supy_conf.supybot.reply.whenAddressedBy.chars
        original_chars = chars_value()
        chars_value.setValue("@")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        # Every key truthy except privacyOptInOnly, so chatter is tracked.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: key != "privacyOptInOnly"
        )
        plugin.context = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._route_addressed_to_assistant = mocker.MagicMock()
        plugin._loom = None
        plugin._loom_bridge = None
        plugin._loom_channel_cache = None
        plugin._loom_network_cache = None
        plugin._loom_bot_nicks_cache = ()
        # Bot-loop guard state (normally built in __init__, which these
        # bare-instance fixtures skip). Empty flags = every speaker reads
        # as a person, so the guard never fires in dispatch tests.
        plugin._bot_flags = {}
        plugin._bot_probed_at = {}
        plugin._bot_reply_counts = {}
        plugin._bot_loop_lock = threading.Lock()

        try:
            yield plugin, mock_irc, mock_msg
        finally:
            chars_value.setValue(original_chars)

    def test_trailing_nick_routes_to_assistant(self, plugin_with_mocks: tuple) -> None:
        """GIVEN 'draw a cat, botname' WHEN doPrivmsg THEN routes without the nick."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "draw a cat, botname")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "draw a cat"
        )
        plugin.context.add_message.assert_not_called()

    def test_trailing_nick_with_sentence_punctuation(self, plugin_with_mocks: tuple) -> None:
        """GIVEN 'what time is it, botname?' WHEN doPrivmsg THEN routes."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "what time is it, botname?")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "what time is it"
        )

    def test_action_with_trailing_nick_routes_as_action(self, plugin_with_mocks: tuple) -> None:
        """GIVEN '/me prods botname' WHEN doPrivmsg THEN routes '* usernick prods'."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "\x01ACTION prods botname\x01")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "* usernick prods"
        )
        plugin.context.add_message.assert_not_called()

    def test_unaddressed_action_is_tracked_as_readable_action(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN '/me waves' WHEN doPrivmsg THEN context stores '* usernick waves'."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "\x01ACTION waves at everyone\x01")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        assert plugin.context.add_message.call_args[0][3] == "* usernick waves at everyone"
        assert plugin.context.add_channel_message.call_args[0][3] == "* usernick waves at everyone"

    def test_action_starting_with_command_char_is_not_a_command(
        self, plugin_with_mocks: tuple
    ) -> None:
        """GIVEN '/me @search things, botname' WHEN doPrivmsg THEN still routes to chat."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "\x01ACTION @search things, botname\x01")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "* usernick @search things"
        )

    def test_third_person_mention_mid_sentence_stays_silent(self, plugin_with_mocks: tuple) -> None:
        """GIVEN 'botnamesomething is weird' WHEN doPrivmsg THEN plain chatter."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "I think botnamesomething is weird")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        plugin.context.add_message.assert_called_once()


class TestInFilterPreservesActions:
    """inFilter must scrub an ACTION's interior without eating its \\x01 frame."""

    @pytest.fixture
    def plugin(self, mocker: MockerFixture) -> object:
        from llm.plugin import LLM

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        return LLM.__new__(LLM)

    @pytest.fixture
    def irc(self, mocker: MockerFixture) -> object:
        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        return mock_irc

    @staticmethod
    def _privmsg(text: str, channel: str = "#test") -> object:
        import supybot.ircmsgs as ircmsgs

        return ircmsgs.IrcMsg(s=f":n!u@h PRIVMSG {channel} :{text}\r\n")

    def test_clean_action_survives_untouched(self, plugin, irc) -> None:
        msg = self._privmsg("\x01ACTION prods botname\x01")

        result = plugin.inFilter(irc, msg)

        assert result.args[1] == "\x01ACTION prods botname\x01"

    def test_action_interior_is_scrubbed_but_stays_an_action(self, plugin, irc) -> None:
        import supybot.ircmsgs as ircmsgs

        msg = self._privmsg("\x01ACTION prods \x1b[6nbotname\x01")

        result = plugin.inFilter(irc, msg)

        assert result.args[1] == "\x01ACTION prods ［6nbotname\x01"
        assert ircmsgs.isAction(result)


class TestNickInMiddleOfAction:
    """A nick mid-way through a ``/me`` addresses the bot too.

    NickInMiddle skips CTCP by design — its front-shifting rewrite would turn
    "asks vibebot for a beer" into "vibebot asks for a beer" and invert who is
    asking. The action path cuts the nick in place instead.
    """

    @pytest.fixture
    def plugin_with_mocks(self, mocker: MockerFixture):
        import supybot.conf as supy_conf
        from llm.plugin import LLM

        mock_irc = mocker.MagicMock()
        mock_irc.nick = "botname"
        mock_irc.state.nickToAccount = mocker.MagicMock(return_value=None)

        mock_msg = mocker.MagicMock()
        mock_msg.prefix = "usernick!user@host"
        mock_msg.nick = "usernick"
        mock_msg.command = "PRIVMSG"
        mock_msg.args = ("#channel", "hello world")
        mock_msg.time = time.time() + 100
        mock_msg.channel = "#channel"
        mock_msg.server_tags = {}

        chars_value = supy_conf.supybot.reply.whenAddressedBy.chars
        original_chars = chars_value()
        chars_value.setValue("@")

        mocker.patch.object(LLM, "__init__", lambda self, irc: None)
        plugin = LLM.__new__(LLM)
        plugin.startup_time = time.time()
        # Every key truthy except privacyOptInOnly, so chatter is tracked.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: key != "privacyOptInOnly"
        )
        plugin.context = mocker.MagicMock()
        plugin.llm_service = mocker.MagicMock()
        plugin.db = mocker.MagicMock()
        plugin._route_addressed_to_assistant = mocker.MagicMock()
        plugin._loom = None
        plugin._loom_bridge = None
        plugin._loom_channel_cache = None
        plugin._loom_network_cache = None
        plugin._loom_bot_nicks_cache = ()
        # Bot-loop guard state (normally built in __init__, which these
        # bare-instance fixtures skip). Empty flags = every speaker reads
        # as a person, so the guard never fires in dispatch tests.
        plugin._bot_flags = {}
        plugin._bot_probed_at = {}
        plugin._bot_reply_counts = {}
        plugin._bot_loop_lock = threading.Lock()

        try:
            yield plugin, mock_irc, mock_msg
        finally:
            chars_value.setValue(original_chars)

    def test_mid_action_nick_is_cut_in_place(self, plugin_with_mocks: tuple) -> None:
        """GIVEN '/me asks botname for a beer' THEN routes '* usernick asks for a beer'."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "\x01ACTION asks botname for a beer\x01")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "* usernick asks for a beer"
        )
        plugin.context.add_message.assert_not_called()

    def test_mid_action_nick_with_comma_separator(self, plugin_with_mocks: tuple) -> None:
        """GIVEN '/me pokes botname, gently' THEN routes '* usernick pokes gently'."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "\x01ACTION pokes botname, gently\x01")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_called_once_with(
            mock_irc, mock_msg, "* usernick pokes gently"
        )

    def test_mid_action_nick_needs_word_boundaries(self, plugin_with_mocks: tuple) -> None:
        """GIVEN '/me waves at botnamesomething today' THEN plain chatter."""
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "\x01ACTION waves at botnamesomething today\x01")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        assert (
            plugin.context.add_message.call_args[0][3]
            == "* usernick waves at botnamesomething today"
        )

    def test_plain_message_mid_nick_is_left_to_nickinmiddle(self, plugin_with_mocks: tuple) -> None:
        """GIVEN a non-action mid-sentence mention THEN this plugin stays out of it.

        NickInMiddle owns that rewrite, and it is per-channel switchable.
        Matching it here too would defeat that switch.
        """
        plugin, mock_irc, mock_msg = plugin_with_mocks
        mock_msg.args = ("#channel", "can you, botname, tell me the weather")

        plugin.doPrivmsg(mock_irc, mock_msg)

        plugin._route_addressed_to_assistant.assert_not_called()
        plugin.context.add_message.assert_called_once()
