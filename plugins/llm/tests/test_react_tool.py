"""The react tool: msgid tracking, emoji validation, sends, and wiring."""

from __future__ import annotations

import json

import pytest
from llm.identity import Caller, Identity
from llm.plugin import AssistantResult
from llm.service import Spend
from supybot import ircmsgs

from .conftest import make_registry_side_effect


def privmsg(channel: str, nick: str, text: str, msgid: str | None) -> ircmsgs.IrcMsg:
    m = ircmsgs.IrcMsg(command="PRIVMSG", args=(channel, text), prefix=f"{nick}!u@host")
    if msgid:
        m.server_tags["msgid"] = msgid
    return m


@pytest.fixture
def react_env(plugin_env):
    plugin, irc, msg = plugin_env
    irc.network = "afternet"
    msg.server_tags = {"msgid": "trigger-1"}
    plugin.registryValue.side_effect = make_registry_side_effect()
    plugin.llm_service.send_reaction.return_value = True
    return plugin, irc, msg


def _call(plugin, irc, msg, **args) -> dict:
    _, handlers = plugin._build_react_tool(irc, msg, Caller.from_msg(msg))
    return json.loads(handlers["react"](args).content)


class TestLastMsgidTracking:
    def test_infilter_records_latest_line_per_nick(self, react_env) -> None:
        plugin, irc, _ = react_env
        plugin.inFilter(irc, privmsg("#test", "Bob", "first", "m1"))
        plugin.inFilter(irc, privmsg("#test", "Bob", "second", "m2"))

        assert plugin._last_msgids[("afternet", "#test", "bob")] == "m2"

    def test_ignores_pms_and_untagged_lines(self, react_env) -> None:
        plugin, irc, _ = react_env
        plugin.inFilter(irc, privmsg("testbot", "bob", "hi", "m1"))
        plugin.inFilter(irc, privmsg("#test", "bob", "hi", None))

        assert not plugin._last_msgids

    @pytest.mark.parametrize(
        ("command", "args"),
        [("TAGMSG", ("#test",)), ("NOTICE", ("#test", "brb")), ("JOIN", ("#test",))],
    )
    def test_only_a_privmsg_is_their_latest_message(self, react_env, command, args) -> None:
        """GIVEN bob's PRIVMSG m1 WHEN a msgid-tagged TAGMSG (a reaction),
        NOTICE or JOIN of his follows THEN "bob's latest message" is still m1."""
        plugin, irc, _ = react_env
        plugin.inFilter(irc, privmsg("#test", "bob", "hot take", "m1"))
        other = ircmsgs.IrcMsg(command=command, args=args, prefix="bob!u@host")
        other.server_tags["msgid"] = "m2"
        plugin.inFilter(irc, other)

        assert plugin._last_msgids[("afternet", "#test", "bob")] == "m1"

    def test_lru_is_bounded(self, react_env, mocker) -> None:
        plugin, irc, _ = react_env
        mocker.patch("llm.plugin._LAST_MSGID_CAP", 2)
        for i in range(3):
            plugin.inFilter(irc, privmsg("#test", f"n{i}", "x", f"m{i}"))

        assert list(plugin._last_msgids.values()) == ["m1", "m2"]


class TestReactHandler:
    def test_no_nick_reacts_to_the_triggering_message(self, react_env) -> None:
        plugin, irc, msg = react_env

        payload = _call(plugin, irc, msg, emoji="👍")

        assert payload == {"status": "ok", "message": "reacted 👍 to testnick's message"}
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "#test", "trigger-1", "👍")

    def test_no_nick_on_a_reply_reacts_to_the_replied_to_message(self, react_env) -> None:
        # rdrake replied to the bot's line with "Praying hands emoji" and the
        # reaction belonged on that line, not on the request (prod 2026-10-10).
        plugin, irc, msg = react_env
        msg.server_tags["+draft/reply"] = "bot-line-7"

        payload = _call(plugin, irc, msg, emoji="🙏")

        assert payload == {
            "status": "ok",
            "message": "reacted 🙏 to the message testnick replied to",
        }
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "#test", "bot-line-7", "🙏")

    def test_nick_reacts_to_their_latest_line(self, react_env) -> None:
        plugin, irc, msg = react_env
        plugin.inFilter(irc, privmsg("#test", "bob", "hot take", "bob-9"))

        payload = _call(plugin, irc, msg, emoji="😂", nick="Bob")

        assert payload["status"] == "ok"
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "#test", "bob-9", "😂")

    def test_line_in_another_channel_does_not_count(self, react_env) -> None:
        plugin, irc, msg = react_env
        plugin.inFilter(irc, privmsg("#other", "bob", "hi", "bob-1"))

        payload = _call(plugin, irc, msg, emoji="😂", nick="bob")

        # A miss puts ❌ on the request and ends the turn like a reaction.
        assert payload["status"] == "ok"
        assert "no recent message from bob" in payload["message"]
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "#test", "trigger-1", "❌")

    def test_miss_with_no_way_to_react_is_an_error(self, react_env) -> None:
        plugin, irc, msg = react_env
        plugin.llm_service.send_reaction.return_value = False

        payload = _call(plugin, irc, msg, emoji="😂", nick="bob")

        assert payload == {"error": "no recent message from bob here"}

    @pytest.mark.parametrize("emoji", ["", ":thumbsup:", "thumbs up", "👍 👍", "👍" * 9, "x"])
    def test_rejects_non_emoji_without_sending(self, react_env, emoji) -> None:
        plugin, irc, msg = react_env

        payload = _call(plugin, irc, msg, emoji=emoji)

        assert "error" in payload
        plugin.llm_service.send_reaction.assert_not_called()

    def test_accepts_zwj_sequence(self, react_env) -> None:
        plugin, irc, msg = react_env

        assert _call(plugin, irc, msg, emoji="🏳️‍🌈")["status"] == "ok"

    def test_send_failure_is_an_error(self, react_env) -> None:
        plugin, irc, msg = react_env
        plugin.llm_service.send_reaction.return_value = False

        payload = _call(plugin, irc, msg, emoji="👍")

        assert "error" in payload

    def test_no_trigger_msgid_is_an_error(self, react_env) -> None:
        plugin, irc, msg = react_env
        msg.server_tags = {}

        assert "error" in _call(plugin, irc, msg, emoji="👍")
        plugin.llm_service.send_reaction.assert_not_called()

    def test_pm_reacts_to_the_sender(self, react_env) -> None:
        plugin, irc, msg = react_env
        msg.args = ("testbot", "react to this")

        assert _call(plugin, irc, msg, emoji="👍")["status"] == "ok"
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "testnick", "trigger-1", "👍")

    def test_pm_cannot_target_another_nick(self, react_env) -> None:
        plugin, irc, msg = react_env
        msg.args = ("testbot", "react to bob")

        _call(plugin, irc, msg, emoji="👍", nick="bob")

        plugin.llm_service.send_reaction.assert_called_once_with(irc, "testnick", "trigger-1", "❌")


class TestReactWiringAndDispatch:
    @staticmethod
    def _result(**kw) -> AssistantResult:
        base = {
            "content": "ok",
            "grounding_used": False,
            "spend": Spend(prompt_tokens=1, completion_tokens=1, cost=0.0),
            "model": "m",
        }
        return AssistantResult(**{**base, **kw})

    def test_ask_advertises_react_when_enabled(self, react_env) -> None:
        plugin, irc, msg = react_env
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = self._result()

        plugin.ask(irc, msg, ["react to bob"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert "react" in [t["function"]["name"] for t in kwargs["extra_tools"]]
        assert "react" in kwargs["extra_handlers"]

    def test_react_only_turn_sends_nothing_and_stores_a_note(self, react_env) -> None:
        plugin, irc, msg = react_env
        result = self._result(
            content="",
            last_successful_tool="react",
            last_tool_message="reacted 👍 to bob's message",
        )

        stored, should_log = plugin._dispatch_assistant_reply(
            irc, msg, result, caller=Caller(Identity("testnick", None), "#test"), response=""
        )

        assert (stored, should_log) == ("[reacted 👍 to bob's message]", True)
        irc.reply.assert_not_called()
        irc.error.assert_not_called()

    @pytest.mark.parametrize(
        "text", ["[reacted 🫡 to Larry's last message]", " [no recent message from bob]"]
    )
    def test_fake_react_note_is_dropped_for_a_cross(self, react_env, text) -> None:
        """2026-10-10 #afternet: grok's react missed and it posted the stored
        note shape as text, claiming a reaction that never happened."""
        plugin, irc, msg = react_env
        result = self._result(content=text, last_successful_tool="")

        _, should_log = plugin._dispatch_assistant_reply(
            irc, msg, result, caller=Caller(Identity("testnick", None), "#test"), response=text
        )

        assert should_log is False
        irc.reply.assert_not_called()
        irc.queueMsg.assert_not_called()
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "#test", "trigger-1", "❌")

    def test_ordinary_bracketed_reply_is_sent(self, react_env) -> None:
        plugin, irc, msg = react_env
        text = "[citation needed] that's not how kernels work"
        result = self._result(content=text)

        _, should_log = plugin._dispatch_assistant_reply(
            irc, msg, result, caller=Caller(Identity("testnick", None), "#test"), response=text
        )

        assert should_log is True
        plugin.llm_service.send_reaction.assert_not_called()


class TestLastMsgidPersistence:
    """The react tool's msgid memory survives a restart."""

    def test_db_round_trip_keeps_newest_and_trims(self, test_db) -> None:
        test_db.save_last_msgids(
            [
                ("afternet", "#a", "bob", "m1", 1.0),
                ("afternet", "#a", "larry", "m2", 2.0),
                ("afternet", "#a", "eve", "m3", 3.0),
            ],
            keep=2,
        )
        test_db.save_last_msgids([("afternet", "#a", "larry", "m4", 4.0)], keep=2)

        assert test_db.load_last_msgids(10) == [
            ("afternet", "#a", "eve", "m3"),
            ("afternet", "#a", "larry", "m4"),
        ]

    def test_flush_writes_only_changes_then_clears(self, react_env) -> None:
        plugin, irc, _ = react_env
        plugin.inFilter(irc, privmsg("#test", "Larry", "hi", "larry-1"))

        plugin._flush_last_msgids()
        plugin._flush_last_msgids()

        plugin.db.save_last_msgids.assert_called_once()
        rows = plugin.db.save_last_msgids.call_args.args[0]
        assert [r[:4] for r in rows] == [("afternet", "#test", "larry", "larry-1")]

    def test_load_refills_the_lru(self, react_env) -> None:
        plugin, irc, msg = react_env
        plugin._last_msgids.clear()
        plugin.db.load_last_msgids.return_value = [("afternet", "#test", "larry", "old-1")]

        plugin._load_last_msgids()
        payload = _call(plugin, irc, msg, emoji="🫡", nick="Larry")

        assert payload["status"] == "ok"
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "#test", "old-1", "🫡")

    def test_die_flushes(self, react_env) -> None:
        plugin, irc, _ = react_env
        plugin.inFilter(irc, privmsg("#test", "bob", "hi", "bob-1"))

        plugin.die()

        plugin.db.save_last_msgids.assert_called_once()
