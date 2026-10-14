"""delete_last_reply (IRCv3 REDACT) and notify_when_online (IRC MONITOR)."""

from __future__ import annotations

import json

import pytest
from supybot import ircmsgs

from .conftest import make_registry_side_effect


def numeric(command: str, *args: str) -> ircmsgs.IrcMsg:
    return ircmsgs.IrcMsg(command=command, args=("testbot", *args), prefix="irc.example.net")


@pytest.fixture
def env(plugin_env, test_db, mocker):
    plugin, irc, msg = plugin_env
    irc.network = "afternet"
    irc.state.capabilities_ack = {"draft/message-redaction"}
    msg.server_tags = {"msgid": "req-1"}
    plugin.db = test_db
    plugin.registryValue.side_effect = make_registry_side_effect(
        {"redactEnabled": True, "notifyOnlineEnabled": True}
    )
    yield plugin, irc, msg
    # test_db closes before plugin_env's teardown runs die().
    plugin.db = mocker.MagicMock()


def sent(irc, command: str) -> list[ircmsgs.IrcMsg]:
    return [c.args[0] for c in irc.queueMsg.call_args_list if c.args[0].command == command]


def echo(plugin, irc, msgid: str, *, at: float, mocker, target: str = "#test") -> None:
    """The server echoing one of the bot's own lines back (echo-message)."""
    mocker.patch("llm.plugin.time.time", return_value=at)
    line = ircmsgs.IrcMsg(prefix="testbot!bot@host", command="PRIVMSG", args=(target, "hi"))
    line.server_tags["msgid"] = msgid
    plugin.inFilter(irc, line)


class TestDeleteLastReply:
    def _delete(self, plugin, irc, msg) -> dict:
        _, handlers = plugin._build_redact_tool(irc, msg)
        return json.loads(handlers["delete_last_reply"]({}).content)

    def test_redacts_the_latest_burst_only(self, env, mocker) -> None:
        plugin, irc, msg = env
        echo(plugin, irc, "old", at=1000.0, mocker=mocker)
        echo(plugin, irc, "new-1", at=2000.0, mocker=mocker)
        echo(plugin, irc, "new-2", at=2001.5, mocker=mocker)

        payload = self._delete(plugin, irc, msg)

        assert payload == {"status": "ok", "message": "deleted my last reply (2 lines)"}
        assert [m.args[:2] for m in sent(irc, "REDACT")] == [
            ("#test", "new-1"),
            ("#test", "new-2"),
        ]

    def test_second_delete_reaches_further_back(self, env, mocker) -> None:
        plugin, irc, msg = env
        echo(plugin, irc, "old", at=1000.0, mocker=mocker)
        echo(plugin, irc, "new", at=2000.0, mocker=mocker)

        self._delete(plugin, irc, msg)
        self._delete(plugin, irc, msg)

        assert [m.args[1] for m in sent(irc, "REDACT")] == ["new", "old"]

    def test_other_channels_lines_are_untouched(self, env, mocker) -> None:
        plugin, irc, msg = env
        echo(plugin, irc, "elsewhere", at=1000.0, mocker=mocker, target="#other")

        assert "error" in self._delete(plugin, irc, msg)
        assert sent(irc, "REDACT") == []

    def test_needs_the_cap(self, env, mocker) -> None:
        plugin, irc, msg = env
        irc.state.capabilities_ack = set()
        echo(plugin, irc, "x", at=1000.0, mocker=mocker)

        assert "not granted" in self._delete(plugin, irc, msg)["error"]

    def test_others_lines_are_not_ours(self, env) -> None:
        plugin, irc, msg = env
        line = ircmsgs.IrcMsg(prefix="bob!u@h", command="PRIVMSG", args=("#test", "x"))
        line.server_tags["msgid"] = "bob-1"
        plugin.inFilter(irc, line)

        assert "error" in self._delete(plugin, irc, msg)


class TestNotifyWhenOnline:
    def _call(self, plugin, irc, msg, **args) -> dict:
        _, handlers = plugin._build_notify_tool(irc, msg)
        return json.loads(handlers["notify_when_online"](args).content)

    def _serve_monitor(self, plugin, irc, reply: str) -> None:
        def on_send(m) -> None:
            if m.command == "MONITOR" and m.args[0] == "+":
                plugin.do730(
                    irc, numeric(reply, f"{m.args[1]}!u@h" if reply == "730" else m.args[1])
                )

        irc.queueMsg.side_effect = on_send

    def test_offline_nick_is_watched_then_announced(self, env) -> None:
        plugin, irc, msg = env
        self._serve_monitor(plugin, irc, "731")

        payload = self._call(plugin, irc, msg, nick="Larry")

        assert payload["online"] is False
        assert plugin.db.list_nick_watches("afternet") == ["larry"]

        irc.queueMsg.side_effect = None
        plugin.do730(irc, numeric("730", "Larry!l@host"))

        lines = [m.args for m in sent(irc, "PRIVMSG")]
        assert lines == [("#test", "testnick: Larry is online now.")]
        assert plugin.db.list_nick_watches("afternet") == []
        assert sent(irc, "MONITOR")[-1].args == ("-", "Larry")

    def test_online_nick_is_not_stored(self, env) -> None:
        plugin, irc, msg = env
        self._serve_monitor(plugin, irc, "730")

        payload = self._call(plugin, irc, msg, nick="larry")

        assert payload["online"] is True
        assert plugin.db.list_nick_watches("afternet") == []
        assert sent(irc, "MONITOR")[-1].args == ("-", "larry")
        # The immediate 730 answered the lookup; nobody gets "is online now".
        assert sent(irc, "PRIVMSG") == []

    def test_lapsed_watch_does_not_fire(self, env) -> None:
        """GIVEN a watch older than the 7-day lapse WHEN the nick signs on
        THEN nobody is told, and the stale row is still removed."""
        plugin, irc, _ = env
        plugin.db.add_nick_watch("afternet", "larry", "rdrake", "#test")
        with plugin.db._write_txn() as conn:
            conn.execute("UPDATE nick_watches SET created_at = created_at - 8 * 86400")

        plugin.do730(irc, numeric("730", "Larry!l@host"))

        assert sent(irc, "PRIVMSG") == []
        assert plugin.db.list_nick_watches("afternet") == []

    def test_cancel_and_list(self, env) -> None:
        plugin, irc, msg = env
        self._serve_monitor(plugin, irc, "731")
        self._call(plugin, irc, msg, nick="larry")

        assert self._call(plugin, irc, msg, action="list")["watching"] == ["larry"]
        assert self._call(plugin, irc, msg, nick="larry", action="cancel")["status"] == "ok"
        assert self._call(plugin, irc, msg, action="list")["watching"] == []
        assert (
            "not watching" in self._call(plugin, irc, msg, nick="larry", action="cancel")["error"]
        )

    def test_per_person_cap(self, env, mocker) -> None:
        plugin, irc, msg = env
        mocker.patch("llm.plugin._WATCHES_PER_REQUESTER", 1)
        self._serve_monitor(plugin, irc, "731")
        self._call(plugin, irc, msg, nick="a")

        assert "cancel one first" in self._call(plugin, irc, msg, nick="b")["error"]

    def test_full_monitor_list_is_an_error(self, env) -> None:
        plugin, irc, msg = env

        def on_send(m) -> None:
            if m.command == "MONITOR":
                plugin.do734(irc, numeric("734", "128", m.args[1], "Monitor list is full"))

        irc.queueMsg.side_effect = on_send

        assert self._call(plugin, irc, msg, nick="larry")["error"] == "the bot's watch list is full"

    def test_reconnect_resends_watches(self, env) -> None:
        plugin, irc, _ = env
        plugin.db.add_nick_watch("afternet", "larry", "rdrake", "#test")
        plugin.db.add_nick_watch("afternet", "eve", "bob", "#test")

        plugin._resend_monitors(irc)

        assert [m.args for m in sent(irc, "MONITOR")] == [("+", "eve,larry")]

    def test_rejects_own_nick(self, env) -> None:
        plugin, irc, msg = env

        assert self._call(plugin, irc, msg, nick="testbot")["error"] == "that is me"


def test_delete_note_shape_counts_as_fake(env) -> None:
    from llm.plugin import _FAKE_REACT_NOTE_RE

    assert _FAKE_REACT_NOTE_RE.match("[deleted my last reply (1 line)]")
