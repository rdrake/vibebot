"""irc_lookup kinds past channels/names/whois: whowas, who, topic, network,
ctcp_version/ctcp_ping and history.

Same simulation as test_irc_lookup.py: ``irc.queueMsg`` feeds the numerics
AfterNET actually sent (captured 2026-10-10) back through the handlers.
"""

from __future__ import annotations

import json

import pytest
from supybot import ircmsgs
from supybot.irclib import Batch

from .conftest import make_registry_side_effect


def numeric(command: str, *args: str) -> ircmsgs.IrcMsg:
    return ircmsgs.IrcMsg(command=command, args=("testbot", *args), prefix="irc.example.net")


@pytest.fixture
def env(plugin_env):
    plugin, irc, msg = plugin_env
    irc.network = "afternet"
    plugin.registryValue.side_effect = make_registry_side_effect()
    return plugin, irc, msg


def lookup(plugin, irc, channel: str = "#test", **args) -> dict:
    _, handlers = plugin._build_irc_lookup_tool(irc, channel)
    return json.loads(handlers["irc_lookup"](args).content)


def serve(irc, command: str, replies) -> None:
    """On the first sent ``command``, call ``replies(sent_msg)``."""

    def on_send(m) -> None:
        if m.command == command:
            replies(m)

    irc.queueMsg.side_effect = on_send


class TestWhowas:
    def test_entries_carry_host_and_last_seen(self, env) -> None:
        plugin, irc, msg = env

        def replies(_m) -> None:
            plugin.do314(
                irc, numeric("314", "rdrake", "rdrake", "rdrake.Users.AfterNET.Org", "*", "R")
            )
            plugin.do312(
                irc, numeric("312", "rdrake", "*.afternet.org", "Fri Oct  9 17:37:42 2026")
            )
            plugin.do301(irc, numeric("301", "rdrake", "Away"))
            plugin.do369(irc, numeric("369", "rdrake", "End of WHOWAS"))

        serve(irc, "WHOWAS", replies)

        payload = lookup(plugin, irc, kind="whowas", target="rdrake")

        assert payload["entries"] == [
            {
                "nick": "rdrake",
                "user": "rdrake",
                "host": "rdrake.Users.AfterNET.Org",
                "realname": "R",
                "server": "*.afternet.org",
                "last_seen": "Fri Oct  9 17:37:42 2026",
                "away": "Away",
            }
        ]

    def test_unknown_nick_is_an_error(self, env) -> None:
        plugin, irc, _ = env

        def replies(_m) -> None:
            plugin.do406(irc, numeric("406", "ghost", "There was no such nickname"))
            plugin.do369(irc, numeric("369", "ghost", "End of WHOWAS"))

        serve(irc, "WHOWAS", replies)

        assert lookup(plugin, irc, kind="whowas", target="ghost") == {
            "error": "There was no such nickname"
        }


class TestWho:
    def test_flags_become_fields(self, env) -> None:
        plugin, irc, _ = env

        def replies(m) -> None:
            assert m.args == ("#afternet", "%tcuhnfar,7")
            for row in (
                ("rdrake", "rdrake.Users.AfterNET.Org", "rdrake", "G*@xz", "rdrake", "ZNC"),
                ("grok", "grok.Bot.AfterNET.Org", "grok", "HxzB", "grok", "Grok"),
                ("user", "7A6318.IP", "juan", "Hxz", "0", "realname"),
            ):
                plugin.do354(irc, numeric("354", "7", "#afternet", *row))
            plugin.do315(irc, numeric("315", "#afternet", "End of /WHO list."))

        serve(irc, "WHO", replies)

        payload = lookup(plugin, irc, kind="who", target="#afternet")

        by_nick = {m["nick"]: m for m in payload["members"]}
        assert by_nick["rdrake"]["away"] and by_nick["rdrake"]["oper"]
        assert by_nick["rdrake"]["status"] == "@"
        assert by_nick["grok"]["bot"] and not by_nick["grok"]["away"]
        assert by_nick["juan"]["account"] is None

    def test_limnoria_join_who_rows_do_not_leak_in(self, env) -> None:
        """Token 1 rows belong to Limnoria's join-time WHO, not the lookup."""
        plugin, irc, _ = env

        def replies(_m) -> None:
            plugin.do354(irc, numeric("354", "1", "u", "1.2.3.4", "h", "eve", "H", "0", "Eve"))
            plugin.do315(irc, numeric("315", "#afternet", "End of /WHO list."))

        serve(irc, "WHO", replies)

        assert lookup(plugin, irc, kind="who", target="#afternet")["members"] == []


class TestTopic:
    def test_joined_channel_gives_setter_and_time(self, env) -> None:
        plugin, irc, _ = env

        def replies(_m) -> None:
            plugin.do332(irc, numeric("332", "#test", "Welcome"))
            plugin.do333(irc, numeric("333", "#test", "rdrake!r@h", "1700000000"))

        serve(irc, "TOPIC", replies)

        payload = lookup(plugin, irc, kind="topic", target="#test")

        assert payload["topic"] == "Welcome"
        assert payload["set_by"] == "rdrake"
        assert payload["set_at"].startswith("2023-11-14")

    def test_unjoined_channel_uses_list(self, env) -> None:
        plugin, irc, _ = env

        def replies(_m) -> None:
            plugin.do322(irc, numeric("322", "#linux", "40", "Kernel talk"))
            plugin.do323(irc, numeric("323", "End of /LIST"))

        serve(irc, "LIST", replies)

        payload = lookup(plugin, irc, kind="topic", target="#linux")

        assert payload["topic"] == "Kernel talk"
        assert "set_by" not in payload


class TestNetwork:
    def test_lusers_version_admin(self, env) -> None:
        plugin, irc, _ = env

        def replies(_m) -> None:
            plugin.do251(irc, numeric("251", "There are 237 users and 93 invisible on 9 servers"))
            plugin.do252(irc, numeric("252", "23", "operator(s) online"))
            plugin.do266(irc, numeric("266", "Current global users: 327 Max: 347"))
            plugin.do351(
                irc, numeric("351", "u2.10.12.14+Nefarious(2.0.0)", "Fractal.AfterNET.Org", "B")
            )
            plugin.do256(irc, numeric("256", "Fractal.AfterNET.Org", "Administrative info"))
            plugin.do257(irc, numeric("257", "Toronto, Canada"))
            plugin.do259(irc, numeric("259", "ibutsu"))

        serve(irc, "ADMIN", replies)

        payload = lookup(plugin, irc, kind="network")

        assert "23 operator(s) online" in payload["users"]
        assert payload["server"] == ["u2.10.12.14+Nefarious(2.0.0) on Fractal.AfterNET.Org"]
        assert payload["admin"] == ["Toronto, Canada", "ibutsu"]


class TestCtcp:
    def _reply(self, plugin, irc, nick: str, body: str) -> None:
        plugin.inFilter(
            irc,
            ircmsgs.IrcMsg(prefix=f"{nick}!u@h", command="NOTICE", args=("testbot", body)),
        )

    def test_version(self, env) -> None:
        plugin, irc, _ = env
        serve(
            irc,
            "PRIVMSG",
            lambda m: self._reply(plugin, irc, "larry", "\x01VERSION HexChat 2.16\x01"),
        )

        payload = lookup(plugin, irc, kind="ctcp_version", target="larry")

        assert payload == {"status": "ok", "nick": "larry", "client": "HexChat 2.16"}

    def test_ping_measures_lag(self, env) -> None:
        plugin, irc, _ = env
        serve(irc, "PRIVMSG", lambda m: self._reply(plugin, irc, "larry", m.args[1]))

        payload = lookup(plugin, irc, kind="ctcp_ping", target="larry")

        assert payload["lag_ms"] >= 0

    def test_silence_is_not_an_error(self, env, mocker) -> None:
        plugin, irc, _ = env
        mocker.patch.object(type(plugin), "_CTCP_TIMEOUT", 0.05)

        payload = lookup(plugin, irc, kind="ctcp_version", target="larry")

        assert payload["reply"] is None

    def test_unsolicited_reply_is_ignored(self, env) -> None:
        plugin, irc, _ = env
        self._reply(plugin, irc, "larry", "\x01VERSION spoof\x01")

        assert not plugin._irc_queries.pending("ctcp-VERSION", "afternet", "larry")


@pytest.fixture
def history_env(env):
    plugin, irc, msg = env
    plugin.registryValue.side_effect = make_registry_side_effect({"privacyOptInOnly": False})
    return plugin, irc, msg


class TestHistoryGate:
    def test_off_by_default_hides_the_kind(self, env) -> None:
        plugin, irc, _ = env

        schemas, _ = plugin._build_irc_lookup_tool(irc, "#test")
        fn = schemas[0]["function"]

        assert "history" not in fn["parameters"]["properties"]["kind"]["enum"]
        assert "history" not in json.dumps(fn)
        assert "count" not in fn["parameters"]["properties"]

    def test_off_refuses_without_sending(self, env) -> None:
        plugin, irc, _ = env
        irc.state.capabilities_ack = {"draft/chathistory"}

        payload = lookup(plugin, irc, kind="history")

        assert "privacyOptInOnly" in payload["error"]
        irc.queueMsg.assert_not_called()

    def test_on_advertises_the_kind(self, history_env) -> None:
        plugin, irc, _ = history_env

        schemas, _ = plugin._build_irc_lookup_tool(irc, "#test")
        fn = schemas[0]["function"]

        assert "history" in fn["parameters"]["properties"]["kind"]["enum"]
        assert "what did I miss" in fn["description"]
        assert "count" in fn["parameters"]["properties"]


class TestHistory:
    @pytest.fixture
    def env(self, history_env):
        return history_env

    @staticmethod
    def _batch(channel: str = "#test") -> Batch:
        return Batch(
            name="ref", type="chathistory", arguments=(channel,), messages=[], parent_batch=None
        )

    def _serve_history(self, plugin, irc, lines) -> None:
        batch = self._batch()
        irc.state.getParentBatches = lambda m: [batch]

        def replies(m) -> None:
            assert m.args[:2] == ("LATEST", "#test")
            for ts, nick, text in lines:
                line = ircmsgs.IrcMsg(prefix=f"{nick}!u@h", command="PRIVMSG", args=("#test", text))
                line.server_tags.update({"batch": "ref", "time": ts, "msgid": nick + ts})
                assert plugin.inFilter(irc, line) is None
            end = ircmsgs.IrcMsg(command="BATCH", args=("-ref",))
            end.tag("batch", batch)
            plugin.inFilter(irc, end)

        serve(irc, "CHATHISTORY", replies)

    def test_lines_come_back_formatted_and_are_dropped(self, env) -> None:
        plugin, irc, _ = env
        irc.state.capabilities_ack = {"draft/chathistory"}
        self._serve_history(
            plugin,
            irc,
            [
                ("2026-10-10T01:11:39.000Z", "rdrake", "vibebot give this a thumbs up"),
                ("2026-10-10T01:12:03.000Z", "Larry", "\x01ACTION salutes\x01"),
            ],
        )

        payload = lookup(plugin, irc, kind="history", count=2)

        assert payload["lines"] == [
            "01:11 <rdrake> vibebot give this a thumbs up",
            "01:12 <Larry> * salutes",
        ]
        # Replayed lines must not refill the react memory either.
        assert not plugin._last_msgids

    def test_other_channel_is_refused(self, env) -> None:
        plugin, irc, _ = env
        irc.state.capabilities_ack = {"draft/chathistory"}

        payload = lookup(plugin, irc, kind="history", target="#secret")

        assert "only covers this channel" in payload["error"]
        irc.queueMsg.assert_not_called()

    def test_needs_the_cap(self, env) -> None:
        plugin, irc, _ = env

        assert "not granted" in lookup(plugin, irc, kind="history")["error"]

    def test_stray_history_line_is_still_dropped(self, env) -> None:
        """A replay nobody is waiting for must not reach doPrivmsg."""
        plugin, irc, _ = env
        batch = self._batch()
        irc.state.getParentBatches = lambda m: [batch]
        line = ircmsgs.IrcMsg(prefix="bob!u@h", command="PRIVMSG", args=("#test", "vibebot hi"))
        line.server_tags["batch"] = "ref"

        assert plugin.inFilter(irc, line) is None


def test_history_and_redaction_caps_are_requested(env) -> None:
    from supybot import irclib

    assert {"draft/chathistory", "draft/message-redaction"} <= set(
        irclib.Irc.REQUEST_EXPERIMENTAL_CAPABILITIES
    )
