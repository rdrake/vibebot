"""The Caller: who sent this message, and where a reply to them goes.

Every test here has an authenticated user whose account differs from their
nick (alice, authenticated as alice_acct). That is the case that broke
delivery three times (fb0be1a, d69f696, 4b7fb80: pages PM'd to the account
name), and in a private message ``msg.args[0]`` is the bot's own nick, the
other half of the same mistake. The reply target is the channel, or the
caller's IRC nick in a PM: never the bot, never the account.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.identity import Caller, Identity
from llm.service import AssistantResult, CompletionResult, Spend, StorybookResult

from .conftest import make_caller, make_registry_side_effect

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

NICK, ACCOUNT = "alice", "alice_acct"


def _result(content: str) -> AssistantResult:
    return AssistantResult(content=content, spend=Spend.ZERO, model="gpt-4")


class TestPreflightResolvesTheCaller:
    def test_channel_caller(self, plugin_env) -> None:
        """GIVEN alice in #test WHEN preflight runs THEN the Caller keys on the
        account and replies to the channel."""
        plugin, irc, msg = plugin_env
        expected = make_caller(irc, msg, NICK, account=ACCOUNT, channel="#test")

        caller = plugin._run_preflight(irc, msg, "hi", "ask", require_account=False).caller

        assert caller == expected
        assert caller.identity.key == ACCOUNT
        assert caller.identity.personal_key == ACCOUNT
        assert caller.reply_target == "#test"

    def test_pm_caller_replies_to_the_nick(self, plugin_env) -> None:
        """GIVEN alice PMs the bot WHEN preflight runs THEN the reply target is
        alice: not the bot's nick in args[0], not her account."""
        plugin, irc, msg = plugin_env
        make_caller(irc, msg, NICK, account=ACCOUNT)
        assert msg.args[0] == irc.nick

        caller = plugin._run_preflight(irc, msg, "hi", "ask", require_account=False).caller

        assert caller.is_private
        assert caller.reply_target == NICK

    def test_unauthenticated_caller_has_no_personal_key(self, plugin_env) -> None:
        plugin, irc, msg = plugin_env
        make_caller(irc, msg, NICK, channel="#test")

        caller = plugin._run_preflight(irc, msg, "hi", "ask", require_account=False).caller

        assert caller.identity.key == "!" + NICK
        assert caller.identity.personal_key is None

    def test_required_account_refusal_logs_the_reply_target(self, plugin_env) -> None:
        """A refused PM is logged under the sender, not the bot's own nick."""
        plugin, irc, msg = plugin_env
        make_caller(irc, msg, NICK)

        pf = plugin._run_preflight(irc, msg, "draw", "draw", require_account=True)

        assert pf.blocked
        irc.error.assert_called_once()
        nick, where = plugin.db.log_usage.call_args.args[:2]
        assert (nick, where) == (NICK, NICK)


class TestDeliveryGoesToTheNick:
    def test_pm_action_reply_goes_to_the_nick_not_the_account(
        self, plugin_env, mocker: MockerFixture
    ) -> None:
        """GIVEN a /me reply in a PM WHEN it is sent THEN the ACTION goes to
        alice. It used to go to ``nick``, which there was the account key."""
        plugin, irc, msg = plugin_env
        caller = make_caller(irc, msg, NICK, account=ACCOUNT)
        action = mocker.patch("llm.plugin.ircmsgs.action", return_value=mocker.sentinel.action)
        mocker.patch.object(plugin, "_safe_queue", return_value=True)

        plugin._dispatch_assistant_reply(
            irc, msg, _result("/me waves"), caller=caller, response="/me waves"
        )

        assert action.call_args.args[0] == NICK

    def test_pm_typing_indicator_goes_to_the_sender(self, make_service, mocker) -> None:
        """GIVEN a PM WHEN the bot starts typing THEN the indicator goes to
        alice, not to the bot's own nick in args[0]."""
        service, plugin = make_service()
        irc = mocker.MagicMock(nick="testbot", network="afternet")
        irc.state.capabilities_ack = {"message-tags"}
        msg = mocker.MagicMock()
        make_caller(irc, msg, NICK, account=ACCOUNT)

        service._begin_typing(irc, msg)()

        targets = {call.args[1].args[0] for call in plugin._safe_queue.call_args_list}
        assert targets == {NICK}

    def test_pm_story_reaches_the_nick_and_books_the_account(
        self, plugin_env, mocker: MockerFixture
    ) -> None:
        """GIVEN @story from a PM WHEN the page is ready THEN the link is PM'd
        to alice and the spend is booked under her storage key."""
        plugin, irc, msg = plugin_env
        caller = make_caller(irc, msg, NICK, account=ACCOUNT)
        plugin.llm_service.generate_storybook.return_value = StorybookResult(
            url="http://h/llm/story_x.html", title="A Tale", image_count=0, dropped=0
        )
        mocker.patch.object(plugin._llm_executor, "submit", side_effect=lambda _l, fn, *a: fn(*a))
        mocker.patch("llm.plugin.world.ircs", [irc])
        mocker.patch.object(plugin, "_safe_queue", return_value=True)
        privmsg = mocker.patch.object(plugin, "_safe_privmsg", side_effect=lambda t, m: (t, m))

        plugin._submit_storybook_job(caller=caller, persona="", brief="a tale")

        assert privmsg.call_args.args[0] == NICK
        assert plugin.db.log_usage.call_args.kwargs["nick"] == ACCOUNT

    def test_pm_reaction_tools_target_the_nick(self, plugin_env) -> None:
        """The react/redact/notify tools are built for the reply target."""
        plugin, irc, msg = plugin_env
        caller = make_caller(irc, msg, NICK, account=ACCOUNT)
        plugin.db.list_nick_watches.return_value = []

        _, handlers = plugin._build_notify_tool(irc, msg, caller)
        handlers["notify_when_online"]({"action": "list"})

        assert plugin.db.list_nick_watches.call_args.args[1] == NICK


class TestPrivateMessagesAreKeyedPerSender:
    def test_one_senders_pm_is_not_another_senders_channel_history(
        self, plugin_env, mocker: MockerFixture
    ) -> None:
        """GIVEN alice and bob both PM the bot WHEN bob's turn gathers history
        THEN alice's PM is not in it. Both used to share the bot's own nick
        as the channel, so each PM read the other's as the channel scene."""
        plugin, irc, _msg = plugin_env
        alice = Caller(Identity(NICK, ACCOUNT), None)
        bob = Caller(Identity("bob", None), None)
        result = mocker.MagicMock(error=None, content="noted", cost=0.0)

        plugin._store_context_and_log_usage(alice, "ask", "my secret", "noted", result, irc)
        _personal, channel_history = plugin._gather_history(bob)

        assert not any("secret" in m["content"] for m in channel_history)

    def test_nick_squatter_pm_sees_none_of_the_owners_pm(self, plugin_env) -> None:
        """GIVEN alice (authenticated) PMs the bot, then someone unauthenticated
        takes the nick alice and PMs it WHEN the second request is built THEN
        neither its history nor its channel history holds alice's line. The PM
        location is the nick, so the shared channel buffer must stay out of
        PMs entirely, and personal history is keyed on the storage key."""
        plugin, irc, msg = plugin_env
        plugin.llm_service.completion.return_value = CompletionResult(
            content="noted", grounding_used=False, model="gpt-4"
        )
        make_caller(irc, msg, NICK, account=ACCOUNT)
        plugin.ask(irc, msg, ["my", "secret", "plan"])
        assert any(
            "secret" in m["content"]
            for m in plugin._gather_history(Caller(Identity(NICK, ACCOUNT), None))[0]
        ), "the owner's own thread keeps the line"

        make_caller(irc, msg, NICK)
        plugin.ask(irc, msg, ["what", "did", "I", "say?"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        seen = [*(kwargs["history"] or []), *(kwargs["channel_history"] or [])]
        assert not any("secret" in m["content"] for m in seen)
        assert kwargs["channel_history"] == []

    def test_channel_lines_still_reach_the_shared_buffer(self, plugin_env, mocker) -> None:
        plugin, irc, _msg = plugin_env
        alice = Caller(Identity(NICK, ACCOUNT), "#test")
        bob = Caller(Identity("bob", None), "#test")
        result = mocker.MagicMock(error=None, content="noted", cost=0.0)

        plugin._store_context_and_log_usage(alice, "ask", "hello all", "noted", result, irc)
        _personal, channel_history = plugin._gather_history(bob)

        assert any("hello all" in m["content"] for m in channel_history)

    def test_duplicate_filter_is_per_sender(self, plugin_env, mocker: MockerFixture) -> None:
        from llm.plugin import PreflightResult

        plugin, irc, msg = plugin_env
        started = mocker.patch("llm.plugin.world.SupyThread")
        for nick in (NICK, "bob"):
            caller = make_caller(irc, msg, nick)
            plugin._dispatch_addressed_async(
                irc, msg, "hi", PreflightResult(False, caller), entry_route="pm"
            )

        assert started.call_count == 2


class TestPersonalDataNeedsAPersonalKey:
    def test_avatar_in_pm_does_not_open_a_verse_store(
        self, plugin_env, mocker: MockerFixture
    ) -> None:
        """@avatar from a PM has no channel, so no verse store is opened (it
        used to look one up under the bot's own nick)."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = make_registry_side_effect({"verseEnabled": True})
        get_store = mocker.patch.object(plugin, "_get_or_create_verse_store")
        make_caller(irc, msg, NICK, account=ACCOUNT)

        plugin.avatar(irc, msg, ["a sly fox"])

        plugin.db.save_avatar_persona.assert_called_once_with(ACCOUNT, "a sly fox")
        get_store.assert_not_called()

    @pytest.mark.parametrize(("account", "reads"), [(None, False), (ACCOUNT, True)])
    def test_storybook_persona_needs_a_personal_key(
        self, plugin_env, mocker: MockerFixture, account: str | None, reads: bool
    ) -> None:
        """GIVEN a verse turn with the storybook tool WHEN the caller is not
        authenticated THEN their nick does not read the persona stored under
        a matching account name."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"verseStorybookEnabled": True}
        )
        caller = make_caller(irc, msg, NICK, account=account, channel="#test")
        mocker.patch.object(plugin, "_build_verse_handlers_for_route", return_value={})
        mocker.patch.object(plugin, "_verse_edit_handler")
        mocker.patch.object(plugin, "_storybook_handler")
        mocker.patch.object(plugin, "_dispatch_assistant_reply", return_value=("", False))
        plugin.db.get_avatar_persona.return_value = "someone's persona"
        plugin.db.get_memories.return_value = []
        plugin.db.get_instruction.return_value = None
        from llm.plugin import PreflightResult

        plugin._ask_impl(
            irc, msg, "hello", PreflightResult(False, caller), verse_route=mocker.MagicMock()
        )

        assert plugin.db.get_avatar_persona.called is reads
