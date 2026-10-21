"""Tests for reminder commands."""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING

import pytest
from llm.identity import Caller
from llm.service import AssistantResult, ReminderParseResult, Spend

from .conftest import (
    back_reminders_with_sqlite,
    due_reminder,
    make_completion_response,
    make_reminder_row,
    put_reminder,
    reminder_names,
)

if TYPE_CHECKING:
    from unittest.mock import MagicMock

    from pytest_mock import MockerFixture


class TestReminderCommands:
    """Tests for the consolidated remind command."""

    def test_remind_command_exists(self) -> None:
        """GIVEN LLM plugin WHEN checking for remind THEN method exists."""
        from llm.plugin import LLM

        assert hasattr(LLM, "remind")
        assert callable(LLM.remind)

    def test_remind_docstring_shows_natural_language_examples(self) -> None:
        """GIVEN remind command WHEN checking docstring THEN shows examples."""
        from llm.plugin import LLM

        doc = LLM.remind.__doc__ or ""
        assert "natural language" in doc.lower() or "in 30 minutes" in doc


class TestReminderParseResult:
    """Tests for ReminderParseResult NamedTuple."""

    def test_schedule_result_with_action_prompt(self) -> None:
        """GIVEN schedule action WHEN creating result THEN stores action_prompt."""
        result = ReminderParseResult(
            action="schedule",
            seconds=1800,
            message="check the build",
            confirmation="Reminder set for 30 minutes from now.",
            note="Assuming UTC timezone.",
            action_prompt="Please post a build status update in #ops.",
        )
        assert result.action == "schedule"
        assert result.action_prompt == "Please post a build status update in #ops."

    def test_schedule_result(self) -> None:
        """GIVEN schedule action WHEN creating result THEN stores all fields."""
        result = ReminderParseResult(
            action="schedule",
            seconds=1800,
            message="check the build",
            confirmation="Reminder set for 30 minutes from now.",
            note="Assuming UTC timezone.",
        )
        assert result.action == "schedule"
        assert result.seconds == 1800
        assert result.message == "check the build"
        assert result.confirmation == "Reminder set for 30 minutes from now."
        assert result.note == "Assuming UTC timezone."

    def test_clarify_result(self) -> None:
        """GIVEN clarify action WHEN creating result THEN stores confirmation."""
        result = ReminderParseResult(
            action="clarify",
            confirmation="When should I remind you?",
        )
        assert result.action == "clarify"
        assert result.seconds is None
        assert result.message is None
        assert result.confirmation == "When should I remind you?"

    def test_default_values(self) -> None:
        """GIVEN minimal args WHEN creating result THEN defaults are correct."""
        result = ReminderParseResult(action="clarify")
        assert result.seconds is None
        assert result.message is None
        assert result.confirmation == ""
        assert result.note is None

    def test_default_action_prompt_is_empty(self) -> None:
        """GIVEN minimal args WHEN creating result THEN action_prompt defaults empty."""
        result = ReminderParseResult(action="clarify")
        assert result.action_prompt == ""


class TestReminderHelperMethods:
    """Tests for reminder helper methods on the plugin."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path) -> MagicMock:
        """Create a plugin instance with reminder support."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        back_reminders_with_sqlite(plugin.db, tmp_path / "reminders.db")

        return plugin

    # Tests for _get_user_reminders

    def test_get_user_reminders_empty(self, plugin: MagicMock) -> None:
        """GIVEN no reminders WHEN queried THEN returns empty list."""
        from llm.plugin import Identity

        result = plugin._get_user_reminders(Identity(raw_nick="testnick", account=None))
        assert result == []

    def test_get_user_reminders_filters_by_nick(self, plugin: MagicMock) -> None:
        """GIVEN reminders from multiple users WHEN queried THEN filters."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_1_100",
                nick="testnick",
                channel="#channel",
                message="my msg",
            ),
        )
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_2_200",
                nick="othernick",
                channel="#channel",
                message="their msg",
            ),
        )

        result = plugin._get_user_reminders(Identity(raw_nick="testnick", account=None))
        assert len(result) == 1
        assert result[0][1].message == "my msg"

    def test_get_user_reminders_matches_by_account(self, plugin: MagicMock) -> None:
        """GIVEN identified caller WHEN nick differs but account matches THEN matches."""
        from llm.plugin import Identity

        # User scheduled while presenting as "testnick"; later changes to "newnick"
        # but still identified as "MyAccount". Account match wins.
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_1_100",
                nick="testnick",
                channel="#channel",
                message="my msg",
                account="MyAccount",
            ),
        )
        result = plugin._get_user_reminders(Identity(raw_nick="newnick", account="MyAccount"))
        assert len(result) == 1
        assert result[0][1].message == "my msg"

    def test_get_user_reminders_account_isolates_users(self, plugin: MagicMock) -> None:
        """GIVEN matching nicks but different accounts WHEN queried THEN isolated."""
        from llm.plugin import Identity

        # Two accounts have happened to share a raw_nick; they must not
        # see each other's reminders.
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_1_100",
                nick="shared",
                channel="#channel",
                message="alpha msg",
                account="AccountAlpha",
            ),
        )
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_2_200",
                nick="shared",
                channel="#channel",
                message="beta msg",
                account="AccountBeta",
            ),
        )
        result = plugin._get_user_reminders(Identity(raw_nick="shared", account="AccountAlpha"))
        assert len(result) == 1
        assert result[0][1].message == "alpha msg"

    # Tests for _format_reminders

    def test_format_reminders_single(self, plugin: MagicMock) -> None:
        """GIVEN single reminder WHEN formatted THEN shows ID and message."""
        reminders = [
            (
                "llm_remind_123_456",
                make_reminder_row(
                    event_name="llm_remind_123_456",
                    nick="nick",
                    channel="#chan",
                    message="test message",
                ),
            )
        ]
        result = plugin._format_reminders(reminders)
        assert "#456" in result
        assert "test message" in result

    def test_format_reminders_multiple(self, plugin: MagicMock) -> None:
        """GIVEN multiple reminders WHEN formatted THEN pipe-separated."""
        reminders = [
            (
                "llm_remind_1_100",
                make_reminder_row(
                    event_name="llm_remind_1_100",
                    nick="nick",
                    channel="#chan",
                    message="first",
                ),
            ),
            (
                "llm_remind_2_200",
                make_reminder_row(
                    event_name="llm_remind_2_200",
                    nick="nick",
                    channel="#chan",
                    message="second",
                ),
            ),
        ]
        result = plugin._format_reminders(reminders)
        assert "#100" in result
        assert "#200" in result
        assert " | " in result

    def test_format_reminders_marks_auto_reminders(self, plugin: MagicMock) -> None:
        """GIVEN mixed reminders WHEN formatted THEN only action ones get [auto] marker."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_aaa1",
                nick="alice",
                channel="#chan",
                message="check CVE",
                action_prompt="check CVE status",
                account="alice",
            ),
        )
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_bbb2",
                nick="alice",
                channel="#chan",
                message="echo this",
            ),
        )
        formatted = plugin._format_reminders(
            plugin._get_user_reminders(Identity(raw_nick="alice", account="alice"))
        )
        assert "[auto]" in formatted
        # Echo reminder must NOT be marked.
        parts = formatted.split(" | ")
        auto_count = sum(1 for p in parts if "[auto]" in p)
        assert auto_count == 1

    def test_format_reminders_truncates_long_message(self, plugin: MagicMock) -> None:
        """GIVEN long message WHEN formatted THEN truncates."""
        long_msg = "x" * 100
        reminders = [
            (
                "llm_remind_1_100",
                make_reminder_row(
                    event_name="llm_remind_1_100",
                    nick="nick",
                    channel="#chan",
                    message=long_msg,
                ),
            )
        ]
        result = plugin._format_reminders(reminders)
        assert "..." in result
        assert len(result) < len(long_msg)

    # Tests for _find_user_reminder

    def test_find_user_reminder_found(self, plugin: MagicMock) -> None:
        """GIVEN matching reminder WHEN searched THEN returns event name."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_123_456",
                nick="testnick",
                channel="#channel",
                message="msg",
            ),
        )
        result = plugin._find_user_reminder(Identity(raw_nick="testnick", account=None), "456")
        assert result == "llm_remind_123_456"

    def test_find_user_reminder_not_found(self, plugin: MagicMock) -> None:
        """GIVEN no matching reminder WHEN searched THEN returns None."""
        from llm.plugin import Identity

        result = plugin._find_user_reminder(Identity(raw_nick="testnick", account=None), "999")
        assert result is None

    def test_find_user_reminder_wrong_owner(self, plugin: MagicMock) -> None:
        """GIVEN reminder owned by other WHEN searched THEN returns None."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_123_456",
                nick="othernick",
                channel="#channel",
                message="msg",
            ),
        )
        result = plugin._find_user_reminder(Identity(raw_nick="testnick", account=None), "456")
        assert result is None

    def test_nick_squatter_cannot_see_or_clear_account_owned_reminder(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN alice's reminder stored under account alice WHEN an unidentified
        caller on nick alice lists, finds, or clears THEN nothing is theirs."""
        from llm.plugin import Identity

        remove = mocker.patch("llm.plugin.schedule.removeEvent")
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_a_111",
                nick="alice",
                channel="#chan",
                message="alice 1",
                account="alice",
            ),
        )
        squatter = Identity(raw_nick="alice", account=None)

        assert plugin._get_user_reminders(squatter) == []
        assert plugin._find_user_reminder(squatter, "111") is None
        plugin._remind_clear_for_assistant(squatter)
        remove.assert_not_called()
        assert "llm_remind_a_111" in reminder_names(plugin)

    def test_owner_keeps_account_owned_reminder_after_nick_change(self, plugin: MagicMock) -> None:
        """GIVEN alice's account-owned reminder WHEN she returns as alice_ THEN she still owns it."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_a_111",
                nick="alice",
                channel="#chan",
                message="alice 1",
                account="alice",
            ),
        )
        owner = Identity(raw_nick="alice_", account="alice")

        assert len(plugin._get_user_reminders(owner)) == 1
        assert plugin._find_user_reminder(owner, "111") == "llm_remind_a_111"

    def test_identified_caller_still_owns_nick_only_reminder(self, plugin: MagicMock) -> None:
        """GIVEN a reminder set before identifying (no account) WHEN the same nick
        asks while identified THEN it matches by nick."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_a_111",
                nick="alice",
                channel="#chan",
                message="alice 1",
            ),
        )

        assert len(plugin._get_user_reminders(Identity(raw_nick="alice", account="alice"))) == 1

    def test_remind_clear_for_assistant_cancels_user_reminders_only(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN reminders from multiple users WHEN clear THEN only caller's are cancelled."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_a_111",
                nick="alice",
                channel="#chan",
                message="alice 1",
                account="alice",
            ),
        )
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_b_222",
                nick="alice",
                channel="#chan",
                message="alice 2",
                account="alice",
            ),
        )
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_c_333",
                nick="bob",
                channel="#chan",
                message="bob 1",
                account="bob",
            ),
        )

        result = plugin._remind_clear_for_assistant(
            Identity(raw_nick="alice", account="alice"),
        )

        assert "2 reminders" in result
        # bob's reminder must remain.
        assert reminder_names(plugin) == {"llm_remind_c_333"}
        assert plugin.db.delete_reminder.call_count == 2

    def test_remind_clear_for_assistant_no_reminders(self, plugin: MagicMock) -> None:
        """GIVEN no reminders WHEN clear THEN returns 'no pending' message."""
        from llm.plugin import Identity

        result = plugin._remind_clear_for_assistant(
            Identity(raw_nick="alice", account=None),
        )
        assert "no pending" in result.lower()
        plugin.db.delete_reminder.assert_not_called()

    def test_remind_clear_for_assistant_singular(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN one reminder WHEN cleared THEN message uses singular phrasing."""
        from llm.plugin import Identity

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_x_999",
                nick="alice",
                channel="#chan",
                message="only one",
            ),
        )

        result = plugin._remind_clear_for_assistant(
            Identity(raw_nick="alice", account=None),
        )
        assert result == "Cancelled 1 reminder."
        assert reminder_names(plugin) == set()

    # Tests for chain caps in _schedule_reminder

    def _amend_msg(self, mocker: MockerFixture) -> MagicMock:
        """An incoming channel message from alice, with no msgid."""
        msg = mocker.MagicMock()
        msg.args = ("#chan", "remind text")
        msg.prefix = "alice!user@host"
        msg.server_tags = {}
        return msg

    def _stub_parse_for_schedule(self, plugin: MagicMock, action_prompt: str = "") -> None:
        """Make parse_reminder return a 60-second schedule result."""
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=60,
            message="ping",
            confirmation="OK.",
            action_prompt=action_prompt,
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

    def test_schedule_reminder_per_user_pending_cap(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN per-user pending cap reached WHEN scheduling fresh THEN refuses."""
        from llm.plugin import LLM

        plugin._MetaSynchronized_rlock = threading.RLock()
        self._stub_parse_for_schedule(plugin)
        mocker.patch("llm.plugin.schedule.addEvent")

        msg = mocker.MagicMock()
        msg.args = ("#chan", "remind text")
        msg.prefix = "alice!user@host"
        msg.server_tags = {}

        # Pre-populate with the cap number of pending reminders for alice.
        for i in range(LLM._REMINDER_MAX_PENDING_PER_USER):
            put_reminder(
                plugin,
                make_reminder_row(
                    event_name=f"llm_remind_{i}_xxx",
                    nick="alice",
                    channel="#chan",
                    message=f"r{i}",
                    account="alice",
                    chain_position=1,
                ),
            )

        result = plugin._schedule_reminder(
            mock_irc,
            msg,
            Caller.from_row("alice", "alice", msg.args[0]),
            "in 1m ping",
        )
        assert result.ok is False
        assert "pending" in result.message.lower()
        plugin.db.save_reminder.assert_not_called()

    def test_schedule_reminder_reports_what_is_now_pending(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a scheduled reminder WHEN the tool returns THEN the summary names the subject and the delay in fixed wording."""

        plugin._MetaSynchronized_rlock = threading.RLock()
        self._stub_parse_for_schedule(plugin)  # message "ping", 60 seconds
        mocker.patch("llm.plugin.schedule.addEvent")

        result = plugin._schedule_reminder(
            mock_irc,
            msg=self._amend_msg(mocker),
            caller=Caller.from_row("alice", "alice", "#chan"),
            text="in 60s ping",
        )

        # The model-authored confirmation names neither reliably; this does.
        assert result.summary == "ping in 60s"

    def test_set_reminder_tool_hands_the_summary_to_the_model(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a scheduled reminder WHEN set_reminder returns THEN its message carries what is pending, so the stored note does too."""

        plugin._MetaSynchronized_rlock = threading.RLock()
        self._stub_parse_for_schedule(plugin)
        mocker.patch("llm.plugin.schedule.addEvent")
        mocker.patch.object(plugin, "_react")

        out = plugin._remind_set_for_assistant(
            mock_irc,
            self._amend_msg(mocker),
            Caller.from_row("alice", "alice", "#chan"),
            "in 60s ping",
        )

        assert out.ok is True
        assert "now pending: ping in 60s" in out.message

    def test_schedule_reminder_amends_a_recent_equivalent_reminder(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a reminder set moments ago WHEN the same one is restated with a new time THEN the old row is replaced, not duplicated."""

        plugin._MetaSynchronized_rlock = threading.RLock()
        self._stub_parse_for_schedule(plugin)  # 60 seconds
        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        now = time.time()

        # "remind me to ping in 10s", 5 seconds ago — still pending.
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_first",
                nick="alice",
                channel="#chan",
                message="ping",
                account="alice",
                fire_at=now + 5,
                created_at=now - 5,
                chain_position=1,
            ),
        )

        result = plugin._schedule_reminder(
            mock_irc,
            msg=self._amend_msg(mocker),
            caller=Caller.from_row("alice", "alice", "#chan"),
            text="no wait, make it a minute",
        )

        assert result.ok is True
        # The original is gone from every layer and exactly one row remains.
        plugin.db.delete_reminder.assert_called_once_with("llm_remind_first")
        assert reminder_names(plugin) != {"llm_remind_first"}
        assert len(reminder_names(plugin)) == 1
        plugin.db.save_reminder.assert_called_once()
        # ...and it fires at the amended time, not the original 5s.
        assert add_event.call_args.args[1] - now == pytest.approx(60, abs=2)

    def test_schedule_reminder_amends_across_a_large_time_change(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a recent reminder WHEN restated far outside the old fire-proximity window THEN it still amends rather than duplicating."""

        plugin._MetaSynchronized_rlock = threading.RLock()
        # 300s — the case that produced two live reminders before amendment.
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=300,
            message="ping",
            confirmation="OK.",
            action_prompt="",
        )
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        now = time.time()

        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_first",
                nick="alice",
                channel="#chan",
                message="ping",
                account="alice",
                fire_at=now + 50,
                created_at=now - 10,
                chain_position=1,
            ),
        )

        result = plugin._schedule_reminder(
            mock_irc,
            msg=self._amend_msg(mocker),
            caller=Caller.from_row("alice", "alice", "#chan"),
            text="make it 5 minutes",
        )

        assert result.ok is True
        assert len(reminder_names(plugin)) == 1
        plugin.db.save_reminder.assert_called_once()
        assert add_event.call_args.args[1] - now == pytest.approx(300, abs=2)

    def test_schedule_reminder_leaves_an_older_reminder_alone(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a same-worded reminder set long ago WHEN a new one is set THEN both stand."""

        plugin._MetaSynchronized_rlock = threading.RLock()
        self._stub_parse_for_schedule(plugin)
        mocker.patch("llm.plugin.schedule.addEvent")
        now = time.time()

        # Set ten minutes ago — a deliberate second reminder, not a correction.
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_old",
                nick="alice",
                channel="#chan",
                message="ping",
                account="alice",
                fire_at=now + 3600,
                created_at=now - 600,
                chain_position=1,
            ),
        )

        result = plugin._schedule_reminder(
            mock_irc,
            msg=self._amend_msg(mocker),
            caller=Caller.from_row("alice", "alice", "#chan"),
            text="in 60s ping",
        )

        assert result.ok is True
        plugin.db.delete_reminder.assert_not_called()
        assert len(reminder_names(plugin)) == 2
        plugin.db.save_reminder.assert_called_once()

    def test_restating_a_recurring_reminder_after_a_fire_keeps_the_chain(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a recurring reminder set long ago that just fired WHEN its owner
        restates it within the amend window THEN it is a second reminder, not
        an amendment that ends the chain: a fire moves the row in place and
        keeps the created_at of the request that set it."""

        plugin._MetaSynchronized_rlock = threading.RLock()
        self._stub_parse_for_schedule(plugin)
        mocker.patch("llm.plugin.world").ircs = [mocker.MagicMock()]
        now = time.time()
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_chain",
                nick="alice",
                channel="#chan",
                message="ping",
                account="alice",
                fire_at=now - 1,
                created_at=now - 3600,
                recurrence_seconds=600,
            ),
        )
        plugin._agenda_tick()
        fired = plugin.db.sqlite_reminders.get_reminder("llm_remind_chain")
        assert fired is not None and fired.chain_position == 2

        result = plugin._schedule_reminder(
            mock_irc,
            msg=self._amend_msg(mocker),
            caller=Caller.from_row("alice", "alice", "#chan"),
            text="in 60s ping",
        )

        assert result.ok is True
        plugin.db.delete_reminder.assert_not_called()
        assert len(reminder_names(plugin)) == 2

    def test_schedule_reminder_fresh_chain_starts_at_position_one(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a new reminder WHEN scheduling THEN its chain starts at position 1."""

        plugin._MetaSynchronized_rlock = threading.RLock()
        self._stub_parse_for_schedule(plugin)
        mocker.patch("llm.plugin.schedule.addEvent")

        msg = mocker.MagicMock()
        msg.args = ("#chan", "remind text")
        msg.prefix = "alice!user@host"
        msg.server_tags = {}

        result = plugin._schedule_reminder(
            mock_irc,
            msg,
            Caller.from_row("alice", "alice", msg.args[0]),
            "in 1m ping",
        )
        assert result.ok is True
        [row] = plugin.db.sqlite_reminders.load_pending_reminders()
        assert row.chain_position == 1

    # Tests for _react helper

    def test_react_returns_false_when_no_msgid(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN msg without msgid WHEN reacting THEN returns False, no TAGMSG queued."""
        irc = mocker.MagicMock()
        msg = mocker.MagicMock()
        msg.server_tags = {}
        msg.args = ("#chan",)
        plugin.llm_service.send_reaction = mocker.MagicMock()

        assert plugin._react(irc, msg, "👍") is False
        plugin.llm_service.send_reaction.assert_not_called()

    def test_react_calls_send_reaction_with_msgid(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN msg with msgid WHEN reacting THEN delegates to send_reaction."""
        irc = mocker.MagicMock()
        msg = mocker.MagicMock()
        msg.server_tags = {"msgid": "abc-123"}
        msg.args = ("#chan",)
        plugin.llm_service.send_reaction = mocker.MagicMock(return_value=True)

        assert plugin._react(irc, msg, "⏰") is True
        plugin.llm_service.send_reaction.assert_called_once_with(irc, "#chan", "abc-123", "⏰")

    # Test database attribute

    def test_plugin_has_db_attribute(self, plugin: MagicMock) -> None:
        """GIVEN plugin WHEN initialized THEN db attribute exists."""
        assert hasattr(plugin, "db")

    def test_schedule_reminder_persists_action_prompt(
        self, plugin: MagicMock, mock_irc: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN action reminder WHEN scheduled THEN persists action_prompt."""

        msg = mocker.MagicMock()
        msg.args = ("#ops", "remind test")
        msg.prefix = "testnick!user@host"
        msg.server_tags = {}

        plugin._MetaSynchronized_rlock = threading.RLock()
        plugin.llm_service.parse_reminder.return_value = ReminderParseResult(
            action="schedule",
            seconds=120,
            message="post status update in #ops",
            confirmation="Reminder set for 2 minutes from now.",
            action_prompt="Post a status update in #ops.",
        )

        mocker.patch("llm.plugin.schedule.addEvent")

        result = plugin._schedule_reminder(
            mock_irc,
            msg,
            Caller.from_row("testnick", "acct-testnick", msg.args[0]),
            "in 2 minutes tell the bot to post status in #ops",
        )

        assert result.ok is True

        [reminder_data] = plugin.db.sqlite_reminders.load_pending_reminders()
        assert reminder_data.event_name.startswith("llm_remind_")
        assert reminder_data.nick == "testnick"
        assert reminder_data.channel == "#ops"
        assert reminder_data.message == "post status update in #ops"
        assert reminder_data.action_prompt == "Post a status update in #ops."
        assert reminder_data.account == "acct-testnick"
        assert reminder_data.chain_position == 1
        plugin.db.save_reminder.assert_called_once()
        assert plugin.db.save_reminder.call_args.kwargs["action_prompt"] == (
            "Post a status update in #ops."
        )
        assert plugin.db.save_reminder.call_args.kwargs["account"] == "acct-testnick"

    # Test plugin cleanup

    def test_plugin_die_cleans_up_reminders(
        self,
        plugin: MagicMock,
        mocker: MockerFixture,
    ) -> None:
        """GIVEN plugin with reminders WHEN die called THEN the wakeup goes and the rows stay.

        The rows are the schedule: they must survive a restart.
        """
        from llm.plugin import LLM

        mock_remove_event = mocker.patch("llm.plugin.schedule.removeEvent")

        # Add some reminders
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_1_100",
                nick="user1",
                channel="#channel",
                message="msg1",
            ),
        )
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_2_200",
                nick="user2",
                channel="#channel",
                message="msg2",
            ),
        )

        mocker.patch.object(LLM.__bases__[0], "die", return_value=None)
        mocker.patch("llm.plugin.httpserver.unhook")
        plugin.die()

        assert mocker.call("llm_agenda_wakeup") in mock_remove_event.call_args_list
        assert reminder_names(plugin) == {"llm_remind_1_100", "llm_remind_2_200"}


class TestParseReminderService:
    """Tests for parse_reminder method in LLMService."""

    @pytest.fixture
    def mock_plugin(self, mocker: MockerFixture) -> MagicMock:
        """Create a mock plugin for service tests."""
        plugin = mocker.MagicMock()
        plugin.registryValue.side_effect = lambda key, *args: {
            "assistantModel": "xai/grok-4.3",
            "timeout": 30,
        }.get(key, "")
        return plugin

    @pytest.fixture
    def service(self, mock_plugin: MagicMock, mocker: MockerFixture) -> MagicMock:
        """Create an LLMService instance."""
        from llm.service import LLMService

        mocker.patch("llm.service.log")
        return LLMService(mock_plugin)

    def test_parse_reminder_empty_text(self, service: MagicMock) -> None:
        """GIVEN empty text WHEN parsing THEN returns clarify without API call."""
        result = service.parse_reminder("")
        assert result.action == "clarify"
        assert (
            "what to remind" in result.confirmation.lower()
            or "tell me" in result.confirmation.lower()
        )

    def test_parse_reminder_whitespace_only(self, service: MagicMock) -> None:
        """GIVEN whitespace-only text WHEN parsing THEN returns clarify without API call."""
        result = service.parse_reminder("   \n\t  ")
        assert result.action == "clarify"

    def test_parse_reminder_too_long(self, service: MagicMock) -> None:
        """GIVEN text over 500 chars WHEN parsing THEN returns clarify without API call."""
        result = service.parse_reminder("x" * 501)
        assert result.action == "clarify"
        assert "too long" in result.confirmation.lower()

    def test_parse_reminder_exactly_500_chars_accepted(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN text at exactly 500 chars WHEN parsing THEN proceeds to API call."""
        text = "x" * 500
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 60, "message": "test", "confirmation": "Set!"}'
        )
        result = service.parse_reminder(text)
        assert result.action == "schedule"

    def test_parse_reminder_no_api_key(
        self, service: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """GIVEN no key for the reminder model's provider THEN clarify with error.

        The fixture's assistantModel is ``xai/grok-4.3``, so XAI_API_KEY — not
        any other provider variable — is what has to be missing for this guard
        to fire.
        """
        monkeypatch.delenv("XAI_API_KEY", raising=False)

        result = service.parse_reminder("in 30 minutes test")
        assert result.action == "clarify"
        assert "XAI_API_KEY" in result.confirmation

    def test_parse_reminder_uses_assistant_model(self, mocker: MockerFixture) -> None:
        """parse_reminder sends assistantModel; provider.chat resolves its key."""
        from llm.service import LLMService

        mock_plugin = mocker.MagicMock()
        mock_plugin.registryValue.side_effect = lambda key, *args: {
            "assistantModel": "anthropic/claude-haiku-4-5",
            "timeout": 30,
        }.get(key, "")
        mocker.patch("llm.service.log")
        service = LLMService(mock_plugin)

        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 60, "message": "x", "confirmation": "ok"}'
        )

        service.parse_reminder("in 60s test")

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["model"] == "anthropic/claude-haiku-4-5"

    def test_parse_reminder_schedule_success(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN valid LLM response WHEN parsing THEN returns schedule result."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 1800, '
            '"message": "check build", "confirmation": "Reminder set for 30m."}'
        )

        result = service.parse_reminder("in 30 minutes check build")

        assert result.action == "schedule"
        assert result.seconds == 1800
        assert result.message == "check build"
        assert "30m" in result.confirmation

    def test_parse_reminder_clarify_response(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN clarify LLM response WHEN parsing THEN returns clarify result."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "clarify", "confirmation": "When should I remind you?"}'
        )

        result = service.parse_reminder("remind me")

        assert result.action == "clarify"
        assert "When" in result.confirmation

    def test_parse_reminder_invalid_json(self, service: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN invalid JSON WHEN parsing THEN returns clarify with error."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response("not valid json")

        result = service.parse_reminder("in 30 minutes test")

        assert result.action == "clarify"
        assert "couldn't understand" in result.confirmation.lower()

    def test_parse_reminder_strips_markdown_fences(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN JSON with markdown fences WHEN parsing THEN strips them."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '```json\n{"action": "schedule", "seconds": 60, '
            '"message": "test", "confirmation": "Set!"}\n```'
        )

        result = service.parse_reminder("in 1 minute test")

        assert result.action == "schedule"
        assert result.seconds == 60

    def test_parse_reminder_with_note(self, service: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN response with note WHEN parsing THEN includes note."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 3600, "message": "meeting", '
            '"confirmation": "Reminder set for 3pm.", "note": "Assuming EST"}'
        )

        result = service.parse_reminder("at 3pm meeting")

        assert result.action == "schedule"
        assert result.note == "Assuming EST"

    def test_parse_reminder_returns_action_prompt_for_imperative(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN imperative reminder WHEN parsing THEN returns action_prompt."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 3600, "message": "post status", '
            '"confirmation": "Reminder set for 1h.", "action_prompt": "Post a status update in #ops."}'
        )

        result = service.parse_reminder("in 1 hour tell the bot to post status in #ops")

        assert result.action == "schedule"
        assert result.action_prompt == "Post a status update in #ops."

    def test_parse_reminder_omits_action_prompt_for_echo(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN passive reminder WHEN parsing THEN action_prompt is empty."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 1200, "message": "check backups", '
            '"confirmation": "Reminder set for 20m.", "action_prompt": ""}'
        )

        result = service.parse_reminder("in 20 minutes remind me to check backups")

        assert result.action == "schedule"
        assert result.action_prompt == ""

    def test_parse_reminder_negative_seconds_rejected(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN negative seconds WHEN parsing THEN returns clarify."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": -100, "message": "test", "confirmation": "Set!"}'
        )

        result = service.parse_reminder("yesterday test")

        assert result.action == "clarify"

    def test_parse_reminder_api_error(self, service: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN API error WHEN parsing THEN returns clarify with error."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.side_effect = Exception("API error")

        result = service.parse_reminder("in 30 minutes test")

        assert result.action == "clarify"
        assert "couldn't parse" in result.confirmation.lower()

    def test_parse_reminder_zero_seconds_rejected(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN zero seconds WHEN parsing THEN returns clarify."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 0, "message": "test", "confirmation": "Set!"}'
        )

        result = service.parse_reminder("now test")

        assert result.action == "clarify"

    def test_parse_reminder_missing_seconds(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN missing seconds field WHEN parsing THEN returns clarify."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "message": "test", "confirmation": "Set!"}'
        )

        result = service.parse_reminder("sometime test")

        assert result.action == "clarify"

    def test_parse_reminder_uses_text_as_fallback_message(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN no message in response WHEN parsing THEN uses input text."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            '{"action": "schedule", "seconds": 60, "confirmation": "Set!"}'
        )

        result = service.parse_reminder("in 1 minute original text")

        assert result.action == "schedule"
        assert result.message == "in 1 minute original text"

    def test_parse_reminder_strips_fences_without_trailing_backticks(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN markdown fence without closing WHEN parsing THEN handles gracefully."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        # Fence without proper closing
        mock_completion.return_value = make_completion_response(
            '```json\n{"action": "schedule", "seconds": 120, "message": "test", "confirmation": "Set!"}'
        )

        result = service.parse_reminder("in 2 minutes test")

        assert result.action == "schedule"
        assert result.seconds == 120

    def test_parse_reminder_numeric_recurrence(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN numeric recurrence WHEN parsing THEN populates recurrence_seconds only."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 300,
                    "message": "check the build",
                    "confirmation": "Reminder set for 5m.",
                    "action_prompt": "check the build",
                    "recurrence_seconds": 300,
                    "recurrence_rrule": None,
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("every 5 minutes check the build")

        assert result.action == "schedule"
        assert result.recurrence_seconds == 300
        assert result.recurrence_rrule is None
        assert result.watch_mode is False
        assert "(recurring" not in result.action_prompt
        assert "(watch" not in result.action_prompt
        assert result.action_prompt == "check the build"

    def test_parse_reminder_sub_minute_recurrence_clarifies(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a 30s cadence WHEN parsing THEN clarify instead of scheduling a
        chain that fires (and bills an LLM turn) twice a minute. Scheduled
        tasks share this parser, so the floor covers both."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 30,
                    "message": "check the build",
                    "confirmation": "Reminder set for 30s.",
                    "action_prompt": "check the build",
                    "recurrence_seconds": 30,
                    "recurrence_rrule": None,
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("every 30 seconds check the build")

        assert result.action == "clarify"
        assert "one minute" in result.confirmation

    def test_parse_reminder_sub_minute_rrule_clarifies(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a FREQ=SECONDLY rule WHEN parsing THEN the one-minute floor
        applies to calendar cadences too."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 10,
                    "message": "check the build",
                    "confirmation": "Set!",
                    "action_prompt": "check the build",
                    "recurrence_seconds": None,
                    "recurrence_rrule": "FREQ=SECONDLY;INTERVAL=10",
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("every 10 seconds check the build")

        assert result.action == "clarify"
        assert "one minute" in result.confirmation

    def test_parse_reminder_calendar_recurrence(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN calendar recurrence WHEN parsing THEN populates recurrence_rrule only."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 3600,
                    "message": "stand-up",
                    "confirmation": "Set!",
                    "action_prompt": "post stand-up reminder",
                    "recurrence_seconds": None,
                    "recurrence_rrule": "FREQ=WEEKLY;BYDAY=MO;BYHOUR=9;BYMINUTE=0",
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("every Monday at 9am stand-up reminder")

        assert result.action == "schedule"
        assert result.recurrence_seconds is None
        assert result.recurrence_rrule is not None
        assert "FREQ=WEEKLY" in result.recurrence_rrule
        assert "BYDAY=MO" in result.recurrence_rrule
        assert "(recurring" not in result.action_prompt

    def test_parse_reminder_watch_mode(self, service: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN watch-style phrasing WHEN parsing THEN watch_mode is True."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 300,
                    "message": "build status",
                    "confirmation": "Set!",
                    "action_prompt": "check whether the build passes",
                    "recurrence_seconds": None,
                    "recurrence_rrule": None,
                    "watch_mode": True,
                }
            )
        )

        result = service.parse_reminder("in 5m let me know if the build passes")

        assert result.action == "schedule"
        assert result.watch_mode is True
        assert "(watch" not in result.action_prompt

    def test_parse_reminder_strips_leaked_recurring_parenthetical(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN model leaks (recurring: ...) into action_prompt WHEN parsing THEN stripped."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 300,
                    "message": "check the build",
                    "confirmation": "ok",
                    "action_prompt": "check the build (recurring: every 5 minutes)",
                    "recurrence_seconds": 300,
                    "recurrence_rrule": None,
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("every 5m check the build")

        assert "(recurring" not in result.action_prompt
        assert result.action_prompt == "check the build"
        assert result.recurrence_seconds == 300

    def test_parse_reminder_strips_leaked_watch_parenthetical(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN model leaks (watch ...) into action_prompt WHEN parsing THEN stripped."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 300,
                    "message": "check CVE",
                    "confirmation": "ok",
                    "action_prompt": "check CVE status (watch — only respond on positive result)",
                    "recurrence_seconds": None,
                    "recurrence_rrule": None,
                    "watch_mode": True,
                }
            )
        )

        result = service.parse_reminder("in 5m let me know when CVE is patched")

        assert "(watch" not in result.action_prompt
        assert result.action_prompt == "check CVE status"
        assert result.watch_mode is True

    def test_parse_reminder_invalid_rrule_falls_back_to_one_shot(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN malformed rrule WHEN parsing THEN clears it and falls back to one-shot."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 60,
                    "message": "x",
                    "confirmation": "ok",
                    "action_prompt": "x",
                    "recurrence_seconds": None,
                    "recurrence_rrule": "FREQ=NONSENSE;BLAH",
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("every monday do x")

        assert result.action == "schedule"
        assert result.recurrence_rrule is None  # invalid rule cleared
        assert result.recurrence_seconds is None  # one-shot fallback

    def test_parse_reminder_both_recurrence_kinds_prefers_rrule(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN model returns both recurrence kinds WHEN parsing THEN keeps rrule and clears seconds."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 86400,
                    "message": "x",
                    "confirmation": "ok",
                    "action_prompt": "x",
                    "recurrence_seconds": 86400,
                    "recurrence_rrule": "FREQ=DAILY;BYHOUR=8",
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("daily at 8am do x")

        assert result.recurrence_rrule == "FREQ=DAILY;BYHOUR=8"
        assert result.recurrence_seconds is None

    def test_parse_reminder_one_shot_has_no_recurrence(
        self, service: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN one-shot reminder WHEN parsing THEN both recurrence fields are None."""
        mock_completion = mocker.patch("llm.service.provider.chat")
        mock_completion.return_value = make_completion_response(
            json.dumps(
                {
                    "action": "schedule",
                    "seconds": 1800,
                    "message": "check build",
                    "confirmation": "Set!",
                    "action_prompt": "check the build",
                    "recurrence_seconds": None,
                    "recurrence_rrule": None,
                    "watch_mode": False,
                }
            )
        )

        result = service.parse_reminder("in 30 minutes check the build")

        assert result.action == "schedule"
        assert result.recurrence_seconds is None
        assert result.recurrence_rrule is None
        assert result.watch_mode is False


class TestReminderEventNaming:
    """Tests for uuid-based reminder event naming (Fix 3)."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path) -> MagicMock:
        """Create a plugin instance."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        back_reminders_with_sqlite(plugin.db, tmp_path / "reminders.db")

        return plugin

    def test_event_name_uses_uuid_format(self, plugin: MagicMock) -> None:
        """GIVEN reminder delivery closure WHEN created THEN event name uses uuid hex."""
        import re

        closure = due_reminder(plugin, "nick", "#chan", "msg", "llm_remind_abc123")
        assert callable(closure)
        # The event name format should be llm_remind_ + 12 hex chars
        assert re.match(r"^[a-f0-9]{12}$", "abc123def456")

    def test_no_reminder_counter_attribute(self, plugin: MagicMock) -> None:
        """GIVEN plugin WHEN initialized THEN no _reminder_counter attribute."""
        assert not hasattr(plugin, "_reminder_counter")


class TestReminderDeliveryClosure:
    """Tests for _make_reminder_delivery_closure (Fix 6)."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path) -> MagicMock:
        """Create a plugin instance."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        back_reminders_with_sqlite(plugin.db, tmp_path / "reminders.db")

        return plugin

    def test_cancel_takes_back_the_clock(self, plugin: MagicMock, mocker: MockerFixture) -> None:
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_a", nick="Rubin", channel="#chan", reply_msgid="m1"
            ),
        )

        plugin._cancel_reminder("llm_remind_a")

        plugin.llm_service.send_reaction.assert_called_once_with(
            mock_irc, "#chan", "m1", "⏰", remove=True
        )

    def test_cancel_in_pm_unreacts_to_the_sender(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]
        put_reminder(
            plugin,
            make_reminder_row(
                event_name="llm_remind_a", nick="Rubin", channel="vibebot", reply_msgid="m1"
            ),
        )

        plugin._cancel_reminder("llm_remind_a")

        assert plugin.llm_service.send_reaction.call_args.args[1] == "Rubin"

    def test_cancel_without_msgid_sends_nothing(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        mocker.patch("llm.plugin.world").ircs = [mocker.MagicMock()]
        put_reminder(
            plugin, make_reminder_row(event_name="llm_remind_a", nick="Rubin", channel="#chan")
        )

        plugin._cancel_reminder("llm_remind_a")

        plugin.llm_service.send_reaction.assert_not_called()

    def test_delivery_cleans_up_on_success(self, plugin: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN delivery closure WHEN queueMsg succeeds THEN cleans up reminder."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]

        event_name = "llm_remind_test123"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="nick",
                channel="#chan",
                message="test msg",
            ),
        )

        deliver = due_reminder(plugin, "nick", "#chan", "test msg", event_name)
        deliver()

        assert event_name not in reminder_names(plugin)
        plugin.db.delete_reminder.assert_called_with(event_name)
        mock_irc.queueMsg.assert_called_once()

    def test_delivery_via_pm_sends_to_user_nick(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN reminder set via PM WHEN delivered THEN sends to user nick, not bot nick."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        event_name = "llm_remind_pm_test"
        # channel="vibebot" simulates PM: _get_channel returns the bot's own nick
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="rdrake",
                channel="vibebot",
                message="eat a sandwich",
            ),
        )

        deliver = due_reminder(plugin, "rdrake", "vibebot", "eat a sandwich", event_name)
        deliver()

        # Should deliver to user's nick, not the bot's nick
        sent_msg = mock_irc.queueMsg.call_args[0][0]
        assert sent_msg.args[0] == "rdrake"
        assert "eat a sandwich" in sent_msg.args[1]

    def test_delivery_in_channel_sends_to_channel(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN reminder set in channel WHEN delivered THEN sends to channel."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_world.ircs = [mock_irc]
        plugin.llm_service.sanitize_output.side_effect = lambda x: x

        event_name = "llm_remind_chan_test"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="rdrake",
                channel="#test",
                message="check build",
            ),
        )

        deliver = due_reminder(plugin, "rdrake", "#test", "check build", event_name)
        deliver()

        sent_msg = mock_irc.queueMsg.call_args[0][0]
        assert sent_msg.args[0] == "#test"

    def test_delivery_cleans_up_even_on_error(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a fire WHEN queueMsg raises THEN the reminder is still settled."""
        mock_world = mocker.patch("llm.plugin.world")
        mock_irc = mocker.MagicMock()
        mock_irc.queueMsg.side_effect = RuntimeError("send failed")
        mock_world.ircs = [mock_irc]

        event_name = "llm_remind_test456"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="nick",
                channel="#chan",
                message="test msg",
            ),
        )

        deliver = due_reminder(plugin, "nick", "#chan", "test msg", event_name)

        deliver()  # the wakeup logs a failed fire; it never raises into the scheduler

        # Cleanup should still happen despite the error
        assert event_name not in reminder_names(plugin)
        plugin.db.delete_reminder.assert_called_with(event_name)


class TestReminderActionDelivery:
    """Tests for reminder action delivery through assistant_completion."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path) -> MagicMock:
        """Create a plugin instance."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        back_reminders_with_sqlite(plugin.db, tmp_path / "reminders.db")
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        return plugin

    def test_accountless_action_reminder_gets_no_memories_or_instruction(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN an action reminder set without an account WHEN it fires THEN
        the nick-keyed memories and @instruct are not loaded: the nick may be
        someone else's account."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        mocker.patch.object(plugin, "_gather_history", return_value=([], []))
        memories = mocker.patch.object(plugin, "_get_user_memories", return_value=["secret"])
        plugin.db.get_instruction.return_value = "secret instruction"
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="ok", spend=Spend.ZERO
        )

        event_name = "llm_remind_action_1"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="check build",
                action_prompt="check build",
            ),
        )
        due_reminder(
            plugin, "alice", "#ops", "check build", event_name, action_prompt="check build"
        )()

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["memories"] == []
        assert kwargs["user_instruction"] is None
        memories.assert_not_called()
        plugin.db.get_instruction.assert_not_called()

    def test_action_delivery_invokes_assistant_with_full_callbacks(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN action reminder WHEN delivered THEN assistant_completion receives full callback set."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]

        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        mocker.patch.object(
            plugin,
            "_gather_history",
            return_value=([{"role": "user", "content": "hi"}], [{"role": "user", "content": "c"}]),
        )
        mocker.patch.object(plugin, "_get_user_memories", return_value=["prefers concise"])
        plugin.db.get_instruction.return_value = "Be direct."
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="Build is green.", spend=Spend.ZERO
        )

        event_name = "llm_remind_action_1"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="check build",
                action_prompt="check build",
                account="acct",
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            event_name,
            action_prompt="check build",
            account="acct",
        )
        deliver()

        plugin.llm_service.assistant_completion.assert_called_once()
        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        # The model is told this turn IS the reminder firing, so "check if
        # you reminded me" gets "yes, this is it" instead of a task list.
        assert kwargs["prompt"].endswith("\n\ncheck build")
        assert "alice's reminder" in kwargs["prompt"]
        assert "firing now" in kwargs["prompt"]
        assert kwargs["entry_route"] == "remind_action"
        assert kwargs["route_profile"] == "remind_action"
        # Authenticated, so the storage key is the account.
        assert kwargs["nick"] == "acct"
        assert kwargs["account"] == "acct"
        assert kwargs["channel"] == "#ops"
        assert kwargs["is_owner"] is False
        # llm.ask/draw/code only — no admin/owner capabilities at fire time.
        assert kwargs["capabilities"] == frozenset({"llm.ask", "llm.draw", "llm.code"})

        synthetic_msg = kwargs["msg"]
        assert synthetic_msg.prefix == "alice!~remind@scheduled"
        assert synthetic_msg.command == "PRIVMSG"
        assert synthetic_msg.args == ("#ops", "")
        assert synthetic_msg.server_tags == {"account": "acct"}

        assert kwargs["history"] == [{"role": "user", "content": "hi"}]
        assert kwargs["channel_history"] == [{"role": "user", "content": "c"}]
        assert kwargs["memories"] == ["prefers concise"]
        # The @instruct now rides as user-role data, not baked into the system
        # overlay (which stays the channel personality prompt).
        assert kwargs["system_prompt"] == "You are helpful."
        assert kwargs["user_instruction"] == "Be direct."
        assert callable(kwargs["search_fn"])
        assert callable(kwargs["fetch_fn"])
        assert callable(kwargs["code_fn"])
        assert callable(kwargs["draw_fn"])
        assert callable(kwargs["cleanup_fn"])
        assert callable(kwargs["list_pending_tasks_fn"])
        assert callable(kwargs["set_reminder_fn"])
        assert callable(kwargs["cancel_pending_task_fn"])
        assert callable(kwargs["cancel_all_pending_tasks_fn"])

        sent = active_irc.queueMsg.call_args[0][0]
        assert sent.args[0] == "#ops"
        assert sent.args[1] == "alice: Reminder (check build): Build is green."

    def test_action_delivery_uses_ask_rate_limit_bucket(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN action reminder WHEN fired THEN uses ask bucket with unregistered fallback tier."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]

        rl_spy = mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="done", spend=Spend.ZERO
        )

        event_name = "llm_remind_action_2"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="check build",
                action_prompt="check build",
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            event_name,
            action_prompt="check build",
            account=None,
        )
        deliver()

        rl_spy.assert_called_once()
        args = rl_spy.call_args.args
        kwargs = rl_spy.call_args.kwargs
        assert args[0] is None
        assert args[1] == "ask"
        assert args[2] == "alice"
        assert kwargs["tier"] == "unregistered"
        assert kwargs["silent"] is True
        assert isinstance(kwargs["now"], float)

    def test_action_delivery_falls_back_on_rate_limit(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN over-limit action reminder WHEN delivered THEN falls back to echo text."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=True)

        event_name = "llm_remind_action_3"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="check build",
                action_prompt="check build",
                account="acct",
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            event_name,
            action_prompt="check build",
            account="acct",
        )
        deliver()

        plugin.llm_service.assistant_completion.assert_not_called()
        sent = active_irc.queueMsg.call_args[0][0]
        assert (
            sent.args[1]
            == "alice: Reminder: check build (action skipped — daily ask limit reached)"
        )

    def test_action_delivery_falls_back_on_exception(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN assistant error WHEN action reminder fires THEN sends generic retry fallback."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]

        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        plugin.llm_service.assistant_completion.side_effect = RuntimeError("boom secret text")

        event_name = "llm_remind_action_4"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="check build",
                action_prompt="check build",
                account="acct",
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            event_name,
            action_prompt="check build",
            account="acct",
        )
        deliver()

        sent = active_irc.queueMsg.call_args[0][0]
        assert (
            sent.args[1]
            == "alice: Reminder: Action 'check build' failed. (Set this reminder again to retry.)"
        )
        assert "secret text" not in sent.args[1]
        plugin.log.exception.assert_called_once()

    def test_action_delivery_threads_under_the_request_and_drops_the_label(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a msgid and message-tags WHEN an action reminder fires THEN it replies to the request instead of restating it."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        active_irc.state.capabilities_ack = {"message-tags"}
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="all green", spend=Spend.ZERO
        )

        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            "llm_remind_thread_1",
            action_prompt="check build",
            account="acct",
            reply_msgid="abc123",
        )
        deliver()

        sent = active_irc.queueMsg.call_args[0][0]
        assert sent.server_tags["+draft/reply"] == "abc123"
        # The client renders the original request, so restating it is noise.
        assert sent.args[1] == "alice: all green"

    def test_action_delivery_keeps_the_label_when_it_cannot_thread(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN no message-tags WHEN an action reminder fires THEN the labelled text is kept and no tag is sent."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        active_irc.state.capabilities_ack = set()
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="all green", spend=Spend.ZERO
        )

        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            "llm_remind_thread_2",
            action_prompt="check build",
            account="acct",
            reply_msgid="abc123",
        )
        deliver()

        sent = active_irc.queueMsg.call_args[0][0]
        # An unattached line has to say what it is answering.
        assert sent.args[1] == "alice: Reminder (check build): all green"
        assert "+draft/reply" not in (sent.server_tags or {})

    def test_long_action_reply_is_pastebinned_not_truncated(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a ~900-char action result WHEN the reminder fires THEN it goes
        out as a teaser + URL instead of one PRIVMSG the server cuts off."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        active_irc.state.capabilities_ack = set()
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        long_reply = " ".join(f"word{i}" for i in range(150))  # ~900 chars
        assert len(long_reply) > 850
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content=long_reply, spend=Spend.ZERO
        )
        plugin.llm_service.save_markdown_to_http.return_value = "http://h/remind.html"

        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            "llm_remind_long_1",
            action_prompt="check build",
            account="acct",
        )
        deliver()

        sent = active_irc.queueMsg.call_args[0][0]
        assert sent.args[1].startswith("alice: ")
        assert "http://h/remind.html" in sent.args[1]
        assert "word149" not in sent.args[1]
        assert long_reply in plugin.llm_service.save_markdown_to_http.call_args.args[0]

    def test_action_delivery_falls_back_on_pre_request_exception(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN exception BEFORE assistant_completion WHEN action reminder fires THEN fallback sent.

        Regression: a narrow try/except around only assistant_completion would
        let history/registry/IrcMsg/rate-limit failures silently lose the
        reminder. The handler must wrap the whole per-irc action body.
        """
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]

        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        # _gather_history runs BEFORE assistant_completion — make it raise.
        plugin._gather_history = mocker.MagicMock(
            side_effect=RuntimeError("internal token leak XYZ")
        )

        event_name = "llm_remind_action_pre"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="check build",
                action_prompt="check build",
                account="acct",
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            event_name,
            action_prompt="check build",
            account="acct",
        )
        deliver()

        plugin.llm_service.assistant_completion.assert_not_called()
        sent = active_irc.queueMsg.call_args[0][0]
        assert (
            sent.args[1]
            == "alice: Reminder: Action 'check build' failed. (Set this reminder again to retry.)"
        )
        assert "XYZ" not in sent.args[1]
        plugin.log.exception.assert_called_once()

    def test_echo_delivery_unchanged(self, plugin: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN non-action reminder WHEN delivered THEN legacy echo path is unchanged."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]

        rl_spy = mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        event_name = "llm_remind_echo_1"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="ping me",
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "ping me",
            event_name,
            action_prompt="",
            account=None,
        )
        deliver()

        plugin.llm_service.assistant_completion.assert_not_called()
        rl_spy.assert_not_called()
        sent = active_irc.queueMsg.call_args[0][0]
        assert sent.args[1] == "alice: Reminder: ping me"

    def test_action_delivery_logs_usage_to_chain_owner(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN successful action fire WHEN delivered THEN log_usage is called under account."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="status: green",
            model="xai/grok-4-1-fast",
            spend=Spend(prompt_tokens=42, completion_tokens=8, cost=0.0001),
        )

        event_name = "llm_remind_action_usage"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="check build",
                action_prompt="check build",
                account="alice-acct",
                chain_position=1,
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "check build",
            event_name,
            action_prompt="check build",
            account="alice-acct",
        )
        deliver()

        plugin.db.log_usage.assert_called_once()
        args = plugin.db.log_usage.call_args.args
        kwargs = plugin.db.log_usage.call_args.kwargs
        assert args[0] == "alice-acct"  # owner_key — account preferred over nick
        assert args[2] == "remind_action"
        assert args[3] == "xai/grok-4-1-fast"
        assert args[4] == 42
        assert args[5] == 8
        assert kwargs["status"] == "success"

    def test_action_delivery_silent_sentinel_skips_send_but_logs_usage(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN [silent] response WHEN delivered THEN no IRC msg sent but usage still logged."""
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="[silent]",
            model="xai/grok-4-1-fast",
            spend=Spend(prompt_tokens=30, completion_tokens=2, cost=0.00005),
        )

        event_name = "llm_remind_action_silent"
        put_reminder(
            plugin,
            make_reminder_row(
                event_name=event_name,
                nick="alice",
                channel="#ops",
                message="watch CVE",
                action_prompt="check CVE (watch — only respond on positive result)",
                account="alice-acct",
                chain_position=1,
            ),
        )
        deliver = due_reminder(
            plugin,
            "alice",
            "#ops",
            "watch CVE",
            event_name,
            action_prompt="check CVE (watch — only respond on positive result)",
            account="alice-acct",
        )
        deliver()

        # No user-visible message — silent fires only react in the log.
        active_irc.queueMsg.assert_not_called()
        # Usage IS logged so cumulative cost stays visible.
        plugin.db.log_usage.assert_called_once()
        kwargs = plugin.db.log_usage.call_args.kwargs
        assert kwargs["status"] == "silent"

    def test_check_rate_limit_silent_enforced_blocks(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN over-limit and enforce=true WHEN _check_rate_limit silent THEN returns True."""
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "askRateLimitCount": 1,
                "askRateLimitWindow": 60,
                "enforceRateLimits": True,
            }.get(key, "")
        )
        now = 1000.0
        plugin._record_rate_limit_hit("ask", "alice", now - 1)

        blocked = plugin._check_rate_limit(
            None,
            "ask",
            "alice",
            "",
            "",
            "",
            tier="registered",
            silent=True,
            now=now,
        )

        assert blocked is True
        assert len(plugin._rate_buckets["ask:alice"]) == 2
        plugin.db.log_usage.assert_not_called()
        assert "rate_limited" in plugin.log.info.call_args.args[0]

    def test_check_rate_limit_silent_shadow_mode(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN over-limit and enforce=false WHEN _check_rate_limit silent THEN logs shadow and allows."""
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: {
                "askRateLimitCount": 1,
                "askRateLimitWindow": 60,
                "enforceRateLimits": False,
            }.get(key, "")
        )
        now = 1000.0
        plugin._record_rate_limit_hit("ask", "alice", now - 1)

        blocked = plugin._check_rate_limit(
            None,
            "ask",
            "alice",
            "",
            "",
            "",
            tier="registered",
            silent=True,
            now=now,
        )

        assert blocked is False
        assert len(plugin._rate_buckets["ask:alice"]) == 2
        plugin.db.log_usage.assert_not_called()
        assert "rate_limit_shadow" in plugin.log.info.call_args.args[0]


class TestRoutingGateAndMechanicalReschedule:
    """Tests for B3.5 strict transition gate + B4 mechanical reschedule."""

    @pytest.fixture
    def plugin(self, mock_irc: MagicMock, mocker: MockerFixture, tmp_path) -> MagicMock:
        """Create a plugin instance with action delivery wiring."""
        from llm.plugin import LLM

        from .conftest import make_registry_side_effect, plugin_init_patches

        mocker.patch.object(LLM, "registryValue", side_effect=make_registry_side_effect())
        plugin_init_patches(mocker)
        plugin = LLM(mock_irc)
        back_reminders_with_sqlite(plugin.db, tmp_path / "reminders.db")
        plugin.llm_service.sanitize_output.side_effect = lambda x: x
        return plugin

    def _wire_active_irc(self, plugin: MagicMock, mocker: MockerFixture) -> MagicMock:
        mock_world = mocker.patch("llm.plugin.world")
        active_irc = mocker.MagicMock()
        active_irc.nick = "testbot"
        mock_world.ircs = [active_irc]
        mocker.patch.object(plugin, "_check_rate_limit", return_value=False)
        mocker.patch.object(plugin, "_gather_history", return_value=([], []))
        mocker.patch.object(plugin, "_get_user_memories", return_value=[])
        plugin.db.get_instruction.return_value = None
        return active_irc

    def _fire(self, plugin: MagicMock, event_name: str, **fields) -> None:
        """Store an action reminder due now and run the agenda wakeup."""
        fields.setdefault("account", "acct")
        due_reminder(plugin, "alice", "#ops", "watch", event_name, **fields)()

    def test_structured_row_excludes_set_reminder_and_reschedules_in_place(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN a recurring row WHEN fired THEN set_reminder is off the tool
        surface and the row moves to its next fire, keeping its id."""
        self._wire_active_irc(plugin, mocker)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="ok", spend=Spend.ZERO
        )
        add_event = mocker.patch("llm.plugin.schedule.addEvent")
        before = time.time()

        self._fire(
            plugin, "llm_remind_struct_1", action_prompt="check build", recurrence_seconds=300
        )

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["exclude_tools"] == frozenset({"set_reminder"})
        row = plugin.db.sqlite_reminders.get_reminder("llm_remind_struct_1")
        assert row is not None
        assert row.chain_position == 2
        assert before + 300 <= row.fire_at <= time.time() + 300
        plugin.db.save_reminder.assert_not_called()
        # The wakeup is re-armed for the next fire.
        assert add_event.call_args.args[1] == row.fire_at

    def test_one_shot_row_is_removed_after_its_fire(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        self._wire_active_irc(plugin, mocker)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="ok", spend=Spend.ZERO
        )

        self._fire(plugin, "llm_remind_oneshot_1", action_prompt="ping me")

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["exclude_tools"] == frozenset({"set_reminder"})
        assert reminder_names(plugin) == set()

    def test_a_firing_one_shot_does_not_list_itself(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """ibutsu 2026-09-25: "remind me to check if you reminded me" fired as
        an action, called list_pending_tasks, and found itself still pending."""
        from llm.plugin import Identity

        self._wire_active_irc(plugin, mocker)
        seen: list[set[str]] = []

        def _list_then_respond(*_a, **_kw) -> AssistantResult:
            owner = Identity(raw_nick="alice", account="acct")
            seen.append({name for name, _row in plugin._get_user_reminders(owner)})
            return AssistantResult(content="ok", spend=Spend.ZERO)

        plugin.llm_service.assistant_completion.side_effect = _list_then_respond

        self._fire(plugin, "llm_remind_self", action_prompt="check")

        assert seen == [set()]

    def test_clear_wins_over_mid_fire(self, plugin: MagicMock, mocker: MockerFixture) -> None:
        """GIVEN a recurring row cancelled while its action runs WHEN the
        action completes THEN the chain stays cancelled."""
        from llm.plugin import Identity

        self._wire_active_irc(plugin, mocker)

        def _clear_then_respond(*_a, **_kw) -> AssistantResult:
            plugin._remind_clear_for_assistant(Identity(raw_nick="alice", account="acct"))
            return AssistantResult(content="ok", spend=Spend.ZERO)

        plugin.llm_service.assistant_completion.side_effect = _clear_then_respond

        self._fire(plugin, "llm_remind_clearwin_1", action_prompt="watch X", recurrence_seconds=600)

        assert reminder_names(plugin) == set()
        plugin.db.update_reminder_fire_at.assert_not_called()

    def test_chain_position_cap_ends_the_chain(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        from llm.agenda import MAX_CHAIN_POSITION, Kind

        self._wire_active_irc(plugin, mocker)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="ok", spend=Spend.ZERO
        )

        self._fire(
            plugin,
            "llm_remind_cap_1",
            action_prompt="capped",
            chain_position=MAX_CHAIN_POSITION[Kind.REMINDER],
            recurrence_seconds=120,
        )

        assert reminder_names(plugin) == set()
        plugin.db.update_reminder_fire_at.assert_not_called()

    def test_structured_watch_mode_silent_response_still_reschedules(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """GIVEN structured watch row WHEN response is [silent] THEN watch keeps watching."""
        active_irc = self._wire_active_irc(plugin, mocker)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="[silent]", spend=Spend.ZERO
        )

        self._fire(
            plugin,
            "llm_remind_watch_1",
            action_prompt="check CVE status",
            recurrence_seconds=900,
            watch_mode=True,
        )

        # No user-visible message — silent fire.
        active_irc.queueMsg.assert_not_called()
        # But the watch keeps watching, on the same row.
        row = plugin.db.sqlite_reminders.get_reminder("llm_remind_watch_1")
        assert row is not None and row.watch_mode is True and row.chain_position == 2

    def test_recurring_fire_keeps_threading_under_the_request(
        self, plugin: MagicMock, mocker: MockerFixture
    ) -> None:
        """The row keeps the setter's msgid across fires, so a later fire
        still threads under the request and a cancel still takes back the ⏰."""
        self._wire_active_irc(plugin, mocker)
        plugin.llm_service.assistant_completion.return_value = AssistantResult(
            content="ok", spend=Spend.ZERO
        )

        self._fire(
            plugin,
            "llm_remind_thread",
            action_prompt="check",
            recurrence_seconds=600,
            reply_msgid="m1",
        )
        plugin._cancel_reminder("llm_remind_thread")

        args = plugin.llm_service.send_reaction.call_args.args
        assert args[2:] == ("m1", "⏰")
