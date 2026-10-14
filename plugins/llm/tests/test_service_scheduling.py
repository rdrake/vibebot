"""Service scheduling: schedule_llm_task, reply-target resolution, scheduled-fire (local db/llm_service fixtures + helpers)."""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

import pytest
from llm.service import LLMService, ReminderParseResult

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


@pytest.fixture
def db(test_db):
    return test_db


@pytest.fixture
def llm_service(make_service, db):
    service, plugin = make_service()
    plugin.db = db
    return service


def _msg_mock(mocker: MockerFixture, *, depth: int | None = None):
    msg = mocker.MagicMock()
    msg.__str__ = lambda self: ":rdrake!u@h PRIVMSG #t :@ask hi"
    msg.nick = "rdrake"
    msg.args = ("#t", "@ask hi")
    msg.tagged.side_effect = lambda key: depth if key == "llm_schedule_depth" else None
    return msg


def _irc_mock(mocker: MockerFixture):
    irc = mocker.MagicMock()
    irc.network = "afternet"
    return irc


def test_schedule_llm_task_creates_db_row_and_schedules_event(
    llm_service, db, mocker: MockerFixture
):
    """B1: a one-shot schedule writes a DB row and registers the event with
    supybot.schedule.addEvent."""
    add_event = mocker.patch("llm.service.schedule.addEvent")

    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)

    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=60,
            message="check build",
            confirmation="ok",
            note=None,
            action_prompt="check the build",
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#test",
        when_natural="in 60s",
        prompt="check the build",
    )

    assert result.status == "ok"
    assert result.event_name.startswith("llm_task_")
    rows = db.load_active_scheduled_llm_tasks()
    [row] = [r for r in rows if r.event_name == result.event_name]
    assert row.creator_nick == "rdrake"
    assert row.account == "rdrake_a"
    assert row.channel == "#test"
    assert row.network == "afternet"
    assert row.prompt == "check the build"
    assert row.recurrence_seconds is None
    assert row.recurrence_rrule is None

    add_event.assert_called_once()
    args = add_event.call_args
    callback = args[0][0]
    fire_at = args[0][1]
    name_kwarg = args.kwargs.get("name") or args[0][2]
    assert callable(callback)
    assert name_kwarg == result.event_name
    assert fire_at == pytest.approx(_time.time() + 60, abs=2)


def test_schedule_llm_task_recurrence_seconds(llm_service, db, mocker: MockerFixture):
    """B1: numeric-cadence recurrence stores recurrence_seconds and schedules
    the FIRST fire at parser.seconds."""
    mocker.patch("llm.service.schedule.addEvent")
    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)
    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=60,
            message="ping me",
            confirmation="ok",
            note=None,
            action_prompt="ping me",
            recurrence_seconds=300,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )
    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="n",
        account="acct",
        channel="#t",
        when_natural="every 5 minutes",
        prompt="ping me",
    )
    assert result.status == "ok"
    [row] = [r for r in db.load_active_scheduled_llm_tasks() if r.event_name == result.event_name]
    assert row.recurrence_seconds == 300
    assert row.recurrence_rrule is None


def test_schedule_llm_task_recurrence_rrule(llm_service, db, mocker: MockerFixture):
    """B1: RRULE recurrence stored as-is; recurrence_seconds remains null."""
    mocker.patch("llm.service.schedule.addEvent")
    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)
    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=3600,
            message="weekly",
            confirmation="ok",
            note=None,
            action_prompt="post the weekly summary",
            recurrence_seconds=None,
            recurrence_rrule="FREQ=WEEKLY;BYDAY=MO;BYHOUR=9",
            watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )
    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="n",
        account="acct",
        channel="#t",
        when_natural="every Monday at 9am",
        prompt="post the weekly summary",
    )
    assert result.status == "ok"
    [row] = [r for r in db.load_active_scheduled_llm_tasks() if r.event_name == result.event_name]
    assert row.recurrence_rrule.startswith("FREQ=WEEKLY")


def test_schedule_llm_task_refuses_when_depth_tag_set(llm_service, mocker: MockerFixture):
    """B1 + D4: a fired task can't recursively call schedule_llm_task."""
    msg = _msg_mock(mocker, depth=1)
    irc = _irc_mock(mocker)
    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="n",
        account=None,
        channel="#t",
        when_natural="in 1m",
        prompt="do something else",
    )
    assert result.status == "error"
    assert "depth" in result.message.lower() or "scheduled" in result.message.lower()


def test_schedule_llm_task_enforces_per_creator_limit(llm_service, db, mocker: MockerFixture):
    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)
    for i in range(5):
        db.save_scheduled_llm_task(
            event_name=f"existing_{i}",
            creator_nick="n",
            account="a",
            channel="#t",
            network="afternet",
            wire_msg=":n!u@h PRIVMSG #t :@ask hi",
            prompt="p",
            fire_at=_time.time() + 60,
        )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=60,
            message="x",
            confirmation="ok",
            note=None,
            action_prompt="x",
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )

    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="n",
        account="a",
        channel="#t",
        when_natural="in 1m",
        prompt="do x",
    )
    assert result.status == "error"
    assert "limit" in result.message.lower()


def test_schedule_llm_task_limit_zero_disables_scheduling(llm_service, mocker: MockerFixture):
    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        0 if k == "bridgeScheduledTaskLimit" else None
    )

    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="n",
        account="a",
        channel="#t",
        when_natural="in 1m",
        prompt="do x",
    )

    assert result.status == "error"
    assert "disabled" in result.message.lower()


def test_schedule_llm_task_clarify_returns_clarify_envelope(llm_service, mocker: MockerFixture):
    """When parse_reminder returns action='clarify', schedule_llm_task surfaces
    the parser's clarification text instead of scheduling."""
    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)
    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="clarify",
            confirmation="When should I run that?",
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="n",
        account="acct",
        channel="#t",
        when_natural="vague request",
        prompt="some action",
    )
    assert result.status == "clarify"
    assert "When should I run that?" in result.message


def test_schedule_llm_task_requires_account(llm_service, mocker: MockerFixture):
    """schedule_llm_task refuses unauthenticated callers (defense in depth)."""
    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )
    result = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="anon",
        account=None,
        channel="#t",
        when_natural="in 1m",
        prompt="do x",
    )
    assert result.status == "error"
    assert "account" in result.message.lower() or "auth" in result.message.lower()


# =============================================================================
# Phase 2 Task 3 / B2 — list + cancel scheduled_llm_task service methods
# =============================================================================


def test_list_scheduled_llm_tasks_filters_by_owner(llm_service, db):
    """B2: list returns only the caller's active tasks. Match policy:
    account-when-account, nick-when-no-account (mirrors reminders)."""
    db.save_scheduled_llm_task(
        event_name="ev1",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=_time.time() + 60,
    )
    db.save_scheduled_llm_task(
        event_name="ev2",
        creator_nick="rdrake_alt",
        account="rdrake_a",
        channel="#t",
        network="afternet",
        wire_msg=":rdrake_alt!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=_time.time() + 600,
    )
    db.save_scheduled_llm_task(
        event_name="other",
        creator_nick="other_user",
        account="other_a",
        channel="#t",
        network="afternet",
        wire_msg=":other!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=_time.time() + 600,
    )

    rows = llm_service.list_scheduled_llm_tasks(creator_nick="rdrake", account="rdrake_a")
    names = {r.event_name for r in rows}
    assert names == {"ev1", "ev2"}


def test_cancel_scheduled_llm_task_owner_scoped(llm_service, db, mocker: MockerFixture):
    """B2: cancelling your own task removes it; cancelling someone else's refuses."""
    remove_event = mocker.patch("llm.service.schedule.removeEvent")
    db.save_scheduled_llm_task(
        event_name="mine",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=_time.time() + 60,
    )
    db.save_scheduled_llm_task(
        event_name="theirs",
        creator_nick="other",
        account="other_a",
        channel="#t",
        network="afternet",
        wire_msg=":other!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=_time.time() + 60,
    )

    ok = llm_service.cancel_scheduled_llm_task(
        event_name="mine", creator_nick="rdrake", account="rdrake_a"
    )
    assert ok.status == "ok"
    # Cancel deleted the row, so a follow-up delete returns False.
    assert db.delete_scheduled_llm_task("mine") is False
    remove_event.assert_called_once_with("mine")

    remove_event.reset_mock()
    foreign = llm_service.cancel_scheduled_llm_task(
        event_name="theirs", creator_nick="rdrake", account="rdrake_a"
    )
    assert foreign.status == "error"
    remove_event.assert_not_called()


def test_cancel_scheduled_llm_task_refuses_unidentified_nick_squatter(
    llm_service, db, mocker: MockerFixture
):
    """An accountless caller cannot cancel a task by holding the owner's nick.

    Scheduling refuses without an account, so every row carries one. A
    nick-fallback owner check would let anyone who takes the owner's nick
    while unidentified destroy a task they cannot even list, because
    ``load_scheduled_llm_tasks_for`` scopes an accountless caller to rows
    with a NULL account.
    """
    remove_event = mocker.patch("llm.service.schedule.removeEvent")
    db.save_scheduled_llm_task(
        event_name="alices",
        creator_nick="alice",
        account="ALICE",
        channel="#t",
        network="afternet",
        wire_msg=":alice!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=_time.time() + 60,
    )

    squatter = llm_service.cancel_scheduled_llm_task(
        event_name="alices", creator_nick="alice", account=None
    )

    assert squatter.status == "error"
    assert "someone else" in squatter.message
    remove_event.assert_not_called()
    # The row survives: delete_scheduled_llm_task returns True only if it
    # actually removed something.
    assert db.delete_scheduled_llm_task("alices") is True


def test_cancel_scheduled_llm_task_unknown_returns_error(llm_service):
    out = llm_service.cancel_scheduled_llm_task(
        event_name="does_not_exist", creator_nick="x", account=None
    )
    assert out.status == "error"


# =============================================================================
# Phase 2 Task 3 / B3 — restore_scheduled_llm_tasks
# =============================================================================


def test_restore_scheduled_llm_tasks_reregisters_events(llm_service, db, mocker: MockerFixture):
    """B3: restore reads active rows, registers each with schedule.addEvent;
    overdue rows fire ~immediately."""
    add_event = mocker.patch("llm.service.schedule.addEvent")
    now = _time.time()
    db.save_scheduled_llm_task(
        event_name="future_ev",
        creator_nick="n",
        account=None,
        channel="#t",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=now + 600,
    )
    db.save_scheduled_llm_task(
        event_name="overdue_ev",
        creator_nick="n",
        account=None,
        channel="#t",
        network="afternet",
        wire_msg=":n!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=now - 60,
    )

    restored, skipped = llm_service.restore_scheduled_llm_tasks()
    assert restored == 2
    assert skipped == 0

    names = set()
    for call in add_event.call_args_list:
        name = call.kwargs.get("name") or call.args[2]
        names.add(name)
    assert names == {"future_ev", "overdue_ev"}

    # Overdue events fire ~immediately (clamped to now+1).
    for call in add_event.call_args_list:
        name = call.kwargs.get("name") or call.args[2]
        fire_at = call.args[1]
        if name == "overdue_ev":
            assert fire_at <= now + 5


# =============================================================================
# Phase 2 Task 3 / D4 — depth-cap end-to-end on fired schedule
# =============================================================================


def test_fired_task_cannot_schedule_a_nested_task(llm_service, db, mocker: MockerFixture):
    """End-to-end: schedule a task; trigger the fire callback; observe that
    within the fired @ask, schedule_llm_task refuses (llm_schedule_depth=1)."""
    add_event = mocker.patch("llm.service.schedule.addEvent")
    msg = _msg_mock(mocker)
    irc = _irc_mock(mocker)
    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=60,
            message="x",
            confirmation="ok",
            note=None,
            action_prompt="check the build",
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
        5 if k == "bridgeScheduledTaskLimit" else None
    )

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="do x",
    )
    assert res.status == "ok"

    # Capture the registered closure so we can fire it manually.
    fire_callable = add_event.call_args.args[0]

    fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
    fake_world.getIrc.return_value = irc
    fake_world.ircs = [irc]

    mocker.patch("llm.service.ircdb.checkCapability", return_value=True)

    from llm.plugin import LLM

    plugin = llm_service.plugin
    plugin._check_rate_limit.return_value = False
    plugin._gather_history.return_value = ([], [])
    plugin._get_user_memories.return_value = []
    mocker.patch.object(plugin.db, "get_instruction", return_value="")
    plugin._pending_task_fns.return_value = {}
    # The fire plumbing lives in plugin._run_unattended_assistant (shared
    # with reminder action fires); bind the real helpers to the mock plugin
    # so the dispatch path reaches the patched assistant_request.
    plugin.llm_service = llm_service
    plugin._unattended_ask_rate_limited = LLM._unattended_ask_rate_limited.__get__(plugin)
    plugin._run_unattended_assistant = LLM._run_unattended_assistant.__get__(plugin)

    captured: dict[str, object] = {}

    def fake_assistant_request(*, msg, **_kwargs):
        captured["depth"] = msg.tagged("llm_schedule_depth")
        nested = llm_service.schedule_llm_task(
            irc=irc,
            msg=msg,
            creator_nick="rdrake",
            account="rdrake_a",
            channel="#t",
            when_natural="in 60s",
            prompt="do y",
        )
        captured["nested_status"] = nested.status
        captured["nested_message"] = nested.message
        return mocker.MagicMock(
            content="",
            model="m",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            error=None,
        )

    mocker.patch.object(llm_service, "assistant_request", side_effect=fake_assistant_request)

    fire_callable()

    assert captured["depth"] == 1
    assert captured["nested_status"] == "error"
    msg_lower = str(captured["nested_message"]).lower()
    assert "depth" in msg_lower or "scheduled" in msg_lower


# =============================================================================
# Phase 2 follow-up B — schedule_llm_task reply_target override
# =============================================================================


class _FakeChannelState:
    def __init__(self, users):
        self.users = set(users)


def _irc_with_channels(mocker: MockerFixture, channels: dict[str, list[str]]):
    irc = _irc_mock(mocker)
    irc.state = mocker.MagicMock()
    irc.state.channels = {name: _FakeChannelState(users) for name, users in channels.items()}
    return irc


def _registry(values):
    """Build a side_effect that returns dict-driven registryValue results."""

    def _lookup(key, ch=None):
        if key == "bridgeScheduledTaskLimit":
            return values.get("bridgeScheduledTaskLimit", 5)
        if key == "bridgeEnabled":
            return values.get(("bridgeEnabled", ch), False)
        if key == "commandPrefixes":
            return values.get("commandPrefixes", "@")
        return values.get(key)

    return _lookup


def _patch_parser(llm_service, mocker: MockerFixture):
    mocker.patch.object(
        llm_service,
        "parse_reminder",
        return_value=ReminderParseResult(
            action="schedule",
            seconds=60,
            message="x",
            confirmation="ok",
            note=None,
            action_prompt="x",
            recurrence_seconds=None,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )


def test_reply_target_channel_membership_ok_persists_override(
    llm_service, db, mocker: MockerFixture
):
    """Cross-channel target where bot+creator are present and bridge is enabled
    persists `reply_target` on the row."""
    mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"], "#t": ["rdrake"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
            "bridgeScheduledTaskLimit": 5,
            ("bridgeEnabled", "#deliver"): True,
        }
    )

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="check the build",
        reply_target="#deliver",
    )
    assert res.status == "ok"
    [row] = [r for r in db.load_active_scheduled_llm_tasks() if r.event_name == res.event_name]
    assert row.reply_target == "#deliver"


def test_reply_target_channel_bot_absent_refused(llm_service, mocker: MockerFixture):
    mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#t": ["rdrake"]})  # bot not in #other
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
            "bridgeScheduledTaskLimit": 5,
        }
    )

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
        reply_target="#other",
    )
    assert res.status == "error"
    assert "not in that channel" in res.message


def test_reply_target_channel_creator_absent_refused(llm_service, mocker: MockerFixture):
    mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#deliver": ["bot"], "#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
            "bridgeScheduledTaskLimit": 5,
            ("bridgeEnabled", "#deliver"): True,
        }
    )

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
        reply_target="#deliver",
    )
    assert res.status == "error"
    assert "you are not in that channel" in res.message


def test_reply_target_channel_bridge_disabled_refused(llm_service, mocker: MockerFixture):
    mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"], "#t": ["rdrake"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
            "bridgeScheduledTaskLimit": 5,
            ("bridgeEnabled", "#deliver"): False,
        }
    )

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
        reply_target="#deliver",
    )
    assert res.status == "error"
    assert "bridge is not enabled" in res.message


def test_reply_target_pm_self_ok(llm_service, db, mocker: MockerFixture):
    mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({"bridgeScheduledTaskLimit": 5})

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
        reply_target="rdrake",
    )
    assert res.status == "ok"
    [row] = [r for r in db.load_active_scheduled_llm_tasks() if r.event_name == res.event_name]
    assert row.reply_target == "rdrake"


def test_reply_target_pm_other_refused(llm_service, mocker: MockerFixture):
    mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({"bridgeScheduledTaskLimit": 5})

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
        reply_target="someone_else",
    )
    assert res.status == "error"
    assert "your own nick" in res.message


def test_reply_target_overrides_dispatch_target(llm_service, db, mocker: MockerFixture):
    """At fire time the privmsg goes to row.reply_target, not row.channel."""
    add_event = mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"], "#origin": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
            "bridgeScheduledTaskLimit": 5,
            ("bridgeEnabled", "#deliver"): True,
        }
    )

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#origin",
        when_natural="in 60s",
        prompt="say hi",
        reply_target="#deliver",
    )
    assert res.status == "ok"

    fire_callable = add_event.call_args.args[0]
    fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
    fake_world.getIrc.return_value = irc
    fake_world.ircs = [irc]
    mocker.patch("llm.service.ircdb.checkCapability", return_value=True)

    plugin = llm_service.plugin
    plugin._check_rate_limit.return_value = False
    plugin._gather_history.return_value = ([], [])
    plugin._get_user_memories.return_value = []
    mocker.patch.object(plugin.db, "get_instruction", return_value="")
    plugin._pending_task_fns.return_value = {}

    # Run the real line formatting so a real PRIVMSG (not a mock) is queued.
    _bind_real_delivery(plugin)

    mocker.patch.object(
        llm_service,
        "assistant_request",
        return_value=mocker.MagicMock(
            content="hi from the future",
            model="m",
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            error=None,
        ),
    )

    fire_callable()

    privmsg_calls = [
        call
        for call in irc.queueMsg.call_args_list
        if getattr(call.args[0], "command", None) == "PRIVMSG"
    ]
    assert privmsg_calls, "expected at least one PRIVMSG queued"
    assert privmsg_calls[-1].args[0].args[0] == "#deliver"


def _bind_real_delivery(plugin) -> None:
    """Run the real one-line formatter (collapse, pastebin, safeArgument)."""
    from llm.plugin import LLM

    plugin._collapse_for_irc.side_effect = LLM._collapse_for_irc
    plugin._safe_privmsg.side_effect = LLM._safe_privmsg
    plugin._fallback_long_reply_teaser.side_effect = LLM._fallback_long_reply_teaser
    plugin._reply_mores_length.return_value = 400
    plugin._finish_irc_line = LLM._finish_irc_line.__get__(plugin)
    plugin._format_pending_completed_reply = LLM._format_pending_completed_reply.__get__(plugin)


def _fire_with_answer(llm_service, mocker: MockerFixture, content: str):
    """Schedule a one-shot task in #deliver, fire it with ``content``, return the irc."""
    add_event = mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"]})
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {"bridgeScheduledTaskLimit": 5, ("bridgeEnabled", "#deliver"): True}
    )
    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=_msg_mock(mocker),
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#deliver",
        when_natural="in 60s",
        prompt="say hi",
    )
    assert res.status == "ok"
    fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
    fake_world.getIrc.return_value = irc
    fake_world.ircs = [irc]
    mocker.patch("llm.service.ircdb.checkCapability", return_value=True)
    plugin = llm_service.plugin
    plugin._check_rate_limit.return_value = False
    plugin._gather_history.return_value = ([], [])
    plugin._get_user_memories.return_value = []
    mocker.patch.object(plugin.db, "get_instruction", return_value="")
    plugin._pending_task_fns.return_value = {}
    plugin.llm_service = llm_service
    from llm.plugin import LLM

    plugin._unattended_ask_rate_limited = LLM._unattended_ask_rate_limited.__get__(plugin)
    plugin._run_unattended_assistant = LLM._run_unattended_assistant.__get__(plugin)
    _bind_real_delivery(plugin)
    mocker.patch.object(
        llm_service,
        "assistant_request",
        return_value=mocker.MagicMock(
            content=content, model="m", prompt_tokens=0, completion_tokens=0, cost=0.0, error=None
        ),
    )
    add_event.call_args.args[0]()
    return irc


def test_long_fired_answer_is_pastebinned_not_truncated(llm_service, db, mocker: MockerFixture):
    """A ~900-char scheduled answer goes out as teaser + URL, not one PRIVMSG
    the server cuts off."""
    mocker.patch.object(llm_service, "save_markdown_to_http", return_value="http://h/task.html")
    long_answer = " ".join(f"word{i}" for i in range(150))
    assert len(long_answer) > 850

    irc = _fire_with_answer(llm_service, mocker, long_answer)

    body = irc.queueMsg.call_args.args[0].args[1]
    assert body.startswith("rdrake: ")
    assert "http://h/task.html" in body
    assert "word149" not in body


def test_fired_answer_neutralizes_irc_injection(llm_service, db, mocker: MockerFixture):
    """A scheduled answer carrying CR/LF/NUL must not smuggle an IRC command.

    The fired-answer path (service.py) sends raw ``ircmsgs.privmsg`` via
    ``_safe_queue``, bypassing the ``safeArgument`` that ``irc.reply`` applies.
    Model output is attacker-influenceable, so the queued body must be free of
    the bytes that frame a second IRC line (\\r, \\n, \\x00).
    """
    irc = _fire_with_answer(llm_service, mocker, "hi from the future\r\nQUIT :pwned\x00")

    body = irc.queueMsg.call_args.args[0].args[1]
    assert body.startswith("rdrake: hi from the future")
    assert "\r" not in body
    assert "\n" not in body
    assert "\x00" not in body


# =============================================================================
# Phase 2 follow-up C — auto-cancel on capability revoke
# =============================================================================


def test_fire_auto_cancels_when_creator_lost_llm_ask(llm_service, db, mocker: MockerFixture):
    """If the creator no longer holds llm.ask at fire time the row is deleted
    and assistant_request is never called."""
    add_event = mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({"bridgeScheduledTaskLimit": 5})

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
    )
    assert res.status == "ok"
    event_name = res.event_name

    fire_callable = add_event.call_args.args[0]
    fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
    fake_world.getIrc.return_value = irc
    fake_world.ircs = [irc]
    mocker.patch("llm.service.ircdb.checkCapability", return_value=False)

    assistant = mocker.patch.object(llm_service, "assistant_request")

    fire_callable()

    assert assistant.call_count == 0
    assert db.get_scheduled_llm_task(event_name) is None
    privmsg_calls = [
        c for c in irc.queueMsg.call_args_list if getattr(c.args[0], "command", None) == "PRIVMSG"
    ]
    assert privmsg_calls, "expected an auto-cancel notice to be queued"
    body = privmsg_calls[-1].args[0].args[1]
    assert "auto-cancelled" in body


class TestScheduleLlmTaskFailurePaths:
    """Coverage for IntegrityError + addEvent failure cleanup."""

    def test_schedule_llm_task_add_event_failure_deletes_db_row(
        self, llm_service, db, mocker: MockerFixture
    ) -> None:
        """If schedule.addEvent raises, the inserted DB row is rolled back."""
        mocker.patch("llm.service.schedule.addEvent", side_effect=RuntimeError("scheduler down"))
        msg = _msg_mock(mocker)
        irc = _irc_mock(mocker)
        mocker.patch.object(
            llm_service,
            "parse_reminder",
            return_value=ReminderParseResult(
                action="schedule",
                seconds=60,
                message="x",
                confirmation="ok",
                note=None,
                action_prompt="x",
                recurrence_seconds=None,
                recurrence_rrule=None,
                watch_mode=False,
            ),
        )
        llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
            5 if k == "bridgeScheduledTaskLimit" else None
        )

        result = llm_service.schedule_llm_task(
            irc=irc,
            msg=msg,
            creator_nick="rdrake",
            account="rdrake_a",
            channel="#test",
            when_natural="in 60s",
            prompt="x",
        )
        assert result.status == "error"
        assert "register" in result.message.lower()
        # No orphan rows.
        assert db.load_active_scheduled_llm_tasks() == []

    def test_schedule_llm_task_integrity_error_returns_collision_message(
        self, llm_service, mocker: MockerFixture
    ) -> None:
        """sqlite IntegrityError on save returns a collision error result."""
        import sqlite3

        msg = _msg_mock(mocker)
        irc = _irc_mock(mocker)
        mocker.patch.object(
            llm_service,
            "parse_reminder",
            return_value=ReminderParseResult(
                action="schedule",
                seconds=60,
                message="x",
                confirmation="ok",
                note=None,
                action_prompt="x",
                recurrence_seconds=None,
                recurrence_rrule=None,
                watch_mode=False,
            ),
        )
        llm_service.plugin.registryValue.side_effect = lambda k, ch=None: (
            5 if k == "bridgeScheduledTaskLimit" else None
        )
        # Replace the db with one that raises IntegrityError on insert.
        bad_db = mocker.MagicMock()
        bad_db.load_scheduled_llm_tasks_for.return_value = []
        bad_db.count_scheduled_llm_tasks_for.return_value = 0
        bad_db.save_scheduled_llm_task.side_effect = sqlite3.IntegrityError("UNIQUE")
        llm_service.plugin.db = bad_db

        result = llm_service.schedule_llm_task(
            irc=irc,
            msg=msg,
            creator_nick="rdrake",
            account="rdrake_a",
            channel="#test",
            when_natural="in 60s",
            prompt="x",
        )
        assert result.status == "error"
        assert "collision" in result.message


class TestMaybeRescheduleOrClean:
    """Coverage for _maybe_reschedule_or_clean cancel-mid-fire and exhausted-rrule."""

    def _make_row(self, **overrides):
        from llm.persistence import ScheduledLlmTaskRow

        defaults = {
            "id": 1,
            "event_name": "ev1",
            "creator_nick": "n",
            "account": None,
            "channel": "#t",
            "network": "afternet",
            "wire_msg": ":n!u@h PRIVMSG #t :@ask hi",
            "prompt": "p",
            "fire_at": 1.0,
            "created_at": 0.0,
            "recurrence_seconds": 300,
            "recurrence_rrule": None,
            "chain_position": 1,
            "watch_mode": False,
            "reply_target": None,
        }
        defaults.update(overrides)
        return ScheduledLlmTaskRow(**defaults)

    def test_one_shot_row_is_deleted_after_fire(self, llm_service, mocker: MockerFixture) -> None:
        """Row with no recurrence is deleted, not rescheduled."""
        add_event = mocker.patch("llm.service.schedule.addEvent")
        bad_db = mocker.MagicMock()
        row = self._make_row(recurrence_seconds=None, recurrence_rrule=None)
        llm_service._maybe_reschedule_or_clean(row, bad_db, 0.0)
        bad_db.delete_scheduled_llm_task.assert_called_once_with("ev1")
        add_event.assert_not_called()

    def test_cancelled_mid_fire_skips_reschedule(self, llm_service, mocker: MockerFixture) -> None:
        """If the row is gone (cancelled mid-fire) reschedule is skipped."""
        add_event = mocker.patch("llm.service.schedule.addEvent")
        bad_db = mocker.MagicMock()
        bad_db.get_scheduled_llm_task.return_value = None  # Cancelled mid-fire.
        row = self._make_row()
        llm_service._maybe_reschedule_or_clean(row, bad_db, 0.0)
        add_event.assert_not_called()

    def test_exhausted_rrule_deletes_row(self, llm_service, mocker: MockerFixture) -> None:
        """recurrence_rrule with no future occurrence triggers row delete."""
        mocker.patch("llm.service.schedule.addEvent")
        bad_db = mocker.MagicMock()
        bad_db.get_scheduled_llm_task.return_value = "row-still-there"
        row = self._make_row(recurrence_seconds=None, recurrence_rrule="FREQ=DAILY")
        # _next_rrule_fire returns None when the rule is exhausted.
        llm_service.plugin._next_rrule_fire = mocker.Mock(return_value=None)
        llm_service._maybe_reschedule_or_clean(row, bad_db, 0.0)
        bad_db.delete_scheduled_llm_task.assert_called_once_with("ev1")

    def test_chain_position_cap_stops_recurring_task(
        self, llm_service, mocker: MockerFixture
    ) -> None:
        """A recurring task at the cap is deleted, not rescheduled."""
        add_event = mocker.patch("llm.service.schedule.addEvent")
        bad_db = mocker.MagicMock()
        bad_db.get_scheduled_llm_task.return_value = "row-still-there"
        cap = llm_service._SCHEDULED_LLM_TASK_MAX_CHAIN_POSITION
        row = self._make_row(chain_position=cap)
        llm_service._maybe_reschedule_or_clean(row, bad_db, 0.0)
        bad_db.delete_scheduled_llm_task.assert_called_once_with("ev1")
        bad_db.update_scheduled_llm_task_fire_at.assert_not_called()
        add_event.assert_not_called()

    def test_below_cap_still_reschedules(self, llm_service, mocker: MockerFixture) -> None:
        """One fire short of the cap still reschedules normally."""
        add_event = mocker.patch("llm.service.schedule.addEvent")
        bad_db = mocker.MagicMock()
        bad_db.get_scheduled_llm_task.return_value = "row-still-there"
        cap = llm_service._SCHEDULED_LLM_TASK_MAX_CHAIN_POSITION
        row = self._make_row(chain_position=cap - 1)
        llm_service._maybe_reschedule_or_clean(row, bad_db, 0.0)
        bad_db.update_scheduled_llm_task_fire_at.assert_called_once()
        _, kwargs = bad_db.update_scheduled_llm_task_fire_at.call_args
        assert kwargs["chain_position"] == cap
        add_event.assert_called_once()
        bad_db.delete_scheduled_llm_task.assert_not_called()

    def test_next_fire_anchors_to_fire_start_not_turn_end(
        self, llm_service, mocker: MockerFixture
    ) -> None:
        """A 100s LLM turn does not push an every-5-minutes task 100s later."""
        mocker.patch("llm.service.schedule.addEvent")
        mocker.patch("llm.service.time.time", return_value=10_100.0)
        bad_db = mocker.MagicMock()
        bad_db.get_scheduled_llm_task.return_value = "row-still-there"
        llm_service._maybe_reschedule_or_clean(self._make_row(), bad_db, 10_000.0)
        assert bad_db.update_scheduled_llm_task_fire_at.call_args.args[1] == 10_300.0

    def test_next_fire_never_in_the_past(self, llm_service, mocker: MockerFixture) -> None:
        """A turn longer than the interval reschedules from now, not behind it."""
        mocker.patch("llm.service.schedule.addEvent")
        mocker.patch("llm.service.time.time", return_value=10_500.0)
        bad_db = mocker.MagicMock()
        bad_db.get_scheduled_llm_task.return_value = "row-still-there"
        llm_service._maybe_reschedule_or_clean(self._make_row(), bad_db, 10_000.0)
        assert bad_db.update_scheduled_llm_task_fire_at.call_args.args[1] >= 10_500.0


def test_cancel_scheduled_llm_task_swallows_keyerror(
    llm_service, db, mocker: MockerFixture
) -> None:
    """If the event is already gone from the scheduler, cancel still succeeds."""
    mocker.patch("llm.service.schedule.removeEvent", side_effect=KeyError("gone"))
    db.save_scheduled_llm_task(
        event_name="mine",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=_time.time() + 60,
    )
    result = llm_service.cancel_scheduled_llm_task(
        event_name="mine", creator_nick="rdrake", account="rdrake_a"
    )
    assert result.status == "ok"
    assert db.get_scheduled_llm_task("mine") is None


# =============================================================================
# _channel_target helper
# =============================================================================


def test_channel_target_passes_through_channel_names() -> None:
    """GIVEN IRC channel names WHEN _channel_target is called THEN returns the name unchanged."""
    assert LLMService._channel_target("#general") == "#general"
    assert LLMService._channel_target("&local") == "&local"


def test_channel_target_returns_none_for_nicks_and_falsy() -> None:
    """GIVEN a nick or falsy value WHEN _channel_target is called THEN returns None."""
    assert LLMService._channel_target("alice") is None
    assert LLMService._channel_target("") is None
    assert LLMService._channel_target(None) is None


# =============================================================================
# Task 11 — scheduled LLM task migration to LLMExecutor
# =============================================================================


def test_scheduled_fire_submits_via_executor(llm_service, db, mocker: MockerFixture):
    """fire() submits the dispatch worker through plugin._llm_executor with a
    scheduled_task: label."""
    add_event = mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({"bridgeScheduledTaskLimit": 5})

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
    )
    assert res.status == "ok"

    fire_callable = add_event.call_args.args[0]
    fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
    fake_world.getIrc.return_value = irc
    fake_world.ircs = [irc]

    plugin = llm_service.plugin
    plugin._check_rate_limit.return_value = False
    plugin._gather_history.return_value = ([], [])
    plugin._get_user_memories.return_value = []
    mocker.patch.object(plugin.db, "get_instruction", return_value="")
    plugin._pending_task_fns.return_value = {}
    mocker.patch("llm.service.ircdb.checkCapability", return_value=True)
    mocker.patch.object(
        llm_service,
        "assistant_request",
        return_value=mocker.MagicMock(
            content="ok", model="m", prompt_tokens=0, completion_tokens=0, cost=0.0, error=None
        ),
    )

    fire_callable()
    plugin._llm_executor.submit.assert_called_once()
    label = plugin._llm_executor.submit.call_args[0][0]
    assert label.startswith("scheduled_task:")


def test_scheduled_fire_short_circuits_when_closing(llm_service, db, mocker: MockerFixture):
    add_event = mocker.patch("llm.service.schedule.addEvent")
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({"bridgeScheduledTaskLimit": 5})

    llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
    )
    fire_callable = add_event.call_args.args[0]

    plugin = llm_service.plugin
    plugin._llm_executor.closing = True

    fire_callable()
    plugin._llm_executor.submit.assert_not_called()


def test_stored_sub_minute_cadence_is_floored_at_fire_time(
    llm_service, mocker: MockerFixture
) -> None:
    """Rows stored before the parse-time floor still fire at most once a minute."""
    row = mocker.MagicMock(recurrence_seconds=5, recurrence_rrule=None)
    assert llm_service._compute_next_fire(row, 1_000.0) == 1_060.0


def test_secondly_rrule_is_floored_at_fire_time() -> None:
    """The parse check sees one gap; the fire-time floor covers every later one."""
    from llm.plugin import LLM

    assert LLM._next_rrule_fire("FREQ=SECONDLY", 1_000.0) == 1_060.0
