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


def _stored_fire(llm_service):
    """The fire the agenda wakeup would run for the one stored task."""
    [(name, _at)] = llm_service.plugin.db.scheduled_llm_task_fire_times()
    row = llm_service.plugin.db.get_scheduled_llm_task(name)
    return lambda: llm_service.fire_scheduled_llm_task(row, _time.time())


def _irc_mock(mocker: MockerFixture):
    irc = mocker.MagicMock()
    irc.network = "afternet"
    return irc


def test_schedule_llm_task_creates_db_row_and_schedules_event(
    llm_service, db, mocker: MockerFixture
):
    """B1: a one-shot schedule writes a DB row and re-arms the agenda wakeup."""

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
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None

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
    rows = db.load_due_scheduled_llm_tasks(float("inf"))
    [row] = [r for r in rows if r.event_name == result.event_name]
    assert row.creator_nick == "rdrake"
    assert row.account == "rdrake_a"
    assert row.channel == "#test"
    assert row.network == "afternet"
    assert row.prompt == "check the build"
    assert row.recurrence_seconds is None
    assert row.recurrence_rrule is None

    # The row is the schedule; the plugin's one agenda wakeup is re-armed.
    llm_service.plugin._arm_agenda.assert_called_once()
    assert row.fire_at == pytest.approx(_time.time() + 60, abs=2)


def test_schedule_llm_task_recurrence_seconds(llm_service, db, mocker: MockerFixture):
    """B1: numeric-cadence recurrence stores recurrence_seconds and schedules
    the FIRST fire at parser.seconds."""
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
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None
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
    [row] = [
        r
        for r in db.load_due_scheduled_llm_tasks(float("inf"))
        if r.event_name == result.event_name
    ]
    assert row.recurrence_seconds == 300
    assert row.recurrence_rrule is None


def test_schedule_llm_task_recurrence_rrule(llm_service, db, mocker: MockerFixture):
    """B1: RRULE recurrence stored as-is; recurrence_seconds remains null."""
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
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None
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
    [row] = [
        r
        for r in db.load_due_scheduled_llm_tasks(float("inf"))
        if r.event_name == result.event_name
    ]
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
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None

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
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None

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
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None
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

    foreign = llm_service.cancel_scheduled_llm_task(
        event_name="theirs", creator_nick="rdrake", account="rdrake_a"
    )
    assert foreign.status == "error"


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
    # The row survives: delete_scheduled_llm_task returns True only if it
    # actually removed something.
    assert db.delete_scheduled_llm_task("alices") is True


def test_cancel_scheduled_llm_task_unknown_returns_error(llm_service):
    out = llm_service.cancel_scheduled_llm_task(
        event_name="does_not_exist", creator_nick="x", account=None
    )
    assert out.status == "error"


def test_cancel_of_an_unknown_id_says_so_to_an_identified_caller(llm_service):
    """Real cancels always carry an account; the id simply does not exist."""
    out = llm_service.cancel_scheduled_llm_task(event_name="nope", creator_nick="x", account="acct")
    assert out.status == "error"
    assert out.message == "No scheduled task with id nope."


# =============================================================================
# Phase 2 Task 3 / D4 — depth-cap end-to-end on fired schedule
# =============================================================================


def test_fired_task_cannot_schedule_a_nested_task(llm_service, db, mocker: MockerFixture):
    """End-to-end: schedule a task; trigger the fire callback; observe that
    within the fired @ask, schedule_llm_task refuses (llm_schedule_depth=1)."""
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
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None

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

    fire_callable = _stored_fire(llm_service)

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
    # so the dispatch path reaches the patched assistant_completion.
    plugin.llm_service = llm_service
    plugin._unattended_ask_rate_limited = LLM._unattended_ask_rate_limited.__get__(plugin)
    plugin._run_unattended_assistant = LLM._run_unattended_assistant.__get__(plugin)

    captured: dict[str, object] = {}

    def fake_assistant_completion(*, msg, **_kwargs):
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

    mocker.patch.object(llm_service, "assistant_completion", side_effect=fake_assistant_completion)

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
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"], "#t": ["rdrake"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
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
    [row] = [
        r for r in db.load_due_scheduled_llm_tasks(float("inf")) if r.event_name == res.event_name
    ]
    assert row.reply_target == "#deliver"


def test_reply_target_channel_bot_absent_refused(llm_service, mocker: MockerFixture):
    irc = _irc_with_channels(mocker, {"#t": ["rdrake"]})  # bot not in #other
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({})

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
    irc = _irc_with_channels(mocker, {"#deliver": ["bot"], "#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
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
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"], "#t": ["rdrake"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
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


def test_reply_target_naming_the_origin_channel_is_always_allowed(
    llm_service, db, mocker: MockerFixture
):
    """The origin is the default target anyway: naming it (in any case) needs
    neither the bridge nor a presence check (scheduled-tasks.md)."""
    irc = _irc_with_channels(mocker, {})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({})

    res = llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#dev",
        when_natural="in 60s",
        prompt="x",
        reply_target="#DEV",
    )

    assert res.status == "ok"


def test_reply_target_pm_self_ok(llm_service, db, mocker: MockerFixture):
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({})

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
    [row] = [
        r for r in db.load_due_scheduled_llm_tasks(float("inf")) if r.event_name == res.event_name
    ]
    assert row.reply_target == "rdrake"


def test_reply_target_pm_other_refused(llm_service, mocker: MockerFixture):
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({})

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
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"], "#origin": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {
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

    fire_callable = _stored_fire(llm_service)
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
        "assistant_completion",
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


def _fire_with_answer(
    llm_service,
    mocker: MockerFixture,
    content: str,
    *,
    instruction: str | None = "",
    overlay: str | None = None,
):
    """Schedule a one-shot task in #deliver, fire it with ``content``, return the irc."""
    irc = _irc_with_channels(mocker, {"#deliver": ["rdrake", "bot"]})
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry(
        {("bridgeEnabled", "#deliver"): True, "assistantSystemPrompt": overlay}
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
    mocker.patch.object(plugin.db, "get_instruction", return_value=instruction)
    plugin._pending_task_fns.return_value = {}
    plugin.llm_service = llm_service
    from llm.plugin import LLM

    plugin._unattended_ask_rate_limited = LLM._unattended_ask_rate_limited.__get__(plugin)
    plugin._run_unattended_assistant = LLM._run_unattended_assistant.__get__(plugin)
    _bind_real_delivery(plugin)
    mocker.patch.object(
        llm_service,
        "assistant_completion",
        return_value=mocker.MagicMock(
            content=content, model="m", prompt_tokens=0, completion_tokens=0, cost=0.0, error=None
        ),
    )
    _stored_fire(llm_service)()
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


@pytest.mark.parametrize("instruction", ["always answer in French", None])
def test_fired_task_sends_the_instruction_as_user_data(
    llm_service, db, mocker: MockerFixture, instruction: str | None
):
    """@instruct rides as fenced user-role data on every path, never inside
    the system prompt, and the channel overlay is always the system prompt
    (rate-limiting-security.md)."""
    _fire_with_answer(llm_service, mocker, "ok", instruction=instruction, overlay="OVERLAY")

    kwargs = llm_service.assistant_completion.call_args.kwargs
    assert kwargs["user_instruction"] == instruction
    assert kwargs["system_prompt"] == "OVERLAY"


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
    and assistant_completion is never called."""
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({})

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

    fire_callable = _stored_fire(llm_service)
    fake_world = mocker.patch("llm.service.world", autospec=False, create=True)
    fake_world.getIrc.return_value = irc
    fake_world.ircs = [irc]
    mocker.patch("llm.service.ircdb.checkCapability", return_value=False)

    assistant = mocker.patch.object(llm_service, "assistant_completion")

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
    """Coverage for IntegrityError cleanup."""

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
        llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None
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


def test_scheduled_fire_short_circuits_when_closing(llm_service, db, mocker: MockerFixture):
    irc = _irc_with_channels(mocker, {"#t": ["rdrake", "bot"]})
    msg = _msg_mock(mocker)
    _patch_parser(llm_service, mocker)
    llm_service.plugin.registryValue.side_effect = _registry({})

    llm_service.schedule_llm_task(
        irc=irc,
        msg=msg,
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        when_natural="in 60s",
        prompt="x",
    )
    fire_callable = _stored_fire(llm_service)

    plugin = llm_service.plugin
    plugin._llm_executor.closing = True

    fire_callable()
    plugin._llm_executor.submit.assert_not_called()


def _stored_task(db, *, fire_at: float = 0.0):
    db.save_scheduled_llm_task(
        event_name="llm_task_abc",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask hi",
        prompt="p",
        fire_at=fire_at,
    )
    return db.get_scheduled_llm_task("llm_task_abc")


def test_scheduled_fire_settles_with_the_agenda_after_the_turn(
    llm_service, db, mocker: MockerFixture
) -> None:
    """GIVEN a fire whose turn raises THEN the worker still hands the fire
    back to the agenda, which reschedules or removes the row."""
    from llm.agenda import Kind

    row = _stored_task(db)
    fake_world = mocker.patch("llm.service.world")
    fake_world.getIrc.return_value = _irc_mock(mocker)
    mocker.patch.object(llm_service, "_dispatch_scheduled_task", side_effect=RuntimeError("boom"))

    llm_service.fire_scheduled_llm_task(row, 123.0)

    llm_service.plugin._finish_agenda.assert_called_once_with(Kind.TASK, "llm_task_abc", 123.0)


def test_scheduled_fire_without_irc_settles_without_a_worker(
    llm_service, db, mocker: MockerFixture
) -> None:
    from llm.agenda import Kind

    row = _stored_task(db)
    fake_world = mocker.patch("llm.service.world")
    fake_world.getIrc.return_value = None
    fake_world.ircs = []

    llm_service.fire_scheduled_llm_task(row, 123.0)

    llm_service.plugin._llm_executor.submit.assert_not_called()
    llm_service.plugin._finish_agenda.assert_called_once_with(Kind.TASK, "llm_task_abc", 123.0)


def test_scheduled_fire_books_usage_while_closing(llm_service, mocker: MockerFixture) -> None:
    """GIVEN shutdown began while a fire's turn ran THEN its spend is still
    booked: the provider billed it whether or not the bot stays up."""
    from llm.persistence import ScheduledLlmTaskRow
    from llm.service import AssistantResult, Spend

    plugin = llm_service.plugin
    plugin._llm_executor.closing = True
    plugin._unattended_ask_rate_limited.return_value = False
    plugin._run_unattended_assistant.return_value = AssistantResult(
        content="ok", spend=Spend(prompt_tokens=5, completion_tokens=2, cost=0.01), model="m"
    )
    mocker.patch("llm.service.ircdb.checkCapability", return_value=True)
    row = ScheduledLlmTaskRow(
        id=1,
        event_name="scheduled_llm_task_1",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask do x",
        prompt="do x",
        fire_at=0.0,
        created_at=0.0,
        recurrence_seconds=None,
        recurrence_rrule=None,
        chain_position=1,
        watch_mode=False,
        reply_target=None,
    )

    llm_service._dispatch_scheduled_task(_irc_mock(mocker), _msg_mock(mocker), row)

    plugin._log_unattended_usage.assert_called_once()
    assert plugin._log_unattended_usage.call_args.kwargs["result"].cost == 0.01


def _task_row(*, watch_mode: bool = False):
    from llm.persistence import ScheduledLlmTaskRow

    return ScheduledLlmTaskRow(
        id=1,
        event_name="llm_task_watch",
        creator_nick="rdrake",
        account="rdrake_a",
        channel="#t",
        network="afternet",
        wire_msg=":rdrake!u@h PRIVMSG #t :@ask check",
        prompt="check whether Ubuntu 24.04 is patched",
        fire_at=0.0,
        created_at=0.0,
        recurrence_seconds=300,
        recurrence_rrule=None,
        chain_position=1,
        watch_mode=watch_mode,
        reply_target=None,
    )


def _dispatch_with_reply(llm_service, mocker: MockerFixture, row, reply: str):
    """Fire ``row`` with the model answering ``reply``; return the irc."""
    from llm.service import AssistantResult, Spend

    plugin = llm_service.plugin
    plugin._unattended_ask_rate_limited.return_value = False
    plugin._run_unattended_assistant.return_value = AssistantResult(
        content=reply, spend=Spend.ZERO, model="m"
    )
    mocker.patch("llm.service.ircdb.checkCapability", return_value=True)
    _bind_real_delivery(plugin)
    irc = _irc_mock(mocker)
    llm_service._dispatch_scheduled_task(irc, _msg_mock(mocker), row)
    return irc


@pytest.mark.parametrize(
    ("watch_mode", "reply"),
    [(True, "[silent]"), (False, ""), (True, "")],
)
def test_a_fire_with_nothing_to_report_posts_nothing(
    llm_service, mocker: MockerFixture, watch_mode: bool, reply: str
) -> None:
    """Only positive results reach the user: a watch's [silent], or an empty
    reply on any task, sends no line at all."""
    irc = _dispatch_with_reply(llm_service, mocker, _task_row(watch_mode=watch_mode), reply)

    irc.queueMsg.assert_not_called()


def test_a_watch_delivers_its_positive_result(llm_service, mocker: MockerFixture) -> None:
    """ "Let me know when X is available": the news itself must be posted."""
    irc = _dispatch_with_reply(
        llm_service, mocker, _task_row(watch_mode=True), "Ubuntu 24.04 is now patched."
    )

    [call] = irc.queueMsg.call_args_list
    assert call.args[0].args == ("#t", "rdrake: Ubuntu 24.04 is now patched.")


def test_fire_uses_the_connection_for_the_stored_network(
    llm_service, db, mocker: MockerFixture
) -> None:
    irc_a, irc_b = _irc_mock(mocker), _irc_mock(mocker)
    fake_world = mocker.patch("llm.service.world")
    fake_world.getIrc.return_value = irc_a
    fake_world.ircs = [irc_b]
    dispatch = mocker.patch.object(llm_service, "_dispatch_scheduled_task")

    llm_service.fire_scheduled_llm_task(_stored_task(db), 123.0)

    fake_world.getIrc.assert_called_once_with("afternet")
    assert dispatch.call_args.args[0] is irc_a


def test_fire_falls_back_to_the_first_connection(llm_service, db, mocker: MockerFixture) -> None:
    irc_b = _irc_mock(mocker)
    fake_world = mocker.patch("llm.service.world")
    fake_world.getIrc.return_value = None
    fake_world.ircs = [irc_b]
    dispatch = mocker.patch.object(llm_service, "_dispatch_scheduled_task")

    llm_service.fire_scheduled_llm_task(_stored_task(db), 123.0)

    assert dispatch.call_args.args[0] is irc_b


def test_a_recurring_task_fires_exactly_five_times(llm_service, db, mocker: MockerFixture) -> None:
    """scheduled-tasks.md: a recurring task fires at most five times and is
    then deleted. Counted from the task as schedule_llm_task stores it."""
    from llm.agenda import Agenda, Kind, Outcome

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
            recurrence_seconds=300,
            recurrence_rrule=None,
            watch_mode=False,
        ),
    )
    llm_service.plugin.registryValue.side_effect = lambda k, ch=None: None
    res = llm_service.schedule_llm_task(
        irc=_irc_mock(mocker),
        msg=_msg_mock(mocker),
        creator_nick="n",
        account="acct",
        channel="#t",
        when_natural="every 5 minutes",
        prompt="ping me",
    )
    assert res.status == "ok"

    now = [0.0]
    schedule = Agenda(db, clock=lambda: now[0])
    fires = 0
    while True:
        row = db.get_scheduled_llm_task(res.event_name)
        now[0] = row.fire_at + 1
        assert res.event_name in [d.row.event_name for d in schedule.claim_due().due]
        fires += 1
        done = schedule.finish(Kind.TASK, res.event_name, fired_at=now[0])
        if done.outcome is not Outcome.RESCHEDULED:
            break

    assert done.outcome is Outcome.CAPPED
    assert fires == 5
    assert db.get_scheduled_llm_task(res.event_name) is None
