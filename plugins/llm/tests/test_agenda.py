"""The agenda: when reminders and scheduled LLM tasks fire.

Runs against a real SQLite store and a fake clock — no IRC, no Limnoria
scheduler. Plugin-level tests cover arming the one wakeup and delivery.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest
from llm.agenda import (
    MAX_CHAIN_POSITION,
    MIN_RECURRENCE_SECONDS,
    Agenda,
    Kind,
    Outcome,
    cadence_too_short,
    next_fire,
    next_rrule_fire,
)
from llm.persistence import EXPIRY_THRESHOLD_SECONDS, LLMDatabase

T0 = 1_800_000_000.0  # 2027-01-15 ~08:00 UTC


class Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def agenda(test_db: LLMDatabase, clock: Clock) -> Agenda:
    return Agenda(test_db, clock=clock)


def _reminder(db: LLMDatabase, name: str, fire_at: float, **kw) -> None:
    db.save_reminder(name, kw.pop("nick", "alice"), "#c", kw.pop("message", "m"), fire_at, **kw)


def _task(db: LLMDatabase, name: str, fire_at: float, **kw) -> None:
    db.save_scheduled_llm_task(
        name,
        "alice",
        kw.pop("account", "alice_a"),
        "#c",
        "afternet",
        ":alice!u@h PRIVMSG #c :@ask hi",
        "p",
        fire_at,
        **kw,
    )


def _fire(agenda: Agenda, clock: Clock, kind: Kind, name: str):
    """Claim ``name`` as due now and finish it, as one fire would."""
    claimed = [d.row.event_name for d in agenda.claim_due().due]
    assert name in claimed
    return agenda.finish(kind, name, fired_at=clock.now)


class TestCadenceFloor:
    """One minute between fires, at parse time and at every fire."""

    def test_sub_minute_numeric_cadence_is_too_short(self) -> None:
        assert cadence_too_short(30, None)
        assert not cadence_too_short(60, None)
        assert not cadence_too_short(None, None)

    def test_sub_minute_rrule_is_too_short(self) -> None:
        assert cadence_too_short(None, "FREQ=SECONDLY;INTERVAL=30")
        assert not cadence_too_short(None, "FREQ=HOURLY")

    def test_malformed_rrule_raises(self) -> None:
        with pytest.raises(ValueError):
            cadence_too_short(None, "FREQ=NONSENSE;BLAH")

    def test_stored_sub_minute_cadence_is_floored_at_fire_time(self, test_db, clock) -> None:
        """Rows stored before the parse-time floor still fire at most once a minute."""
        _reminder(test_db, "r", T0, recurrence_seconds=5)
        row = test_db.get_reminder("r")
        assert next_fire(row, fired_at=T0, now=T0) == T0 + MIN_RECURRENCE_SECONDS

    def test_secondly_rrule_is_floored_at_fire_time(self) -> None:
        """The parse check sees one gap; the fire-time floor covers every later one."""
        assert next_rrule_fire("FREQ=SECONDLY", 1_000.0) == 1_060.0


class TestNextRruleFire:
    def test_spring_forward_moves_a_day(self) -> None:
        rule = "FREQ=DAILY;BYHOUR=6;BYMINUTE=30"
        now = datetime(2027, 3, 13, 6, 0, tzinfo=UTC).timestamp()
        first = next_rrule_fire(rule, now)
        assert first is not None and first > now
        second = next_rrule_fire(rule, first)
        assert second is not None and second - first > 23 * 3600

    def test_fall_back_has_no_duplicate_fire(self) -> None:
        rule = "FREQ=DAILY;BYHOUR=5;BYMINUTE=30"
        now = datetime(2027, 11, 6, 5, 0, tzinfo=UTC).timestamp()
        first = next_rrule_fire(rule, now)
        assert first is not None
        second = next_rrule_fire(rule, first)
        assert second is not None and second - first > 23 * 3600

    def test_byhour_is_the_owners_wall_clock(self) -> None:
        toronto = ZoneInfo("America/Toronto")
        now = datetime(2027, 1, 15, 12, 0, tzinfo=UTC).timestamp()
        nxt = next_rrule_fire("FREQ=DAILY;BYHOUR=9;BYMINUTE=0", now, toronto)
        assert nxt is not None
        local = datetime.fromtimestamp(nxt, tz=toronto)
        assert (local.hour, local.minute) == (9, 0)

    def test_malformed_rule_has_no_next_fire(self) -> None:
        assert next_rrule_fire("FREQ=NONSENSE;BLAH", T0) is None


class TestNextFire:
    def test_anchors_to_fire_start_not_turn_end(self, test_db) -> None:
        """A 100s LLM turn does not push an every-5-minutes chain 100s later."""
        _task(test_db, "t", T0, recurrence_seconds=300)
        row = test_db.get_scheduled_llm_task("t")
        assert next_fire(row, fired_at=T0, now=T0 + 100) == T0 + 300

    def test_never_in_the_past(self, test_db) -> None:
        """A turn longer than the interval reschedules from now, not behind it."""
        _task(test_db, "t", T0, recurrence_seconds=300)
        row = test_db.get_scheduled_llm_task("t")
        assert next_fire(row, fired_at=T0, now=T0 + 500) == T0 + 501

    def test_one_shot_has_none(self, test_db) -> None:
        _reminder(test_db, "r", T0)
        assert next_fire(test_db.get_reminder("r"), fired_at=T0, now=T0) is None


class TestClaim:
    def test_claims_only_what_is_due(self, agenda, test_db, clock) -> None:
        _reminder(test_db, "due", T0 - 1)
        _reminder(test_db, "later", T0 + 60)
        _task(test_db, "task", T0)
        claim = agenda.claim_due()
        assert [(d.kind, d.row.event_name) for d in claim.due] == [
            (Kind.REMINDER, "due"),
            (Kind.TASK, "task"),
        ]

    def test_a_firing_entry_is_not_claimed_twice(self, agenda, test_db, clock) -> None:
        """A slow fire is still due by its row; the next wakeup must not start it again."""
        _reminder(test_db, "r", T0, recurrence_seconds=60)
        assert len(agenda.claim_due().due) == 1
        clock.now += 30
        assert agenda.claim_due().due == []
        assert agenda.firing(Kind.REMINDER, "r")

    def test_stale_one_shot_is_dropped_not_fired(self, agenda, test_db, clock) -> None:
        """Restart after a long outage: a one-shot a day late is not delivered."""
        _reminder(test_db, "old", T0 - EXPIRY_THRESHOLD_SECONDS - 1)
        _task(test_db, "old_task", T0 - EXPIRY_THRESHOLD_SECONDS - 1)
        claim = agenda.claim_due()
        assert claim.due == []
        assert {d.row.event_name for d in claim.expired} == {"old", "old_task"}
        assert test_db.get_reminder("old") is None
        assert test_db.get_scheduled_llm_task("old_task") is None

    def test_overdue_recurring_survives_a_long_outage(self, agenda, test_db, clock) -> None:
        """4830d4f: a restart must not end a chain, however overdue."""
        for kind, add in ((Kind.REMINDER, _reminder), (Kind.TASK, _task)):
            add(
                test_db, f"chain_{kind}", T0 - 3 * EXPIRY_THRESHOLD_SECONDS, recurrence_seconds=3600
            )
        claim = agenda.claim_due()
        assert claim.expired == []
        assert len(claim.due) == 2
        for d in claim.due:
            done = agenda.finish(d.kind, d.row.event_name, fired_at=clock.now)
            assert done.outcome is Outcome.RESCHEDULED
            assert done.next_fire_at == T0 + 3600

    def test_slightly_overdue_one_shot_still_fires(self, agenda, test_db, clock) -> None:
        _reminder(test_db, "r", T0 - 3600)
        assert [d.row.event_name for d in agenda.claim_due().due] == ["r"]


class TestFinish:
    def test_one_shot_is_deleted(self, agenda, test_db, clock) -> None:
        _task(test_db, "t", T0)
        assert _fire(agenda, clock, Kind.TASK, "t").outcome is Outcome.DONE
        assert test_db.get_scheduled_llm_task("t") is None

    def test_recurring_reminder_moves_in_place(self, agenda, test_db, clock) -> None:
        """Same row, same id, next position; created_at stays the request's."""
        _reminder(test_db, "llm_remind_abc", T0, recurrence_seconds=600, reply_msgid="m1")
        before = test_db.get_reminder("llm_remind_abc")
        done = _fire(agenda, clock, Kind.REMINDER, "llm_remind_abc")
        assert done.outcome is Outcome.RESCHEDULED
        after = test_db.get_reminder("llm_remind_abc")
        assert after is not None
        assert after.id == before.id
        assert after.fire_at == T0 + 600
        assert after.chain_position == 2
        assert after.created_at == before.created_at
        assert after.reply_msgid == "m1"

    def test_cancelled_mid_fire_stays_cancelled(self, agenda, test_db, clock) -> None:
        """The row is the cancel truth: deleted while firing means no reschedule."""
        _reminder(test_db, "r", T0, recurrence_seconds=600)
        agenda.claim_due()
        test_db.delete_reminder("r")
        done = agenda.finish(Kind.REMINDER, "r", fired_at=T0)
        assert done.outcome is Outcome.CANCELLED
        assert test_db.get_reminder("r") is None

    @pytest.mark.parametrize("kind", [Kind.REMINDER, Kind.TASK])
    def test_chain_cap_ends_the_chain(self, agenda, test_db, clock, kind) -> None:
        cap = MAX_CHAIN_POSITION[kind]
        add = _reminder if kind is Kind.REMINDER else _task
        add(test_db, "c", T0, recurrence_seconds=60, chain_position=cap)
        done = _fire(agenda, clock, kind, "c")
        assert done.outcome is Outcome.CAPPED
        assert done.cap == cap
        get = test_db.get_reminder if kind is Kind.REMINDER else test_db.get_scheduled_llm_task
        assert get("c") is None

    def test_caps_are_fifty_reminders_and_five_tasks(self) -> None:
        assert MAX_CHAIN_POSITION == {Kind.REMINDER: 50, Kind.TASK: 5}

    def test_one_short_of_the_cap_reschedules(self, agenda, test_db, clock) -> None:
        cap = MAX_CHAIN_POSITION[Kind.TASK]
        _task(test_db, "t", T0, recurrence_seconds=60, chain_position=cap - 1)
        done = _fire(agenda, clock, Kind.TASK, "t")
        assert done.outcome is Outcome.RESCHEDULED
        assert test_db.get_scheduled_llm_task("t").chain_position == cap

    def test_exhausted_rrule_deletes_the_row(self, agenda, test_db, clock) -> None:
        _task(test_db, "t", T0, recurrence_rrule="FREQ=DAILY;COUNT=1")
        done = _fire(agenda, clock, Kind.TASK, "t")
        assert done.outcome is Outcome.EXHAUSTED
        assert test_db.get_scheduled_llm_task("t") is None

    def test_rrule_uses_the_owners_zone(self, test_db, clock) -> None:
        toronto = ZoneInfo("America/Toronto")
        seen: list[tuple[str, str | None]] = []

        def owner_tz(nick: str, account: str | None):
            seen.append((nick, account))
            return toronto

        agenda = Agenda(test_db, clock=clock, owner_tz=owner_tz)
        _task(test_db, "t", T0, recurrence_rrule="FREQ=DAILY;BYHOUR=9;BYMINUTE=0")
        done = _fire(agenda, clock, Kind.TASK, "t")
        assert seen == [("alice", "alice_a")]
        local = datetime.fromtimestamp(done.next_fire_at, tz=toronto)
        assert (local.hour, local.minute) == (9, 0)

    def test_second_finish_is_a_no_op(self, agenda, test_db, clock) -> None:
        """A fire settled twice must not advance its chain twice."""
        _reminder(test_db, "r", T0, recurrence_seconds=600)
        _fire(agenda, clock, Kind.REMINDER, "r")
        done = agenda.finish(Kind.REMINDER, "r", fired_at=T0)
        assert done.outcome is Outcome.NOT_FIRING
        assert test_db.get_reminder("r").chain_position == 2


class TestNextDue:
    def test_earliest_across_both_kinds(self, agenda, test_db) -> None:
        assert agenda.next_due_at() is None
        _reminder(test_db, "r", T0 + 50)
        _task(test_db, "t", T0 + 20)
        assert agenda.next_due_at() == T0 + 20

    def test_skips_a_firing_entry(self, agenda, test_db, clock) -> None:
        """Otherwise the wakeup re-arms every second for the length of a fire."""
        _reminder(test_db, "r", T0, recurrence_seconds=600)
        _reminder(test_db, "later", T0 + 90)
        agenda.claim_due()
        assert agenda.next_due_at() == T0 + 90


class TestPendingReminders:
    def test_a_firing_one_shot_is_no_longer_pending(self, agenda, test_db) -> None:
        """An action that lists its owner's reminders must not list itself."""
        _reminder(test_db, "one", T0)
        _reminder(test_db, "chain", T0, recurrence_seconds=600)
        agenda.claim_due()
        assert [r.event_name for r in agenda.pending_reminders()] == ["chain"]


class TestClaimHeldUntilSettled:
    """A wakeup between "fire finished" and "row moved" must not refire it."""

    def test_reschedule_holds_the_claim_until_the_row_moves(self, test_db, clock) -> None:
        _reminder(test_db, "r", T0, recurrence_seconds=600)
        nested: list[list[str]] = []
        real_update = test_db.update_reminder_fire_at
        agenda: Agenda

        def update_with_a_wakeup_in_between(*args, **kwargs):
            nested.append([d.row.event_name for d in agenda.claim_due().due])
            real_update(*args, **kwargs)

        test_db.update_reminder_fire_at = update_with_a_wakeup_in_between
        agenda = Agenda(test_db, clock=clock)
        agenda.claim_due()

        agenda.finish(Kind.REMINDER, "r", fired_at=T0)

        assert nested == [[]]
        assert not agenda.firing(Kind.REMINDER, "r")

    def test_delete_holds_the_claim_until_the_row_goes(self, test_db, clock) -> None:
        _reminder(test_db, "r", T0)
        nested: list[list[str]] = []
        real_delete = test_db.delete_reminder
        agenda: Agenda

        def delete_with_a_wakeup_in_between(name):
            nested.append([d.row.event_name for d in agenda.claim_due().due])
            return real_delete(name)

        test_db.delete_reminder = delete_with_a_wakeup_in_between
        agenda = Agenda(test_db, clock=clock)
        agenda.claim_due()

        agenda.finish(Kind.REMINDER, "r", fired_at=T0)

        assert nested == [[]]

    def test_a_raising_store_still_releases_the_claim(self, test_db, clock) -> None:
        _reminder(test_db, "r", T0)

        def broken_delete(_name):
            raise RuntimeError("database is locked")

        test_db.delete_reminder = broken_delete
        agenda = Agenda(test_db, clock=clock)
        agenda.claim_due()

        with pytest.raises(RuntimeError):
            agenda.finish(Kind.REMINDER, "r", fired_at=T0)

        assert not agenda.firing(Kind.REMINDER, "r")
