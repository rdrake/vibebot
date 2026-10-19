"""The agenda: when reminders and scheduled LLM tasks fire — no supybot imports.

Both kinds of agenda entry live in their own table (``reminders`` and
``scheduled_llm_tasks``) and are delivered by their own code (the plugin
echoes or runs a reminder; the service dispatches a task). Everything about
*when* they fire lives here, once, for both:

- the one-minute cadence floor, checked when a cadence is parsed and again
  for every computed fire;
- the next fire of a recurring entry, anchored to the start of the fire that
  just ran (so a long LLM turn does not drift the cadence) and never earlier
  than a second from now;
- the chain cap, per kind (a reminder chain may run 50 times, a scheduled
  task 5), enforced in one place;
- the cancel check: the database row is the only truth about whether an
  entry is still wanted. Cancelling deletes the row; a fire that finds its
  row gone when it finishes does not reschedule;
- restart recovery: an overdue entry is simply due. A one-shot more than
  ``EXPIRY_THRESHOLD_SECONDS`` overdue is dropped instead of fired late; a
  recurring entry is never dropped for being overdue, so an outage cannot
  end a chain;
- what is due now, and when the next entry comes due, so the plugin can run
  one wakeup timer instead of one scheduler event per row.

Recurring entries keep their row (and so their id) across fires: each fire
moves ``fire_at`` and ``chain_position`` in place.

The only in-memory state is the set of entries being fired right now. It
stops the next wakeup from firing an entry a second time while its first
fire (an LLM turn, possibly a minute long) is still running, and hides a
firing one-shot from the owner's pending list. It is never consulted to
decide whether an entry was cancelled — that is the row's job — so losing
it (a restart) can cause at most one repeated fire of an entry that was
mid-fire, never a lost or resurrected one.

Kept free of Limnoria imports so it is tested against a real SQLite store
and a fake clock with no IRC scaffolding.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from enum import StrEnum
from typing import Protocol

from .persistence import EXPIRY_THRESHOLD_SECONDS, ReminderRow, ScheduledLlmTaskRow

# Floor between two fires of one recurring chain: each action fire bills an LLM turn.
MIN_RECURRENCE_SECONDS = 60


class Kind(StrEnum):
    """Which table an agenda entry lives in."""

    REMINDER = "reminder"
    TASK = "scheduled_task"


# How many times a chain may fire before its owner has to set it again.
# Tasks are capped much tighter than reminders: they fire LLM completions and
# can target other channels or users through reply_target, so a recurring
# task left running turns into harassment fast. Five fires forces a re-arm
# long before "every few minutes" becomes "ran for hours".
MAX_CHAIN_POSITION: dict[Kind, int] = {Kind.REMINDER: 50, Kind.TASK: 5}

AgendaRow = ReminderRow | ScheduledLlmTaskRow


def owner_nick(row: AgendaRow) -> str:
    """The nick that set the entry."""
    return row.nick if isinstance(row, ReminderRow) else row.creator_nick


def is_recurring(row: AgendaRow) -> bool:
    """True when the entry has a cadence (numeric or RRULE)."""
    return row.recurrence_seconds is not None or row.recurrence_rrule is not None


def cadence_too_short(recurrence_seconds: int | None, recurrence_rrule: str | None) -> bool:
    """True when a parsed cadence's first gap is under the one-minute floor.

    The parser is the only way a cadence reaches storage, so this is the
    check that refuses it up front. An RRULE's later gaps can still be
    shorter than its first; :func:`next_fire` floors every fire as well.

    Raises:
        ValueError: ``recurrence_rrule`` is not a rule dateutil can read.
    """
    gap: float | None = recurrence_seconds
    if recurrence_rrule is not None:
        from dateutil.rrule import rrulestr

        try:
            first_two = rrulestr(recurrence_rrule)[:2]
        except TypeError as exc:  # rrulestr raises both for malformed input
            raise ValueError(str(exc)) from exc
        if len(first_two) == 2:
            gap = (first_two[1] - first_two[0]).total_seconds()
    return gap is not None and gap < MIN_RECURRENCE_SECONDS


def next_rrule_fire(rule_str: str, now: float, tz: tzinfo = UTC) -> float | None:
    """The next fire after ``now`` for an RRULE string.

    BYHOUR/BYMINUTE are wall-clock times in ``tz`` (the owner's zone, as the
    parser was told), so "daily at 8am" stays at 8am across a DST change.
    Returns None when the rule is malformed or has no future occurrence.
    """
    from dateutil.rrule import rrulestr

    try:
        now_local = datetime.fromtimestamp(now, tz=tz)
        rule = rrulestr(rule_str, dtstart=now_local)
        next_dt = rule.after(now_local)
    except (ValueError, TypeError):
        return None
    if next_dt is None:
        return None
    # Parse-time checks see only the first gap; this holds for every fire.
    return max(next_dt.timestamp(), now + MIN_RECURRENCE_SECONDS)


def next_fire(row: AgendaRow, *, fired_at: float, now: float, tz: tzinfo = UTC) -> float | None:
    """When a recurring entry fires next, or None when it never will.

    Anchored to ``fired_at``, the start of the fire that just ran, so the
    cadence holds however long the turn took; clamped to a second after
    ``now`` so a turn longer than the interval cannot schedule a fire in the
    past. Every fire is at least :data:`MIN_RECURRENCE_SECONDS` after the
    anchor. None for a one-shot, an exhausted RRULE, or a malformed one.
    """
    if row.recurrence_seconds is not None:
        nxt: float | None = fired_at + max(row.recurrence_seconds, MIN_RECURRENCE_SECONDS)
    elif row.recurrence_rrule is not None:
        nxt = next_rrule_fire(row.recurrence_rrule, fired_at, tz)
    else:
        nxt = None
    if nxt is None:
        return None
    return max(nxt, now + 1)


class Store(Protocol):
    """The persistence calls the agenda makes (``LLMDatabase`` satisfies it)."""

    def get_reminder(self, event_name: str) -> ReminderRow | None: ...
    def load_due_reminders(self, now: float) -> list[ReminderRow]: ...
    def reminder_fire_times(self) -> list[tuple[str, float]]: ...
    def update_reminder_fire_at(
        self, event_name: str, fire_at: float, *, chain_position: int
    ) -> None: ...
    def delete_reminder(self, event_name: str) -> bool: ...
    def load_pending_reminders(self) -> list[ReminderRow]: ...

    def get_scheduled_llm_task(self, event_name: str) -> ScheduledLlmTaskRow | None: ...
    def load_due_scheduled_llm_tasks(self, now: float) -> list[ScheduledLlmTaskRow]: ...
    def scheduled_llm_task_fire_times(self) -> list[tuple[str, float]]: ...
    def update_scheduled_llm_task_fire_at(
        self, event_name: str, fire_at: float, *, chain_position: int | None = None
    ) -> None: ...
    def delete_scheduled_llm_task(self, event_name: str) -> bool: ...


@dataclass(frozen=True)
class _Table:
    """One kind's store calls, so the rules below are written once."""

    get: Callable[[str], AgendaRow | None]
    due: Callable[[float], Sequence[AgendaRow]]
    fire_times: Callable[[], Sequence[tuple[str, float]]]
    advance: Callable[[str, float, int], object]
    delete: Callable[[str], object]


class Outcome(StrEnum):
    """What :meth:`Agenda.finish` did with an entry that just fired."""

    DONE = "done"  # a one-shot fired; its row is gone
    RESCHEDULED = "rescheduled"  # a recurring entry moved to its next fire
    CANCELLED = "cancelled"  # the row was deleted while the fire ran
    CAPPED = "capped"  # the chain reached its kind's cap; row deleted
    EXHAUSTED = "exhausted"  # the RRULE has no next fire (or is malformed); row deleted
    NOT_FIRING = "not_firing"  # nothing claimed under that name; nothing changed


@dataclass(frozen=True)
class Finished:
    """The result of :meth:`Agenda.finish`."""

    outcome: Outcome
    next_fire_at: float | None = None
    chain_position: int | None = None
    cap: int | None = None


@dataclass(frozen=True)
class Due:
    """An entry claimed for firing. Hand it back to :meth:`Agenda.finish`."""

    kind: Kind
    row: AgendaRow


@dataclass(frozen=True)
class Claim:
    """What :meth:`Agenda.claim_due` found."""

    due: list[Due]
    expired: list[Due]  # one-shots too far overdue to fire; already deleted


class Agenda:
    """Reminders and scheduled LLM tasks: what is due, and what follows a fire.

    Thread-safe: :meth:`claim_due` runs on the scheduler thread and
    :meth:`finish` on whichever thread ran the fire.
    """

    def __init__(
        self,
        store: Store,
        *,
        clock: Callable[[], float] = time.time,
        owner_tz: Callable[[str, str | None], tzinfo] = lambda _nick, _account: UTC,
    ) -> None:
        self._clock = clock
        self._owner_tz = owner_tz
        self._store = store
        self._tables: dict[Kind, _Table] = {
            Kind.REMINDER: _Table(
                get=store.get_reminder,
                due=store.load_due_reminders,
                fire_times=store.reminder_fire_times,
                advance=lambda name, at, pos: store.update_reminder_fire_at(
                    name, at, chain_position=pos
                ),
                delete=store.delete_reminder,
            ),
            Kind.TASK: _Table(
                get=store.get_scheduled_llm_task,
                due=store.load_due_scheduled_llm_tasks,
                fire_times=store.scheduled_llm_task_fire_times,
                advance=lambda name, at, pos: store.update_scheduled_llm_task_fire_at(
                    name, at, chain_position=pos
                ),
                delete=store.delete_scheduled_llm_task,
            ),
        }
        self._lock = threading.Lock()
        self._firing: set[tuple[Kind, str]] = set()

    def claim_due(self) -> Claim:
        """Claim every entry whose fire time has come and that is not firing.

        Claimed entries stay hidden from :meth:`claim_due` and
        :meth:`next_due_at` until :meth:`finish` releases them, so a slow
        fire is never started twice. One-shots more than a day overdue (the
        bot was down) are deleted rather than delivered that late.
        """
        now = self._clock()
        stale_before = now - EXPIRY_THRESHOLD_SECONDS
        claim = Claim(due=[], expired=[])
        with self._lock:
            for kind, table in self._tables.items():
                for row in table.due(now):
                    key = (kind, row.event_name)
                    if key in self._firing:
                        continue
                    if not is_recurring(row) and row.fire_at <= stale_before:
                        table.delete(row.event_name)
                        claim.expired.append(Due(kind, row))
                        continue
                    self._firing.add(key)
                    claim.due.append(Due(kind, row))
        claim.due.sort(key=lambda d: d.row.fire_at)
        return claim

    def finish(self, kind: Kind, event_name: str, *, fired_at: float) -> Finished:
        """Settle an entry after its fire: reschedule it in place or remove it.

        Re-reads the row first. A row deleted while the fire ran was
        cancelled, and stays cancelled. Otherwise a one-shot is deleted, and
        a recurring entry either moves to its next fire (same row, same id,
        ``chain_position`` + 1) or is deleted at its kind's cap or when its
        RRULE runs out. Calling it for an entry that is not firing is a no-op,
        so a double finish cannot advance a chain twice.
        """
        key = (kind, event_name)
        with self._lock:
            if key not in self._firing:
                return Finished(Outcome.NOT_FIRING)
            self._firing.discard(key)
        table = self._tables[kind]
        row = table.get(event_name)
        if row is None:
            return Finished(Outcome.CANCELLED)
        if not is_recurring(row):
            table.delete(event_name)
            return Finished(Outcome.DONE)
        cap = MAX_CHAIN_POSITION[kind]
        position = row.chain_position + 1
        if position > cap:
            table.delete(event_name)
            return Finished(Outcome.CAPPED, chain_position=position, cap=cap)
        tz = self._owner_tz(owner_nick(row), row.account)
        at = next_fire(row, fired_at=fired_at, now=self._clock(), tz=tz)
        if at is None:
            table.delete(event_name)
            return Finished(Outcome.EXHAUSTED)
        table.advance(event_name, at, position)
        return Finished(Outcome.RESCHEDULED, next_fire_at=at, chain_position=position, cap=cap)

    def next_due_at(self) -> float | None:
        """The earliest fire time among entries not already firing."""
        with self._lock:
            firing = set(self._firing)
        earliest: float | None = None
        for kind, table in self._tables.items():
            for name, at in table.fire_times():
                if (kind, name) in firing:
                    continue
                if earliest is None or at < earliest:
                    earliest = at
                break  # fire_times() is ordered: the first unclaimed row wins
        return earliest

    def firing(self, kind: Kind, event_name: str) -> bool:
        """True while a claimed entry's fire has not finished."""
        with self._lock:
            return (kind, event_name) in self._firing

    def pending_reminders(self) -> list[ReminderRow]:
        """Reminders still to come, oldest first.

        A one-shot that is firing right now is no longer pending (an action
        that lists its owner's reminders must not list itself). A firing
        recurring reminder still is: it has fires to come.
        """
        return [
            row
            for row in self._store.load_pending_reminders()
            if is_recurring(row) or not self.firing(Kind.REMINDER, row.event_name)
        ]
