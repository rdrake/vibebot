"""Property-based tests for reminder & scheduled-LLM-task CRUD.

Covers ``persistence.py:545-880``: ``save_reminder``,
``delete_reminder``, ``load_pending_reminders``,
the agenda reads and the parallel ``*_scheduled_llm_task``
family. Properties focus on the invariants the audit flagged:
recurrence mutual exclusion (seconds xor rrule), case-insensitive
owner matching (the migration history shows this has churned), and
``fire_at`` ordering on the load paths.

Custom strategies are local to this module rather than being shared
with the other property files so that the constraint set (no NUL
bytes, fire_at offsets bounded by ``EXPIRY_THRESHOLD_SECONDS``) reads
in one place.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies import (
    booleans,
    integers,
    lists,
    none,
    one_of,
    sampled_from,
    text,
)
from llm.persistence import EXPIRY_THRESHOLD_SECONDS, LLMDatabase

# Bounded enough to keep generated event_names distinct via index suffix.
_NICK_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
nicks = text(alphabet=_NICK_ALPHABET, min_size=1, max_size=15)
channels = sampled_from(["#a", "#b", "#c", "#priv1", "#priv2"])
networks = sampled_from(["afternet", "freenode"])
accounts = one_of(none(), nicks)

# fire_at offsets relative to ``time.time()``. The lower bound is well
# inside the 24h ``EXPIRY_THRESHOLD_SECONDS`` cutoff so saved rows
# survive ``load_pending_reminders`` / ``load_scheduled_llm_tasks_for``;
# the upper bound is ~2 days in the future. Using ``time.time()`` (read
# at the top of each test) rather than a fixed anchor is required: the
# implementation hard-codes ``time.time() - EXPIRY_THRESHOLD_SECONDS``
# at the cutoff, so any fixed anchor would drift relative to wall clock
# and silently filter out the row.
_MIN_OFFSET = -EXPIRY_THRESHOLD_SECONDS // 2
_MAX_OFFSET = 2 * EXPIRY_THRESHOLD_SECONDS
fire_at_offsets = integers(min_value=_MIN_OFFSET, max_value=_MAX_OFFSET)


def _new_db() -> tuple[LLMDatabase, Path]:
    work_dir = Path(tempfile.mkdtemp(prefix="sched-prop-"))
    return LLMDatabase(str(work_dir / "p.db")), work_dir


def _save_task(
    db: LLMDatabase,
    *,
    event_name: str,
    creator_nick: str,
    account: str | None,
    channel: str,
    fire_at: float,
    recurrence_seconds: int | None = None,
    recurrence_rrule: str | None = None,
) -> int:
    return db.save_scheduled_llm_task(
        event_name=event_name,
        creator_nick=creator_nick,
        account=account,
        channel=channel,
        network="afternet",
        wire_msg=f":{creator_nick}!u@h PRIVMSG {channel} :@ask hi",
        prompt="p",
        fire_at=fire_at,
        recurrence_seconds=recurrence_seconds,
        recurrence_rrule=recurrence_rrule,
    )


# ----------------------------------------------------------------------
# Reminder properties
# ----------------------------------------------------------------------


@given(nick=nicks, channel=channels, offset=fire_at_offsets)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_reminder_round_trip(nick: str, channel: str, offset: float) -> None:
    """``save → load → delete → load`` removes only the saved row."""
    db, work_dir = _new_db()
    try:
        fire_at = time.time() + offset
        db.save_reminder("evt", nick, channel, "msg", fire_at)
        loaded = db.load_pending_reminders()
        names_after_save = {r.event_name for r in loaded}
        assert "evt" in names_after_save
        assert db.delete_reminder("evt") is True
        names_after_delete = {r.event_name for r in db.load_pending_reminders()}
        assert "evt" not in names_after_delete
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    seconds=one_of(none(), integers(min_value=60, max_value=3600)),
    rrule=one_of(none(), sampled_from(["FREQ=DAILY", "FREQ=HOURLY"])),
)
@settings(max_examples=20, deadline=None)
def test_reminder_recurrence_mutual_exclusion(seconds: int | None, rrule: str | None) -> None:
    """``save_reminder`` raises ``ValueError`` iff both recurrence kinds are set."""
    db, work_dir = _new_db()
    try:
        if seconds is not None and rrule is not None:
            with pytest.raises(ValueError, match="mutually exclusive"):
                db.save_reminder(
                    "evt",
                    "n",
                    "#x",
                    "m",
                    time.time() + 60,
                    recurrence_seconds=seconds,
                    recurrence_rrule=rrule,
                )
        else:
            db.save_reminder(
                "evt",
                "n",
                "#x",
                "m",
                time.time() + 60,
                recurrence_seconds=seconds,
                recurrence_rrule=rrule,
            )
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(offsets=lists(fire_at_offsets, min_size=1, max_size=10, unique=True))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_reminder_load_is_sorted_by_fire_at(offsets: list[float]) -> None:
    """``load_pending_reminders`` returns rows ordered by ``fire_at`` ascending."""
    db, work_dir = _new_db()
    try:
        for i, off in enumerate(offsets):
            db.save_reminder(f"evt_{i}", "n", "#x", "m", time.time() + off)
        loaded = db.load_pending_reminders()
        # Strategy keeps every row inside the 24h cutoff, so all inserted
        # rows survive the load. Without this, an empty list would
        # vacuously satisfy the sort assertion.
        assert len(loaded) == len(offsets)
        fire_ats = [r.fire_at for r in loaded]
        assert fire_ats == sorted(fire_ats)
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# Scheduled-LLM-task properties
# ----------------------------------------------------------------------


@given(
    creator=nicks,
    account=accounts,
    channel=channels,
    offset=fire_at_offsets,
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_scheduled_task_round_trip(
    creator: str, account: str | None, channel: str, offset: float
) -> None:
    """``save → load → delete → load`` is a clean round-trip."""
    db, work_dir = _new_db()
    try:
        _save_task(
            db,
            event_name="evt",
            creator_nick=creator,
            account=account,
            channel=channel,
            fire_at=time.time() + offset,
        )
        names_after_save = {name for name, _at in db.scheduled_llm_task_fire_times()}
        assert "evt" in names_after_save
        row = db.get_scheduled_llm_task("evt")
        assert row is not None and row.creator_nick == creator and row.account == account
        assert db.delete_scheduled_llm_task("evt") is True
        names_after_delete = {name for name, _at in db.scheduled_llm_task_fire_times()}
        assert "evt" not in names_after_delete
        assert db.get_scheduled_llm_task("evt") is None
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    seconds=one_of(none(), integers(min_value=60, max_value=3600)),
    rrule=one_of(none(), sampled_from(["FREQ=DAILY", "FREQ=HOURLY"])),
)
@settings(max_examples=20, deadline=None)
def test_scheduled_task_recurrence_mutual_exclusion(seconds: int | None, rrule: str | None) -> None:
    """``save_scheduled_llm_task`` raises ``ValueError`` iff both recurrence kinds are set."""
    db, work_dir = _new_db()
    try:
        if seconds is not None and rrule is not None:
            with pytest.raises(ValueError, match="mutually exclusive"):
                _save_task(
                    db,
                    event_name="evt",
                    creator_nick="n",
                    account=None,
                    channel="#x",
                    fire_at=time.time() + 60,
                    recurrence_seconds=seconds,
                    recurrence_rrule=rrule,
                )
        else:
            _save_task(
                db,
                event_name="evt",
                creator_nick="n",
                account=None,
                channel="#x",
                fire_at=time.time() + 60,
                recurrence_seconds=seconds,
                recurrence_rrule=rrule,
            )
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(offsets=lists(fire_at_offsets, min_size=1, max_size=10, unique=True))
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_scheduled_task_load_is_sorted_by_fire_at(offsets: list[float]) -> None:
    """``load_due_scheduled_llm_tasks`` returns rows ordered by ``fire_at`` ascending."""
    db, work_dir = _new_db()
    try:
        for i, off in enumerate(offsets):
            _save_task(
                db,
                event_name=f"evt_{i}",
                creator_nick="n",
                account=None,
                channel="#x",
                fire_at=time.time() + off,
            )
        loaded = db.load_due_scheduled_llm_tasks(time.time() + 30 * 86400)
        assert len(loaded) == len(offsets)
        fire_ats = [r.fire_at for r in loaded]
        assert fire_ats == sorted(fire_ats)
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    account=nicks,
    channel=channels,
    creator=nicks,
    upper_account=booleans(),
    upper_lookup=booleans(),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_count_scheduled_tasks_account_match_is_case_insensitive(
    account: str,
    channel: str,
    creator: str,
    upper_account: bool,
    upper_lookup: bool,
) -> None:
    """``count_scheduled_llm_tasks_for(account=A, …)`` is invariant under case of ``A``."""
    db, work_dir = _new_db()
    try:
        stored_account = account.upper() if upper_account else account.lower()
        _save_task(
            db,
            event_name="evt",
            creator_nick=creator,
            account=stored_account,
            channel=channel,
            fire_at=time.time() + 60,
        )
        lookup_account = account.upper() if upper_lookup else account.lower()
        count = db.count_scheduled_llm_tasks_for(
            account=lookup_account, nick="anything", channel=channel
        )
        assert count == 1
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    creator=nicks,
    channel=channels,
    upper_creator=booleans(),
    upper_lookup=booleans(),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_count_scheduled_tasks_nick_match_is_case_insensitive_when_account_none(
    creator: str,
    channel: str,
    upper_creator: bool,
    upper_lookup: bool,
) -> None:
    """When ``account is None``, owner match falls back to nick (case-insensitive)."""
    db, work_dir = _new_db()
    try:
        stored_creator = creator.upper() if upper_creator else creator.lower()
        _save_task(
            db,
            event_name="evt",
            creator_nick=stored_creator,
            account=None,
            channel=channel,
            fire_at=time.time() + 60,
        )
        lookup_nick = creator.upper() if upper_lookup else creator.lower()
        count = db.count_scheduled_llm_tasks_for(account=None, nick=lookup_nick, channel=channel)
        assert count == 1
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)


@given(
    creator=nicks,
    account=nicks,
    channel=channels,
)
@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_count_scheduled_tasks_account_overrides_nick_when_present(
    creator: str, account: str, channel: str
) -> None:
    """A row stored with ``account`` is matched by account, not by raw nick.

    Mirror of ``Identity.matches`` semantics: when the stored row has
    an account, lookups by ``account=None, nick=creator`` must not see
    it (to avoid leaking ownership across identities).
    """
    db, work_dir = _new_db()
    try:
        _save_task(
            db,
            event_name="evt",
            creator_nick=creator,
            account=account,
            channel=channel,
            fire_at=time.time() + 60,
        )
        nick_only_count = db.count_scheduled_llm_tasks_for(
            account=None, nick=creator, channel=channel
        )
        account_count = db.count_scheduled_llm_tasks_for(
            account=account, nick="anything", channel=channel
        )
        assert nick_only_count == 0
        assert account_count == 1
    finally:
        db.close()
        shutil.rmtree(work_dir, ignore_errors=True)
