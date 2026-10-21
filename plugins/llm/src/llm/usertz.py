"""Per-user time zones: validating a zone name and reading a CTCP TIME reply.

IRC carries no time zone. Every ``server-time`` tag is UTC, so "tomorrow
morning" parsed without help lands at 09:00 UTC — 05:00 for someone in
Ontario (observed 2026-09-26). Two sources fill the gap, in this order:

1. A zone the user set with ``@tz`` (an IANA name, so DST is handled).
2. The user's client clock, read from a CTCP TIME reply. That gives an
   offset only, not a zone: good enough for a one-shot reminder, and a
   recurring one drifts by an hour across a DST change until ``@tz`` is set.

Everything here is pure; the plugin owns the probe and the cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as date_parser

# Real zones span UTC-12 to UTC+14. A client clock outside that is wrong,
# not exotic.
_MAX_OFFSET = timedelta(hours=14)
# Every zone in use today sits on a 15-minute boundary. Rounding to it
# absorbs clock skew and the seconds the reply spent in transit.
_OFFSET_STEP_SECONDS = 15 * 60

SOURCE_SET = "set"
SOURCE_CTCP = "ctcp"
SOURCE_DEFAULT = "default"


@dataclass(frozen=True)
class UserTz:
    """A resolved zone plus where it came from, for the parser prompt."""

    tz: tzinfo
    source: str  # SOURCE_SET | SOURCE_CTCP | SOURCE_DEFAULT

    def label(self, now: datetime | None = None) -> str:
        """``America/Toronto (UTC-04:00)``, ``UTC-04:00`` or ``UTC``."""
        now = now or datetime.now(UTC)
        offset = format_offset(now.astimezone(self.tz).utcoffset() or timedelta(0))
        if isinstance(self.tz, ZoneInfo) and self.tz.key != "UTC":
            return f"{self.tz.key} ({offset})"
        return offset


UTC_DEFAULT = UserTz(tz=UTC, source=SOURCE_DEFAULT)


_WALL_CLOCK_RE = re.compile(
    r"""
    \b\d{1,2}(?::\d{2})?\s*(?:a\.?m\b\.?|p\.?m\b\.?)   # 5pm, 5:30 a.m.
    | \b\d{1,2}:\d{2}\b                               # 17:30
    | \bat\s+\d{1,2}\b                                # at 5
    | \b\d{4}-\d{1,2}-\d{1,2}\b | \b\d{1,2}/\d{1,2}\b  # 2026-09-28, 9/28
    | \b\d{1,2}(?:st|nd|rd|th)\b                      # the 3rd
    | \b(?:o'?clock|noon|midnight|today|tonight|tomorrow|tmrw|tmr
        |morning|afternoon|evening|night|lunch|dinner|breakfast|bedtime
        |eod|weekday|weekend|next\s+(?:week|month|year)
        |mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun
        |(?:mon|tues|wednes|thurs|fri|satur|sun)days?
        |jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec
        |january|february|march|april|june|july|august|september
        |october|november|december)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


def needs_zone(text: str) -> bool:
    """True when ``text`` names a wall-clock time or calendar day.

    "in 20 minutes" and "every hour" land the same instant in any zone, so
    asking the user's client for its clock is wasted (and, observed
    2026-09-28, visible: Rubin got TIME'd for "remind d0nk when it's going
    to rain"). Errs towards True: a spare probe costs 3 s at most, a missed
    one parses "at 5pm" as UTC.
    """
    return _WALL_CLOCK_RE.search(text) is not None


def parse_zone(name: str) -> ZoneInfo | None:
    """Return the IANA zone for ``name``, or None when it is not one."""
    name = name.strip()
    # ZoneInfo reads a path under the tz database; refuse anything that
    # could walk out of it rather than trusting the library to.
    if not name or ".." in name or name.startswith("/") or len(name) > 64:
        return None
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError, ValueError:
        return None


def format_offset(offset: timedelta) -> str:
    """``UTC``, ``UTC+05:30`` or ``UTC-04:00``."""
    total = int(offset.total_seconds())
    if total == 0:
        return "UTC"
    sign = "+" if total > 0 else "-"
    hours, rem = divmod(abs(total), 3600)
    return f"UTC{sign}{hours:02d}:{rem // 60:02d}"


def offset_from_ctcp_time(reply: str, now_utc: datetime) -> timezone | None:
    """Turn a CTCP TIME reply into the sender's UTC offset.

    Clients disagree on the format: HexChat and irssi send a bare local
    ``Sat Sep 26 05:12:03 2026``, WeeChat an RFC 2822 date with ``-0400``.
    An explicit offset is used as given; a bare time is compared with
    ``now_utc``. Returns None for anything unparseable or out of range.
    """
    try:
        # A reply with no date takes now_utc's, not the host calendar's.
        parsed = date_parser.parse(reply.strip(), fuzzy=True, default=now_utc.replace(tzinfo=None))
    except ValueError, OverflowError:
        return None

    explicit = parsed.utcoffset() if parsed.tzinfo is not None else None
    if explicit is not None:
        offset = explicit
    else:
        offset = parsed.replace(tzinfo=None) - now_utc.replace(tzinfo=None)
        # A reply with no date gets today's date filled in, which is a day
        # off either side of midnight. Fold that back; a clock wrong by
        # days stays out of range and is rejected below.
        if _MAX_OFFSET < offset < timedelta(hours=36):
            offset -= timedelta(days=1)
        elif -timedelta(hours=36) < offset < -timedelta(hours=12):
            offset += timedelta(days=1)

    steps = round(offset.total_seconds() / _OFFSET_STEP_SECONDS)
    offset = timedelta(seconds=steps * _OFFSET_STEP_SECONDS)
    if abs(offset) > _MAX_OFFSET:
        return None
    return timezone(offset)
