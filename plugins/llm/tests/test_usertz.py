"""Tests for llm.usertz: zone validation and CTCP TIME offset parsing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from llm.usertz import (
    SOURCE_CTCP,
    SOURCE_SET,
    UTC_DEFAULT,
    UserTz,
    format_offset,
    needs_zone,
    offset_from_ctcp_time,
    parse_zone,
)

# 09:00 UTC on the morning the "strike" reminder fired at 05:00 Ontario time.
NOW = datetime(2026, 9, 26, 9, 0, 0, tzinfo=UTC)


class TestParseZone:
    def test_iana_name(self) -> None:
        assert parse_zone("America/Toronto") == ZoneInfo("America/Toronto")

    def test_strips_whitespace(self) -> None:
        assert parse_zone("  Europe/London ") == ZoneInfo("Europe/London")

    @pytest.mark.parametrize(
        "name", ["", "Mars/Olympus", "EDT-ish", "../../etc/passwd", "/etc/localtime"]
    )
    def test_rejects_non_zones(self, name: str) -> None:
        assert parse_zone(name) is None


class TestFormatOffset:
    @pytest.mark.parametrize(
        ("offset", "expected"),
        [
            (timedelta(0), "UTC"),
            (timedelta(hours=-4), "UTC-04:00"),
            (timedelta(hours=5, minutes=30), "UTC+05:30"),
        ],
    )
    def test_format(self, offset: timedelta, expected: str) -> None:
        assert format_offset(offset) == expected


class TestUserTzLabel:
    def test_named_zone_shows_name_and_current_offset(self) -> None:
        tz = UserTz(tz=ZoneInfo("America/Toronto"), source=SOURCE_SET)
        assert tz.label(NOW) == "America/Toronto (UTC-04:00)"

    def test_named_zone_follows_dst(self) -> None:
        tz = UserTz(tz=ZoneInfo("America/Toronto"), source=SOURCE_SET)
        assert tz.label(datetime(2026, 12, 1, tzinfo=UTC)) == "America/Toronto (UTC-05:00)"

    def test_fixed_offset_shows_offset_only(self) -> None:
        tz = UserTz(tz=timezone(timedelta(hours=-4)), source=SOURCE_CTCP)
        assert tz.label(NOW) == "UTC-04:00"

    def test_utc_default_is_labelled_utc(self) -> None:
        # The fallback for a user with no zone: absolute times are UTC.
        assert UTC_DEFAULT.label(NOW) == "UTC"
        assert UserTz(tz=ZoneInfo("UTC"), source=SOURCE_SET).label(NOW) == "UTC"

    def test_named_zone_at_zero_offset_shows_utc(self) -> None:
        # London in winter sits at UTC+0; the label never shows a nonzero offset.
        tz = UserTz(tz=ZoneInfo("Europe/London"), source=SOURCE_SET)
        assert tz.label(datetime(2026, 12, 1, tzinfo=UTC)) == "Europe/London (UTC)"


class TestOffsetFromCtcpTime:
    def test_bare_local_time_hexchat_style(self) -> None:
        tz = offset_from_ctcp_time("Sat Sep 26 05:00:02 2026", NOW)
        assert tz == timezone(timedelta(hours=-4))

    def test_explicit_offset_weechat_style(self) -> None:
        tz = offset_from_ctcp_time("Sat, 26 Sep 2026 14:30:00 +0530", NOW)
        assert tz == timezone(timedelta(hours=5, minutes=30))

    def test_rounds_skew_to_quarter_hour(self) -> None:
        # 3 minutes of clock skew on a UTC+1 client.
        tz = offset_from_ctcp_time("Sat Sep 26 10:03:00 2026", NOW)
        assert tz == timezone(timedelta(hours=1))

    def test_time_without_date_across_midnight(self) -> None:
        # 23:30 local while UTC is already 03:30 the next day: UTC-4, not UTC+20.
        now = datetime(2026, 9, 27, 3, 30, tzinfo=UTC)
        tz = offset_from_ctcp_time("23:30:00", now)
        assert tz == timezone(timedelta(hours=-4))

    def test_clock_days_off_is_rejected(self) -> None:
        assert offset_from_ctcp_time("Mon Sep 21 09:00:00 2026", NOW) is None

    @pytest.mark.parametrize("reply", ["", "no idea", "\x01"])
    def test_garbage_is_rejected(self, reply: str) -> None:
        assert offset_from_ctcp_time(reply, NOW) is None


class TestNeedsZone:
    @pytest.mark.parametrize(
        "text",
        [
            "at 5pm call Bob",
            "5:30 a.m. stretch",
            "17:30 standup",
            "at 9 check the build",
            "tomorrow morning strike",
            "tonight feed the cat",
            "every Monday post the agenda",
            "on the 3rd pay rent",
            "Sept 30 renew the cert",
            "2026-10-01 renew the cert",
            "at noon lunch",
            "next week review the PR",
        ],
    )
    def test_wall_clock_times(self, text: str) -> None:
        assert needs_zone(text)

    @pytest.mark.parametrize(
        "text",
        [
            "d0nk when its going to rain",
            "in 20 minutes check the oven",
            "every hour tag rdrake with a penguin gif",
            "in 2 days renew the cert",
            "let me know when the build is green",
            "every 5 minutes check the Debian CVE-2026-31431 status",
        ],
    )
    def test_relative_times(self, text: str) -> None:
        assert not needs_zone(text)
