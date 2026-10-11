"""Solar-wind parsing and the sustained-southward-Bz episode lifecycle.

Pure-function tests for llm.spaceweather: RTSW rows in, Samples and Episodes
out. No network, no IRC scaffolding.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from llm import spaceweather
from llm.spaceweather import Episode, InvalidPayload, Sample

FIXTURES = Path(__file__).parent / "fixtures" / "spaceweather"
NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)


def _load(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text())


def _series(minutes, bz, speed=600.0, *, end=NOW):
    """`minutes` one-minute Samples ending at `end` (inclusive), constant bz."""
    return [
        Sample(ts=end - timedelta(minutes=i), bz=bz, bt=abs(bz), speed=speed)
        for i in range(minutes - 1, -1, -1)
    ]


def _mag_row(ts: str, bz, *, active=True, bt=5.0):
    return {"time_tag": ts, "active": active, "source": "ACE", "bz_gsm": bz, "bt": bt}


def _wind_row(ts: str, speed, *, active=True):
    return {"time_tag": ts, "active": active, "source": "ACE", "proton_speed": speed}


STORM = {"threshold_nt": -10.0, "hours": 3.0}


def test_inactive_spacecraft_rows_ignored():
    mag = _load("rtsw_mag_1m.json")
    active_minutes = {r["time_tag"][:16] for r in mag if r["active"] is True}
    assert {r["source"] for r in mag} >= {"ACE", "IMAP", "SOLAR1"}
    samples = spaceweather.parse_rtsw(mag, [])
    assert len(samples) == len(active_minutes)
    assert [s.ts for s in samples] == sorted(s.ts for s in samples)
    assert all(s.ts.tzinfo is UTC and s.ts.second == 0 for s in samples)


def test_fixture_wind_speed_joins_mag_minute():
    samples = spaceweather.parse_rtsw(_load("rtsw_mag_1m.json"), _load("rtsw_wind_1m.json"))
    assert any(s.bz is not None and s.speed is not None for s in samples)


def test_duplicate_minutes_count_once():
    mag = [_mag_row("2026-10-11T11:59:00", -12.0), _mag_row("2026-10-11T11:59:30", -11.0)]
    samples = spaceweather.parse_rtsw(mag, [])
    assert len(samples) == 1
    assert samples[0].bz == -11.0  # later row wins

    # 30 distinct minutes, each present twice: still only 30 minutes of coverage.
    padded = _series(30, -12.0) * 2
    assert not spaceweather.storm_conditions(padded, threshold_nt=-10.0, hours=1.0, now=NOW)
    assert spaceweather.storm_conditions(_series(60, -12.0), threshold_nt=-10.0, hours=1.0, now=NOW)


def test_null_nan_and_zero_speed_are_missing():
    mag = [
        _mag_row("2026-10-11T11:57:00", None),
        _mag_row("2026-10-11T11:58:00", float("nan")),
        _mag_row("2026-10-11T11:59:00", float("inf")),
    ]
    wind = [
        _wind_row("2026-10-11T11:57:00", 0),
        _wind_row("2026-10-11T11:58:00", -3.0),
        _wind_row("2026-10-11T11:59:00", 450.0),
    ]
    by_ts = {s.ts.minute: s for s in spaceweather.parse_rtsw(mag, wind)}
    assert by_ts[57].bz is None
    assert by_ts[58].bz is None
    assert by_ts[59].bz is None
    assert by_ts[57].speed is None
    assert by_ts[58].speed is None
    assert by_ts[59].speed == 450.0


def test_non_numeric_fields_are_missing():
    mag = [_mag_row("2026-10-11T11:59:00", "-12"), _mag_row("2026-10-11T11:58:00", True)]
    assert [s.bz for s in spaceweather.parse_rtsw(mag, [])] == [None, None]


@pytest.mark.parametrize("bad", [{"a": 1}, "x", None, [1], [[]]])
def test_invalid_payload_raises(bad):
    with pytest.raises(InvalidPayload):
        spaceweather.parse_rtsw(bad, [])
    with pytest.raises(InvalidPayload):
        spaceweather.parse_rtsw([], bad)


def test_unparseable_time_tag_row_is_skipped():
    mag = [_mag_row("garbage", -12.0), _mag_row("2026-10-11T11:59:00", -12.0)]
    assert len(spaceweather.parse_rtsw(mag, [])) == 1


def test_sustained_minus_12_for_3h_starts_episode():
    ep = spaceweather.advance_episode(None, _series(190, -12.0), now=NOW, **STORM)
    assert ep is not None
    assert ep.min_bz == -12.0
    assert ep.mean_ey == pytest.approx(7.2)
    assert ep.latest_speed == pytest.approx(600.0)
    assert ep.lead_minutes == 42
    # earliest minute whose 30-min mean exists: feed start + 23 min
    assert ep.start == NOW - timedelta(minutes=189) + timedelta(minutes=23)


def test_short_dip_does_not_start():
    samples = _series(120, 2.0, end=NOW) + _series(60, -15.0, end=NOW - timedelta(minutes=120))
    assert (
        spaceweather.advance_episode(None, sorted(samples, key=lambda s: s.ts), now=NOW, **STORM)
        is None
    )


def test_stale_feed_does_not_start():
    samples = _series(190, -12.0, end=NOW - timedelta(minutes=20))
    assert not spaceweather.storm_conditions(samples, now=NOW, **STORM)
    assert spaceweather.advance_episode(None, samples, now=NOW, **STORM) is None


def test_wind_only_minutes_do_not_keep_stalled_bz_fresh():
    """A stalled magnetometer with a live plasma feed is still stale for Bz."""
    bz = _series(190, -12.0, end=NOW - timedelta(minutes=20))
    wind_only = [
        Sample(ts=NOW - timedelta(minutes=i), bz=None, bt=None, speed=600.0)
        for i in range(19, -1, -1)
    ]
    assert not spaceweather.storm_conditions(bz + wind_only, now=NOW, **STORM)


def test_trailing_mean_needs_min_points():
    assert spaceweather.trailing_mean_bz(_series(23, -12.0), now=NOW) is None
    assert spaceweather.trailing_mean_bz(_series(24, -12.0), now=NOW) == pytest.approx(-12.0)


def _running(start=None):
    return Episode(
        start=start or NOW - timedelta(hours=4),
        min_bz=-14.0,
        latest_speed=600.0,
        mean_ey=7.0,
        lead_minutes=42,
    )


def test_episode_latches_through_brief_recovery():
    cur = _running()
    samples = _series(20, -12.0, end=NOW - timedelta(minutes=10)) + _series(10, -4.0)
    ep = spaceweather.advance_episode(cur, samples, now=NOW, **STORM)
    assert ep is not None and ep.start == cur.start
    assert ep.min_bz == -14.0  # never rises


def test_episode_min_bz_deepens():
    ep = spaceweather.advance_episode(_running(), _series(60, -20.0), now=NOW, **STORM)
    assert ep is not None and ep.min_bz == -20.0


def test_episode_survives_data_gap():
    cur = _running()
    samples = _series(120, -12.0, end=NOW - timedelta(minutes=31))
    ep = spaceweather.advance_episode(cur, samples, now=NOW, **STORM)
    assert ep is not None and ep.start == cur.start


def test_episode_ends_when_mean_above_minus_5():
    samples = _series(60, 1.0)
    assert spaceweather.advance_episode(_running(), samples, now=NOW, **STORM) is None


def test_episode_start_stable_past_24h():
    cur = _running(start=NOW - timedelta(hours=30))
    samples = _series(24 * 60, -9.0)  # feed's oldest row is 24 h old, after start
    ep = spaceweather.advance_episode(cur, samples, now=NOW, **STORM)
    assert ep is not None and ep.start == cur.start


def test_northward_minutes_excluded_from_ey():
    # 30-minute mean stays at -7 nT, so the episode continues.
    samples = _series(80, -12.0, speed=500.0, end=NOW - timedelta(minutes=10)) + _series(
        10, 3.0, speed=500.0
    )
    ep = spaceweather.advance_episode(_running(), samples, now=NOW, **STORM)
    assert ep is not None
    assert ep.mean_ey == pytest.approx(6.0)  # 500 * 12 / 1000, northward minutes skipped


def test_nan_never_reaches_samples():
    samples = spaceweather.parse_rtsw([_mag_row("2026-10-11T11:59:00", float("nan"))], [])
    assert samples[0].bz is None and not math.isnan(samples[0].bt)
