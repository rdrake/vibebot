"""Aurora and space-weather model — pure, no supybot imports.

This module turns NOAA SWPC real-time solar-wind (RTSW) rows into minute
``Sample`` records and tracks "storm conditions" episodes: sustained strongly
southward interplanetary Bz. It is a heuristic for sustained storm-driving
conditions, not a guarantee of a geomagnetic storm.

Kept free of Limnoria and ``llm.service`` imports for the same reason as
``llm.statuspage``: it unit-tests against plain dicts and datetimes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

# Distance from L1 to Earth, km. Lead time = distance / wind speed.
L1_DISTANCE_KM = 1.5e6

# An episode ends when the 30-minute mean Bz rises above this.
EPISODE_END_BZ_NT = -5.0

# Fraction of the window that must have data, and must meet the threshold.
_COVERAGE = 0.8

# Newest sample older than this means the feed is stale.
_STALE = timedelta(minutes=15)

_ONE_MINUTE = timedelta(minutes=1)


class InvalidPayload(ValueError):  # noqa: N818
    """An RTSW payload was not the list-of-objects shape the feed promises."""


@dataclass(frozen=True)
class Sample:
    """One UTC minute of solar-wind data from the active spacecraft."""

    ts: datetime
    bz: float | None
    bt: float | None
    speed: float | None


@dataclass(frozen=True)
class Episode:
    """A running spell of sustained southward Bz."""

    start: datetime
    min_bz: float
    latest_speed: float | None
    mean_ey: float | None
    lead_minutes: int | None


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    f = float(value)
    return f if math.isfinite(f) else None


def _minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def _parse_ts(tag: Any) -> datetime | None:
    if not isinstance(tag, str):
        return None
    try:
        ts = datetime.fromisoformat(tag)
    except ValueError:
        return None
    ts = ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts.astimezone(UTC)
    return _minute(ts)


def _active_rows(rows: Any, field: str) -> dict[datetime, dict[str, Any]]:
    """Active rows keyed by UTC minute; a later row for a minute replaces an earlier one."""
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        raise InvalidPayload(f"{field} is not a list of objects")
    out: dict[datetime, dict[str, Any]] = {}
    for row in rows:
        if row.get("active") is not True:
            continue
        ts = _parse_ts(row.get("time_tag"))
        if ts is not None:
            out[ts] = row
    return out


def parse_rtsw(mag: list[dict], wind: list[dict]) -> list[Sample]:
    """Active-spacecraft rows to one Sample per UTC minute, oldest first.

    Null, NaN, infinite and non-numeric values become None; a proton speed
    of zero or less is a dropout, so it is None too. A missing minute is
    never reported as 0.
    """
    mag_rows = _active_rows(mag, "mag")
    wind_rows = _active_rows(wind, "wind")
    samples = []
    for ts in sorted(mag_rows.keys() | wind_rows.keys()):
        m = mag_rows.get(ts, {})
        speed = _finite(wind_rows.get(ts, {}).get("proton_speed"))
        samples.append(
            Sample(
                ts=ts,
                bz=_finite(m.get("bz_gsm")),
                bt=_finite(m.get("bt")),
                speed=speed if speed is not None and speed > 0 else None,
            )
        )
    return samples


def _by_minute(samples: Sequence[Sample]) -> dict[datetime, Sample]:
    """Deduplicate by UTC minute (later entry wins)."""
    return {_minute(s.ts): s for s in samples}


def _window(
    samples: Sequence[Sample], start_exclusive: datetime, end_inclusive: datetime
) -> list[Sample]:
    return [s for ts, s in _by_minute(samples).items() if start_exclusive < ts <= end_inclusive]


def storm_conditions(
    samples: Sequence[Sample], *, threshold_nt: float, hours: float, now: datetime
) -> bool:
    """True when Bz stayed at or below the threshold for most of the trailing window."""
    if not samples or now - max(s.ts for s in samples) > _STALE:
        return False
    window_minutes = hours * 60
    bz = [
        s.bz
        for s in _window(samples, now - timedelta(minutes=window_minutes), now)
        if s.bz is not None
    ]
    if len(bz) < _COVERAGE * window_minutes:
        return False
    return sum(1 for v in bz if v <= threshold_nt) >= _COVERAGE * len(bz)


def trailing_mean_bz(
    samples: Sequence[Sample], *, now: datetime, minutes: int = 30, min_points: int = 24
) -> float | None:
    """Mean Bz over the trailing window, or None with fewer than min_points minutes."""
    bz = [s.bz for s in _window(samples, now - timedelta(minutes=minutes), now) if s.bz is not None]
    return sum(bz) / len(bz) if len(bz) >= min_points else None


def _stats(
    samples: Sequence[Sample], start: datetime, now: datetime
) -> tuple[float | None, float | None, float | None]:
    """(min Bz, mean Ey in mV/m, mean speed over last 30 min) for [start, now]."""
    in_episode = [s for s in _by_minute(samples).values() if start <= s.ts <= now]
    bz = [s.bz for s in in_episode if s.bz is not None]
    ey = [s.speed * -s.bz / 1000 for s in in_episode if s.bz is not None and s.bz < 0 and s.speed]
    recent = [
        s.speed for s in _window(samples, now - timedelta(minutes=30), now) if s.speed is not None
    ]
    return (
        min(bz) if bz else None,
        sum(ey) / len(ey) if ey else None,
        sum(recent) / len(recent) if recent else None,
    )


def _lead_minutes(speed: float | None) -> int | None:
    return round(L1_DISTANCE_KM / speed / 60) if speed else None


def _walk_back_start(samples: Sequence[Sample], now: datetime) -> datetime | None:
    """Earliest minute of the unbroken run, ending at the newest sample, whose
    30-minute mean Bz exists and is at or below the episode-end level."""
    by_min = _by_minute(samples)
    t = max(ts for ts in by_min if ts <= now)
    oldest = min(by_min)
    start = None
    while t >= oldest:
        mean = trailing_mean_bz(samples, now=t)
        if mean is None or mean > EPISODE_END_BZ_NT:
            break
        start = t
        t -= _ONE_MINUTE
    return start


def advance_episode(
    current: Episode | None,
    samples: Sequence[Sample],
    *,
    threshold_nt: float,
    hours: float,
    now: datetime,
) -> Episode | None:
    """Next state of the storm-conditions episode.

    Starts when storm_conditions holds, with the start walked back to where
    the 30-minute mean Bz first sank to -5 nT. Once running, the start is
    latched: it never moves, recoveries that keep the 30-minute mean at or
    below -5 nT continue the episode, and a data gap keeps it alive. It ends
    when the 30-minute mean rises above -5 nT.
    """
    if current is None:
        if not storm_conditions(samples, threshold_nt=threshold_nt, hours=hours, now=now):
            return None
        start = _walk_back_start(samples, now)
        if start is None:
            return None
        min_bz, ey, speed = _stats(samples, start, now)
        return Episode(
            start=start,
            min_bz=min_bz if min_bz is not None else threshold_nt,
            latest_speed=speed,
            mean_ey=ey,
            lead_minutes=_lead_minutes(speed),
        )

    mean = trailing_mean_bz(samples, now=now)
    if mean is not None and mean > EPISODE_END_BZ_NT:
        return None
    min_bz, ey, speed = _stats(samples, current.start, now)
    speed = speed if speed is not None else current.latest_speed
    return Episode(
        start=current.start,
        min_bz=current.min_bz if min_bz is None else min(current.min_bz, min_bz),
        latest_speed=speed,
        mean_ey=ey if ey is not None else current.mean_ey,
        lead_minutes=_lead_minutes(speed),
    )
