"""Aurora and space-weather model — pure, no supybot imports.

This module turns NOAA SWPC real-time solar-wind (RTSW) rows into minute
``Sample`` records and tracks "storm conditions" episodes: sustained strongly
southward interplanetary Bz. It is a heuristic for sustained storm-driving
conditions, not a guarantee of a geomagnetic storm.

Kept free of Limnoria and ``llm.service`` imports for the same reason as
``llm.statuspage``: it unit-tests against plain dicts and datetimes.
"""

from __future__ import annotations

import http.client
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

# Distance from L1 to Earth, km. Lead time = distance / wind speed.
L1_DISTANCE_KM = 1.5e6

# An episode ends when the 30-minute mean Bz rises above this.
EPISODE_END_BZ_NT = -5.0

# Fraction of the window that must have data, and must meet the threshold.
_COVERAGE = 0.8

# Newest Bz minute older than this means the feed is stale.
STALE_AFTER = timedelta(minutes=15)

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
    """True when Bz stayed at or below the threshold for most of the trailing window.

    Staleness is judged on the newest minute that has Bz: plasma-only minutes
    must not keep a stalled magnetometer feed looking fresh.
    """
    bz_times = [s.ts for s in samples if s.bz is not None]
    if not bz_times or now - max(bz_times) > STALE_AFTER:
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


# --- Watches -----------------------------------------------------------------


@dataclass(frozen=True)
class Watch:
    """One NOAA geomagnetic storm watch issuance (WATA*). Level 0 is below G1."""

    serial: int
    issued: datetime
    levels: dict[date, int]


_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_MONTHS = {m: i for i, m in enumerate(_MONTH_NAMES, 1)}
_SERIAL_RE = re.compile(r"Serial Number:\s*(\d+)")
_ISSUE_RE = re.compile(r"Issue Time:\s*(\d{4}) ([A-Z][a-z]{2}) (\d{2}) (\d{2})(\d{2}) UTC")
_BY_DAY_RE = re.compile(
    r"Predicted by Day:\s*\n(.*?)(?:\nTHIS SUPERSEDES|\nComment:|\Z)", re.DOTALL
)
_DAY_RE = re.compile(r"([A-Z][a-z]{2}) (\d{2}):\s+(?:G(\d)\b|None)")
# A cancellation is a line that starts with CANCEL ("CANCEL WATCH: ..."); the
# word inside comment text is not one.
_CANCEL_RE = re.compile(r"^[ \t]*CANCEL", re.MULTILINE)


def _day_levels(text: str, issued: datetime) -> dict[date, int]:
    levels: dict[date, int] = {}
    block = _BY_DAY_RE.search(text)
    if block is None:
        return levels
    for mon, day, g in _DAY_RE.findall(block.group(1)):
        month = _MONTHS.get(mon)
        if month is None:
            continue
        try:
            d = date(issued.year, month, int(day))
            if d < issued.date() - timedelta(days=180):  # December issue, January day
                d = date(issued.year + 1, month, int(day))
        except ValueError:
            continue
        levels[d] = int(g) if g else 0
    return levels


def parse_watches(alerts: list[dict]) -> list[Watch]:
    """Geomagnetic storm watches (message code WATA*) from the SWPC alerts feed.

    A cancellation yields ``levels == {}``. Entries that are not watches, or
    whose serial or issue time cannot be read, are skipped.
    """
    watches = []
    for alert in alerts:
        text = alert.get("message") if isinstance(alert, dict) else None
        if not isinstance(text, str) or "Space Weather Message Code: WATA" not in text:
            continue
        serial = _SERIAL_RE.search(text)
        issue = _ISSUE_RE.search(text)
        if serial is None or issue is None or issue.group(2) not in _MONTHS:
            continue
        try:
            issued = datetime(
                int(issue.group(1)),
                _MONTHS[issue.group(2)],
                int(issue.group(3)),
                int(issue.group(4)),
                int(issue.group(5)),
                tzinfo=UTC,
            )
        except ValueError:
            continue
        levels = {} if _CANCEL_RE.search(text) else _day_levels(text, issued)
        watches.append(Watch(serial=int(serial.group(1)), issued=issued, levels=levels))
    return watches


def effective_levels(watches: list[Watch], *, today: date) -> dict[date, int]:
    """Level per date from the newest issuance alone; dates before today dropped."""
    if not watches:
        return {}
    newest = max(watches, key=lambda w: (w.issued, w.serial))
    return {d: lvl for d, lvl in newest.levels.items() if d >= today}


# --- Darkness ----------------------------------------------------------------

# Dark means the sun is more than 12 degrees below the horizon (nautical twilight).
DARK_ALTITUDE_DEG = -12.0


def sun_altitude(lat: float, lon: float, when: datetime) -> float:
    """Geometric solar altitude in degrees (NOAA solar-position equations, no refraction)."""
    utc = when.astimezone(UTC)
    jd = (utc - datetime(2000, 1, 1, 12, tzinfo=UTC)).total_seconds() / 86400 + 2451545.0
    t = (jd - 2451545.0) / 36525
    l0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    m = math.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    ecc = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    center = (
        math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m) * 0.000289
    )
    true_long = l0 + center
    omega = math.radians(125.04 - 1934.136 * t)
    app_long = math.radians(true_long - 0.00569 - 0.00478 * math.sin(omega))
    obliq = (
        23
        + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60
        + 0.00256 * math.cos(omega)
    )
    obliq_r = math.radians(obliq)
    decl = math.asin(math.sin(obliq_r) * math.sin(app_long))
    y = math.tan(obliq_r / 2) ** 2
    l0_r = math.radians(l0)
    eq_time = 4 * math.degrees(
        y * math.sin(2 * l0_r)
        - 2 * ecc * math.sin(m)
        + 4 * ecc * y * math.sin(m) * math.cos(2 * l0_r)
        - 0.5 * y * y * math.sin(4 * l0_r)
        - 1.25 * ecc * ecc * math.sin(2 * m)
    )
    minutes = utc.hour * 60 + utc.minute + utc.second / 60
    true_solar = (minutes + eq_time + 4 * lon) % 1440
    hour_angle = math.radians(true_solar / 4 - 180)
    lat_r = math.radians(lat)
    cos_zenith = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(
        hour_angle
    )
    return 90 - math.degrees(math.acos(max(-1.0, min(1.0, cos_zenith))))


def is_dark(lat: float, lon: float, when: datetime) -> bool:
    return sun_altitude(lat, lon, when) < DARK_ALTITUDE_DEG


def dark_window(
    lat: float, lon: float, night: date, tz: str
) -> tuple[datetime | None, datetime | None]:
    """First dark minute and the first light minute after it, as UTC datetimes.

    Scans from local noon on ``night`` to local noon the next day. ``(None, None)``
    means it never gets dark (midnight sun); ``(start, None)`` means it is still
    dark at the end of the scan (polar night).
    """
    zone = ZoneInfo(tz)
    begin = datetime.combine(night, time(12), zone).astimezone(UTC)
    end = datetime.combine(night + timedelta(days=1), time(12), zone).astimezone(UTC)
    step = timedelta(minutes=10)

    def first(want_dark: bool, since: datetime) -> datetime | None:
        t = since
        while t <= end:
            if is_dark(lat, lon, t) == want_dark:
                if t == since:
                    return t
                back = t
                while (
                    back - _ONE_MINUTE > since
                    and is_dark(lat, lon, back - _ONE_MINUTE) == want_dark
                ):
                    back -= _ONE_MINUTE
                return back
            t += step
        return None

    start = first(True, begin)
    if start is None:
        return None, None
    return start, first(False, start)


# --- OVATION -----------------------------------------------------------------

_POLEWARD_SPAN_DEG = 10


def ovation_at(grid: dict, lat: float, lon: float) -> dict:
    """Aurora percentage overhead and the best within 10 degrees poleward.

    The OVATION grid is 1-degree cells with longitude 0..359. Poleward is toward
    the place's own pole, so a southern place searches south. Cells absent from
    the grid count as 0.
    """
    lon_i = round(lon) % 360
    lat_i = round(lat)
    column = {int(la): pct for lo, la, pct in grid["coordinates"] if lo == lon_i}
    pole = 90 if lat_i >= 0 else -90
    sign = 1 if pole > 0 else -1
    lats = [lat_i + sign * i for i in range(_POLEWARD_SPAN_DEG + 1) if abs(lat_i + sign * i) <= 90]
    best_lat = max(lats, key=lambda la: (column.get(la, 0), -abs(la - lat_i)))
    return {
        "overhead_pct": column.get(lat_i, 0),
        "poleward_max_pct": column.get(best_lat, 0),
        "poleward_max_lat": best_lat,
        "observed": grid.get("Observation Time"),
        "forecast_time": grid.get("Forecast Time"),
    }


# --- Fetching ----------------------------------------------------------------

SWPC = "https://services.swpc.noaa.gov"
MAG_PATH = "/json/rtsw/rtsw_mag_1m.json"
WIND_PATH = "/json/rtsw/rtsw_wind_1m.json"
ALERTS_PATH = "/products/alerts.json"
KP_PATH = "/products/noaa-planetary-k-index-forecast.json"
OVATION_PATH = "/json/ovation_aurora_latest.json"

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

ALLOWED_HOSTS = frozenset(
    {"services.swpc.noaa.gov", "geocoding-api.open-meteo.com", "api.open-meteo.com"}
)

# Wire cap for a gzip body. The live RTSW feeds are 1.6-2.7 MB of JSON but
# compress about tenfold; OVATION gzips to ~140 KB.
MAX_BYTES = 1024 * 1024
OVATION_MAX_BYTES = 1024 * 1024
# Cap on the decoded JSON: a gzip body after inflating, or an identity body
# from a server that ignored Accept-Encoding.
MAX_DECOMPRESSED = 8 * 1024 * 1024

USER_AGENT = "VibeBot/8 (aurora)"


class FetchError(RuntimeError):
    """A fixed-host fetch failed, was refused, or returned unusable data."""


def _no_redirect_opener():
    """An opener that refuses redirects (same idea as ``statuspage``)."""

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *_args: object, **_kwargs: object) -> None:
            return None

    return urllib.request.build_opener(_NoRedirect())


def _gunzip(body: bytes, limit: int) -> bytes:
    """Inflate one gzip member, never producing more than ``limit + 1`` bytes."""
    inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = inflater.decompress(body, limit + 1)
    except zlib.error as exc:
        raise FetchError(f"bad gzip body: {exc}") from exc
    if len(out) > limit:
        raise FetchError(f"decompressed response exceeds {limit} bytes")
    if not inflater.eof:
        raise FetchError("bad gzip body: truncated")
    return out


def fetch_json(
    url: str,
    *,
    timeout: float = 10.0,
    max_bytes: int,
    opener: Any = None,
) -> Any:
    """GET ``url`` and decode JSON; only the three allowed hosts, no redirects.

    Asks for gzip. A gzip body may be at most ``max_bytes`` on the wire and
    ``MAX_DECOMPRESSED`` once inflated; an identity body is held to
    ``MAX_DECOMPRESSED`` directly.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS:
        raise FetchError(f"host not allowed: {parts.hostname}")
    if opener is None:
        opener = _no_redirect_opener()
    req = urllib.request.Request(  # noqa: S310
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "User-Agent": USER_AGENT,
        },
    )
    limit = MAX_DECOMPRESSED
    try:
        with opener.open(req, timeout=timeout) as resp:
            encoding = (resp.headers.get("Content-Encoding") or "identity").strip().lower()
            if encoding not in ("gzip", "identity"):
                raise FetchError(f"unsupported content-encoding: {encoding}")
            cap = max_bytes if encoding == "gzip" else limit
            body = resp.read(cap + 1)
        if len(body) > cap:
            raise FetchError(f"response exceeds {cap} bytes")
        if encoding == "gzip":
            body = _gunzip(body, limit)
        return json.loads(body)
    except (OSError, http.client.HTTPException, json.JSONDecodeError, UnicodeDecodeError) as exc:
        # OSError covers URLError/HTTPError/TimeoutError/ConnectionResetError.
        raise FetchError(f"{type(exc).__name__}: {exc}") from exc


@dataclass(frozen=True)
class Place:
    name: str
    lat: float
    lon: float
    tz: str


def geocode(query: str, *, opener: Any = None) -> Place | None:
    """Resolve a place name via Open-Meteo; ``None`` when nothing matches."""
    qs = urllib.parse.urlencode({"name": query, "count": 1})
    data = fetch_json(f"{GEOCODE_URL}?{qs}", max_bytes=MAX_BYTES, opener=opener)
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        return None
    r = results[0] if isinstance(results, list) else None
    if not isinstance(r, dict):
        raise FetchError(f"unexpected geocoding results: {type(results).__name__}")
    try:
        region = r.get("admin1") or r.get("country")
        name = f"{r['name']}, {region}" if region else str(r["name"])
        return Place(name, float(r["latitude"]), float(r["longitude"]), str(r["timezone"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise FetchError(f"unexpected geocoding result: {exc}") from exc


def cloud_series(lat: float, lon: float, *, opener: Any = None) -> list[tuple[datetime, int]]:
    """Hourly cloud cover percent (UTC) for five forecast days from today, oldest first.

    Five days covers a NOAA watch night up to D+3 at any UTC offset.
    """
    qs = urllib.parse.urlencode(
        {
            "latitude": lat,
            "longitude": lon,
            "hourly": "cloud_cover",
            "timezone": "UTC",
            "forecast_days": 5,
        }
    )
    data = fetch_json(f"{FORECAST_URL}?{qs}", max_bytes=MAX_BYTES, opener=opener)
    try:
        times = data["hourly"]["time"]
        covers = data["hourly"]["cloud_cover"]
    except (KeyError, TypeError) as exc:
        raise FetchError(f"unexpected forecast shape: {exc}") from exc
    if not isinstance(times, list) or not isinstance(covers, list):
        raise FetchError("unexpected forecast shape: hourly time/cloud_cover not lists")
    out: list[tuple[datetime, int]] = []
    for tag, pct in zip(times, covers, strict=False):
        value = _finite(pct)
        if value is None:
            continue
        try:
            ts = datetime.fromisoformat(tag).replace(tzinfo=UTC)
        except TypeError, ValueError:
            continue
        out.append((ts, round(value)))
    return out


def hours_within(
    rows: Sequence[tuple[datetime, int]], *, start: datetime, end: datetime
) -> list[tuple[datetime, int]]:
    """The rows whose hour falls within ``[start, end]``."""
    return [(ts, pct) for ts, pct in rows if start <= ts <= end]


_KP_BLOCK = timedelta(hours=3)


def parse_kp(rows: list[dict], *, now: datetime, hours: int = 24) -> list[dict]:
    """3-hour Kp blocks overlapping ``[now, now + hours]``."""
    horizon = now + timedelta(hours=hours)
    out: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        start = _parse_ts(row.get("time_tag"))
        kp = _finite(row.get("kp"))
        if start is None or kp is None:
            continue
        if start + _KP_BLOCK > now and start < horizon:
            out.append(
                {
                    "start": start,
                    "kp": kp,
                    "scale": row.get("noaa_scale"),
                    "kind": row.get("observed"),
                }
            )
    out.sort(key=lambda b: b["start"])
    return out


# --- Announcements -----------------------------------------------------------


@dataclass
class SwState:
    """What the storm announcer has already told each channel.

    ``seen_channels`` is the set of announcing channels on the last pass that
    read the solar wind. A channel outside it is new, so an episode already
    running when it arrives is acked silently instead of announced late.
    """

    watches_seeded: bool = False
    episode_seeded: bool = False
    episode: Episode | None = None
    # The episode that last ended and when. A new one starting soon after is
    # the same storm resuming after a brief northward turn.
    last_episode: Episode | None = None
    last_end: datetime | None = None
    announced_levels: dict[str, dict[date, int]] = field(default_factory=dict)
    episode_acked: dict[str, datetime] = field(default_factory=dict)
    seen_channels: set[str] = field(default_factory=set)


def watch_news(
    levels: dict[date, int], told: dict[date, int], *, min_level: int
) -> dict[date, int]:
    """Dates at or above ``min_level`` whose level rose past what was told."""
    return {d: lvl for d, lvl in levels.items() if lvl >= min_level and lvl > told.get(d, 0)}


def mean_cloud(rows: Sequence[tuple[datetime, int]]) -> int | None:
    """Rounded mean cloud percent, or None with no rows."""
    return round(sum(pct for _ts, pct in rows) / len(rows)) if rows else None


def _day(d: date) -> str:
    return f"{_MONTH_NAMES[d.month - 1]} {d.day}"


def _date_span(dates: Sequence[date]) -> str:
    """``Oct 12–13``; runs of consecutive dates joined with commas."""
    runs: list[list[date]] = []
    for d in sorted(dates):
        if runs and d - runs[-1][-1] == timedelta(days=1):
            runs[-1].append(d)
        else:
            runs.append([d])
    out = []
    for run in runs:
        first, last = run[0], run[-1]
        if first == last:
            out.append(_day(first))
        elif first.month == last.month:
            out.append(f"{_day(first)}\u2013{last.day}")
        else:
            out.append(f"{_day(first)}\u2013{_day(last)}")
    return ", ".join(out)


def _clouds(label: str, clouds: dict[str, int | None]) -> str:
    if not clouds:
        return ""
    parts = ", ".join(
        f"{name} {pct}%" if pct is not None else f"{name} ?" for name, pct in clouds.items()
    )
    return f" {label}: {parts}."


def render_watch_line(levels: dict[date, int], clouds: dict[str, int | None]) -> str:
    return (
        f"\U0001f30c NOAA G{max(levels.values())} storm watch for {_date_span(list(levels))}."
        + _clouds(f"Clouds night of {_day(min(levels))}", clouds)
    )


def render_storm_line(
    episode: Episode, *, now: datetime, clouds: dict[str, int | None], direction: str
) -> str:
    """Deterministic storm-conditions line. A missing speed drops the speed clause."""
    minutes = max(0, int((now - episode.start).total_seconds() // 60))
    line = (
        f"\U0001f30c Aurora storm conditions: Bz {episode.min_bz:.0f} nT "
        f"for {minutes // 60}h{minutes % 60:02d}m"
    )
    if episode.latest_speed is not None:
        line += f" at {episode.latest_speed:.0f} km/s"
        if episode.lead_minutes is not None:
            line += f" (~{episode.lead_minutes} min to Earth)"
    return line + f". Look {direction}." + _clouds("Clouds now", clouds)
