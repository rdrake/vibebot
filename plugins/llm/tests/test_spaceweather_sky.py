"""Watches, darkness, and the OVATION lookup in llm.spaceweather.

Pure-function tests: alerts.json / OVATION payloads in, Watches, dark windows
and aurora percentages out. No network.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from llm import spaceweather
from llm.spaceweather import Watch

FIXTURES = Path(__file__).parent / "fixtures" / "spaceweather"


def _load(name: str):
    return json.loads((FIXTURES / name).read_text())


def _watch_alert(serial: int, issue: str, days: str, *, product: str = "A30F") -> dict:
    msg = (
        f"Space Weather Message Code: WATA30\r\nSerial Number: {serial}\r\n"
        f"Issue Time: {issue} UTC\r\n\r\nWATCH: Geomagnetic Storm Category G2 Predicted \n"
        f"Highest Storm Level Predicted by Day:\n{days}\n"
        "THIS SUPERSEDES ANY/ALL PRIOR WATCHES IN EFFECT\nComment: \r\n"
    )
    return {"product_id": product, "issue_datetime": "x", "message": msg}


def _by_serial(alerts) -> dict[int, Watch]:
    return {w.serial: w for w in spaceweather.parse_watches(alerts)}


def test_parse_real_g2_watch():
    w = _by_serial(_load("alerts.json"))[285]
    assert w.issued == datetime(2026, 10, 9, 16, 36, tzinfo=UTC)
    assert w.levels == {date(2026, 10, 9): 1, date(2026, 10, 10): 1, date(2026, 10, 11): 2}


def test_none_below_g1_is_zero():
    w = _by_serial(_load("alerts.json"))[284]
    assert w.levels[date(2026, 10, 10)] == 0
    assert w.levels[date(2026, 10, 11)] == 2


def test_non_watch_products_ignored():
    alerts = _load("alerts.json")
    assert any(a["product_id"] == "K05A" for a in alerts)
    watches = spaceweather.parse_watches(alerts)
    assert watches
    assert {w.serial for w in watches} >= {284, 285}
    only_k = [a for a in alerts if a["product_id"] == "K05A"]
    assert spaceweather.parse_watches(only_k) == []


def test_comment_dates_are_not_levels():
    # Serial 285's comment mentions "09 Oct" / "10 Oct"; only the by-day line counts.
    assert len(_by_serial(_load("alerts.json"))[285].levels) == 3


def test_cancel_is_empty_levels():
    alert = {
        "product_id": "A30F",
        "message": (
            "Space Weather Message Code: WATA30\r\nSerial Number: 286\r\n"
            "Issue Time: 2026 Oct 10 1200 UTC\r\n\r\nCANCEL WATCH: Geomagnetic Storm "
            "Category G2 Predicted\nCancels all prior watches.\n"
        ),
    }
    (w,) = spaceweather.parse_watches([alert])
    assert w.serial == 286
    assert w.levels == {}


def test_year_rollover():
    alert = _watch_alert(
        1,
        "2026 Dec 31 1200",
        "Dec 31:  G1 (Minor)   Jan 01:  G3 (Strong)   Jan 02:  None (Below G1)   ",
    )
    (w,) = spaceweather.parse_watches([alert])
    assert w.levels == {date(2026, 12, 31): 1, date(2027, 1, 1): 3, date(2027, 1, 2): 0}


def test_newest_watch_supersedes_and_past_dates_expire():
    old = _watch_alert(10, "2026 Oct 09 0800", "Oct 12:  G4 (Severe)   ")
    new = _watch_alert(11, "2026 Oct 10 0800", "Oct 12:  G2 (Moderate)   ")
    today = date(2026, 10, 11)
    for feed in ([old, new], [new, old]):
        got = spaceweather.effective_levels(spaceweather.parse_watches(feed), today=today)
        assert got == {date(2026, 10, 12): 2}
    assert (
        spaceweather.effective_levels(
            spaceweather.parse_watches([old, new]), today=date(2026, 10, 13)
        )
        == {}
    )


def test_effective_levels_cancel_clears_older_watch():
    old = _watch_alert(10, "2026 Oct 09 0800", "Oct 12:  G4 (Severe)   ")
    cancel = {
        "message": (
            "Space Weather Message Code: WATA30\r\nSerial Number: 11\r\n"
            "Issue Time: 2026 Oct 10 0800 UTC\r\n\r\nCANCEL WATCH\n"
        )
    }
    ws = spaceweather.parse_watches([old, cancel])
    assert spaceweather.effective_levels(ws, today=date(2026, 10, 11)) == {}


def _utc_from_local(y, m, d, hh, mm, offset_hours):
    return datetime(y, m, d, hh, mm, tzinfo=UTC) - timedelta(hours=offset_hours)


def test_toronto_dark_window():
    # Reference: sun at -12 deg (nautical twilight) for Toronto (43.65, -79.38),
    # EDT (UTC-4). timeanddate.com is Cloudflare-blocked and NOAA's calculator is
    # an interactive JS page with no API, so these come from the independent
    # USNO/Almanac-for-Computers sunrise/sunset algorithm at zenith 102 deg,
    # validated against the USNO API (aa.usno.navy.mil/api/rstt) for rise/set and
    # civil twilight on the same dates (all within 1 min):
    #   dusk 2026-10-11 19:43.3 EDT, dawn 2026-10-12 06:25.8 EDT.
    dusk = _utc_from_local(2026, 10, 11, 19, 43, -4)
    dawn = _utc_from_local(2026, 10, 12, 6, 26, -4)
    start, end = spaceweather.dark_window(43.65, -79.38, date(2026, 10, 11), "America/Toronto")
    assert start is not None
    assert end is not None
    assert start.tzinfo is not None
    assert abs(start - dusk) <= timedelta(minutes=3)
    assert abs(end - dawn) <= timedelta(minutes=3)


def test_polar_night_and_midnight_sun():
    start, end = spaceweather.dark_window(69.65, 18.96, date(2026, 12, 21), "Europe/Oslo")
    assert start is not None
    assert start.astimezone(UTC) < datetime(2026, 12, 21, 15, 0, tzinfo=UTC)  # before 16:00 CET
    assert end is not None  # polar-night Tromso still has a bright midday

    assert spaceweather.dark_window(69.65, 18.96, date(2026, 6, 21), "Europe/Oslo") == (None, None)


def test_dark_all_night_returns_open_end(monkeypatch):
    monkeypatch.setattr(spaceweather, "sun_altitude", lambda lat, lon, when: -30.0)
    start, end = spaceweather.dark_window(0.0, 0.0, date(2026, 12, 21), "UTC")
    assert start == datetime(2026, 12, 21, 12, 0, tzinfo=UTC)
    assert end is None


def test_is_dark():
    assert spaceweather.is_dark(43.65, -79.38, datetime(2026, 10, 12, 5, 0, tzinfo=UTC))
    assert not spaceweather.is_dark(43.65, -79.38, datetime(2026, 10, 11, 17, 0, tzinfo=UTC))


def test_sun_altitude_known_values():
    # Equinox-ish noon at the equator: sun near overhead.
    alt = spaceweather.sun_altitude(0.0, 0.0, datetime(2026, 3, 20, 12, 0, tzinfo=UTC))
    assert 85 < alt <= 90
    # Same moment at 180 E: local midnight, sun far below.
    assert spaceweather.sun_altitude(0.0, 180.0, datetime(2026, 3, 20, 12, 0, tzinfo=UTC)) < -80


def test_ovation_overhead_and_poleward():
    grid = _load("ovation_trimmed.json")
    got = spaceweather.ovation_at(grid, 43.7, -79.4)
    cells = {(lo, la): p for lo, la, p in grid["coordinates"]}
    assert got["overhead_pct"] == cells[(281, 44)]
    window = [cells[(281, la)] for la in range(44, 55)]
    assert got["poleward_max_pct"] == max(window)
    assert 44 <= got["poleward_max_lat"] <= 54
    assert cells[(281, got["poleward_max_lat"])] == max(window)
    assert got["observed"] == grid["Observation Time"]
    assert got["forecast_time"] == grid["Forecast Time"]


def test_negative_longitude_maps_into_grid():
    grid = {
        "Observation Time": "o",
        "Forecast Time": "f",
        "coordinates": [[281, 44, 37], [280, 44, 5], [79, 44, 99]],
    }
    assert spaceweather.ovation_at(grid, 43.7, -79.4)["overhead_pct"] == 37


def test_southern_hemisphere_searches_south():
    grid = {
        "Observation Time": "o",
        "Forecast Time": "f",
        "coordinates": [[147, la, 90 if la == -50 else 1] for la in range(-60, -30)]
        + [[147, la, 99] for la in range(-42, -30)],  # decoy equatorward of Hobart
    }
    got = spaceweather.ovation_at(grid, -42.9, 147.3)
    assert got["poleward_max_lat"] <= -42
    assert got["poleward_max_pct"] == 90
    assert got["poleward_max_lat"] == -50


def test_ovation_missing_cells_are_zero_and_clamped_at_pole():
    grid = {"Observation Time": "o", "Forecast Time": "f", "coordinates": [[0, 88, 4]]}
    got = spaceweather.ovation_at(grid, 85.0, 360.0)
    assert got["overhead_pct"] == 0
    assert got["poleward_max_pct"] == 4
    assert got["poleward_max_lat"] == 88
