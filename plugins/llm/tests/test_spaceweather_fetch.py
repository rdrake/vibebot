"""Fixed-host fetch, geocoding, cloud cover and Kp parsing. No network."""

from __future__ import annotations

import gzip
import http.client
import io
import json
import os
import random
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pytest
from llm import spaceweather as sw

FIXTURES = Path(__file__).parent / "fixtures" / "spaceweather"


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, headers: dict | None = None):
        super().__init__(body)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class FakeOpener:
    def __init__(
        self,
        body: bytes = b"{}",
        raises: Exception | None = None,
        headers: dict | None = None,
    ):
        self.body = body
        self.raises = raises
        self.headers = headers
        self.request = None

    def open(self, req, timeout=None):  # noqa: ARG002
        self.request = req
        if self.raises:
            raise self.raises
        return FakeResponse(self.body, self.headers)


GZIP = {"Content-Encoding": "gzip"}


def opener_for(payload) -> FakeOpener:
    return FakeOpener(json.dumps(payload).encode())


def test_disallowed_host_rejected():
    opener = FakeOpener()
    with pytest.raises(sw.FetchError):
        sw.fetch_json("https://evil.example/x.json", max_bytes=1000, opener=opener)
    assert opener.request is None


def test_non_https_rejected():
    with pytest.raises(sw.FetchError):
        sw.fetch_json("http://services.swpc.noaa.gov/x.json", max_bytes=1000, opener=FakeOpener())


def test_redirect_refused():
    import urllib.request

    handler = sw._no_redirect_opener().handlers
    nr = next(h for h in handler if isinstance(h, urllib.request.HTTPRedirectHandler))
    req = urllib.request.Request(sw.SWPC + sw.MAG_PATH)
    assert nr.redirect_request(req, None, 302, "Found", {}, "http://169.254.169.254/") is None


def test_oversize_gzip_body_raises():
    # Random digits barely compress, so the wire body is well over 1000 bytes.
    body = gzip.compress(json.dumps([int(b) for b in os.urandom(1000)]).encode())
    assert len(body) > 1000
    with pytest.raises(sw.FetchError, match="exceeds 1000"):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=FakeOpener(body, headers=GZIP))


def test_oversize_identity_body_raises(monkeypatch):
    monkeypatch.setattr(sw, "MAX_DECOMPRESSED", 1000)
    opener = FakeOpener(b"[" + b"1," * 600 + b"1]")
    with pytest.raises(sw.FetchError, match="exceeds 1000"):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=opener)


def _rtsw_body(target: int) -> bytes:
    """An RTSW-shaped mag list at least ``target`` bytes long."""
    row = {
        "time_tag": "2026-10-11T12:00:00",
        "active": True,
        "source": "ACE",
        "range": 2,
        "scale": None,
        "sensitivity": 0,
        "manual_mode": False,
        "sample_size": 60,
        "bt": 5.1,
        "bx_gse": -1.2,
        "by_gse": 3.4,
        "bz_gse": -2.1,
        "theta_gse": -24.5,
        "phi_gse": 109.4,
        "bx_gsm": -1.2,
        "by_gsm": 3.9,
        "bz_gsm": -1.1,
        "theta_gsm": -12.4,
        "phi_gsm": 107.0,
        "max_telemetry_flag": 0,
        "max_data_flag": 0,
        "overall_quality": 0,
    }
    n = target // len(json.dumps(row)) + 1
    return json.dumps([row] * n).encode()


@pytest.mark.parametrize("gzipped", [True, False])
def test_full_size_rtsw_feed_fetches(gzipped):
    # Live rtsw_wind_1m.json is ~2.7 MB uncompressed, over the 1 MB wire cap.
    raw = _rtsw_body(2_800_000)
    assert len(raw) > 2_700_000
    opener = FakeOpener(gzip.compress(raw), headers=GZIP) if gzipped else FakeOpener(raw)
    got = sw.fetch_json(sw.SWPC + sw.WIND_PATH, max_bytes=sw.MAX_BYTES, opener=opener)
    assert len(got) == len(json.loads(raw))


def test_full_size_ovation_grid_fetches():
    # The live OVATION grid is every 1-degree cell: ~918 KB raw, ~140 KB gzip.
    rng = random.Random(0)
    grid = {
        "Observation Time": "2026-10-11T13:24:00Z",
        "Forecast Time": "2026-10-11T14:12:00Z",
        "Data Format": "[Longitude, Latitude, Aurora]",
        "coordinates": [
            [lo, la, rng.randint(0, 100) if abs(la) >= 50 else rng.randint(0, 3)]
            for lo in range(360)
            for la in range(-90, 91)
        ],
        "type": "MultiPoint",
    }
    raw = json.dumps(grid, separators=(",", ":")).encode()
    assert len(raw) > 700_000
    body = gzip.compress(raw)
    assert len(body) > 100_000
    opener = FakeOpener(body, headers=GZIP)
    got = sw.fetch_json(sw.SWPC + sw.OVATION_PATH, max_bytes=sw.OVATION_MAX_BYTES, opener=opener)
    assert len(got["coordinates"]) == 360 * 181
    cells = {(lo, la): p for lo, la, p in grid["coordinates"]}
    assert sw.ovation_at(got, 64.8, -147.7)["overhead_pct"] == cells[(212, 65)]


def test_no_opener_uses_the_no_redirect_opener(monkeypatch):
    opener = opener_for({"ok": True})
    monkeypatch.setattr(sw, "_no_redirect_opener", lambda: opener)
    assert sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000) == {"ok": True}
    assert opener.request is not None


def test_gzip_bomb_raises():
    bomb = gzip.compress(b" " * (sw.MAX_DECOMPRESSED + 1))
    assert len(bomb) < sw.MAX_BYTES
    with pytest.raises(sw.FetchError, match="decompressed"):
        sw.fetch_json(
            sw.SWPC + sw.KP_PATH, max_bytes=sw.MAX_BYTES, opener=FakeOpener(bomb, headers=GZIP)
        )


@pytest.mark.parametrize(
    "body",
    [b"not gzip at all", gzip.compress(b"[1, 2, 3]")[:-6], gzip.compress(b"[1]")[:5]],
    ids=["garbage", "truncated-trailer", "truncated-header"],
)
def test_corrupt_gzip_raises(body):
    with pytest.raises(sw.FetchError):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=FakeOpener(body, headers=GZIP))


def test_unsupported_encoding_raises():
    opener = FakeOpener(b"[1]", headers={"Content-Encoding": "br"})
    with pytest.raises(sw.FetchError, match="br"):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=opener)


def test_request_asks_for_gzip_and_names_itself():
    opener = opener_for([1])
    sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=opener)
    assert opener.request.get_header("Accept-encoding") == "gzip"
    assert opener.request.get_header("User-agent") == "VibeBot/8 (aurora)"
    assert opener.request.get_header("Accept") == "application/json"


def test_http_500_is_fetch_error():
    err = urllib.error.HTTPError(sw.SWPC, 500, "boom", {}, None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(sw.FetchError):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=FakeOpener(raises=err))


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError("down"),
        TimeoutError(),
    ],
)
def test_network_errors_are_fetch_error(exc):
    with pytest.raises(sw.FetchError):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=FakeOpener(raises=exc))


def test_bad_json_is_fetch_error():
    with pytest.raises(sw.FetchError):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=FakeOpener(b"<html>"))


def test_fetch_ok_returns_parsed():
    assert sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=opener_for([1, 2])) == [1, 2]


def test_geocode_toronto():
    opener = opener_for(
        {
            "results": [
                {
                    "name": "Toronto",
                    "latitude": 43.7,
                    "longitude": -79.42,
                    "timezone": "America/Toronto",
                    "admin1": "Ontario",
                    "country": "Canada",
                }
            ]
        }
    )
    place = sw.geocode("Toronto", opener=opener)
    assert place == sw.Place("Toronto, Ontario", 43.7, -79.42, "America/Toronto")
    assert opener.request.full_url.startswith("https://geocoding-api.open-meteo.com/v1/search?")
    assert "name=Toronto" in opener.request.full_url


def test_geocode_falls_back_to_country_and_encodes():
    opener = opener_for(
        {
            "results": [
                {
                    "name": "Tromsø",
                    "latitude": 69.6,
                    "longitude": 18.9,
                    "timezone": "Europe/Oslo",
                    "country": "Norway",
                }
            ]
        }
    )
    place = sw.geocode("Tromsø & co", opener=opener)
    assert place is not None
    assert place.name == "Tromsø, Norway"
    assert "%C3%B8" in opener.request.full_url
    assert "%26" in opener.request.full_url


def test_geocode_no_results_is_none():
    assert sw.geocode("Nowhereville", opener=opener_for({"generationtime_ms": 0.5})) is None


@pytest.mark.parametrize("results", [["Toronto"], [None], {"0": {}}])
def test_geocode_junk_results_is_fetch_error(results):
    with pytest.raises(sw.FetchError):
        sw.geocode("Toronto", opener=opener_for({"results": results}))


@pytest.mark.parametrize(
    "hourly",
    [{"time": None, "cloud_cover": [1]}, {"time": ["x"], "cloud_cover": 5}, None, []],
)
def test_cloud_series_junk_hourly_is_fetch_error(hourly):
    with pytest.raises(sw.FetchError):
        sw.cloud_series(0, 0, opener=opener_for({"hourly": hourly}))


def test_cloud_series_whole_five_days_then_window_filter():
    times = [f"2026-10-11T{h:02d}:00" for h in range(0, 8)]
    opener = opener_for(
        {"hourly": {"time": times, "cloud_cover": [10, 20, 30, 40, 50, 60, 70, 80]}}
    )
    rows = sw.cloud_series(43.7, -79.42, opener=opener)
    assert len(rows) == 8
    got = sw.hours_within(
        rows,
        start=datetime(2026, 10, 11, 2, 0, tzinfo=UTC),
        end=datetime(2026, 10, 11, 5, 0, tzinfo=UTC),
    )
    assert got == [(datetime(2026, 10, 11, h, tzinfo=UTC), 30 + 10 * (h - 2)) for h in (2, 3, 4, 5)]
    url = opener.request.full_url
    assert url.startswith("https://api.open-meteo.com/v1/forecast?")
    assert "hourly=cloud_cover" in url and "timezone=UTC" in url and "forecast_days=5" in url


def test_cloud_series_skips_null_and_junk_hours():
    opener = opener_for(
        {
            "hourly": {
                "time": ["2026-10-11T00:00", "2026-10-11T01:00", "junk"],
                "cloud_cover": [None, 5, 7],
            }
        }
    )
    assert sw.cloud_series(0, 0, opener=opener) == [(datetime(2026, 10, 11, 1, tzinfo=UTC), 5)]


def test_parse_kp_next_24h():
    rows = json.loads((FIXTURES / "kp_forecast.json").read_text())
    now = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    blocks = sw.parse_kp(rows, now=now)
    first = blocks[0]
    assert first["start"] == datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
    assert first["kp"] == 5.33
    assert first["scale"] == "G1"
    assert first["kind"] == "estimated"
    assert len(blocks) == 8
    assert all(b["start"] < datetime(2026, 10, 12, 12, 0, tzinfo=UTC) for b in blocks)


def test_parse_kp_includes_block_in_progress_and_skips_junk():
    rows = [
        {"time_tag": "2026-10-11T09:00:00", "kp": 3.0, "observed": "observed", "noaa_scale": None},
        {"time_tag": "garbage", "kp": 1, "observed": "observed", "noaa_scale": None},
        {
            "time_tag": "2026-10-11T12:00:00",
            "kp": None,
            "observed": "predicted",
            "noaa_scale": None,
        },
    ]
    blocks = sw.parse_kp(rows, now=datetime(2026, 10, 11, 10, 30, tzinfo=UTC))
    assert [b["kp"] for b in blocks] == [3.0]
    assert blocks[0]["scale"] is None


def test_parse_kp_skips_header_row_and_missing_time_tag():
    # The Kp product was once a list of lists with a header row.
    rows = [
        ["time_tag", "kp", "observed", "noaa_scale"],
        {"time_tag": None, "kp": 5.0, "observed": "predicted", "noaa_scale": "G1"},
        {"kp": 6.0, "observed": "predicted", "noaa_scale": "G2"},
        {"time_tag": "2026-10-11T09:00:00", "kp": 3.0, "observed": "observed", "noaa_scale": None},
    ]
    blocks = sw.parse_kp(rows, now=datetime(2026, 10, 11, 10, 30, tzinfo=UTC))
    assert [(b["start"], b["kp"]) for b in blocks] == [
        (datetime(2026, 10, 11, 9, 0, tzinfo=UTC), 3.0)
    ]


class _BrokenRead:
    def __init__(self, exc: Exception):
        self.exc = exc
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self, _n=-1):
        raise self.exc


class _BrokenOpener:
    def __init__(self, exc: Exception):
        self.exc = exc

    def open(self, req, timeout=None):  # noqa: ARG002
        return _BrokenRead(self.exc)


@pytest.mark.parametrize(
    "exc",
    [
        ConnectionResetError("reset"),
        http.client.IncompleteRead(b""),
        http.client.RemoteDisconnected("closed"),
        http.client.BadStatusLine("x"),
    ],
)
def test_mid_read_transport_errors_are_fetch_error(exc):
    with pytest.raises(sw.FetchError):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=_BrokenOpener(exc))
