"""Fixed-host fetch, geocoding, cloud cover and Kp parsing. No network."""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
from datetime import UTC, datetime
from pathlib import Path

import pytest
from llm import spaceweather as sw

FIXTURES = Path(__file__).parent / "fixtures" / "spaceweather"


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


class FakeOpener:
    def __init__(self, body: bytes = b"{}", raises: Exception | None = None):
        self.body = body
        self.raises = raises
        self.request = None

    def open(self, req, timeout=None):  # noqa: ARG002
        self.request = req
        if self.raises:
            raise self.raises
        return FakeResponse(self.body)


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


def test_oversize_body_raises():
    opener = FakeOpener(b"[" + b"1," * 600 + b"1]")
    with pytest.raises(sw.FetchError):
        sw.fetch_json(sw.SWPC + sw.KP_PATH, max_bytes=1000, opener=opener)


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


def test_cloud_cover_window_filter():
    times = [f"2026-10-11T{h:02d}:00" for h in range(0, 8)]
    opener = opener_for(
        {"hourly": {"time": times, "cloud_cover": [10, 20, 30, 40, 50, 60, 70, 80]}}
    )
    got = sw.cloud_cover(
        43.7,
        -79.42,
        start=datetime(2026, 10, 11, 2, 0, tzinfo=UTC),
        end=datetime(2026, 10, 11, 5, 0, tzinfo=UTC),
        opener=opener,
    )
    assert got == [(datetime(2026, 10, 11, h, tzinfo=UTC), 30 + 10 * (h - 2)) for h in (2, 3, 4, 5)]
    url = opener.request.full_url
    assert url.startswith("https://api.open-meteo.com/v1/forecast?")
    assert "hourly=cloud_cover" in url and "timezone=UTC" in url and "forecast_days=7" in url


def test_cloud_cover_skips_null_hours():
    opener = opener_for(
        {"hourly": {"time": ["2026-10-11T00:00", "2026-10-11T01:00"], "cloud_cover": [None, 5]}}
    )
    got = sw.cloud_cover(
        0,
        0,
        start=datetime(2026, 10, 11, tzinfo=UTC),
        end=datetime(2026, 10, 12, tzinfo=UTC),
        opener=opener,
    )
    assert got == [(datetime(2026, 10, 11, 1, tzinfo=UTC), 5)]


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


class _BrokenRead:
    def __init__(self, exc: Exception):
        self.exc = exc

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
