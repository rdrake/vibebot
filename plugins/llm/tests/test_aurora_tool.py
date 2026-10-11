"""The aurora tool: payload assembly, caches, per-feed failure, and wiring."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta

import pytest
from llm import spaceweather
from llm.spaceweather import FetchError, Place

from .conftest import make_registry_side_effect

NOW = datetime(2026, 10, 12, 0, 0, tzinfo=UTC)  # 20:00 EDT on Oct 11
OSHAWA = Place("Oshawa, Ontario", 43.9, -78.9, "America/Toronto")


def _grid(observed: datetime) -> dict:
    return {
        "Observation Time": observed.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Forecast Time": (observed + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "coordinates": [[281, 44, 4], [281, 50, 35], [281, 53, 20]],
    }


def _rtsw(now: datetime, minutes: int = 200) -> tuple[list[dict], list[dict]]:
    mag, wind = [], []
    for i in range(minutes):
        tag = (now - timedelta(minutes=minutes - i)).strftime("%Y-%m-%dT%H:%M:%S")
        mag.append({"time_tag": tag, "active": True, "bz_gsm": -12.0, "bt": 14.0})
        wind.append({"time_tag": tag, "active": True, "proton_speed": 600.0})
    return mag, wind


class Feeds:
    """Router for the patched ``spaceweather.fetch_json``."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.down: set[str] = set()
        self.observed = NOW - timedelta(minutes=5)
        self.delay = 0.0

    def __call__(self, url, *, timeout=10.0, max_bytes, opener=None):
        self.calls.append(url)
        if self.delay:
            time.sleep(self.delay)
        paths = {
            spaceweather.OVATION_PATH: "ovation",
            spaceweather.MAG_PATH: "mag",
            spaceweather.WIND_PATH: "wind",
            spaceweather.ALERTS_PATH: "alerts",
            spaceweather.KP_PATH: "kp",
        }
        tag = next((t for path, t in paths.items() if url.endswith(path)), None)
        assert tag is not None, url
        if tag in self.down:
            raise FetchError("down")
        if tag == "ovation":
            return _grid(self.observed)
        if tag in ("mag", "wind"):
            return _rtsw(NOW)[0 if tag == "mag" else 1]
        if tag == "alerts":
            return []
        return [
            {
                "time_tag": (NOW + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S"),
                "kp": 6.0,
                "observed": "predicted",
                "noaa_scale": "G2",
            }
        ]

    def count(self, path: str) -> int:
        return sum(1 for c in self.calls if c.endswith(path))


@pytest.fixture
def env(plugin_env, mocker):
    plugin, _irc, _msg = plugin_env
    plugin.registryValue.side_effect = make_registry_side_effect({"auroraEnabled": True})
    feeds = Feeds()
    mocker.patch("llm.spaceweather.fetch_json", side_effect=feeds)
    geo = mocker.patch("llm.spaceweather.geocode", return_value=OSHAWA)
    start = datetime(2026, 10, 11, 0, 0, tzinfo=UTC)
    clouds = mocker.patch(
        "llm.spaceweather.cloud_series",
        return_value=[(start + timedelta(hours=h), 20) for h in range(72)],
    )
    return plugin, feeds, geo, clouds


@pytest.fixture
def window(mocker):
    """Spy on the dark-window slice of the cached cloud series."""
    return mocker.spy(spaceweather, "hours_within")


def _payload(plugin, place="Oshawa", now=NOW) -> dict:
    return plugin._aurora_payload(place, now=now)


def test_payload_shape(env) -> None:
    plugin, _feeds, _geo, _clouds = env
    p = _payload(plugin)

    for key in (
        "place",
        "lat",
        "lon",
        "now",
        "ovation",
        "kp_next_24h",
        "dark",
        "cloud_pct_while_dark",
    ):
        assert key in p, key
    assert "unavailable" not in p
    assert p["place"] == "Oshawa, Ontario"
    assert p["now"]["bz_nt"] == -12.0
    assert p["now"]["speed_kms"] == 600.0
    assert p["now"]["storm_minutes"] > 0
    assert p["now"]["lead_minutes"] == 42
    assert p["ovation"]["overhead_pct"] == 4
    assert p["ovation"]["poleward_max_pct"] == 35
    assert p["ovation"]["poleward_max_lat"] == 50
    assert "poleward_max is the strongest aurora" in p["ovation"]["note"]
    assert "stale" not in p["ovation"]
    assert p["kp_next_24h"][0]["scale"] == "G2"
    assert "active_watch" not in p
    json.dumps(p)


def test_dark_is_tonight_in_place_tz(env, window) -> None:
    plugin, _feeds, _geo, _clouds = env
    p = _payload(plugin)

    assert p["dark"]["tz"] == "local"
    assert p["dark"]["from"] > "19:00"
    assert p["dark"]["until"] < "08:00"
    assert p["cloud_pct_while_dark"][0]["pct"] == 20
    kwargs = window.call_args.kwargs
    assert kwargs["start"].tzinfo is UTC
    assert kwargs["end"] > kwargs["start"]


def test_unknown_place_errors(env) -> None:
    plugin, _feeds, geo, _clouds = env
    geo.return_value = None

    assert _payload(plugin, "Nowhereville") == {"error": "couldn't find that place"}


def test_ovation_down_drops_only_ovation(env) -> None:
    plugin, feeds, _geo, _clouds = env
    feeds.down.add("ovation")
    p = _payload(plugin)

    assert p["unavailable"] == ["ovation"]
    assert "ovation" not in p
    for key in ("now", "kp_next_24h", "dark", "cloud_pct_while_dark"):
        assert key in p


def test_other_feeds_report_unavailable(env, mocker) -> None:
    plugin, feeds, _geo, clouds = env
    feeds.down.update({"mag", "kp", "alerts"})
    clouds.side_effect = FetchError("down")
    p = _payload(plugin)

    assert p["unavailable"] == ["solar_wind", "kp", "alerts", "clouds"]
    assert "now" not in p
    assert "kp_next_24h" not in p
    assert "cloud_pct_while_dark" not in p
    assert "ovation" in p


def test_stale_solar_wind_is_unavailable(env) -> None:
    plugin, _feeds, _geo, _clouds = env
    p = _payload(plugin, now=NOW + timedelta(minutes=20))  # newest Bz is 21 min old

    assert "now" not in p
    assert "solar_wind" in p["unavailable"]


def test_ovation_cache_10_min(env) -> None:
    plugin, feeds, _geo, _clouds = env
    _payload(plugin, now=NOW)
    _payload(plugin, now=NOW + timedelta(minutes=5))
    assert feeds.count(spaceweather.OVATION_PATH) == 1

    _payload(plugin, now=NOW + timedelta(minutes=11))
    assert feeds.count(spaceweather.OVATION_PATH) == 2


_SECOND = timedelta(seconds=1)


@pytest.mark.parametrize(
    ("path", "ttl"),
    [
        (spaceweather.MAG_PATH, timedelta(minutes=2)),
        (spaceweather.WIND_PATH, timedelta(minutes=2)),
        (spaceweather.KP_PATH, timedelta(hours=3)),
        (spaceweather.ALERTS_PATH, timedelta(minutes=5)),
    ],
    ids=["mag", "wind", "kp", "alerts"],
)
def test_feed_cache_ttl(env, path, ttl) -> None:
    plugin, feeds, _geo, _clouds = env
    _payload(plugin, now=NOW)
    _payload(plugin, now=NOW + ttl - _SECOND)
    assert feeds.count(path) == 1

    _payload(plugin, now=NOW + ttl + _SECOND)
    assert feeds.count(path) == 2


def test_cloud_cache_one_hour_per_place(env) -> None:
    plugin, _feeds, geo, clouds = env
    _payload(plugin, now=NOW)
    _payload(plugin, now=NOW + timedelta(hours=1) - _SECOND)
    assert clouds.call_count == 1
    assert clouds.call_args.args == (43.9, -78.9)

    _payload(plugin, now=NOW + timedelta(hours=1) + _SECOND)
    assert clouds.call_count == 2

    geo.return_value = Place("Ottawa, Ontario", 45.42, -75.7, "America/Toronto")
    _payload(plugin, "Ottawa", now=NOW + timedelta(hours=1) + _SECOND)
    assert clouds.call_count == 3


def test_expired_cloud_entries_evicted(env) -> None:
    plugin, _feeds, geo, _clouds = env
    _payload(plugin, now=NOW)
    assert [k for k in plugin._sw_cache if k.startswith("clouds:")] == ["clouds:43.9,-78.9"]

    geo.return_value = Place("Ottawa, Ontario", 45.42, -75.7, "America/Toronto")
    _payload(plugin, "Ottawa", now=NOW + timedelta(minutes=30))
    assert len([k for k in plugin._sw_cache if k.startswith("clouds:")]) == 2  # both fresh

    geo.return_value = Place("Toronto, Ontario", 43.7, -79.42, "America/Toronto")
    _payload(plugin, "Toronto", now=NOW + timedelta(hours=1, seconds=1))
    assert sorted(k for k in plugin._sw_cache if k.startswith("clouds:")) == [
        "clouds:43.7,-79.42",
        "clouds:45.42,-75.7",
    ]
    assert "ovation" in plugin._sw_cache  # other feeds untouched


def test_failed_fetch_not_cached(env) -> None:
    plugin, feeds, _geo, _clouds = env
    feeds.down.add("kp")
    assert "kp" in _payload(plugin)["unavailable"]
    feeds.down.clear()
    assert "unavailable" not in _payload(plugin, now=NOW + _SECOND)
    assert feeds.count(spaceweather.KP_PATH) == 2


def test_feeds_fetched_in_parallel(env, mocker) -> None:
    plugin, feeds, _geo, clouds = env
    feeds.delay = 0.5
    rows = clouds.return_value

    def slow_clouds(*_a, **_kw):
        time.sleep(0.5)
        return rows

    clouds.side_effect = slow_clouds
    t0 = time.monotonic()
    p = _payload(plugin)
    elapsed = time.monotonic() - t0

    assert "unavailable" not in p
    assert elapsed < 2.0, elapsed  # serial is ~3 s


@pytest.mark.parametrize("feed", ["solar_wind", "ovation", "kp", "alerts", "clouds"])
def test_one_failing_feed_is_alone_in_unavailable(env, feed) -> None:
    plugin, feeds, _geo, clouds = env
    feeds.delay = 0.05  # overlap the fetches
    if feed == "clouds":
        clouds.side_effect = FetchError("down")
    else:
        feeds.down.add({"solar_wind": "wind"}.get(feed, feed))
    p = _payload(plugin)

    assert p["unavailable"] == [feed]
    present = {
        "solar_wind": "now",
        "ovation": "ovation",
        "kp": "kp_next_24h",
        "clouds": "cloud_pct_while_dark",
    }
    for name, key in present.items():
        assert (key in p) is (name != feed), key


def test_payload_key_order_unchanged(env, mocker) -> None:
    plugin, _feeds, _geo, _clouds = env
    from llm.spaceweather import Watch

    watch = Watch(serial=1, issued=NOW, levels={NOW.date(): 3})
    mocker.patch("llm.spaceweather.parse_watches", return_value=[watch])
    assert list(_payload(plugin)) == [
        "place",
        "lat",
        "lon",
        "now",
        "ovation",
        "kp_next_24h",
        "active_watch",
        "dark",
        "cloud_pct_while_dark",
    ]


def test_announcer_and_tool_share_alerts_cache(env, mocker) -> None:
    plugin, feeds, _geo, _clouds = env
    mocker.patch.object(plugin, "_sw_announce_channels", return_value=["#a"])
    mocker.patch.object(plugin, "_sw_now", return_value=NOW)
    plugin._run_spaceweather_poll()
    assert feeds.count(spaceweather.ALERTS_PATH) == 1
    assert feeds.count(spaceweather.MAG_PATH) == 1

    _payload(plugin, now=NOW + timedelta(minutes=1))
    assert feeds.count(spaceweather.ALERTS_PATH) == 1
    assert feeds.count(spaceweather.MAG_PATH) == 1


def test_ovation_stale_flag(env) -> None:
    plugin, feeds, _geo, _clouds = env
    feeds.observed = NOW - timedelta(minutes=90)

    assert _payload(plugin)["ovation"]["stale"] is True


def test_geocode_fetch_error_not_cached(env) -> None:
    plugin, _feeds, geo, _clouds = env
    geo.side_effect = [FetchError("boom"), OSHAWA]

    assert "error" in _payload(plugin)
    assert _payload(plugin)["place"] == "Oshawa, Ontario"
    assert geo.call_count == 2


def test_geocode_hit_and_miss_cached(env) -> None:
    plugin, _feeds, geo, _clouds = env
    _payload(plugin)
    _payload(plugin, "oshawa")
    assert geo.call_count == 1

    geo.return_value = None
    _payload(plugin, "Nowhere")
    _payload(plugin, "Nowhere")
    assert geo.call_count == 2


def test_active_watch_reported(env, mocker) -> None:
    plugin, _feeds, _geo, _clouds = env
    from llm.spaceweather import Watch

    watch = Watch(serial=1, issued=NOW, levels={NOW.date(): 2, NOW.date() + timedelta(days=1): 3})
    mocker.patch("llm.spaceweather.parse_watches", return_value=[watch])

    assert _payload(plugin)["active_watch"] == {
        "level": "G3",
        "dates": [(NOW.date() + timedelta(days=1)).isoformat()],
    }


def test_handler_returns_tool_result(env, mocker) -> None:
    plugin, _feeds, _geo, _clouds = env
    schemas, handlers = plugin._build_aurora_tool()

    assert schemas[0]["function"]["name"] == "aurora"
    assert schemas[0]["function"]["parameters"]["required"] == ["place"]
    out = json.loads(handlers["aurora"]({"place": "Oshawa"}).content)
    assert out["place"] == "Oshawa, Ontario"
    assert json.loads(handlers["aurora"]({"place": ""}).content)["error"]


def _ask_tools(plugin_env, enabled: bool):
    from llm.service import AssistantResult

    plugin, irc, msg = plugin_env
    irc.state.nickToAccount.return_value = "test_account"
    plugin.registryValue.side_effect = make_registry_side_effect(
        {"auroraEnabled": enabled, "ircLookupEnabled": False}
    )
    plugin.llm_service.detect_images.return_value = []
    plugin.llm_service.assistant_request.side_effect = None
    plugin.llm_service.assistant_request.return_value = AssistantResult(
        content="ok",
        grounding_used=False,
        prompt_tokens=1,
        completion_tokens=1,
        cost=0.0,
        model="m",
    )
    plugin.ask(irc, msg, ["any aurora tonight in Oshawa"])
    return plugin.llm_service.assistant_request.call_args.kwargs


def test_tool_advertised_when_enabled(plugin_env) -> None:
    kwargs = _ask_tools(plugin_env, True)

    assert [t["function"]["name"] for t in kwargs["extra_tools"]] == ["aurora"]
    assert "aurora" in kwargs["extra_handlers"]


def test_tool_on_by_default(plugin_env) -> None:
    from llm import config  # noqa: F401  (registers the keys)
    from supybot import conf

    assert conf.supybot.plugins.LLM.auroraEnabled() is True
    assert conf.supybot.plugins.LLM.auroraAnnounce() is False
    assert conf.supybot.plugins.LLM.auroraAnnouncePlaces() == []


def test_tool_hidden_when_disabled(plugin_env) -> None:
    assert _ask_tools(plugin_env, False)["extra_tools"] is None


class TestTonightWindow:
    def test_before_dawn_gets_night_in_progress(self, env, window) -> None:
        plugin, _feeds, _geo, _clouds = env
        now = datetime(2026, 10, 12, 7, 0, tzinfo=UTC)  # 03:00 EDT, still dark
        p = _payload(plugin, now=now)

        kwargs = window.call_args.kwargs
        assert kwargs["start"] < now < kwargs["end"]
        assert p["dark"]["from"] > "19:00"

    def test_morning_after_dawn_gets_this_evening(self, env, window) -> None:
        plugin, _feeds, _geo, _clouds = env
        now = datetime(2026, 10, 12, 14, 0, tzinfo=UTC)  # 10:00 EDT
        _payload(plugin, now=now)

        kwargs = window.call_args.kwargs
        assert kwargs["start"] > now
        assert kwargs["start"].date() == now.date()

    def test_no_dawn_clouds_cover_twelve_hours(self, env, window, mocker) -> None:
        plugin, _feeds, _geo, _clouds = env
        start = NOW + timedelta(hours=1)
        mocker.patch("llm.spaceweather.dark_window", return_value=(start, None))
        p = _payload(plugin)

        kwargs = window.call_args.kwargs
        assert kwargs["start"] == start
        assert kwargs["end"] == start + timedelta(hours=12)
        assert "until" not in p["dark"]

    def test_never_dark_has_no_cloud_key(self, env, mocker) -> None:
        plugin, _feeds, _geo, clouds = env
        mocker.patch("llm.spaceweather.dark_window", return_value=(None, None))
        p = _payload(plugin)

        assert p["dark"]["never_dark"] is True
        assert "cloud_pct_while_dark" not in p
        clouds.assert_not_called()
