"""The storm announcer: G3+ watches and sustained-Bz storm conditions.

Drives the real poll methods on the bare ``spaceweather_plugin`` stand-in.
Alerts, cloud cover and geocoding are stubbed; the solar wind is real
``Sample`` series fed through the real ``advance_episode``. No network.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from llm import spaceweather
from llm.spaceweather import Episode, Place, Sample

FIXTURES = Path(__file__).parent / "fixtures" / "spaceweather"
NOW = datetime(2026, 10, 11, 12, 0, tzinfo=UTC)
OTTAWA = Place("Ottawa, Ontario", 45.42, -75.70, "America/Toronto")
HOBART = Place("Hobart, Tasmania", -42.88, 147.33, "Australia/Hobart")


def _series(end: datetime, bz: float, *, minutes: int = 200, speed: float = 620.0):
    return [
        Sample(ts=end - timedelta(minutes=i), bz=bz, bt=abs(bz), speed=speed)
        for i in range(minutes - 1, -1, -1)
    ]


def _watch(serial: int, issue: str, days: str) -> dict:
    return {
        "product_id": "A30F",
        "message": (
            f"Space Weather Message Code: WATA30\r\nSerial Number: {serial}\r\n"
            f"Issue Time: {issue} UTC\r\n\r\nWATCH: Geomagnetic Storm Category G3 Predicted\n"
            f"Highest Storm Level Predicted by Day:\n{days}\n"
            "THIS SUPERSEDES ANY/ALL PRIOR WATCHES IN EFFECT\nComment: \r\n"
        ),
    }


G3_OCT12 = _watch(10, "2026 Oct 11 0600", "Oct 12:  G3 (Strong)   Oct 13:  G3 (Strong)   ")


NIGHT = datetime(2026, 10, 12, 1, 0, tzinfo=UTC)  # 21:00 EDT, dark in Ottawa
TORONTO = Place("Toronto, Ontario", 43.70, -79.42, "America/Toronto")


def _clouds_on(day: str) -> str:
    return f" Clouds night of {day}: Ottawa 40%."


@pytest.fixture
def sw(spaceweather_plugin, mocker):
    """The announcer with #a watching Ottawa, quiet wind, no watches."""
    p = spaceweather_plugin
    p._channels = {"#a"}
    p._registry["auroraAnnounce"] = True
    p._registry["auroraAnnouncePlaces"] = ["Ottawa"]
    p._places = {"Ottawa": OTTAWA}
    p._samples = _series(NOW, 2.0)
    p._alerts = []

    def fetch(url, **_kw):
        assert url.endswith(spaceweather.ALERTS_PATH)
        if isinstance(p._alerts, BaseException):
            raise p._alerts
        return p._alerts

    p._fetch = mocker.patch("llm.spaceweather.fetch_json", side_effect=fetch)
    june = datetime(2026, 6, 1, tzinfo=UTC)  # through December: the polar tests
    p._series = mocker.patch(
        "llm.spaceweather.cloud_series",
        return_value=[(june + timedelta(hours=h), 40) for h in range(214 * 24)],
    )
    p._clouds = mocker.spy(spaceweather, "hours_within")
    return p


def _pass(p, now: datetime, *, alerts=None, wind=None) -> list:
    """Run one pass at ``now``; return the lines it sent."""
    p._now = now
    if alerts is not None:
        p._alerts = alerts
    if wind is not None:
        p._samples = _series(now, wind)
    before = len(p.sent)
    p._run_spaceweather_poll()
    return p.sent[before:]


# --- seeding ------------------------------------------------------------------


def test_cold_start_seeds_watches_silently(sw) -> None:
    assert _pass(sw, NOW, alerts=[G3_OCT12]) == []
    assert sw._sw_state.watches_seeded
    assert sw._sw_state.announced_levels["#a"] == {date(2026, 10, 12): 3, date(2026, 10, 13): 3}
    assert _pass(sw, NOW + timedelta(minutes=5)) == []


def test_cold_start_mid_episode_silent(sw) -> None:
    assert _pass(sw, NIGHT, wind=-14.0) == []
    state = sw._sw_state
    assert state.episode is not None
    assert state.episode_acked == {"#a": state.episode.start}
    assert _pass(sw, NIGHT + timedelta(minutes=5), wind=-14.0) == []


def test_feeds_seed_independently(sw) -> None:
    assert _pass(sw, NOW, alerts=spaceweather.FetchError("down")) == []
    assert sw._sw_state.episode_seeded
    assert not sw._sw_state.watches_seeded
    assert _pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12]) == []
    assert sw._sw_state.watches_seeded
    assert _pass(sw, NOW + timedelta(minutes=10)) == []


def test_new_channel_seeded_silently_for_watches(sw) -> None:
    _pass(sw, NOW)
    sw._channels = {"#a", "#b"}
    sent = _pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12])
    assert [ch for ch, _ in sent] == ["#a"]  # #b arrived with the watch already out


# --- watches ------------------------------------------------------------------


def test_watch_announced_once(sw) -> None:
    _pass(sw, NOW)
    sent = _pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12])
    assert sent == [("#a", "🌌 NOAA G3 storm watch for Oct 12–13." + _clouds_on("Oct 12"))]
    assert _pass(sw, NOW + timedelta(minutes=10)) == []


def test_superseding_watch_same_level_not_reannounced(sw) -> None:
    _pass(sw, NOW)
    assert len(_pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12])) == 1
    reissue = _watch(11, "2026 Oct 11 1100", "Oct 12:  G3 (Strong)   Oct 13:  G3 (Strong)   ")
    assert _pass(sw, NOW + timedelta(minutes=10), alerts=[G3_OCT12, reissue]) == []


def test_upgrade_g3_to_g4_reannounced(sw) -> None:
    _pass(sw, NOW)
    _pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12])
    upgrade = _watch(11, "2026 Oct 11 1100", "Oct 12:  G4 (Severe)   Oct 13:  G3 (Strong)   ")
    sent = _pass(sw, NOW + timedelta(minutes=10), alerts=[G3_OCT12, upgrade])
    assert sent == [("#a", "🌌 NOAA G4 storm watch for Oct 12." + _clouds_on("Oct 12"))]
    downgrade = _watch(12, "2026 Oct 11 1130", "Oct 12:  G3 (Strong)   ")
    assert _pass(sw, NOW + timedelta(minutes=15), alerts=[downgrade]) == []


def test_g2_watch_below_min_level_ignored(sw) -> None:
    real = json.loads((FIXTURES / "alerts.json").read_text())  # serial 285: Oct 11 G2
    _pass(sw, NOW)
    assert _pass(sw, NOW + timedelta(minutes=5), alerts=real) == []
    sw._SW_MIN_WATCH_LEVEL = 2  # the gate, not the parse
    sent = _pass(sw, NOW + timedelta(minutes=10))
    assert sent == [("#a", "🌌 NOAA G2 storm watch for Oct 11." + _clouds_on("Oct 11"))]


def test_partial_delivery_retries_only_failed_channel(sw) -> None:
    sw._channels = {"#a", "#b"}
    _pass(sw, NOW)
    sw._queue_ok = {"#b": False}
    sent = _pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12])
    assert [ch for ch, _ in sent] == ["#a"]
    sw._queue_ok = {}
    sent = _pass(sw, NOW + timedelta(minutes=10))
    assert [ch for ch, _ in sent] == ["#b"]


def test_watch_clouds_over_dark_window(sw) -> None:
    sw._registry["auroraAnnouncePlaces"] = ["Ottawa", "Sault_Ste._Marie"]
    sw._places = {"Ottawa": OTTAWA, "Sault Ste. Marie": spaceweather.FetchError("x")}
    _pass(sw, NOW)
    sent = _pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12])
    assert sent == [("#a", "🌌 NOAA G3 storm watch for Oct 12–13." + _clouds_on("Oct 12"))]
    assert sw._clouds.call_count == 1  # Sault Ste. Marie's lookup failed: left out


def test_watch_clouds_use_local_night_covering_the_utc_day(sw) -> None:
    """NOAA watch days are UTC days. In Toronto, UTC Oct 12 is mostly the
    local night of Oct 11 (23:43Z Oct 11 to 10:27Z Oct 12), not Oct 12's."""
    sw._sw_night_clouds([("Toronto", TORONTO)], date(2026, 10, 12), now=NOW)
    kw = sw._clouds.call_args.kwargs
    assert datetime(2026, 10, 11, 23, 30, tzinfo=UTC) < kw["start"]
    assert kw["start"] < datetime(2026, 10, 11, 23, 59, tzinfo=UTC)
    assert datetime(2026, 10, 12, 10, 15, tzinfo=UTC) < kw["end"]
    assert kw["end"] < datetime(2026, 10, 12, 10, 40, tzinfo=UTC)
    later = datetime(2026, 10, 12, 2, 0, tzinfo=UTC)
    sw._sw_night_clouds([("Toronto", TORONTO)], date(2026, 10, 12), now=later)
    assert sw._clouds.call_args.kwargs["start"] == later  # clipped to now
    assert sw._clouds.call_args.kwargs["end"] == kw["end"]


def test_watch_clouds_start_now_when_dark_already_began(sw) -> None:
    now = datetime(2026, 10, 12, 23, 58, tzinfo=UTC)  # Ottawa dark since about 23:43
    start, _end = sw._sw_dark_hours(OTTAWA, date(2026, 10, 12))
    assert start < now
    sw._sw_night_clouds([("Ottawa", OTTAWA)], date(2026, 10, 12), now=now)
    kw = sw._clouds.call_args.kwargs
    assert kw["start"] == now
    assert kw["end"] > now


def test_watch_clouds_use_next_night_when_window_is_over(sw) -> None:
    now = datetime(2026, 10, 11, 20, 0, tzinfo=UTC)  # 07:00 in Hobart: the night is over
    _start, end = sw._sw_dark_hours(HOBART, date(2026, 10, 11))
    assert end <= now
    sw._sw_night_clouds([("Hobart", HOBART)], date(2026, 10, 11), now=now)
    kw = sw._clouds.call_args.kwargs
    assert kw["start"] > now
    assert kw["start"] == sw._sw_dark_hours(HOBART, date(2026, 10, 12))[0]


def test_polar_places_cloud_windows(sw, mocker) -> None:
    tromso = Place("Tromsø, Troms", 69.65, 18.96, "Europe/Oslo")
    june = sw._sw_night_clouds([("Tromsø", tromso)], date(2026, 6, 20), now=NOW)
    assert june == {"Tromsø": None}  # midnight sun: no dark window, no figure
    assert not sw._clouds.called
    dusk = datetime(2026, 12, 20, 12, 0, tzinfo=UTC)
    mocker.patch("llm.spaceweather.dark_window", return_value=(dusk, None))  # polar night
    december = sw._sw_night_clouds([("Tromsø", tromso)], date(2026, 12, 20), now=NOW)
    assert december == {"Tromsø": 40}
    kw = sw._clouds.call_args.kwargs
    assert kw["end"] - kw["start"] == timedelta(hours=12)  # polar night: bounded


def test_departed_channels_and_past_dates_pruned(sw) -> None:
    sw._channels = {"#a", "#b"}
    _pass(sw, NOW, alerts=[G3_OCT12])
    sw._channels = {"#a"}
    _pass(sw, datetime(2026, 10, 13, 12, 0, tzinfo=UTC))
    assert sw._sw_state.announced_levels == {"#a": {date(2026, 10, 13): 3}}


def test_unreadable_alerts_payload_skips_watches(sw) -> None:
    _pass(sw, NOW, alerts={"not": "a list"})
    assert not sw._sw_state.watches_seeded
    assert sw._sw_state.episode_seeded


# --- storm conditions -----------------------------------------------------------


def test_storm_held_in_daylight_then_released_at_dark(sw) -> None:
    day = datetime(2026, 10, 11, 16, 0, tzinfo=UTC)  # noon in Ottawa
    _pass(sw, day - timedelta(hours=1))
    assert _pass(sw, day, wind=-14.0) == []
    assert _pass(sw, day + timedelta(hours=2), wind=-14.0) == []
    (sent,) = _pass(sw, NIGHT, wind=-14.0)
    assert sent[0] == "#a"
    assert "Look north. Clouds now: Ottawa 40%." in sent[1]
    assert _pass(sw, NIGHT + timedelta(minutes=5), wind=-14.0) == []


def test_one_line_per_channel_per_episode(sw) -> None:
    sw._channels = {"#a", "#b"}
    _pass(sw, NIGHT)
    sent = []
    for i in range(1, 4):
        sent += _pass(sw, NIGHT + timedelta(minutes=5 * i), wind=-14.0)
    assert sorted(ch for ch, _ in sent) == ["#a", "#b"]
    assert sent[0][1].startswith("🌌 Aurora storm conditions: Bz -14 nT for 2h56m at 620 km/s")


def test_new_episode_after_end_announces_again(sw) -> None:
    _pass(sw, NIGHT)
    assert len(_pass(sw, NIGHT + timedelta(minutes=5), wind=-14.0)) == 1
    assert _pass(sw, NIGHT + timedelta(hours=2), wind=2.0) == []
    assert sw._sw_state.episode is None
    assert sw._sw_state.episode_acked  # kept: a storm resuming must stay told
    assert _pass(sw, NIGHT + timedelta(hours=5, minutes=10), wind=2.0) == []
    assert len(_pass(sw, NIGHT + timedelta(hours=9), wind=-14.0)) == 1  # >3 h later: new


def _storm_with_gap(sw, gap_minutes: int) -> tuple[list, list]:
    """180 min at -15, ``gap_minutes`` at +5, then -15 until the episode is
    running again; a pass every 5 min. Returns (lines sent, episode starts)."""
    sw._channels = {"#a", "#b"}
    sw._registry["auroraAnnouncePlaces"] = []  # always dark: the gap is the variable
    t0 = NIGHT - timedelta(hours=1)
    resume = 180 + gap_minutes

    def bz_at(minute: float) -> float:
        if minute < 0:
            return 2.0
        if minute < 180 or minute >= resume:
            return -15.0
        return 5.0

    sent: list = []
    starts: list = []
    now = NIGHT
    while now <= NIGHT + timedelta(minutes=resume + 200):
        count = int((now - t0).total_seconds() // 60) + 1
        sw._samples = [
            Sample(ts=(ts := t0 + timedelta(minutes=i)),
                   bz=bz_at((ts - NIGHT).total_seconds() / 60), bt=15.0, speed=620.0)
            for i in range(count)
        ]  # fmt: skip
        sent += _pass(sw, now)
        if sw._sw_state.episode is not None:
            starts.append(sw._sw_state.episode.start)
        now += timedelta(minutes=5)
    assert sw._sw_state.episode is not None  # the storm did come back
    return sent, starts


@pytest.mark.parametrize("gap", [20, 90])
def test_northward_turn_within_window_is_the_same_storm(sw, gap) -> None:
    """A gap shorter than 3 h: one storm, one line per channel, and the
    duration still runs from the original start."""
    sent, starts = _storm_with_gap(sw, gap)
    assert sorted(ch for ch, _ in sent) == ["#a", "#b"]
    assert len(set(starts)) == 1


def test_northward_turn_past_window_is_a_new_storm(sw) -> None:
    sent, starts = _storm_with_gap(sw, 240)
    assert sorted(ch for ch, _ in sent) == ["#a", "#a", "#b", "#b"]
    assert len(set(starts)) == 2


def test_channel_new_mid_episode_acked_silently(sw) -> None:
    _pass(sw, NIGHT)
    assert len(_pass(sw, NIGHT + timedelta(minutes=5), wind=-14.0)) == 1
    sw._channels = {"#a", "#b"}
    assert _pass(sw, NIGHT + timedelta(minutes=10), wind=-14.0) == []
    assert sw._sw_state.episode_acked["#b"] == sw._sw_state.episode.start


def test_southern_places_look_south(sw) -> None:
    sw._registry["auroraAnnouncePlaces"] = ["Hobart"]
    sw._places = {"Hobart": HOBART}
    _pass(sw, NOW)
    night = datetime(2026, 10, 11, 13, 0, tzinfo=UTC)  # 00:00 AEDT
    (sent,) = _pass(sw, night, wind=-14.0)
    assert "Look south. Clouds now: Hobart 40%." in sent[1]


def test_unmatched_places_treated_as_placeless(sw) -> None:
    """A typo ("Otawa") must not block storm lines forever: no place raised,
    none resolved, so the channel is placeless (always dark, no clouds)."""
    sw._registry["auroraAnnouncePlaces"] = ["Otawa"]
    sw._places = {"Otawa": None}
    _pass(sw, NOW)
    (sent,) = _pass(sw, NOW + timedelta(minutes=5), wind=-14.0)  # daytime in Ottawa
    assert sent[1].endswith("Look north.")
    assert not sw._clouds.called
    sw._sw_channel_places("#a")
    assert sw.log.warning.call_count == 1  # warned once, not per pass


def test_geocode_outage_holds_storm_line(sw) -> None:
    sw._registry["auroraAnnouncePlaces"] = ["Ottawa", "Otawa"]
    sw._places = {"Ottawa": spaceweather.FetchError("down"), "Otawa": None}
    _pass(sw, NIGHT)
    assert _pass(sw, NIGHT + timedelta(minutes=5), wind=-14.0) == []  # transient: held
    sw._places = {"Ottawa": OTTAWA, "Otawa": None}
    (sent,) = _pass(sw, NIGHT + timedelta(minutes=10), wind=-14.0)
    assert sent[1].endswith("Clouds now: Ottawa 40%.")


def test_cloud_failure_renders_question_mark(sw, mocker) -> None:
    sw._series.side_effect = spaceweather.FetchError("x")
    _pass(sw, NOW)
    (sent,) = _pass(sw, NIGHT, wind=-14.0)
    assert sent[1].endswith("Clouds now: Ottawa ?.")


def test_wind_outage_skips_only_storms(sw) -> None:
    sw._samples = spaceweather.InvalidPayload("bad")
    _pass(sw, NOW)
    assert sw._sw_state.watches_seeded
    assert not sw._sw_state.episode_seeded


def test_announce_off_fetches_nothing(sw) -> None:
    sw._registry["auroraAnnounce"] = False
    _pass(sw, NIGHT, alerts=[G3_OCT12], wind=-14.0)
    assert not sw._fetch.called
    assert not sw._sw_state.watches_seeded
    assert not sw._sw_state.episode_seeded


def test_no_places_announces_at_once_without_clouds(sw) -> None:
    sw._registry["auroraAnnouncePlaces"] = []
    _pass(sw, NOW)
    sent = _pass(sw, NOW + timedelta(minutes=5), alerts=[G3_OCT12], wind=-14.0)  # daytime
    assert sent == [
        ("#a", "🌌 NOAA G3 storm watch for Oct 12–13."),
        (
            "#a",
            "🌌 Aurora storm conditions: Bz -14 nT for 2h56m at 620 km/s "
            "(~40 min to Earth). Look north.",
        ),
    ]
    assert not sw._clouds.called


def test_pass_budget_stops_sending(sw) -> None:
    sw._channels = {"#a", "#b"}
    _pass(sw, NIGHT)
    real_deliver = sw._deliver_spaceweather_line

    def slow_deliver(channel, line):
        sw._mono += sw._SPACEWEATHER_PASS_BUDGET + 1
        return real_deliver(channel, line)

    sw._deliver_spaceweather_line = slow_deliver
    sent = _pass(sw, NIGHT + timedelta(minutes=5), alerts=[G3_OCT12], wind=-14.0)
    assert [ch for ch, _ in sent] == ["#a"]  # out of budget before #b, and before storms
    sent = _pass(sw, NIGHT + timedelta(minutes=10), wind=-14.0)
    assert [ch for ch, _ in sent] == ["#b"]


def test_unexpected_error_is_logged_not_raised(sw) -> None:
    sw._sw_announce_channels = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    _pass(sw, NOW)
    assert sw.log.error.called


def test_deliver_without_irc_returns_false(sw) -> None:
    sw._irc_for_channel = lambda _ch: None
    assert sw._deliver_spaceweather_line("#a", "x") is False
    assert sw.sent == []


# --- rendering ------------------------------------------------------------------


def test_render_storm_line_exact() -> None:
    ep = Episode(
        start=NOW - timedelta(hours=3, minutes=10),
        min_bz=-14.0,
        latest_speed=620.0,
        mean_ey=None,
        lead_minutes=40,
    )
    line = spaceweather.render_storm_line(
        ep, now=NOW, clouds={"Ottawa": 10, "Toronto": 90}, direction="north"
    )
    assert line == (
        "🌌 Aurora storm conditions: Bz -14 nT for 3h10m at 620 km/s (~40 min to Earth). "
        "Look north. Clouds now: Ottawa 10%, Toronto 90%."
    )


def test_render_storm_line_without_speed_or_clouds() -> None:
    ep = Episode(
        start=NOW - timedelta(minutes=5),
        min_bz=-11.4,
        latest_speed=None,
        mean_ey=None,
        lead_minutes=None,
    )
    assert spaceweather.render_storm_line(ep, now=NOW, clouds={}, direction="south") == (
        "🌌 Aurora storm conditions: Bz -11 nT for 0h05m. Look south."
    )


@pytest.mark.parametrize(
    ("levels", "span"),
    [
        ({date(2026, 10, 12): 3}, "Oct 12"),
        ({date(2026, 10, 31): 3, date(2026, 11, 1): 4}, "Oct 31–Nov 1"),
        ({date(2026, 10, 12): 3, date(2026, 10, 14): 3}, "Oct 12, Oct 14"),
    ],
)
def test_render_watch_line_spans(levels, span) -> None:
    top = max(levels.values())
    assert spaceweather.render_watch_line(levels, {}) == f"🌌 NOAA G{top} storm watch for {span}."


def test_render_watch_line_unknown_cloud() -> None:
    line = spaceweather.render_watch_line({date(2026, 10, 12): 3}, {"Ottawa": None, "Toronto": 5})
    assert line.endswith("Clouds night of Oct 12: Ottawa ?, Toronto 5%.")


def test_watch_news_only_rises_at_or_above_min() -> None:
    levels = {date(2026, 10, 12): 4, date(2026, 10, 13): 3, date(2026, 10, 14): 2}
    told = {date(2026, 10, 12): 3, date(2026, 10, 13): 3}
    assert spaceweather.watch_news(levels, told, min_level=3) == {date(2026, 10, 12): 4}


def test_mean_cloud() -> None:
    assert spaceweather.mean_cloud([]) is None
    assert spaceweather.mean_cloud([(NOW, 10), (NOW, 21)]) == 16


# --- scheduling -----------------------------------------------------------------


def test_enqueue_dedupes_inflight(spaceweather_plugin, mocker) -> None:
    mocker.patch("llm.plugin.schedule")
    spaceweather_plugin._sw_poll_inflight.set()
    spaceweather_plugin._enqueue_spaceweather_poll()
    assert not spaceweather_plugin._llm_executor.submit.called


def test_enqueue_submits_and_rearms_when_done(spaceweather_plugin, mocker) -> None:
    sched = mocker.patch("llm.plugin.schedule")
    p = spaceweather_plugin
    p._enqueue_spaceweather_poll()
    name, fn = p._llm_executor.submit.call_args.args
    assert name == "spaceweather_poll"
    assert fn == p._run_spaceweather_poll
    assert p._sw_poll_inflight.is_set()
    fut = p._llm_executor.submit.return_value
    for call in fut.add_done_callback.call_args_list:
        call.args[0](fut)
    assert not p._sw_poll_inflight.is_set()
    assert sched.addEvent.call_args.kwargs["name"] == "llm_spaceweather_poll"
    assert sched.addEvent.call_args.args[1] == 1000.0 + 300


def test_enqueue_rearms_after_submit_failure(spaceweather_plugin, mocker) -> None:
    sched = mocker.patch("llm.plugin.schedule")
    p = spaceweather_plugin
    p._llm_executor.submit.side_effect = RuntimeError("pool gone")
    with pytest.raises(RuntimeError):
        p._enqueue_spaceweather_poll()
    assert not p._sw_poll_inflight.is_set()
    assert sched.addEvent.called


def test_closing_executor_noop(spaceweather_plugin, mocker) -> None:
    sched = mocker.patch("llm.plugin.schedule")
    p = spaceweather_plugin
    p._llm_executor.closing = True
    p._enqueue_spaceweather_poll()
    p._schedule_spaceweather_poll()
    assert not p._llm_executor.submit.called
    assert not sched.addEvent.called


def test_teardown_removes_event(plugin_env, mocker) -> None:
    plugin, _irc, _msg = plugin_env
    remove = mocker.spy(__import__("llm.plugin", fromlist=["schedule"]).schedule, "removeEvent")
    plugin.die()
    assert any(c.args == ("llm_spaceweather_poll",) for c in remove.call_args_list)
