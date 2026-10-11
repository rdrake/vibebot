# Aurora and space weather — design

Date: 2026-10-11
Status: draft, awaiting review

## Goal

Tell IRC users when a major geomagnetic storm makes the northern lights
likely, and answer "can I see it tonight from <place>?" with aurora odds,
darkness, and cloud cover. Major events only: no routine Kp chatter.

## Decisions already made

- Location: the user names a place; otherwise the model takes it from the
  user's saved memories (already injected into the prompt at
  `service.py:8524`); otherwise it asks. No new location store.
- Announcements report clouds for a per-channel configured list of places,
  never for places pulled from user memories.
- No extras (aurora meter, photo reactions, morning recap) in this cut.

## The signal (research summary)

| Signal | Lead time | Role |
|---|---|---|
| NOAA G3+ watch (`alerts.json`) | 1–3 days | Day-ahead heads-up |
| Bz ≤ −10 nT sustained ≥ 3 h at L1 | ~45 min after arrival at L1 | "Storm is underway" trigger |
| OVATION aurora grid | ~30–90 min | Per-place odds in the tool |
| Kp forecast / observed | 3 h blocks | Context in the tool; not an announcement trigger |

The Bz rule is Gonzalez & Tsurutani (1987): every intense storm
(Dst < −100 nT) in their set followed southward IMF Bz below −10 nT lasting
more than 3 hours, equivalent to a dawn–dusk electric field
`Ey = V × Bs` above 5 mV/m. A 1976–2002 follow-up found the same
condition always produced a Dst depression. The IRC source suggested 4 h;
the paper says 3 h. Default 3 h, configurable.

OVATION Prime is driven by the Newell (2007) coupling function, built from
the same L1 wind and field data. It is the better per-place estimate; the
Bz rule is the better "big event, announce it" trigger because it is
simple, published, and computable from 24 h of 1-minute data.

Speed matters for timing: at 582 km/s, L1 → Earth takes ~43 min
(1.5 million km / speed). The announcement states this lead time.

## Feeds (all verified live 2026-10-11)

- `https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json` — 24 h,
  1-minute, newest first. Multiple spacecraft interleaved (ACE, IMAP,
  SOLAR1); **use only rows with `active: true`** (SOLAR1 today). Fields:
  `bz_gsm`, `by_gsm`, `bt`.
- `https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json` — same
  layout; `proton_speed`, `proton_density`.
- `https://services.swpc.noaa.gov/products/alerts.json` — watches,
  warnings, alerts as free text with a `Serial Number`.
- `https://services.swpc.noaa.gov/products/noaa-planetary-k-index-forecast.json`
  — `kp`, `observed` (`observed` | `estimated` | `predicted`), `noaa_scale`.
- `https://services.swpc.noaa.gov/json/ovation_aurora_latest.json` —
  `coordinates: [[lon 0..359, lat -90..90, pct], ...]`, 918 KB. Fetched
  on tool call only, cached until its `Forecast Time`.
- Open-Meteo geocoding and forecast (`hourly=cloud_cover`), free, no key.

## Components

### `plugins/llm/src/llm/spaceweather.py`

Same shape as `statuspage.py`: pure functions plus one fetch layer.

- `parse_rtsw(mag, wind) -> list[Sample]` — active rows only, sorted
  oldest first, `Sample(ts, bz, bt, speed)`.
- `bz_episode(samples, *, threshold_nt, hours, now) -> Episode | None` —
  qualifies when, over the trailing `hours` window, at least 80% of
  minutes have data and at least 80% of those are `bz ≤ threshold_nt`.
  Episode starts at the first qualifying minute; ends when the trailing
  30-min mean Bz rises above −5 nT. Reports duration, min Bz, mean speed,
  mean `Ey` (mV/m), and lead time `1.5e6 / speed` minutes.
- `parse_watches(alerts) -> list[Watch]` — messages containing `WATCH`
  and `Category G<n>`; extracts serial, level, and the dates watched.
- `sun_altitude(lat, lon, when) -> float` — NOAA solar position formula,
  no new dependency. Dark means altitude < −12°.
- `ovation_at(grid, lat, lon) -> (overhead_pct, poleward_max_pct,
  poleward_max_lat)` — nearest cell, and the maximum within 10° poleward
  at the same longitude (a stand-in for "on your northern horizon";
  labelled as an approximation in the payload).
- `render_watch_line(...)`, `render_storm_line(...)` — deterministic
  announcement text (see Announcements).

### Tool: `aurora(place)`

Registered with the chat tools. Returns JSON facts; the model writes the
reply in the channel's voice.

```json
{
  "place": "Toronto, ON", "lat": 43.7, "lon": -79.4,
  "now": {"bz_nt": -12.1, "speed_kms": 610, "storm_minutes": 95,
          "lead_minutes": 41},
  "ovation": {"overhead_pct": 4, "poleward_max_pct": 35,
              "poleward_max_lat": 51, "valid_until": "...", "note": "..."},
  "kp_next_24h": [{"start": "...", "kp": 6.0, "scale": "G2"}],
  "dark": {"from": "19:42", "until": "06:31", "tz": "local"},
  "cloud_pct_while_dark": [{"hour": "21:00", "pct": 20}],
  "active_watch": {"level": "G3", "dates": ["2026-10-12"]}
}
```

Geocoding failure returns an error the model relays ("couldn't find
that place"). Any single feed failing drops only its key.

### Announcer

A poller thread modelled on the status announcer, every 5 min.

1. **Watch:** a new G3+ watch serial → one line per configured channel.
   Clouds: for each configured place, mean cloud cover over that place's
   dark hours on the watched night(s), when within Open-Meteo's range.
2. **Storm:** a Bz episode qualifies **and** at least one configured place
   for the channel is dark → one line per channel per episode. If the
   episode qualifies in daylight, it announces once darkness arrives,
   provided the episode is still running. Clouds: next 3 h per place.

Dedupe state is in memory. Like the status announcer, a cold start seeds
current watches and any running episode as already announced, so a
restart never repeats a line.

Example lines (deterministic; the model does not write these):

- `🌌 NOAA G3 storm watch for Oct 12–13. Clouds tonight: Ottawa 20%,
  Toronto 85%.`
- `🌌 Major storm underway: Bz −14 nT for 3h10m at 620 km/s (~40 min to
  Earth). Look north. Clouds now: Ottawa 10%, Toronto 90%.`

### Config (no backwards compatibility)

- `auroraEnabled` (channel, bool, default off) — exposes the tool
- `spaceWeatherAnnounce` (channel, bool, default off)
- `spaceWeatherPlaces` (channel, space-separated list of places, geocoded
  once and cached)
- `spaceWeatherBzThreshold` (global, default −10)
- `spaceWeatherBzHours` (global, default 3)
- `spaceWeatherMinWatchLevel` (global, default 3)

Prod change: set `spaceWeatherAnnounce` and `spaceWeatherPlaces` in
`bot.conf` with the bot stopped.

## Testing

- Fixtures: today's captured `rtsw_mag_1m`, `rtsw_wind_1m`, `alerts`,
  `kp forecast`, and a trimmed OVATION grid.
- `bz_episode`: synthetic series for qualify, short dip, gap-heavy data,
  inactive-spacecraft rows ignored, episode end and re-arm.
- `parse_watches`: real G1 watch text (not announced) and a G3 sample.
- `sun_altitude`: known sunrise/sunset values for Toronto to ±2 min.
- Announcer: fake opener (as in the statuspage tests); daylight-hold then
  dark-release; restart does not re-announce.
- Tool: payload shape, single-feed failure, unknown place.

## Out of scope

Aurora meter, photo reactions, morning recap, per-user alert
subscriptions, moon phase, magnetic-latitude Kp tables.
