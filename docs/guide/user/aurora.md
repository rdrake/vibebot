# Aurora

Ask the bot whether the northern lights are likely at a place. It reads
NOAA's live aurora forecast, the current solar wind, and the Kp forecast for
the next 24 hours, then adds when it gets dark there and how cloudy it will
be while it is. It is on in every channel unless an operator turns
`auroraEnabled` off; see [Aurora](../operator/configuration.md#aurora).

## Quick how-to

**Users.** Mention the bot and name a place:

```
<you> VibeBot, any aurora tonight in Toronto?
```

The reply covers the aurora odds overhead and toward the pole, the solar wind
right now, Kp for the next 24 hours, when it gets dark there, and the cloud
cover while it is.

**Operators.** Turn on announcements in a channel, and optionally name places
for cloud cover and to hold storm alerts until dark:

```
@config channel #yourchan plugins.LLM.auroraAnnounce True
@config channel #yourchan plugins.LLM.auroraAnnouncePlaces Ottawa Toronto
```

Turn the tool off in a channel with
`@config channel #yourchan plugins.LLM.auroraEnabled False`. A runtime
`@config` change is lost when the bot restarts; to keep it, set it in
`bot.conf` with the bot stopped.

## Asking

Ask in plain language. There is no command.

```
<you>     VibeBot, any aurora tonight in Toronto?
<VibeBot> Unlikely overhead (4%), but up to 35% toward the northern horizon
          around 50°N. Bz is -12 nT at 610 km/s, so storm conditions have
          held for about 95 minutes. Dark from 19:42, clouds 20% at 21:00.
```

```
<you>     VibeBot, can I see the aurora from Tromsø this week?
<VibeBot> Good odds: 70% overhead right now. It is dark all evening, and
          Kp is forecast at 5 for the next three hours. Clouds are 85%, so
          look for a gap.
```

If you name no place, the bot uses one it knows about you. If it knows none,
it asks.

## What the numbers mean

- **Overhead** is NOAA's aurora probability at the place itself.
- **Toward the pole** is the strongest probability within 10 degrees toward
  the pole. It stands in for what you would see low on the horizon.
- **Storm conditions** means the solar wind's Bz has stayed strongly
  southward for hours: Bz at or below -10 nT for most of the last 3 hours.
  That is a heuristic, not a guarantee of a storm.
- **Dark** means the sun is more than 12 degrees below the horizon.

When a source can't be reached, the bot leaves that part out and says so.
An aurora forecast more than an hour old is marked stale. Answers can lag by
a few minutes: the bot reuses solar wind for 2 minutes, storm alerts for 5,
the aurora forecast for 10, cloud cover for an hour and the Kp forecast for 3
hours.

## Announcements

An operator can have the bot post to a channel on its own with
`auroraAnnounce`. Announcements go to every channel with `auroraAnnounce` on;
set `auroraAnnouncePlaces` to add cloud cover and hold storm alerts until it is
dark there. It posts when NOAA issues a strong geomagnetic storm watch (G3 or higher), and
again if a watch is raised. It also posts once when storm conditions set in, as soon as it is dark
at one of the channel's places (at once if it has none).

```
<VibeBot> 🌌 NOAA G3 storm watch for Oct 12–13. Clouds night of Oct 12:
          Ottawa 20%, Toronto 85%.
<VibeBot> 🌌 Aurora storm conditions: Bz -14 nT for 3h10m at 620 km/s (~40 min
          to Earth). Look north. Clouds now: Ottawa 10%, Toronto 90%.
```

The places are the channel's, set by the operator (`auroraAnnouncePlaces`),
never ones the bot knows about you. A watch is for NOAA's day, which runs
midnight to midnight UTC, so "night of Oct 12" is the local night that falls
mostly inside it. A `?` means there is no cloud figure for that place: the
forecast couldn't be read, the night is past its 3 days, or it has no dark
hours then. A place the bot
couldn't look up is left out.
