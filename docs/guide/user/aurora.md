# Aurora

Ask the bot whether the northern lights are likely at a place. It reads
NOAA's live aurora forecast, the current solar wind, and the Kp forecast for
the next 24 hours, then adds when it gets dark there and how cloudy it will
be while it is. An operator turns this on per channel with `auroraEnabled`;
see [Aurora](../operator/configuration.md#aurora).

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
  southward for hours. That is a heuristic, not a guarantee of a storm. The
  threshold and duration are operator settings (`spaceWeatherBzThreshold`,
  `spaceWeatherBzHours`).
- **Dark** means the sun is more than 12 degrees below the horizon.

When a source can't be reached, the bot leaves that part out and says so.
An aurora forecast more than an hour old is marked stale.

## Announcements

An operator can have the bot post to a channel on its own, with
`spaceWeatherAnnounce`. It posts when NOAA issues a strong geomagnetic storm
watch (G3 or higher unless the operator changes it), and again if a watch is
raised. It also posts once when storm conditions set in, as soon as it is dark
at one of the channel's places.

```
<VibeBot> 🌌 NOAA G3 storm watch for Oct 12–13. Clouds tonight: Ottawa 20%,
          Toronto 85%.
<VibeBot> 🌌 Aurora storm conditions: Bz -14 nT for 3h10m at 620 km/s (~40 min
          to Earth). Look north. Clouds now: Ottawa 10%, Toronto 90%.
```

The places are the channel's, set by the operator (`spaceWeatherPlaces`),
never ones the bot knows about you. A `?` means the cloud forecast couldn't be
read.
