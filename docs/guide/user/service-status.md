# Service status

The bot watches one or more live status pages — Claude's, GitHub's, and
OpenAI's by default — and can answer questions about any of them in
conversation, announcing incidents on its own where a channel has opted in.
Both Atlassian Statuspage pages and
incident.io pages work; incident.io is read through its Atlassian-compatible
endpoints, so it needs no separate setup. Which pages it watches
(`statusPageUrls`, up to 5) and whether a channel gets
announcements (`statusAnnounce`) are operator settings: see
[Status pages](../operator/tuning-monitoring.md#status-pages). With no
pages configured the bot has no status awareness at all and falls back to
whatever the chat model already believes, which is usually months out of
date.

## Asking

Ask in plain language. There's no command.

```
<you>     VibeBot, is Claude down?
<VibeBot> Yeah, they're investigating elevated error rates on Opus 4.5.
          The API's showing degraded; everything else looks fine.
```

Ask about any configured page by name — "is GitHub down?" is meant to get
GitHub's reading, not Claude's. A question about a watched page is normally
answered from the usual all-services reading rather than a page fetched
alone; that's the same one-call reading that lets "are Claude and Codex up?"
get answered in a single reply. It's the model deciding which page a
question means and which name to use, not anything that pins the page down
for it, so an oddly phrased question can occasionally answer from the wrong
entry. A page the operator hasn't configured can't be checked at all.

The bot reads the live pages rather than answering from memory. With more
than one configured, it polls them in rotation within a single pass, so a
slow or unreachable page can't stall the others — a given source might
occasionally wait longer than one poll cycle before its turn comes round
again. A reading older than 10 minutes is reported as stale rather than
current. When a page can't be reached at all, the bot says so and reports the
last reading it has, rather than guessing; before the first successful read
it says only that it hasn't read the page yet.

A scheduled check works too. `@remind in 2 hours check whether Claude is still
having problems` runs the same lookup at fire time — see
[Action reminders](reminders.md#action-reminders). Where an operator has enabled
conversational scheduling and you're logged in, "VibeBot, every weekday at 9
a.m. tell me if Claude is having problems" makes it recurring — see
[Recurring reminders and scheduled tasks](reminders.md#recurring-reminders-and-scheduled-tasks).

Roleplay replies can't reach the status pages. Ask outside the story.

## Past incidents

Questions about the past pull a second reading — the recent incident list,
not just what's broken now. It comes back for every watched page at once,
the same as a current-status question would, and the reply still focuses on
whichever page was asked about:

```
<you>     VibeBot, has Claude been flaky lately?
<VibeBot> Three times this week: elevated API errors on Tuesday (about 40
          minutes), a Sonnet latency spike Thursday that ran two hours, and
          degraded search yesterday, cleared in 25 minutes.
```

Five incidents at most, newest first, each with its name, impact level, how
long ago it started, and how long it lasted. Update text and per-incident links
are left out, so the bot can't quote you a postmortem or link a single
incident — only that status page as a whole. An incident still open has no
duration to report.

That history is fetched only when the question calls for it, and cached for an
hour afterwards, so it can lag the live page by that much. If the fetch fails
the bot falls back to the last history it read — past that hour, with nothing
marking it stale — and says nothing about the past if it has none. Current
status still answers.

## Announcements

`statusAnnounce` is a per-channel, all-or-nothing switch: where an operator
has enabled it, the channel hears incidents from every `statusPageUrls` page
as they open, and again as they clear. There's no picking and choosing
individual pages per channel. Asking, by contrast, always works regardless of this
setting — a channel that never announces can still ask "is GitHub down?".

```
<VibeBot> Claude status: Elevated error rates on the API (investigating) — https://status.claude.com/incidents/005ym4vzrq2w
<VibeBot> Claude status: Elevated error rates on the API resolved after 1h 23m — https://status.claude.com/incidents/005ym4vzrq2w
```

The link points at the incident itself, not the status page's front door. It
is built from that page's own configured URL plus the incident's own id, so it
stays on the page it came from; an id in any unexpected shape drops back to
the bare page URL.

That template is the floor, not the usual output. Up to six times an hour the
bot restates the same facts in the channel's own voice instead; a rewrite is
thrown away and the template sent if it drops the service name, carries a link
to anywhere but the incident's own configured page, or the model ran out of
room and stopped mid-sentence. Openings and all-clears across all configured
pages draw on that same hourly budget.

The duration runs from the incident's own start time to the poll that saw it
clear, so it tracks the polling rotation's own granularity — with several
pages configured, a given one might go more than one 2-minute cycle between
reads — and rounds to whole minutes. An incident the status page never dated
is announced without one.

Openings and resolutions are the whole vocabulary. Status moves within an
incident (investigating → identified → monitoring), component-only changes,
and scheduled maintenance are not announced.

Incidents already open when the bot starts count as already announced, so a
restart mid-outage doesn't replay one into the channel. Their all-clear still
fires, though — the bot watched that incident end even if it never announced
the start. An incident that both opened and closed while the bot was down is
silent in both directions. Ask the bot if you need its current state.
