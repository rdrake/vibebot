# Tuning and monitoring

## Choosing models

The model surface splits by workload so cost and quality can differ per task. Every model setting below is channel-overridable except `verseCompactionModel`, which is global.

| Setting | Used for |
|---------|----------|
| `assistantModel` | Chat, the planner loop, bridge tool selection, reminder parsing, memory cleanup, image-prompt rewrites, scheduled tasks |
| `memoryModel` | Memory extraction after every reply. Falls back to `assistantModel` when empty |
| `codeModel` | The code-generation one-shot behind `@code` and the `generate_code` tool. The `@code` planner loop that calls it runs on `assistantModel` |
| `imageModel` | Image generation |
| `searchModel` | Web search and URL fetch tools. Falls back to `assistantModel` when empty. GitHub links are read through GitHub's API instead |
| `subjectResearchModel` | The subject-research pre-stage in front of the `@draw` planner. Falls back to `searchModel`, then `assistantModel` |
| `verseModel` | Verse-mode narration. Falls back to `assistantModel` when empty |
| `verseCompactionModel` | The daily verse compaction digest |

Guidance from operating this surface:

- `assistantModel` needs vision support if users paste image URLs into chat.
- Reasoning models make terse, flat narrators. If the assistant model is a reasoning model, point `verseModel` at a non-reasoning model for verse prose.
- Compaction is a cheap summarization job; leave `verseCompactionModel` on a flash-lite class model.
- Memory extraction runs after every reply and its output is never shown. Put `memoryModel` on a flash-lite class model. On Gemini's free tier each model has its own daily quota, so a different model also keeps extraction from using up the chat model's.
- Subject research is a lookup, not a writing job, so a flash-class model is enough. What matters is whether it will describe real people at all: a model that declines contributes nothing and the picture loses its likeness. If dossiers come back empty on obvious subjects, point `subjectResearchModel` at a more permissive model than `searchModel`.

Keys are not part of this per-command surface: one environment variable per *provider* covers every model on that provider, regardless of which setting names the model. Point a channel at a different model and, if that model's provider is already configured, no key change is needed. See [Configuration → API keys](configuration.md#api-keys) for the variable names.

```
@config channel #yourchan plugins.LLM.assistantModel anthropic/claude-haiku-5-5
```

## Context tuning

Conversation context lets the bot follow recent exchanges in a channel.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `contextEnabled` | `True` | channel | Enable conversation context |
| `contextMaxMessages` | `20` | channel | Messages kept in per-user history |
| `contextTimeoutMinutes` | `5` | channel | Clear context after this much inactivity |
| `privacyOptInOnly` | `True` | channel | Send only bot-addressed lines to the provider; `False` tracks every message and enables history lookup |
| `channelContextMaxMessages` | `10` | channel | Messages in the shared channel context |

The shared channel context is what lets the bot connect a question from one user with a follow-up from another.

**Privacy note.** With `privacyOptInOnly` turned off, every message in the channel is sent to third-party LLM providers as context, and the bot can pull up to 100 lines of server history on request. The setting is on by default. Turn it off only where users know and consent.

## Limnoria tool bridge

The bridge exposes loaded Limnoria plugin commands to the LLM as two tools: `run_limnoria_command` dispatches one command, and `search_bridge_commands` substring-searches the exposed set by plugin name, command name, argument syntax and description. When the model needs something a stock plugin already does, a ping, the time, a seen lookup, it defers to that plugin instead of guessing. The bridge is opt-in per channel.

### Enabling

```
@config channel #yourchan plugins.LLM.bridgeEnabled True
```

That activates the curated default plugin set. To take full control, set an explicit list; a non-empty value replaces the default:

```
@config channel #yourchan plugins.LLM.bridgeAllowedPlugins Misc Time
```

Names must match Limnoria's exact form (`Misc`, `Time`, `Math`); mismatched names are ignored silently. The bridge can only expose plugins that are already loaded, so `@load` them first.

### Curated default set

With `bridgeAllowedPlugins` empty, the bridge exposes:

`Misc Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG Network`

Each is pure-read or has its write commands gated by `bridgeAllowMutating`. The list lives in `DEFAULT_ALLOWED_PLUGINS` (`plugins/llm/src/llm/limnoria_bridge.py`).

### Security model

Three independent layers limit what the bridge reaches:

1. **Hard-denied plugins**, never bridged regardless of config: `LLM` (recursion), `Owner`, `Admin`, `Config`, `Channel`, `User` (bot, channel, and account management).
2. **Hard-denied commands**, blocked even inside allowlisted plugins: every URL-fetching `Web` read — `fetch`, `location`, `headers`, `doctype`, `size`, `title` — because each hands the bot's network position to a caller-supplied URL; `Utilities.apply` and `Utilities.let` (re-dispatch bypass); `Misc.more` and `Misc.clearmores` (interactive scrollback only). Those six `Web` reads are not writes, so `bridgeAllowMutating` does not gate them, and `Web` ships in the curated default set, so the deny is unconditional.
3. **Limnoria capabilities**, applied per command at dispatch time. Commands guarded by default-deny anti-capabilities are refused unless explicitly granted.

### Write-command gate

Commands that change persistent state (offline notes, RSS registrations, karma changes, queued PMs) stay hidden from the LLM by default, even when their plugin is allowlisted. Any hallucinated dispatch of a hidden write returns a denial. Open the gate per channel:

```
@config channel #yourchan plugins.LLM.bridgeAllowMutating True
```

The classification lives in `MUTATING_COMMANDS` (`plugins/llm/src/llm/limnoria_bridge.py`). When the gate is closed and an allowlisted plugin has hidden writes, the tool description tells the model the gate exists.

### Scheduled LLM tasks

The `schedule_llm_task` tool schedules a future assistant turn. At fire time the bot replays the creator's IRC identity and runs the prompt on the reminder-action profile, which carries every native tool: search, fetch, draw, code, memory, usage, and pending-task management. It does not carry the bridge — bridged Limnoria commands ride `@ask`, mentions and PMs only. It differs from `set_reminder`:

- `set_reminder` delivers fixed text at fire time. "Remind me to switch the laundry at 6."
- `schedule_llm_task` runs an LLM turn at fire time. "Every Monday at 9, check my open PRs and tell me which are stale." Use it when the task needs tools when it fires.

Operational properties:

- `bridgeScheduledTaskLimit` (default 5, channel) caps each creator's active schedules; `0` disables scheduling. Each fire still counts against the creator's `ask` rate-limit bucket.
- Schedules bind to the creator's account at create time; the tool refuses unauthenticated callers.
- A schedule auto-cancels the first time it fires after its creator loses `llm.ask`: the row is deleted, the reply target gets a one-line notice, and an `INFO` line records `scheduled_llm_task fire: <event> creator <nick> lost llm.ask; auto-cancelling`. At the default `WARNING` that notice is the only trace.
- A fired task cannot schedule another task (depth cap of 1), which blocks recursion.
- Users cannot list or cancel schedules by talking to the bot — the unified pending-task tools are off the chat surface — but `@remind list`, `del` and `clear` reach their own tasks as well as their reminders. Owners use `@remind admin list <nick>` and `@remind admin del <nick> <id>` to reach anyone's. Limnoria's own `@scheduler list` does not show these tasks; they live in the plugin's database.

### Known limitation

Plugins that nest sub-leaves under a `Commands` group (notably `RSS`'s `announce add` and friends) are not surfaced by the bridge. Manage those with the native IRC commands.

## HTTP output

`@code`, `@draw`, and `@story` save output as files served over HTTP.

- **Built-in server (default).** With `httpRoot` empty, files land in Limnoria's web directory (`data/web/llm/`) and its built-in HTTP server serves them. See the [Limnoria HTTP server documentation](https://docs.limnoria.net/use/httpserver.html) for port and public URL setup.
- **External server.** Set `httpRoot` to a filesystem path (for example `/var/www/llm`) and `httpUrlBase` to the matching public URL. You serve the directory with nginx, caddy, or similar.

| Setting | Default | Description |
|---------|---------|-------------|
| `httpRoot` | empty | Filesystem path for output files. Empty uses the built-in server |
| `httpUrlBase` | empty | Public base URL. Empty uses Limnoria's public URL plus `/llm/` |
| `longReplyTeaserMaxChars` | `220` | Characters in the one-line teaser when a long reply is saved as HTML and linked |
| `multilineMaxLines` | `3` | Replies up to this many IRC lines go inline as one `draft/multiline` batch; longer ones get the teaser and link. `0` turns it off |
| `fileCleanupAge` | `720` | Delete output files older than this many hours (30 days) |
| `fileCleanupMax` | `1000` | Cap on files kept in the output directory |

## Concurrency

Every outbound LLM call, foreground or background, takes a slot from one bounded executor, so a single knob caps how much load reaches a provider no matter which thread started the call. Keeping a slow call off the IRC driver is a separate mechanism: nick-addressed messages are dispatched on their own thread so the driver can flush typing notifications while the call is in flight.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `maxConcurrentLLMCalls` | `16` | global | Cap on simultaneous outbound LLM calls |

Lower it on small hosts or when a provider rate-limits aggressively. The global `@usage` report (admin, by PM) appends an `executor: running/queued/max` field. `queued` counts background submissions only: a foreground command waiting for a slot blocks inside `permit()` before either counter moves, so `16/0/16` does not mean nothing is waiting. `running` pinned at `max` is the signal, not `queued`.

## Status pages

The bot polls up to 5 status pages and can announce incidents from them on
its own as they open and resolve. It can also be **asked** about up to 20
further pages it never polls or announces — the long tail like Cloudflare or
AWS, where polling forever just to answer an occasional question is the
wrong trade. Both Atlassian Statuspage and incident.io pages work —
incident.io is read through its Atlassian-compatible endpoints, no separate
configuration needed. See [Service status](../user/service-status.md) for
the user-facing behaviour.

| Setting | Default | Scope | Description |
|---------|---------|-------|-------------|
| `statusPageUrls` | `Claude=https://status.claude.com GitHub=https://www.githubstatus.com OpenAI=https://status.openai.com` | global | Space-separated status pages to poll and announce, each `Name=url` or a bare `scheme://host` that takes its host as its name. Entries that don't parse are dropped with a warning; duplicate names and same-page duplicates (trailing slash, case, default port) collapse to one; at most 5 are polled. The bot polls each `{url}/api/v2/summary.json` to answer status questions and to announce new incidents. Empty stops polling and announcing, but `check_service_status` stays available if `statusQueryablePages` holds anything. |
| `statusQueryablePages` | empty | global | Same `Name=url` grammar as `statusPageUrls`, for pages the bot only answers about when asked — never polled, never announced, no incident lifecycle. Fetched on-demand and cached for 5 minutes, with a failure backoff on an unreachable page. At most 20. Names must be unique, case-insensitively, across both keys; a name already used by `statusPageUrls` is dropped with a warning. Same rule for the page itself: an entry pointing at a URL already covered by `statusPageUrls` is dropped, regardless of its name, since that page is already reachable — and fresher — as the polled entry. |
| `statusAnnounce` | `False` | channel | Announce incidents from every configured `statusPageUrls` page in this channel as they open and again as they resolve — all-or-nothing per channel, not selectable per source. `statusQueryablePages` pages never announce. Both draw on one budget of six LLM rewrites an hour; over budget the deterministic template still sends. |

Sources are polled in rotation inside a single 45-second-per-pass budget on a
2-minute schedule, with a cursor so one slow or failing source can't starve
the others — with several sources configured, a given one might wait more than
one 2-minute cycle for its turn. A reading older than 10 minutes is reported
to the model as stale. `statusQueryablePages` pages take no part in that
rotation and carry no staleness state of their own — each is fetched fresh,
subject only to its own 5-minute cache, at the moment a question needs it.

!!! warning "Turn off the RSS feed first"

    If the channel already announces the same status page through the RSS
    plugin, remove that first, or every incident is reported twice:

    ```
    @rss announce remove #yourchan <feedname>
    @config channel #yourchan plugins.LLM.statusAnnounce True
    ```

## Monitoring

### Logs

```bash
journalctl --user -u vibebot                    # service logs
journalctl --user -u vibebot -f                 # follow in real time
tail -f ~/.config/vibebot/logs/messages.log     # bot message log
```

At startup, an `INFO`-level line reports secret redaction coverage, naming which environment variable *names* it covers (never their values):

```
secret redaction: 4 handler(s) filtered, 2 variable(s) covered: GEMINI_API_KEY, XAI_API_KEY
```

That line is the only positive confirmation that redaction is actually installed and live — worth checking after any change that touches logging handlers, given that credentials also sit directly in the container's environment (see [Operations → Credentials in the environment](operations.md#credentials-in-the-environment)).

!!! warning "Use `%i`, never `%d`, in log format strings"

    Supybot routes log arguments through `supybot.utils.str.format`, which is not printf. It supports `%s`, `%r`, `%i`, `%f` and `%.3f` — but has **no `%d`**. An unsupported `%d` is left in the output literally and the remaining arguments shift left into whichever slots *are* supported, silently producing a wrong line with no exception and no warning. Because `supybot.log` calls `logging.setLoggerClass` at import, this affects every logger in the process, not just Supybot's own. `test_log_format_specifiers.py` fails the suite on any `%d` in a logging call.

### Per-call log lines

Three structured lines record what a call cost, what the history guards removed, and how long a background call waited for a slot. A line emitted from a turn is prefixed with that turn's eight-character request id (`[3f9a1c04] completion_timing op=…`), so one turn can be pulled out of an interleaved log with a single grep. The daily verse compaction runs off a timer with no request id and its lines carry no prefix.

| Line | Level | Emitted | Fields |
|------|-------|---------|--------|
| `completion_timing` | `WARNING` | Once per model call | `op` (call-site label: `assistant_step_N`, `run_completion_<command>`, `grounded_<kind>` — `xai_responses_<kind>` instead when the resolved model is an xAI one — `image_generation`, `prompt_rewrite`, `reminder_parse`, `pending_retry`, `status_announce`, `compaction:<op>`), `model`, `msgs`, `msg_chars`, `tools`, `prefix_hash`, `gap_s`, `elapsed_ms`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `tool_calls`. A failed call ends `result=error error_type=<class>` in place of the token fields |
| `history_strip` | `WARNING` | Once per turn that dropped poisoned history; silent when nothing was stripped | `model`, `channel`, `route`, `assistant_turns`, `removed`, then one field per guard that fired: `safety_refusal`, `image_failure`, `tool_complaint`, `degraded`, `repeat`, `verse_denial` |
| `prompt_rewrite_fidelity` | `WARNING` | Only when a safety rewording came back padded | `orig_chars`, `new_chars`. Needs both 20% growth and 40 absolute characters, so an ordinary rewording that runs a little long stays silent |
| `llm_executor submit` and `llm_executor done` | `INFO` | Once each per background submission | `submit`: `label`, `running`, `queued`, `max`. `done`: `label`, `elapsed_ms`, `queued_ms` |

`completion_timing` and `history_strip` sit at `WARNING`, so they are present at the default log level; the `llm_executor` pair needs `logLevel` at `INFO`. Three `completion_timing` variants carry a reduced field set: `op=image_generation` logs `prompt_chars` and `elapsed_ms` only, since an image response carries no message or token shape; `op=xai_responses_<kind>` omits `prefix_hash` and `gap_s`; `op=compaction:<op>` logs `elapsed_ms` and the two token counts plus a `cost` field found on no other line.

A refused draw that recovered is three `completion_timing` lines under one request id, in order: `op=image_generation` ending `result=error`, then `op=prompt_rewrite`, then a second `op=image_generation` with no `result=error`. Counting those is how to tell whether the rewrite loop is working. Do not grep for `Image generation blocked, attempting auto-rewrite` or `Rewrite attempt N`: both are `INFO`, so they are absent at the default log level, and their absence reads as a loop that never fired.

On `completion_timing`, read `gap_s` next to `prefix_hash` when `cached_tokens` is 0: `gap_s` is the seconds since the last call on the same model and cache lane (`-1` on the first call) and is the dominant predictor of a cache hit, so a large gap means a cold cache while a changed `prefix_hash` means the cacheable head of the request moved.

### Log level

| Level | Use case |
|-------|----------|
| `WARNING` (default) | Normal operation: warnings and errors only |
| `INFO` | Request flow and timing |
| `DEBUG` | Verbose tracing, including provider response detail |

```
@config plugins.LLM.logLevel DEBUG
```

### Usage statistics

`@usage` shows API usage for yourself and the channel; the global overview by PM is admin-only. One caveat: verse turns are recorded under the `ask` label, so verse traffic is displayed as `ask` rows in the report.

A `@draw` request with subject research enabled writes a `dossier` row alongside the command's own row. That is the research call in front of the planner, and it is booked separately because it usually runs on a different model — see [Subject research](configuration.md#subject-research). A request that named nobody real still writes the row: asking and being told there is nothing to look up costs the same tokens as asking and getting an answer.

One draw can write more than one `draw:image` row. Each row is one call to the image provider, so a prompt the filter refused and a rewording of it that succeeded are two rows: the refusal carries `status='content_blocked'` and the exact prompt the filter rejected, the delivered image carries `status='success'`. Counting rows therefore counts calls, not pictures. The bill is split rather than duplicated, so `SUM(cost)` is still the true total.

Every chat turn writes a row, even when its reply never reached the channel. A turn that spent tokens and then delivered nothing, such as an echo of the prompt or an empty reply, carries `status='error'`. A turn that timed out carries `status='timeout'`, including one the bot queued for a retry; the retry writes its own row when it delivers. `@usage` totals count every status.

The dollar figures come from LiteLLM's built-in price table, not from any provider billing API. A model LiteLLM has no price for records `$0.0000` and logs `completion_cost failed for model=…`; image models are priced from the two-entry `IMAGE_COST_PER_IMAGE` table in `plugins/llm/src/llm/service.py`. Read `@usage` as a relative signal and reconcile against the provider invoice.

### Storage

Reminders, usage statistics, and memories live in one SQLite database: `data/LLM.db` by default, or the path in `databasePath`. Each verse-enabled channel adds its own database under `data/verse/`.

### Common issues

**API key not working.**

- The error names the missing variable: `no API key configured for provider 'xai' (set XAI_API_KEY)`. Grep logs for `no API key configured` rather than the old "API key not configured" wording, which no longer appears anywhere.
- Confirm the key is set in the container, without printing it, by comparing a hash rather than a prefix:

  ```bash
  docker exec vibebot python3 -c "import hashlib,os;k=os.environ.get('XAI_API_KEY','');print(len(k), hashlib.sha256(k.encode()).hexdigest()[:12])"
  ```

  A length of `0` means the variable is unset or empty. Compare the hash against a known-good value computed the same way, rather than eyeballing the key itself.
- Check the key matches the model's provider; a Gemini key does not work with an `anthropic/` model.
- Set `logLevel` to `DEBUG` and retry to see the full provider error.

**A reply arrived with a 🔁 in front of it.**

- The image the user asked for was refused by a content filter, the bot reworded the prompt, and the second attempt landed. The icon says the wording behind the picture moved, so a picture that reads slightly off is explained rather than mysterious.
- It is the same mechanism as the 🌐 grounding icon and rides the same path: the draw callback flags the rework, the tool executor latches it for the turn, and the reply prefixes one icon per signal raised. Both can appear together.
- The matching usage rows are two, not one: the refusal with `status='content_blocked'` and the delivered image with `status='success'`.

**The bot says a request was refused.**

- A refusal is not a fault. The provider turned the content down, and the reply says so and asks the user to reword it or pick a different subject. Nothing needs restarting.
- The provider's own category for the refusal is deliberately kept out of the channel, because these filters produce false positives and the category reads as an accusation of whoever happened to be talking. It goes to the log instead: grep `refused on content grounds`, which carries the full provider error.
- A refusal is not the same event as an outage. Every other failure class now names itself in the reply — a timeout says it timed out, a bad key names the variable — so a generic `Error: Unable to complete` is the one that means look at the logs.
- Refusals can arrive on the chat call itself, not just on a draw, in which case no image was ever attempted. The log line is `assistant_completion failed`, and it is one turn, not a stuck bot: successful turns in the same channel either side of it are normal, because the refused message is never written to history.

**Bot does not remember previous messages.**

- Confirm context is on: `@config channel #yourchannel plugins.LLM.contextEnabled`.
- Check `contextTimeoutMinutes`; a short timeout expires context between messages.
- Users clear their own context with `@forget`.

**No link from `@code` or `@draw`.**

- Built-in server: confirm Limnoria's HTTP server is enabled and its public URL is set.
- External server: confirm the bot can write to `httpRoot` and that `httpUrlBase` matches what your web server serves.
- Check `DEBUG` logs for file-save errors.

**A drawn image link 404s, or shows the wrong picture.**

The image was never generated. A non-reasoning model will sometimes answer a draw request by writing a plausible image URL instead of calling the `generate_image` tool — either inventing a path outright, or reusing a real URL from an earlier turn, which is harder to spot because the link works and only the picture is wrong.

The reliable tell is latency against the logs. A genuine image takes roughly 15–20 seconds and always leaves a matching line:

```bash
docker logs vibebot 2>&1 | grep "op=image_generation"
```

A reply that arrived in two or three seconds with no corresponding `op=image_generation` entry was fabricated.

The bot detects this itself and recovers: any image URL on a host only the bot publishes to (`httpUrlBase` or `imageUploadUrl`) that the current turn did not generate is rejected, the `generate_image` tool is forced, and the turn is retried once. Links to other hosts are left alone, so quoting somebody else's image still works. Look for:

```
assistant_completion: reply invented image URL <url> without calling generate_image;
forcing the tool and retrying (1/1)
```

Seeing that line means the guard worked and the user received a real image. If the model still refuses to call the tool on the forced retry, the reply is replaced with a plain failure message rather than a link that does not work.

**The bot repeats a stock failure line instead of retrying.**

The model imitates its own recent output, so a message like "Image generation failed." left in the conversation history teaches it to answer the *next* request the same way, without calling the tool at all. The bot strips its own past refusals, policy-refusals, collapsed replies, repeated replies, image-failure reports, and tool complaints in any wording from the history before each turn, precisely so they cannot seed the next one.

The last of those is the general case, and it also catches a reply that blames a tool the turn never called at all — the wording drifts ("the tool's broken", "tool refused", "still choking on the request") long after the failure that started it. When one is caught mid-turn the bot re-rolls the reply once and logs:

```
assistant_completion: reply blamed a tool that never ran, nudging and retrying (1/1)
```

If a stock phrase does get stuck, run `@forget` in the affected channel: that clears the caller's own thread *and* the channel's shared recent history. Run from PM, or with a channel argument from somewhere else, it clears only the caller's thread and leaves the bad line in the shared window. A bot reply is itself stored as history, so it persists until cleared or aged out.

To measure how often the guards fire, grep for `history_strip`:

```bash
docker logs vibebot 2>&1 | grep history_strip
```

```
history_strip model=xai/grok-4-1-fast-reasoning channel=#afternet route=chat assistant_turns=14 removed=2 image_failure=1 tool_complaint=1
```

`assistant_turns` is the denominator within one turn: the bot's own turns visible across the personal thread and the shared channel window before anything was stripped, against `removed` as the numerator. The line is silent on clean turns, so for a rate across a whole log use the `op=assistant_step_1` lines as the turn count:

```bash
docker logs vibebot 2>&1 | grep -c "op=assistant_step_1"
```
