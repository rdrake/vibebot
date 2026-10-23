# Tuning and monitoring

Which model to put where, how much load the bot puts on providers, and what the logs and usage table say about it. Every setting named here is described in full under [Configuration](configuration.md); the bridge has its own page, [Bridge tools](../reference/bridge-tools.md).

## Choosing models

The model surface splits by workload so cost and quality can differ per task. The three model settings below are channel-overridable; the other roles run on fixed models listed under [Configuration → Model selection](configuration.md#model-selection): images on `xai/grok-imagine-image`, search and subject research on `xai/grok-4.3`, the `@meme` picker and verse compaction on `anthropic/claude-haiku-5-5`, and refused chat turns are re-run once on `xai/grok-4-1-fast-non-reasoning`.

| Setting | Used for |
|---------|----------|
| `assistantModel` | Chat, the planner loop, bridge tool selection, reminder parsing, memory cleanup, image-prompt rewrites, scheduled tasks |
| `memoryModel` | Memory extraction after a stored reply to an authenticated user. Falls back to `assistantModel` when empty |
| `verseModel` | Verse-mode narration. Falls back to `assistantModel` when empty |

Guidance from operating this surface:

- `assistantModel` needs vision support if users paste image URLs into chat.
- Reasoning models make terse, flat narrators. If the assistant model is a reasoning model, point `verseModel` at a non-reasoning model for verse prose.
- Memory extraction runs after most replies and its output is never shown. Put `memoryModel` on a cheap model.

Keys are not part of this per-command surface: one environment variable per *provider* covers every model on that provider, regardless of which setting names the model. Point a channel at a different model and, if that model's provider is already configured, no key change is needed. See [Configuration → API keys](configuration.md#api-keys) for the variable names.

```
@config channel #yourchan plugins.LLM.assistantModel anthropic/claude-haiku-5-5
```

## Concurrency

Every outbound LLM call, foreground or background, takes a slot from one bounded executor of 16 slots, so a single cap limits how much load reaches a provider no matter which thread started the call. Keeping a slow call off the IRC driver is a separate mechanism: nick-addressed messages are dispatched on their own thread so the driver can flush typing notifications while the call is in flight.

The global `@usage` report (admin, by PM) appends an `executor: running/queued/max` field. `queued` counts background submissions only: a foreground command waiting for a slot blocks inside `permit()` before either counter moves, so `16/0/16` does not mean nothing is waiting. `running` pinned at `max` is the signal, not `queued`.

## Monitoring

Where the logs are and how to read them is under [Operations → Log locations](operations.md#log-locations); the startup `secret redaction:` line is explained under [Installation → First start](installation.md#first-start). For a specific symptom, start at [Troubleshooting](troubleshooting.md).

### Per-call log lines

Three structured lines record what a call cost, what the history guards removed, and how long a background call waited for a slot. A line emitted from a turn is prefixed with that turn's eight-character request id (`[3f9a1c04] completion_timing op=…`), so one turn can be pulled out of an interleaved log with a single grep. The daily verse compaction runs off a timer with no request id and its lines carry no prefix.

| Line | Level | Emitted | Fields |
|------|-------|---------|--------|
| `completion_timing` | `WARNING` | Once per model call | `op` (call-site label: `assistant_step_N`, `xai_responses_<kind>` for search, URL fetch and subject research, `image_generation`, `image_edit`/`image_generate` for meme pictures, `prompt_rewrite`, `reminder_parse`, `pending_retry`, `status_announce`, `ask_helper`, `extract_memories`, `cleanup_memories`, `meme_pick`, `compaction:<op>`), `model`, `msgs`, `msg_chars`, `tools`, `prefix_hash`, `gap_s`, `elapsed_ms`, `prompt_tokens`, `cached_tokens`, `completion_tokens`, `tool_calls`. A failed call ends `result=error error_type=<class>` in place of the token fields |
| `history_strip` | `WARNING` | Once per turn that dropped poisoned history; silent when nothing was stripped | `model`, `channel`, `route`, `assistant_turns`, `removed`, then one field per guard that fired: `safety_refusal`, `image_failure`, `job_marker`, `tool_complaint`, `degraded`, `repeat`, `verse_denial` |
| `prompt_rewrite_fidelity` | `WARNING` | Only when a safety rewording came back padded | `orig_chars`, `new_chars`. Needs both 20% growth and 40 absolute characters, so an ordinary rewording that runs a little long stays silent |
| `llm_executor submit` and `llm_executor done` | `INFO` | Once each per background submission | `submit`: `label`, `running`, `queued`, `max`. `done`: `label`, `elapsed_ms`, `queued_ms` |

`completion_timing` and `history_strip` sit at `WARNING`, so they are present in the log; the `llm_executor` pair is `INFO`, below the plugin's fixed `WARNING` level, so it is not emitted. Some `completion_timing` variants carry a reduced field set: `op=image_generation` logs `prompt_chars` and `elapsed_ms` only, since an image response carries no message or token shape; `op=image_edit` and `op=image_generate` (meme pictures) log `model` and `elapsed_ms` only; `op=xai_responses_<kind>` carries `prefix_hash=-` and `gap_s=-1` as placeholders, since no prefix fingerprint or cache gap is measured; `op=compaction:<op>` logs `elapsed_ms` and the two token counts plus a `cost` field found on no other line.

A refused draw that recovered is three `completion_timing` lines under one request id, in order: `op=image_generation` (ending `result=error` for a filter rejection, or with no error field when the provider returned no picture), then `op=prompt_rewrite`, then a second `op=image_generation` with no `result=error`. Counting those is how to tell whether the rewrite loop is working. Do not grep for `Image generation blocked, attempting auto-rewrite` or `Rewrite attempt N`: both are `INFO`, so they never reach the log, and their absence reads as a loop that never fired.

On `completion_timing`, read `gap_s` next to `prefix_hash` when `cached_tokens` is 0: `gap_s` is the seconds since the last call on the same model and cache lane (`-1` on the first call) and is the dominant predictor of a cache hit, so a large gap means a cold cache while a changed `prefix_hash` means the cacheable head of the request moved.

### Log level

The plugin logs at `WARNING`: warnings and errors only. Its `INFO` request-flow lines and `DEBUG` tracing are not emitted.

### Usage statistics

`@usage` shows API usage for yourself and the channel; the global overview by PM is admin-only. One caveat: verse turns are recorded under the `ask` label, so verse traffic is displayed as `ask` rows in the report.

A `@draw` request writes a `dossier` row alongside the command's own row. That is the research call in front of the planner, and it is booked separately because it usually runs on a different model — see [Subject research](configuration.md#subject-research). A request that named nobody real still writes the row: asking and being told there is nothing to look up costs the same tokens as asking and getting an answer.

One draw can write more than one `draw:image` row. Each row is one call to the image provider, so a prompt the filter refused and a rewording of it that succeeded are two rows: the refusal carries `status='content_blocked'` and the exact prompt the filter rejected, the delivered image carries `status='success'`. Counting rows therefore counts calls, not pictures. The bill is split rather than duplicated, so `SUM(cost)` is still the true total.

Every chat turn writes a row, even when its reply never reached the channel. A turn that spent tokens and then delivered nothing, such as an echo of the prompt or an empty reply, carries `status='error'`. A turn that timed out carries `status='timeout'`, including one the bot queued for a retry; the retry writes its own row when it delivers. `@usage` totals count every status.

To read costs straight from the table, open `~/.config/vibebot/data/LLM.db` with `sqlite3` on the host. `timestamp` is Unix seconds and `cost` is in US dollars; `unixepoch()` needs SQLite 3.38 or newer. Cost per model per day over the last week:

```sql
SELECT date(timestamp, 'unixepoch') AS day, model,
       COUNT(*) AS calls, ROUND(SUM(cost), 4) AS usd
FROM usage
WHERE timestamp >= unixepoch('now', '-7 days')
GROUP BY day, model
ORDER BY day DESC, usd DESC;
```

And per status, which shows what failed calls cost. `status` is `success` or a failure class such as `error`, `timeout` or `content_blocked`:

```sql
SELECT status, command, COUNT(*) AS calls, ROUND(SUM(cost), 4) AS usd
FROM usage
WHERE timestamp >= unixepoch('now', '-7 days')
GROUP BY status, command
ORDER BY usd DESC;
```

xAI calls book the cost xAI reports in the response. Anthropic calls, and xAI calls without a reported cost, are priced from the `PRICES` and `IMAGE_PRICES` tables in `plugins/llm/src/llm/provider.py`. A model with no entry there books `$0.00` and logs `has no price in provider.py` once. Read `@usage` as a relative signal and reconcile against the provider invoice.

### Storage

Reminders, usage statistics, and memories live in one SQLite database, `data/LLM.db` in Limnoria's data directory. Each verse-enabled channel adds its own database under `data/verse/`.
