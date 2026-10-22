# LLM plugin

The LLM plugin is VibeBot's main AI surface. It provides
natural-language chat, image generation, illustrated stories,
persistent memory, and reminders. It also runs scheduled agentic
tasks, hosts a persistent roleplay world (the verse), and carries a
curated bridge that lets the model call other Limnoria plugins as
tools.

## Command families

| Family | Commands | Notes |
|--------|----------|-------|
| Core AI | `@ask`, `@draw`, `@meme`, `@story` | Chat with vision, images, captioned memes, illustrated pages. |
| Memory | `@forget`, `@memories`, `@instruct` | Volatile context, durable facts, persistent instructions. |
| Reminders | `@remind`, `@tz` | Natural-language reminders with recurring support, read in your time zone. |
| Accounting | `@usage` | Per-account and per-channel API usage. |
| Verse (user) | `@verseopt`, `@rp`, `@verse`, `@look`, `@who`, `@avatar` | Opt in, roleplay, inspect the scene, set a persona. |
| Verse (editor) | `@canon`, `@versedit` | Curate durable canon. Requires `llm.verse.edit`. |
| Verse (GM) | `@versedump`, `@versepurge`, `@versecompact` | Inspect, wipe, and compact a channel's verse. Requires `llm.verse.gm`. |

See the [command reference](../reference/commands.md) for full syntax
and capability requirements, and the
[configuration guide](../operator/configuration.md) for every registry
key.

`@ask` and `@draw` also work through plain language: mention
the bot by name or send it a PM. `@story` does not — an illustrated page
comes from the command, or from an explicit illustrate cue in a verse
channel with `verseStorybookEnabled` on. Memory, Reminders and Accounting are
command-only, with two exceptions — the model can save a memory, and
where `pendingTasksEnabled` is on it can set, list and cancel reminders
and scheduled tasks. `@remind list`, `del` and `clear` cover both kinds
anywhere.
[Bridge tools](../reference/bridge-tools.md) lists the tool surface each
route sees.

## Capabilities

Limnoria capabilities gate the AI and verse commands: `llm.ask` (`@ask`,
`@forget`), `llm.draw` (also covers `@story`), `llm.verse`
(`@rp`, `@verseopt`, `@verse`, `@look`, `@who`), `llm.verse.edit`
(`@canon`, `@versedit`), and `llm.verse.gm` (`@versedump`,
`@versepurge`, `@versecompact`).

`@remind` also needs `llm.ask`, since it manages persisted reminders and
scheduled tasks. `@memories`, `@instruct`, `@tz`, `@usage`, and `@avatar` carry
no capability of their own. The paths that reach past your own data check
inline instead: `owner` for `@memories <nick>` and `@remind admin`,
`admin` for a bare `@usage` by PM. Owner and admin accounts bypass rate limits.

## Internal layout

```
plugins/llm/src/llm/
├── plugin.py          # IRC command surface + Limnoria glue
├── service.py         # Completions, tool loop, sanitisation, output shaping
├── provider.py        # Stdlib HTTP to Anthropic and xAI
├── assistant.py       # Tool schemas and per-route visibility (function calling)
├── profile.py         # Route profiles: model, prompt, overlay, and caps per mode
├── prompts.py         # Shared system-prompt fragments
├── executor.py        # LLMExecutor: global concurrency cap
├── persistence.py     # SQLite store (memories, reminders, schedules, usage)
├── agenda.py          # When reminders and scheduled tasks fire: cadence, caps, restart
├── limnoria_bridge.py # Allowlisted "Limnoria as tools" surface
├── statuspage.py      # Statuspage v2 polling, parsing, and rendering
├── context.py         # Conversation history with TTL
├── config.py          # Limnoria registry options
├── apikeys.py         # Provider-scoped API keys and secret redaction
├── tracing.py         # Per-request IDs threaded through the logs
└── verse/             # Verse store, avatar engine, compaction, taste tools
```

## Concurrency model

Every blocking LLM call goes through `LLMExecutor` (in `executor.py`),
which combines a `BoundedSemaphore` and a `ThreadPoolExecutor`. This
design:

- caps total in-flight LLM I/O at 16 calls,
- keeps the IRC main thread responsive, because no provider call can
  block it directly, and
- shares one shutdown gate, so plugin reloads drain in-flight work
  cleanly.

The global `@usage` report — a bare `@usage` by PM, admin only —
appends an `executor: running/queued/max` field for tuning.

## Storage

Persistent state lives in a single SQLite database managed by
`persistence.py`. All writes go through the `_write_txn` context
manager, so a failure mid-update rolls back rather than leaving partial
state. Tables cover memories, reminders, scheduled tasks, conversation
snapshots, and per-account usage. Each verse-enabled channel gets its
own SQLite store under `data/verse/`.

## See also

- [Memory and instructions](../user/memory.md)
- [Reminders](../user/reminders.md)
- [Scheduled tasks](../user/scheduled-tasks.md)
- [Bridge tools](../reference/bridge-tools.md)
- [Verse operations](../operator/verse.md)
- [Tuning and monitoring](../operator/tuning-monitoring.md)
