# Bridge tools

The assistant carries tools that wrap Limnoria's stock plugin commands.
When a user asks in natural language ("what time is it in Tokyo?",
"have you seen alice?"), the model calls the corresponding Limnoria
command directly instead of relying on a custom implementation.

This page describes the bridge's gating rules and the knobs operators
use to control what the model sees.

## How it works

The bridge injects two tools: `run_limnoria_command` and
`search_bridge_commands`. `enumerate_commands` in
`plugins/llm/src/llm/limnoria_bridge.py` walks `irc.callbacks` each time
the pair is built, so the tool surface tracks whatever Limnoria has
loaded at that moment. Newly loaded plugins become visible without a bot
restart.

`run_limnoria_command`'s description carries the whole exposed command
list, one line per command with its argument syntax and docstring.
`search_bridge_commands` substring-searches that same list
(`limit` 1 to 25, default 10); it exists because Limnoria's
`Misc.apropos` matches command names only, not descriptions.

Both tools ride `@ask`, nick-addressed mentions, PMs, and `@rp`/verse
turns — everything that reaches `_ask_impl`, which builds the pair
unconditionally. `@draw`, reminder fires, and scheduled tasks
never carry them.

The bridge exposes a command when **all** of these hold:

1. `bridgeEnabled` is `True` for the channel (default `False`).
2. The owning plugin is in the operator allowlist
   (`bridgeAllowedPlugins`), or in the curated
   `DEFAULT_ALLOWED_PLUGINS` set when the allowlist is empty.
3. The command is **not** in `DENY_COMMANDS`. Examples: `misc more`
   (interactive pagination) and the `web` fetch-style commands, which
   the bridge denies as server-side request forgery risks.
4. The command is read-only: it is **not** in `MUTATING_COMMANDS`.
5. The caller holds the Limnoria capability for the command, the same
   check Limnoria runs for IRC dispatch.

## Default allowlist

The curated set applies when `bridgeAllowedPlugins` is empty. Every
plugin in the set is either read-only by nature or has its writes listed
in `MUTATING_COMMANDS`, which the bridge never exposes.

`Misc`, `Time`, `Math`, `Utilities`, `Seen`, `Web`, `Later`, `Note`,
`Karma`, `QuoteGrabs`, `RSS`, `DDG`, `Network`

`Network` is in the set for `whois`, `whowas`, `latency` and `uptime` —
the "who is alice?" and "are you lagging?" questions. Its connection
management (`connect`, `disconnect`, `reconnect`, `authenticate`) and
its re-dispatch commands (`command`, `cmdall`) are in `DENY_COMMANDS`,
so the model never sees them whatever the caller's capabilities.

For the canonical, current list see `DEFAULT_ALLOWED_PLUGINS` in
`plugins/llm/src/llm/limnoria_bridge.py`.

## Loading the plugins

`bridgeAllowedPlugins` only matters for plugins Limnoria has loaded. Out
of the curated set, only `Misc` loads on its own: `alwaysLoadImportant`
covers Admin, Channel, Config, Misc, Owner and User, and the bridge
hard-denies every one of those but `Misc`. Everything else, `Utilities`
included, needs an explicit load as bot owner:

```
@load Utilities
@load Time
@load Math
@load Seen
@load Web
@load Later
@load Note
@load Karma
@load QuoteGrabs
@load RSS
@load DDG
@load Network
```

Limnoria persists each `load` to `bot.conf` automatically. The bridge
picks them up on the next assistant turn, with no bot restart needed.

To see the commands a plugin exports, ask the bot:

```
@list Time
@help time
```

Whatever shows up under `@list <Plugin>` is what the model can call,
minus anything held back by `DENY_COMMANDS` or `MUTATING_COMMANDS`.

## Mutating commands

The bridge never exposes commands that change persistent state, send
messages on behalf of another user, or read with side effects. Filtering
happens at enumeration *and* at dispatch, as a defence-in-depth measure.
The full set lives at the top of `limnoria_bridge.py` under
`MUTATING_COMMANDS`. Examples:

- `RSS` mutations, which can announce feed entries to every channel
  subscribed to the feed.
- `Karma` `clear`, `dump`, and `load`, which are destructive against the
  karma database.
- `Factoids` `whatis`, which counts as mutating because the default
  `keepRankInfo` writes a usage counter on every read.

## Refused calls

A refusal is not silent. Dispatch hands the model a JSON envelope, so it
can correct itself or tell the user why. A call that runs returns
`{"status": "ok", "reply": "<captured text>"}`; a refusal returns one of:

- `{"error": "denied: write commands disabled"}` — the command is in
  `MUTATING_COMMANDS`.
- `{"error": "not permitted: Plugin.command"}` — the caller lacks the
  Limnoria capability for that command.
- `{"error": "denied: Plugin.command"}` — a `DENY_COMMANDS` entry, or a
  plugin in `DENY_PLUGINS`.
- `{"error": "unknown plugin: X"}` or
  `{"error": "unknown command: Plugin.command"}` — the model invented a
  name.
- `{"error": "command failed"}` — the plugin raised. The exception text
  goes to the log, not to the model; anything the plugin printed before
  raising comes back under `partial_output`.
- `{"error": "<tokeniser message>"}` — the argument string failed
  Limnoria's tokeniser: an unbalanced quote, bracket, or pipe. The
  message passes through verbatim so the model can correct its own call.

## Live IRC lookups

`irc_lookup` is a third per-request tool that rides with the bridge
pair, bridge or no bridge. It exists because nothing stock answers the first
two questions — Limnoria has no `LIST` command at all, and
`Channel.nicks` only reads channels the bot has joined — and because a
`whois` buried in the bridge's command table is one the model tends to
skip; as a named kind on its own tool it gets picked.

| `kind` | What it sends | What comes back |
|--------|---------------|-----------------|
| `channels` | `LIST` | Up to 25 public channels sorted by user count, each with `name`, `users` and a cleaned `topic`, plus `total`. `target` filters by exact name or glob (`#linux*`). |
| `names` | `NAMES <target>` | `count` and up to 100 nicks with their `@`/`+` prefixes, for any channel the server will show — joined or not. |
| `whois` | `WHOIS <target> <target>` | `user`, `host`, `realname`, `server`, `channels` (with prefixes), `account`, `oper`, `away`, `idle_seconds`, `signon`. The doubled nick asks the user's own server, so idle time comes back. |
| `whowas` | `WHOWAS <target> 3` | Up to 3 `entries` for a nick that has left, each with `user`, `host`, `realname`, and where the server says so `server`, `last_seen` and `away`. |
| `who` | `WHO <target>` (WHOX) | `count` and up to 100 `members`, each with `away`, `oper`, `status` (`@`/`%`/`+`), `bot`, `account` (`null` when not logged in), `host` and `realname`. |
| `topic` | `TOPIC <target>`, or `LIST` when the bot is not in the channel | `topic`, plus `set_by` and `set_at` for a joined channel. For a channel the bot is not in, the setter is unknown and only public channels answer. |
| `network` | `LUSERS`, `VERSION`, `ADMIN` | `users` (user, oper and channel counts), `server` (software and server name) and `admin` (contact lines). |
| `ctcp_version` / `ctcp_ping` | CTCP `VERSION` / `PING` to `<target>` | `client`, or `lag_ms`. Many clients ignore CTCP; then `reply` is `null` with a note. |
| `history` | `CHATHISTORY LATEST <this channel> * <count>` | `lines` from the server's history, newest last, as `HH:MM <nick> text` in UTC. `count` is 1–100, default 50. Only the channel the request came from; refused in a PM. |

`history` is the "what did I miss" kind. It ships ordinary channel
chatter to the provider, so it obeys `privacyOptInOnly`. With that
setting on (the default) the schema leaves out the kind, its `count`
parameter and every mention of history, and the handler refuses a call
that names it anyway without sending anything. It also needs the server to grant `draft/chathistory`, which
the bot requests at connect.

The server enforces its own visibility rules: a `+s` channel is absent
from `LIST` and answers `NAMES` with nothing, exactly as it would for a
user typing `/list` or `/names`. The bot adds no privilege.

`LIST` is the one expensive request here, so the reply is cached for
60 seconds per network and concurrent askers share one in-flight
request. A server that never closes the reply (or answers `RPL_TRYAGAIN`)
surfaces as an error to the model after 15 seconds rather than a hung
turn. The same code backs the `@channels` and `@names` commands. Every
dispatch logs one `irc_lookup: kind=… target=…` line, so "did the model
call it?" is a grep.

## Native tools

Beyond the bridge, the assistant carries its own tools. Users never
call these directly; the model picks them from conversation. Which ones
it is offered depends on the route that handled the message:

| Route | Entry points | Tools offered |
|-------|--------------|---------------|
| chat | `@ask`, nick-addressed mentions, PMs | `check_service_status`, `fetch_url`, `generate_image`, `save_memory`, `search_web`; plus the five pending-task tools (`set_reminder`, `schedule_llm_task`, `list_pending_tasks`, `cancel_pending_task`, `cancel_all_pending_tasks`) where `pendingTasksEnabled` is on. Every request also carries `irc_lookup`, `react`, `delete_last_reply`, `notify_when_online`, `make_meme` and `aurora`, plus the bridge pair where the bridge is on. |
| verse | `@rp`, a live sticky `@rp` session, the one-shot ambient-prose promotion | `fetch_url`, `generate_image`, `save_memory`, `search_web` (4) |
| draw | `@draw` | `generate_image` (1) |
| remind_action | reminder and scheduled-task fires | all 20 |

`assistant.py` defines 20 tool schemas, but defining one is not
advertising it. Ten are hidden from chat and verse because each
duplicates a command the user can already type:

- `list_memories`, `update_memory`, `delete_memory`, `clear_memories`,
  `cleanup_memories` → `@memories`, `@memories edit <id> <text>`,
  `@memories delete <id>`, `@memories clear`, `@memories cleanup`
- `set_instruction`, `clear_instruction` → `@instruct <instruction>`,
  `@instruct clear`
- `get_usage`, `get_channel_usage` → `@usage [<nick or #channel>]`
- `forget_context` → `@forget [<channel>]`

`list_pending_tasks`, `cancel_pending_task` and `cancel_all_pending_tasks`
used to be on this list. They are back in chat: with `set_reminder`
visible and the cancel tools hidden, chat answered "cancel my reminder"
with a made-up confirmation and the reminder fired anyway.

Keeping the advertised surface small is a correctness measure, not
tidiness. `xai/grok-4-1-fast-reasoning` starts returning empty
completions once the tool count climbs past roughly 25 — four
empty-response incidents on 2026-05-10, more than any day in the
preceding 30 — and a non-reasoning model asked to pick one tool out of
twenty will sometimes pick none and answer from its own invention, which
is how a draw request came back with a fabricated image URL on
2026-08-01. Hiding
these costs no capability: the handlers still exist, the commands above
still work, and memories still arrive on their own through the
background extraction pass. `save_memory` stays on the chat surface
because it is the one write extraction cannot replace — an explicit
"remember this" should stick at once rather than wait for the
candidate-reinforcement threshold.

Reminder fires keep all 20. A scheduled task runs as its creator with no
user present to type a command, so it may legitimately need to tidy up
after itself.

The five pending-task tools appear on the chat surface only when
`pendingTasksEnabled` is on for the channel, and it defaults to off.
Their schemas and prompt rules cost roughly 1,100 prompt tokens on every
completion, so channels that never schedule anything should not pay for
them. The `@remind` command and already-scheduled fires ignore the gate:
users list and cancel both kinds with `@remind list`, `@remind del` and
`@remind clear` in any channel.

`check_service_status` answers from the poller's cached snapshot of the
`statusPageUrls` pages: one entry per page when the model omits `service`,
or a single named lookup. The pages are named in the tool's description
text, in a "Monitored services" sentence, and the `service` argument's enum
lists the same names. The tool leaves the request when `statusPageUrls` is
empty, rather than shipping a tool that could only answer "not configured".

Verse is a strict subset of chat: everything chat hides, plus
`set_reminder`, `schedule_llm_task` and `check_service_status`, none of
which has an in-character use. The subset relation is asserted by
`test_verse_profile_is_strict_subset_of_chat`.

On any turn in a verse-enabled channel the model also receives the verse
tool set, advertised to every speaker whether or not they have opted in,
so the channel's cacheable prompt prefix stays byte-identical across
users; calls from a speaker with no avatar land on denial handlers. The
set is `verse_act`, `verse_move`, `verse_look`, `verse_recall`,
`verse_record`, `verse_edit` (refused unless the caller holds
`llm.verse.edit`), and `verse_storybook` (present only when
`verseStorybookEnabled` is on). See
[verse operations](../operator/verse.md).

## Scheduled LLM tasks

`schedule_llm_task` lets the assistant create recurring agentic work
("every weekday at 09:00, summarise the news"). A creator can hold at
most 5 active schedules in a channel. Each fire still counts
against the user's normal `@ask` rate-limit bucket; the cap limits the
number of pending schedules, not their cumulative cost. Owners can
list and cancel any user's tasks with `@remind admin`.

## Operator config

| Setting | Default | Description |
|---------|---------|-------------|
| `bridgeEnabled` | `False` | Primary switch, per channel. `False` removes every bridge tool. |
| `bridgeAllowedPlugins` | empty (curated default applies) | Space-separated plugin names. A non-empty value replaces the curated set entirely. Names use camel case to match `cb.name()`. |
| `pendingTasksEnabled` | `False` | Per channel. Advertises the five pending-task tools on the chat surface, so the model can create, list and cancel reminders and schedules from plain language. `@remind` works either way. |
| `bridgeDebugInChannel` | `False` | Per channel. Appends a one-line footer listing every bridge call made during the turn, as `Plugin.command (N chars) [ok]`, or `[err:<reason>]` when the call was refused. Argument text is never echoed, only its length: bridge arguments are model-generated and can carry secrets. Full arguments are logged only at `DEBUG`, below the plugin's fixed `WARNING` level. |

To add a plugin outside the curated default, list every plugin you want
exposed, because the registry value replaces the default rather than
extending it. For example, to add `Factoids` and `Dict`:

```
@config plugins.LLM.bridgeAllowedPlugins Misc Time Math Utilities Seen Web Later Note Karma QuoteGrabs RSS DDG Factoids Dict
```

Once a plugin is in `bridgeAllowedPlugins` and loaded into Limnoria,
every read-only command it exports becomes available to the model on
the next turn.
