# Configuration

VibeBot uses Limnoria's registry for all configuration. See the [Limnoria configuration documentation](https://docs.limnoria.net/use/configuration.html) for how the registry works in general.

Set values from IRC with:

```
@config plugins.LLM.<settingName> <value>
```

View a current value by omitting the value. Channel-overridable settings also accept a per-channel form:

```
@config channel #channel plugins.LLM.<settingName> <value>
```

**Scope** in the tables below means: *channel* settings accept per-channel overrides, which a channel op can set; *global* settings apply everywhere and need `owner`. New to `@config`? The [Limnoria crash course](limnoria.md) covers reading, setting, resetting and who may change what.

!!! warning "A new default in a release does not change a running bot"

    Limnoria writes the whole registry to `bot.conf` when the bot shuts down, so every setting the bot has ever loaded exists there as an explicit line. That line wins over the default shipped in a later release.

    So when an upgrade changes a **Default** in the tables below, the new value applies to fresh installs only. An existing deployment keeps whatever was persisted, silently and with no error — the feature simply behaves as it did before.

    To adopt a new default, set it explicitly with `@config` and then `@flush`, or stop the bot, edit `bot.conf`, and start it again. Editing `bot.conf` while the bot runs does not work: the next shutdown flush overwrites it.

    Verify by reading back the value with `@config plugins.LLM.<settingName>` — a green deploy and a quiet log do not prove the new value took effect.

!!! tip "Every `@config` change needs an `@flush`"

    `@config` changes the running registry only. Restarts on this deployment have come without a shutdown flush, and unflushed changes were lost. Run `@flush` (owner) after any change you want to keep. See [Making a change stick](limnoria.md#making-a-change-stick).

## API keys

API keys are **not** part of the Limnoria registry. They come from environment variables, one per provider, read at the point each provider call is made and selected by the provider of the model being called — not by which command or channel triggered the call. The old `assistantApiKey` / `codeApiKey` / `imageApiKey` / `searchApiKey` registry settings were removed, not deprecated: `@config` on them now errors.

| Variable | Provider |
|----------|----------|
| `ANTHROPIC_API_KEY` | Anthropic (`anthropic/` models) |
| `XAI_API_KEY` | xAI (`xai/` models) |

These are the only two providers the bot can call. With the defaults, chat and the `@meme` template picker need `ANTHROPIC_API_KEY` (a named template with captions renders locally and calls no model), and search, subject research, `@draw`, meme picture edits and the refusal fallback need `XAI_API_KEY`, because they run on fixed xAI models (see [Model selection](#model-selection)).

Copy `.env.example` to `~/.config/vibebot/env` and set the variables there; the systemd unit passes that file to the container with `--env-file` (see [Installation](installation.md)). A model whose provider has no key configured fails with an error naming the missing variable, for example `no API key configured for provider 'xai' (set XAI_API_KEY)`.

## Model selection

Models are `provider/name` strings, and the provider must be `anthropic/` or `xai/`; `@config` rejects any other prefix. A model with no price in `provider.py` is accepted with a one-time warning, because its spend books as $0.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `assistantModel` | channel | `anthropic/claude-haiku-5-5` | All assistant text and tool work: chat, planner, memory, reminders, scheduled tasks. Needs vision support if users paste image URLs |
| `advisorModel` | channel | `anthropic/claude-opus-5-5` | A stronger model chat may consult at most once a turn, through Anthropic's advisor tool. The assistant model decides when; most chat never calls it. A consult runs inside the same request, so a step that may consult gets a 120-second timeout instead of `timeout`, and an Opus consult costs about $0.05, booked at Opus rates. Chat turns only, never verse or `@draw`. Must be an `anthropic/` model at least as capable as `assistantModel`; any other pairing is skipped with a log warning, and an `xai/` assistant model never consults. Empty turns it off |
| `memoryModel` | channel | empty | Model for the memory extraction that runs after a stored `@ask` or chat reply to an authenticated user. Falls back to `assistantModel`. A cheap model is enough |
| `verseModel` | channel | empty | Model for verse-mode replies. Falls back to `assistantModel`. Set this when the assistant model is a terse reasoning model that writes poor prose |

The other roles run on fixed models:

| Role | Model | Notes |
|------|-------|-------|
| Refusal fallback | `xai/grok-4-1-fast-non-reasoning` | Re-runs a chat turn once when its model refuses: an Anthropic `stop_reason` refusal (at once, with no retry), a policy refusal that survives its one retry, or a provider refusal error. The fallback gets the original request without the refusal. Skipped when the turn already runs on this model, when `XAI_API_KEY` is unset, or when a tool that changes something (a reminder, an image, a memory) already worked this turn, since the re-run would repeat it. Its cost books at `xai/grok-4.3` rates, the model behind the alias |
| Images | `xai/grok-imagine-image` | `@draw`, `@story` and the `generate_image` tool |
| Search | `xai/grok-4.3` | Web search, URL fetch and [subject research](#subject-research). GitHub links skip it: they are read through GitHub's API. A request can run at most three searches |
| Meme picker | `anthropic/claude-haiku-5-5` | Picks a template and writes captions when `@meme` (or `make_meme`'s `brief`) names no template the catalog knows. It opens on a short list (the 40 most popular templates plus up to 20 sharing a word with the request, about 5K prompt tokens) and can call a `search_templates` tool up to twice to reach the rest of the catalog. No search: one call, about 5K tokens, 2 s. Each search is another round trip: about 12K tokens and 4 s with one, 19K and 6 s with two. Under $0.003 either way. [Subject research](#subject-research) runs first, adding one search-model call and a few seconds to every picker request |
| Meme picture edits | `xai/grok-imagine-image` | The picker's `draw` (and `make_meme`'s). Templates with picture slots get one generated picture per slot (`/v1/images/generations`), masked into the blank; other templates send the captioned meme to `/v1/images/edits`. About $0.02 per picture, refusals billed too |
| Verse compaction | `anthropic/claude-haiku-5-5` | The daily verse compaction job |

Channel overrides let different channels run different models:

```
@config channel #dev plugins.LLM.assistantModel xai/grok-4.3
@config channel #casual plugins.LLM.assistantModel anthropic/claude-haiku-5-5
```

### Subject research

Image generators know nothing by name. A bare "Churchill" comes out as
nothing at all, or as the word printed across the frame — which is why `@draw`
runs a planner turn that rewrites the request before the generator ever sees
it. The planner is told to
keep the name the user typed and put a description of the subject beside it,
but for a real person or a real event it has nothing to build that description
from: the verse canon block covers characters the channel invented, not people
the world already contains.

Subject research closes that gap with one grounded
completion in front of the planner. It identifies the real subjects the request
names, looks up how each actually looks, and layers the answer onto the
planner's prompt. A request that names nobody real — "a pine forest at
sunrise" — comes back with nothing and adds nothing to the prompt.

The cost is one extra call and a few seconds, so `@draw` is visibly
slower than the image call alone. The `@meme` template picker runs the same
research before it picks, so a meme of a real person carries what they look like. The typing indicator covers the wait. The usage table
records research costs under the command name `dossier`, separately from the
planner's own row, because the two run on different models.

Only the dash-prefixed fact lines the research prompt asks for are kept, and
only those that end in a `(source: …)` tag. A model that declines the
question, or wraps its answer in commentary, contributes nothing rather than
contributing its refusal to the next model's prompt. The source tag guards the
other failure: asked about a name that exists nowhere, a grounded model guesses
an appearance from the words in it rather than answering `NONE`, and a guess
has nothing to cite. The tag is stripped before the line reaches the planner.

The research runs on the search model (`xai/grok-4.3`), which in most
deployments means the researcher is a different provider from the planner.

## System prompts

The system prompt defines the bot's personality and constraints. It is channel-overridable.

| Setting | Description |
|---------|-------------|
| `assistantSystemPrompt` | Personality and constraints for all assistant work. Verse mode layers scene context on top of this prompt rather than replacing it |

`assistantSystemPrompt` is a personality overlay: the bot appends it to a fixed framework prompt and never lets it replace that prompt. The structural rules (the three-line cap, plain text, no markdown) live in the framework and survive any replacement, so this setting changes the bot's voice, not its output format. Read the shipped text before you replace it: `@config default plugins.LLM.assistantSystemPrompt`.

Example: give a channel a specialised personality:

```
@config channel #python plugins.LLM.assistantSystemPrompt You are a Python expert. Answer with Python examples. Be concise.
```

## Verse

The verse is the per-channel world model. See [The verse](verse.md) for concepts and operation; this table is the key reference.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `verseEnabled` | channel | `False` | Main switch. When off, verse commands are unavailable and no events are recorded |
| `verseTriggerRegex` | channel | `\bverse\b` | Case-insensitive pattern that marks a message as a canon reference, alongside entity mentions. A match grounds the reply in canon and, for an avatar holder, turns it into a prose tale. It does not enter roleplay mode; that needs `@rp`. Empty disables the keyword signal |
| `verseChatRecordEnabled` | channel | `False` | Let an opted-in avatar's ordinary chat turn call `verse_record`. Off keeps canon growing only during roleplay |
| `verseRosterMaxChars` | channel | `4000` | Character cap on the canon-roster block in the verse system prompt |
| `verseStyleExemplars` | channel | `[]` | Curated style exemplars injected into the verse prompt. Populate offline with the taste miner |

Fixed behaviour: sticky `@rp on` roleplay lapses after 15 minutes without an in-character turn, and each such turn restarts the clock. Compaction runs daily at 03:00 local time, compacts events older than 30 days, and skips verses with fewer than 20 events in total. Auto-created non-player characters (NPCs) retire after 14 days without a mention; pinned entities are exempt. One `verse_record` call names at most 8 actors. IRCv3 emoji reactions to verse lines are captured as an offline approval signal.

### Storybook

These caps cover both illustrated-page paths: the `verse_storybook` tool on verse turns, and the standalone `@story` command. `@story` needs no verse — it is gated like `@draw`, on `llm.draw` plus an authenticated account — so these limits still bind in a channel where `verseEnabled` is `False`. The `verse` prefix is historical; renaming the keys would orphan existing `bot.conf` values.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `verseStorybookEnabled` | channel | `False` | Expose the `verse_storybook` tool on verse turns. It does **not** gate `@story` |
| `verseStorybookMaxImages` | channel | `5` | Cap on illustrations per storybook |
| `verseStorybookCooldownSeconds` | channel | `300` | Per-account cooldown between storybooks |

A storybook body is capped at 6,000 characters, each illustration gets 45 seconds, and a verse turn makes at most one storybook.

Turning `verseStorybookEnabled` off does not stop image costs on its own, because `@story` still runs. What bounds them is `verseStorybookMaxImages` per page and the per-account cooldown between pages; there is no daily image cap.

## Memory

Memory extraction runs after a stored `@ask` or chat reply to an authenticated user, and stops once that user holds 50 durable memories; `memoryModel` (under [Model selection](#model-selection)) picks its model. A candidate fact needs 2 mentions before it is promoted, an unreinforced candidate is pruned after 14 days, durable memories are capped at 50 per user, and cleanup runs every 3 extraction passes that promote something.

See [Memory promotion](memory-promotion.md) for how the two-stage pipeline works.

## Conversation context

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `contextTimeoutMinutes` | channel | `5` | Clear context after this much inactivity |
| `privacyOptInOnly` | channel | `True` | Send only lines addressed to the bot to the provider. Set `False` to track every channel message as context and let `irc_lookup` read the channel's server history (`kind=history`, up to 100 lines) for "what did I miss". Those lines go to Anthropic or xAI, whichever serves the channel's models |

Per-user history keeps the last 20 messages; the shared channel context keeps 10. The shared context is what lets the bot connect a question from one user with a follow-up from another.

!!! warning "Privacy"

    With `privacyOptInOnly` off, every message in the channel goes to a third-party LLM provider as context, and the bot can pull up to 100 lines of server history on request. Turn it off only where users know and consent.

## Rate limiting

Rate limits follow the pattern `{command}{tier}RateLimitCount` and `{command}{tier}RateLimitWindow` for the commands `ask`, `draw`, `meme`, and `story`. All rate-limit settings are global. `ask` has unregistered, registered and Trusted tiers; `draw`, `meme` and `story` require an authenticated account, so only the registered and Trusted tiers apply to them. See [Rate limiting and security](rate-limiting-security.md) for tiers and defaults.

## Bot-to-bot loops

Two bots that answer each other never get bored, so replies to another bot are capped: after 3 consecutive replies to one bot the bot goes quiet toward it, and the count resets after 300 seconds of quiet.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `botLoopHostSuffixes` | global | `Bot.AfterNET.Org` | Host suffixes the network reserves for bots. Matched case-insensitively against the sender's host. A sender matched here, or one with the +B mode, cannot address the bot by replying to its lines. A full host works too, for a bot on a user host |

A nick counts as a bot only when the network says so. AfterNET advertises `BOT=B`, so a `+B` sender's lines carry the IRCv3 `bot` message tag, which the bot reads on the first line; once a WHO answers, the flag in the WHOX status field is the fallback — six nicks carry it today, long-running service bots among them. Anyone unflagged is treated as a person and is never capped, and a human *addressing the bot* clears the counts in that channel, so a conversation someone has joined is never cut short. Only addressed lines reset it: what people say during a bot loop is commentary on the noise ("wtf", "jesus"), and treating that as a reset would license three more replies after every such line.

The `+B` mode is opt-in, and the bot most likely to loop with yours may not set it — `grok` answers on `Hxz` from `grok.Bot.AfterNET.Org`. So the sender's host counts too: AfterNET only issues `*.Bot.AfterNET.Org` to registered bots, and across a full day of channel logs the only nicks on one were `grok`, `grook5`, and `vibebot` itself. The host is checked first because it is free and decides on the very first line, with no WHO round trip.

Because the bot skips Limnoria's automatic WHO on join when `account-tag` and `extended-join` are active, the flag is fetched on demand: the first time an unknown nick addresses the bot, a WHO goes out for that nick. The answer is kept until restart, and a WHO that gets no answer is re-sent at most once an hour. That first line is always answered — the guard fails open, so the worst case is one extra reply, never a silenced user. A bot that never sets `+B` is invisible to this and will not be capped.

## Web and HTTP output

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `httpRoot` | global | empty | Filesystem path for output files. Empty keeps them under Limnoria's `data/web/llm/` and serves them from its built-in HTTP server. Setting it **unhooks that server** — an external server such as nginx must serve the path, and `httpUrlBase` must be set to match, or every generated link 404s |
| `httpUrlBase` | global | empty | Public base URL for output files. Empty uses Limnoria's `publicUrl` plus `/llm`; if `publicUrl` is also unset the bot falls back to `http://localhost:<http port>/llm`, which nobody off the host can open |
| `imageUploadUrl` | global | empty | External image host for generated images. Empty stores them locally |
| `fileCleanupAge` | global | `720` | Delete output files older than this many hours (30 days) |

Replies that wrap to at most 3 IRC lines go inline as one IRCv3 `draft/multiline` batch (this needs `supybot.protocols.irc.experimentalExtensions`); longer replies are saved as HTML and announced with a one-line teaser of up to 220 characters and a link. The output directory keeps at most 1,000 files.

### Offloading images to an external host

Set `imageUploadUrl` to an uploader that takes a multipart `POST` on the `images[]` field and answers with JSON:

```json
{"results": [{"success": true, "filePath": "/img/img_abc123.png"}]}
```

Generated images then live on that host instead of under `httpRoot`, and storybook pages embed the absolute URL. Pages themselves (long answers, stories) still come from your own HTTP server, so this offloads bandwidth rather than replacing it.

Every failure falls back to local storage: endpoint unreachable, upload rejected, image over 10 MB, or a reply naming a URL that is not an image on the configured host. An outage costs a slower draw, nothing more.

[paste.boxlabs.uk/img/](https://paste.boxlabs.uk/img/) is an externally operated uploader, the host this contract follows, and it needs no key. Ask the host's owner before pointing the bot at somebody else's uploader.

## Limnoria tool bridge

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `bridgeEnabled` | channel | `False` | Expose loaded Limnoria plugin commands to the LLM as a tool |
| `bridgeAllowedPlugins` | channel | empty | Plugin allowlist. Empty uses the curated read-safe default set |
| `bridgeDebugInChannel` | channel | `False` | Append a bridge-call debug footer to replies |

The bridge only reaches plugins Limnoria has loaded, and only `Misc` loads on its own, so `@load` the rest first: see [Bridge tools → Loading the plugins](../reference/bridge-tools.md#loading-the-plugins). A non-empty `bridgeAllowedPlugins` replaces the curated set rather than extending it. The bridge never exposes the state-changing commands in `MUTATING_COMMANDS`; allowlisting a plugin outside the curated set exposes its other commands, subject only to Limnoria capabilities. [Bridge tools](../reference/bridge-tools.md) has the security model and every refusal.

The `irc_lookup` tool (live `LIST`, `NAMES`, `WHOIS`, `WHOWAS`, `WHO`, topic, network info, CTCP) is on the chat tool surface whether the bridge is on or not; its `history` kind also needs `privacyOptInOnly` off.

## Status pages

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `statusPageUrls` | global | `Claude=https://status.claude.com GitHub=https://www.githubstatus.com OpenAI=https://status.openai.com` | Space-separated status pages to poll and announce — both Atlassian Statuspage and incident.io pages work — each written as `Name=url` or as a bare url, which takes its host as its name. Names are 1-32 chars of `[A-Za-z0-9._-]`, case-insensitively unique, and are what the model uses to ask about one service. Unusable entries are dropped with a warning and the rest survive; at most 5 are polled |
| `statusAnnounce` | channel | `False` | Announce incidents from every configured `statusPageUrls` page in this channel as they open and again as they resolve — all or nothing per channel. Openings and resolutions share a budget of six LLM rewrites an hour; over budget the fixed template still goes out |

`check_service_status` is on the model's tool surface whenever `statusPageUrls` holds at least one page. An empty `statusPageUrls` removes the tool and stops polling and announcing. Duplicate names, and the same page written two ways (trailing slash, case, default port), collapse to one.

The bot polls `{url}/api/v2/summary.json` for each page every 2 minutes, in rotation inside one 45-second budget per pass, so a slow or failing page can't starve the others; with several pages configured, one may wait more than a cycle for its turn. A reading older than 10 minutes is reported to the model as stale. [Service status](../user/service-status.md) covers what users see.

!!! warning "Turn off the RSS feed first"

    If the channel already announces the same status page through the RSS plugin, remove that before turning `statusAnnounce` on, or every incident is reported twice:

    ```
    @rss announce remove #yourchan <feedname>
    @config channel #yourchan plugins.LLM.statusAnnounce True
    @flush
    ```

## Aurora

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `auroraAnnounce` | channel | `False` | Announce G3+ storm watches and sustained storm conditions in this channel |
| `auroraAnnouncePlaces` | channel | empty | Optional space-separated places for this channel's announcements; `_` stands for a space, as in `Sault_Ste._Marie`. Adds cloud cover for them and holds a storm line until one is dark. A place that matches nothing is skipped (warned once in the log) |

The chat model always has the `aurora` tool: NOAA aurora probability, solar wind, Kp, darkness, and cloud cover for a named place. The tool reads NOAA SWPC (`services.swpc.noaa.gov`) and Open-Meteo (`geocoding-api.open-meteo.com`, `api.open-meteo.com`). No API keys are needed. The tool and the announcer share one cache: solar wind for 2 minutes, aurora probability for 10 minutes, storm alerts for 5 minutes, the Kp forecast for 3 hours, cloud cover for 1 hour per place (a 5-day forecast), and place lookups for 24 hours (misses for 1 hour). A failed fetch is not cached. See [Aurora](../user/aurora.md).

Announcements go to every channel with `auroraAnnounce` on; set `auroraAnnouncePlaces` to add cloud cover and hold storm alerts until it is dark there. The bot checks every 5 minutes and posts two kinds of line, both fixed templates with no model rewrite:

- **Storm watch**: when NOAA's newest geomagnetic storm watch puts a UTC day at G3 or higher. A re-issued watch at the same level is not announced again; a rise (G3 to G4) is. With places set, the line adds "Clouds night of <first date>": the mean over each place's remaining dark hours in the local night that overlaps that UTC day most, or the next night if that one is already over.
- **Storm conditions**: once per episode of Bz at or below -10 nT for most of 3 hours. With places set, the line waits until at least one of them is dark, and cloud figures are the mean over the next 3 hours; without places it goes out at once with no cloud figures. A place that matches nothing is skipped (warned once in the log); if a lookup service outage stops every place resolving, the storm line waits for the next check.

The first check after a restart, and the first check after a channel turns announcements on, only records what is already in effect; nothing already running is announced. A channel whose send fails is retried on the next check. If no channel announces, nothing is fetched.

## Reminders and scheduled tasks

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `pendingTasksEnabled` | channel | `False` | Advertise the five pending-task tools (create, list and cancel reminders and scheduled tasks) to the model, so natural-language scheduling works. Off by default: the schemas and their rules cost roughly 1,100 prompt tokens on every completion in the channel |

The `@remind` command and the firing of reminders and tasks already created work regardless. This key only controls whether the model can handle reminders and scheduled tasks from plain language. Users reach both with `@remind list`, `del` and `clear` at any setting; owners reach anyone's with `@remind admin list|del|clear <nick>`. Limnoria's own `@scheduler list` does not show them, because they live in the plugin's database.

Fixed limits: 25 pending reminders per user, 5 active scheduled tasks per creator per channel, 50 fires for a recurring reminder and 5 for a recurring task. A fire checks the creator's `ask` rate limit without using it up. A task whose creator has lost `llm.ask` is deleted at its next fire, with a one-line notice to its delivery target and nothing in the log at the plugin's `WARNING` level. The user side is in [Reminders](../user/reminders.md) and [Scheduled tasks](../user/scheduled-tasks.md).

## Timeouts and retries

When an `@ask` or `@draw` call times out, the bot keeps retrying in the background for 60 seconds. Image generation has its own 120-second timeout.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `timeout` | global | `30` | Timeout for LLM API calls, in seconds |

## Draw behaviour

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `drawAutoRewriteMax` | channel | `1` | Automatic prompt rewordings when a safety filter blocks a request. `0` disables. Each rewording costs a second billed image call on top of the refused one, and every recovery observed in prod landed on the first attempt, so raising this mostly buys repeat refusals of a prompt that was never going to pass |

`@draw` passes conversation context to the planner only when the last activity was within 60 seconds. When `XAI_API_KEY` is unset, `@draw` refuses and the `generate_image` tool is hidden from chat.

## Other settings

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `commandPrefixes` | global | `. @` | Space-separated prefixes sanitised in output to prevent IRC command injection. The default covers both the conventional `.` and VibeBot's own `@` |
| `memeAliases` | global | `tyrone=yallgot biggums=yallgot chappelle=yallgot` | Extra `@meme` template names, `name=id` or `name=https://image`. An id target adds the name as a keyword on that template; a URL target adds a two-caption template drawn on that image. Hyphens for spaces in the name. The models behind `@meme` are under [Model selection](#model-selection) |

Fixed limits: an image prompt is at most 10,000 characters (chat text is not measured), an assistant turn makes at most 12 tool-call round trips, and at most 16 outbound LLM calls run at once across all surfaces. The SQLite database is `data/LLM.db` in Limnoria's data directory, and the plugin logs at `WARNING`. Limnoria's automatic `WHO` on join is skipped when the `account-tag` and `extended-join` capabilities are active.
