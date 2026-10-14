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

**Scope** in the tables below means: *channel* settings accept per-channel overrides; *global* settings apply everywhere.

!!! warning "A new default in a release does not change a running bot"

    Limnoria writes the whole registry to `bot.conf` when the bot shuts down, so every setting the bot has ever loaded exists there as an explicit line. That line wins over the default shipped in a later release.

    So when an upgrade changes a **Default** in the tables below, the new value applies to fresh installs only. An existing deployment keeps whatever was persisted, silently and with no error — the feature simply behaves as it did before.

    To adopt a new default, set it explicitly with `@config` (which writes through the registry), or stop the bot, edit `bot.conf`, and start it again. Editing `bot.conf` while the bot runs does not work: the next shutdown flush overwrites it.

    Verify by reading back the value with `@config plugins.LLM.<settingName>` — a green deploy and a quiet log do not prove the new value took effect.

## API keys

API keys are **not** part of the Limnoria registry. They come from environment variables, one per provider, read at the point each provider call is made and selected by the provider of the model being called — not by which command or channel triggered the call. The old `assistantApiKey` / `codeApiKey` / `imageApiKey` / `searchApiKey` registry settings were removed, not deprecated: `@config` on them now errors.

| Variable | Provider |
|----------|----------|
| `XAI_API_KEY` | xAI (Grok) |
| `GEMINI_API_KEY` | Google Gemini |
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic (Claude) |

Any other provider LiteLLM recognises — `vertex_ai`, `openrouter`, `azure`, `bedrock`, and so on — resolves to no key from this plugin, so LiteLLM falls back to that provider's own native credentials: Application Default Credentials, IAM, or its own environment variables. With the defaults, a fresh install needs `ANTHROPIC_API_KEY` for chat and `@code`, `XAI_API_KEY` for search, and `GEMINI_API_KEY` for `@draw` (the default `imageModel` is `gemini/imagen-4.0-fast-generate-001`).

### Running on local models

A local server (Ollama, vLLM, LM Studio) takes no key from this plugin. Set its address in the env file and pick models with the matching LiteLLM prefix:

| Server | Env variable | Model prefix |
|--------|--------------|--------------|
| Ollama | `OLLAMA_API_BASE=http://host:11434` | `ollama_chat/<model>` |
| vLLM | `HOSTED_VLLM_API_BASE=http://host:8000/v1` | `hosted_vllm/<model>` |
| LM Studio | `LM_STUDIO_API_BASE=http://host:1234/v1` | `lm_studio/<model>` |

```
@config channel #chan supybot.plugins.LLM.assistantModel ollama_chat/qwen3:14b
```

Don't use `openai/<model>` with `OPENAI_API_BASE` for a local server. That redirects every OpenAI call in the bot and sends `OPENAI_API_KEY` to your box.

The chat path calls tools, so the local model must support function calling. Each role has its own model setting (`assistantModel`, `searchModel`, `codeModel`, `memoryModel`, `verseModel` and the rest under [Model selection](#model-selection)). Any role left on a hosted provider still sends its requests there. That matters when you turn off `privacyOptInOnly`: keep every model a channel uses local, or overheard lines can still reach a hosted provider.

### Opting in to Vertex AI

Vertex AI is not the default, but it remains a supported `imageModel` (and general model) choice for anyone who wants it — for example to bill image generation to a separate GCP project. Credentials (ADC, IAM) are enough for authentication, but LiteLLM's Vertex Imagen path separately requires the project and region, and it fails closed with `vertex_project and vertex_location are required for Vertex AI` if they are missing:

| Variable | Required | Notes |
|----------|----------|-------|
| `VERTEXAI_PROJECT` | Yes | Your GCP project ID |
| `VERTEXAI_LOCATION` | Yes | Region, e.g. `us-central1`. `VERTEX_LOCATION` also works as a fallback name |

Vertex AI takes no bearer key from this plugin — no `VERTEX_AI_API_KEY` exists; it authenticates through ADC and IAM only.

### Drawing against a self-hosted endpoint

`imageApiBase` points `@draw` at any OpenAI-shaped `/v1/images/generations` endpoint — including the same box `@animate` uses, which serves images as well as video. Set it and three things change: the request carries an `api_base`, it authenticates with `ANIMATE_API_KEY` instead of the provider key `imageModel`'s name would imply, and `imageSteps`/`imageSize` ride along in the request body. Leave it empty and nothing changes; the hosted provider path is untouched.

```
@config channel #chan supybot.plugins.LLM.imageApiBase http://host:port/v1
@config channel #chan supybot.plugins.LLM.imageModel openai//path/to/model
@config channel #chan supybot.plugins.LLM.imageSteps 8
@config channel #chan supybot.plugins.LLM.imageSize 1024x576
```

The `openai/` prefix tells LiteLLM which wire format to speak; everything after it is the model name your server expects, which is why a filesystem path with slashes in it is fine.

Two things to weigh before switching a busy channel over:

- **Steps are the whole latency budget.** Measured against the reference box at 1024x576 on an idle server: 8 steps ≈ 32 seconds, 25 steps ≈ 94 seconds. Compare a hosted provider answering in a few seconds. Raise `drawTimeout` (default 120) to cover whatever you pick, or draws will fall into the stash-and-deliver-later path.
- **A busy box is much slower than an idle one.** That server runs jobs concurrently rather than queueing, so a draw submitted while clips are rendering can take well over twice as long, and every other job on the box slows down too.

Spend is recorded as `$0.00`, which is accurate for your own hardware — and unlike an unpriced hosted model, it does not log the "no price in `IMAGE_COST_PER_IMAGE`" warning, so that warning still means what it says.

### Falling back when a provider refuses

The same endpoint can sit behind a hosted provider instead of replacing it. Set `imageFallbackApiBase` and a refusal on content grounds is redrawn against your own box rather than reaching the user as an error:

```
@config channel #chan supybot.plugins.LLM.imageFallbackApiBase http://host:port/v1
@config channel #chan supybot.plugins.LLM.imageFallbackModel openai//path/to/model
@config channel #chan supybot.plugins.LLM.imageFallbackSteps 8
@config channel #chan supybot.plugins.LLM.imageFallbackSize 1024x576
```

The fallback is sent the **original prompt**, never a safety rewrite. A rewrite exists to talk a filter around, and an endpoint with no filter has nothing to be talked around — so the picture that arrives is the one that was asked for, and it carries no 🔁 marker because nothing was reworded.

This interacts with `drawAutoRewriteMax` (default `1`), and the two orderings buy different things:

| `drawAutoRewriteMax` | What a refused prompt does |
|---|---|
| `1` (default) | Rewrites once and redraws on the provider. If that is refused too, the fallback draws the original. Fast when it works, but the user gets a softened version of their request, and the rewrite costs a second billed call |
| `0` | Skips rewriting and goes straight to the fallback. Slower per refusal (a self-hosted draw is tens of seconds), free, and faithful to what was asked |

Refusals are still recorded either way. A recovered draw carries its `blocked_attempts` through to the usage table, so the refusal rate stays measurable, and the fallback logs `image_fallback_served` at WARNING so you can count how often it earns its keep.

If the fallback also fails — refused again, misconfigured, or the box is down — the user gets the original refusal message. Which of two backends disappointed them is not something they asked about.

### Video generation

`@animate` talks to a self-hosted vLLM video server rather than a LiteLLM provider, so it needs both halves of its own credential and stays off until it has them:

| Setting | Where | Notes |
|---------|-------|-------|
| `ANIMATE_API_KEY` | Environment | Bearer token for the video server. Covered by log redaction like every other `*_API_KEY` |
| `animateApiUrl` | Registry | Base URL, e.g. `http://videohost:14205` — no trailing `/v1`. Empty (the default) disables `@animate` and hides the `generate_video` tool |

With either half missing the command returns a configuration error and the tool never reaches the model, so it cannot promise a clip the bot has no way to render.

`@animate` never sends the words the user typed. A planner turn rewrites the ask into a script the video server can render, so the usage row records both: `prompt` holds the request and `rendered_prompt` holds what was submitted. Every other command leaves `rendered_prompt` empty. Rows written before this landed have it empty too — the rendered wording was only ever held in `pending_tasks`, which is cleared as each clip is delivered.

Copy `.env.example` to `~/.config/vibebot/env` and set the variables there; the systemd unit passes that file to the container with `--env-file` (see [Installation](installation.md)). A model whose provider has no key configured fails with an error naming the missing variable, for example `no API key configured for provider 'xai' (set XAI_API_KEY)`.

## Model selection

Models follow [LiteLLM's provider/model format](https://docs.litellm.ai/docs/providers). Model names are validated against LiteLLM's known providers; the bot rejects unrecognised names and suggests alternatives.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `assistantModel` | channel | `anthropic/claude-haiku-5-5` | All assistant text and tool work: chat, planner, memory, reminders, scheduled tasks. Needs vision support if users paste image URLs |
| `codeModel` | channel | `anthropic/claude-haiku-5-5` | The code-generation one-shot behind `@code` and the `generate_code` tool. The `@code` planner loop itself runs on `assistantModel` |
| `imageModel` | channel | `gemini/imagen-4.0-fast-generate-001` | Model for `@draw` |
| `imageApiBase` | channel | empty | Draw against an OpenAI-shaped endpoint of your own instead of the provider `imageModel` names — see [Drawing against a self-hosted endpoint](#drawing-against-a-self-hosted-endpoint) |
| `imageSteps` | channel | `0` | Denoising steps for a self-hosted endpoint. `0` lets the server choose. Ignored unless `imageApiBase` is set |
| `imageSize` | channel | empty | Output geometry as `WxH` for a self-hosted endpoint. Ignored unless `imageApiBase` is set |
| `imageFallbackApiBase` | channel | empty | Endpoint to redraw against when the primary provider refuses a prompt — see [Falling back when a provider refuses](#falling-back-when-a-provider-refuses) |
| `imageFallbackModel` | channel | empty | Model for `imageFallbackApiBase`. A plain string, not a validated model name |
| `imageFallbackSteps` | channel | `0` | Denoising steps for the fallback endpoint |
| `imageFallbackSize` | channel | empty | Output geometry as `WxH` for the fallback endpoint |
| `searchModel` | channel | `xai/grok-4.3` | Model for web search and URL fetch. Falls back to `assistantModel`. Only xAI and Gemini models search; any other provider answers from its own knowledge. GitHub links skip it: they are read through GitHub's API. A request can run at most three searches |
| `memoryModel` | channel | empty | Model for the memory extraction that runs after every reply. Falls back to `assistantModel`. A flash-lite class model is enough |
| `subjectResearchEnabled` | channel | `True` | Research the real people, places and events a `@draw` or `@animate` request names before the planner writes the prompt — see [Subject research](#subject-research) |
| `subjectResearchModel` | channel | empty | Model for the subject-research pre-stage. Falls back to `searchModel`, then `assistantModel` |
| `verseModel` | channel | empty | Model for verse-mode replies. Falls back to `assistantModel`. Set this when the assistant model is a terse reasoning model that writes poor prose |
| `verseCompactionModel` | global | `anthropic/claude-haiku-5-5` | Cheap model for the daily verse compaction job. Unlike the channel model keys this one is a plain string, so a misspelled model name is accepted at `@config` time and only fails when the nightly job runs |

Channel overrides let different channels run different models:

```
@config channel #dev plugins.LLM.assistantModel anthropic/claude-sonnet-4-20250514
@config channel #casual plugins.LLM.assistantModel anthropic/claude-haiku-5-5
```

### Subject research

Image and video generators know nothing by name. `@animate` renders a bare
"Churchill" as nothing at all, or as the word printed across the frame, and
`@draw` is little better — which is why both commands run a planner turn that
rewrites the request before the generator ever sees it. The planner is told to
keep the name the user typed and put a description of the subject beside it,
but for a real person or a real event it has nothing to build that description
from: the verse canon block covers characters the channel invented, not people
the world already contains.

`subjectResearchEnabled` (on by default) closes that gap with one grounded
completion in front of the planner. It identifies the real subjects the request
names, looks up how each actually looks, and layers the answer onto the
planner's prompt. A request that names nobody real — "a slow aerial shot over a
pine forest" — comes back with nothing and adds nothing to the prompt.

The cost is one extra call and a few seconds. `@animate` hides that behind a
render measured at 135 seconds; `@draw` does not, so `@draw` gets visibly
slower with this on. The typing indicator covers the wait. The usage table
records research costs under the command name `dossier`, separately from the
planner's own row, because the two run on different models.

Only the dash-prefixed fact lines the research prompt asks for are kept, and
only those that end in a `(source: …)` tag. A model that declines the
question, or wraps its answer in commentary, contributes nothing rather than
contributing its refusal to the next model's prompt. The source tag guards the
other failure: asked about a name that exists nowhere, a grounded model guesses
an appearance from the words in it rather than answering `NONE`, and a guess
has nothing to cite. The tag is stripped before the line reaches the planner.

`subjectResearchModel` picks the model. Left empty it falls back to
`searchModel` and then `assistantModel`, which in most deployments means the
researcher is a different provider from the planner. Set it when the search
model is more reticent about describing real people than the generator
downstream is.

## System prompts

System prompts define the bot's personality and constraints. Both are channel-overridable.

| Setting | Description |
|---------|-------------|
| `assistantSystemPrompt` | Personality and constraints for all assistant work. Verse mode layers scene context on top of this prompt rather than replacing it |
| `codeSystemPrompt` | Instructions for `@code` output format |

Both defaults carry constraints worth keeping: `assistantSystemPrompt` holds replies to one to three lines of plain text with no markdown, and `codeSystemPrompt` keeps the code body in the tool's URL rather than in the channel. Read the shipped text before you replace it — `@config default plugins.LLM.assistantSystemPrompt` — because a replacement drops those constraints with it.

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
| `verseRoleplayStickyTtlSeconds` | channel | `900` | Sliding inactivity timeout for sticky `@rp on` roleplay. Each in-character turn refreshes it. `0` never expires |
| `verseChatRecordEnabled` | channel | `False` | Let an opted-in avatar's ordinary chat turn call `verse_record`. Off keeps canon growing only during roleplay |
| `verseStoryAmbientMaxImages` | channel | `1` | Vestigial. Since ambient canon mentions became inline prose, only an explicit illustrate cue reaches the storybook, and that path reads `verseStorybookMaxImages` instead. `0` does not give text only — the read falls back to `1` |
| `verseEventRetentionDays` | channel | `30` | Events older than this are eligible for compaction. `0` disables compaction for the channel |
| `verseAutoEntityRetireDays` | channel | `14` | Days without a mention before auto-created non-player characters (NPCs) retire. `0` disables the sweep. Pinned entities are exempt |
| `verseAutoEntityMaxNamesPerCall` | channel | `8` | Cap on the `actors` array in one `verse_record` call |
| `verseRosterMaxChars` | channel | `4000` | Character cap on the canon-roster block in the verse system prompt |
| `verseStyleExemplars` | channel | `[]` | Curated style exemplars injected into the verse prompt. Populate offline with the taste miner |
| `verseReactionCaptureEnabled` | channel | `True` | Capture IRCv3 emoji reactions to verse lines as an offline approval signal |
| `verseCompactionDailyAt` | global | `03:00` | Local `HH:MM` time for the daily compaction sweep. Empty falls back to `03:00`. A malformed value does not — it defers the next run by an hour, each time, and logs a warning |
| `verseCompactionMinKeepEvents` | global | `20` | Verses with fewer total events are skipped by compaction |

### Storybook

These caps cover both illustrated-page paths: the `verse_storybook` tool on verse turns, and the standalone `@story` command. `@story` needs no verse — it is gated like `@draw`, on `llm.draw` plus an authenticated account — so these limits still bind in a channel where `verseEnabled` is `False`. The `verse` prefix is historical; renaming the keys would orphan existing `bot.conf` values.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `verseStorybookEnabled` | channel | `False` | Expose the `verse_storybook` tool on verse turns. It does **not** gate `@story` |
| `verseStorybookMaxImages` | channel | `5` | Cap on illustrations per storybook |
| `verseStorybookMaxPerTurn` | channel | `1` | Cap on storybooks per verse turn. Tool-only; `@story` does not read it |
| `verseStorybookCooldownSeconds` | channel | `300` | Per-account cooldown between storybooks |
| `verseStorybookDailyImageCap` | channel | `30` | Registered but **not yet enforced** — there is no per-account daily image count to check it against |
| `verseStorybookMaxChars` | channel | `6000` | Character cap on the storybook body |
| `verseStorybookImageTimeout` | channel | `45` | Seconds to wait for each illustration |

Turning `verseStorybookEnabled` off does not stop image costs on its own, because `@story` still runs. What bounds them today is `verseStorybookMaxImages` per page and the per-account cooldown between pages, not the daily cap.

## Memory

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `memoryEnabled` | channel | `True` | Enable automatic memory extraction |
| `memoryMaxPerUser` | global | `50` | Cap on durable memories per user |
| `memoryCleanupInterval` | global | `3` | Extraction passes between cleanup runs. `0` disables |
| `memoryPromotionThreshold` | global | `2` | Mentions a candidate fact needs before promotion. `1` restores single-stage saving |
| `memoryCandidateTTLDays` | global | `14` | Days an unreinforced candidate survives. `0` disables pruning |

See [Memory promotion](memory-promotion.md) for how the two-stage pipeline works.

## Conversation context

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `contextEnabled` | channel | `True` | Enable conversation context |
| `contextMaxMessages` | channel | `20` | Messages kept in per-user history |
| `contextTimeoutMinutes` | channel | `5` | Clear context after this much inactivity |
| `privacyOptInOnly` | channel | `True` | Send only lines addressed to the bot to the provider. Set `False` to track every channel message as context and let `irc_lookup` read the channel's server history (`kind=history`, up to 100 lines) for "what did I miss". Those lines go to whichever provider serves the channel's models. See [Running on local models](#running-on-local-models) |
| `channelContextMaxMessages` | channel | `10` | Messages in the shared channel context |

## Rate limiting

Rate limits follow the pattern `{command}{tier}RateLimitCount` and `{command}{tier}RateLimitWindow` for the commands `ask`, `code`, `draw`, `story`, and `animate`. All rate-limit settings are global. See [Rate limiting and security](rate-limiting-security.md) for tiers, defaults, and the observe-only switch `enforceRateLimits`.

## Bot-to-bot loops

Two bots that answer each other never get bored, so replies to another bot are capped.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `botLoopReplyLimit` | channel | `3` | Consecutive replies to one bot before the bot goes quiet. `0` disables the guard |
| `botLoopWindow` | channel | `300` | Seconds of quiet after which that bot's count resets |
| `botLoopHostSuffixes` | global | `Bot.AfterNET.Org` | Host suffixes the network reserves for bots. Matched case-insensitively against the sender's host. A sender matched here, or one with the +B mode, cannot address the bot by replying to its lines. A full host works too, for a bot on a user host |

A nick counts as a bot only when the network says so, by either of two signals. AfterNET advertises `BOT=B` and marks flagged users in the WHOX status field, which is where the bot reads it — six nicks carry it today, long-running service bots among them. Anyone unflagged is treated as a person and is never capped, and a human *addressing the bot* clears the counts in that channel, so a conversation someone has joined is never cut short. Only addressed lines reset it: what people say during a bot loop is commentary on the noise ("wtf", "jesus"), and treating that as a reset would license three more replies after every such line.

The `+B` mode is opt-in, and the bot most likely to loop with yours may not set it — `grok` answers on `Hxz` from `grok.Bot.AfterNET.Org`. So the sender's host counts too: AfterNET only issues `*.Bot.AfterNET.Org` to registered bots, and across a full day of channel logs the only nicks on one were `grok`, `grook5`, and `vibebot` itself. The host is checked first because it is free and decides on the very first line, with no WHO round trip.

Because `skipAutoWhoOnJoin` suppresses the WHO on join, the flag is fetched on demand: the first time an unknown nick addresses the bot, a WHO goes out for that nick and the answer is cached for an hour. That first line is always answered — the guard fails open, so the worst case is one extra reply, never a silenced user. A bot that never sets `+B` is invisible to this and will not be capped.

## Web and HTTP output

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `httpRoot` | global | empty | Filesystem path for output files. Empty keeps them under Limnoria's `data/web/llm/` and serves them from its built-in HTTP server. Setting it **unhooks that server** — an external server such as nginx must serve the path, and `httpUrlBase` must be set to match, or every generated link 404s |
| `httpUrlBase` | global | empty | Public base URL for output files. Empty uses Limnoria's `publicUrl` plus `/llm`; if `publicUrl` is also unset the bot falls back to `http://localhost:<http port>/llm`, which nobody off the host can open |
| `imageUploadUrl` | global | empty | External image host for generated images. Empty stores them locally |
| `helpUrl` | global | `https://rdrake.github.io/vibebot-v8/` | URL shown in help output |
| `longReplyTeaserMaxChars` | channel | `220` | Characters in the one-line teaser for long replies saved as HTML |
| `multilineMaxLines` | channel | `3` | Replies that wrap to at most this many IRC lines go inline as one IRCv3 `draft/multiline` batch instead of a teaser and link. Needs `supybot.protocols.irc.experimentalExtensions`. `0` turns it off |
| `fileCleanupAge` | global | `720` | Delete output files older than this many hours (30 days) |
| `fileCleanupMax` | global | `1000` | Cap on files kept in the output directory |

### Offloading images to an external host

Set `imageUploadUrl` to an uploader that takes a multipart `POST` on the `images[]` field and answers with JSON:

```json
{"results": [{"success": true, "filePath": "/img/img_abc123.png"}]}
```

Generated images then live on that host instead of under `httpRoot`, and storybook pages embed the absolute URL. Pages themselves (answers, code, stories) still come from your own HTTP server, so this offloads bandwidth rather than replacing it.

Every failure falls back to local storage: endpoint unreachable, upload rejected, image over 10 MB, or a reply naming a URL that is not an image on the configured host. An outage costs a slower draw, nothing more.

[paste.boxlabs.uk/img/](https://paste.boxlabs.uk/img/) is an externally operated uploader, the host this contract follows, and it needs no key. Ask the host's owner before pointing the bot at somebody else's uploader.

## Limnoria tool bridge

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `bridgeEnabled` | channel | `False` | Expose loaded Limnoria plugin commands to the LLM as a tool |
| `bridgeAllowedPlugins` | channel | empty | Plugin allowlist. Empty uses the curated read-safe default set |
| `bridgeAllowMutating` | channel | `False` | Expose state-changing commands through the bridge |
| `bridgeScheduledTaskLimit` | channel | `5` | Active scheduled LLM tasks per creator per channel. `0` disables scheduling |
| `bridgeDebugInChannel` | channel | `False` | Append a bridge-call debug footer to replies |
| `ircLookupEnabled` | channel | `True` | Advertise the `irc_lookup` tool (live `LIST`, `NAMES`, `WHOIS`, `WHO`, topic, network info, CTCP) to chat. Its `history` kind also needs `privacyOptInOnly` off. `@channels` and `@names` work regardless |
| `memeModel` | channel | `anthropic/claude-haiku-5-5` | Model that picks a template and writes captions when `@meme` (or `make_meme`'s `brief`) names no template the catalog knows. It opens on a short list (the 40 most popular templates plus up to 20 sharing a word with the request, about 5K prompt tokens) and can call a `search_templates` tool up to twice to reach the rest of the catalog. No search: one call, about 5K tokens, 2 s. Each search is another round trip: about 12K tokens and 4 s with one, 19K and 6 s with two. Under $0.003 on Haiku either way. Empty falls back to `assistantModel`; grok-4-1-fast copies the template's example captions instead of writing new ones |
| `memeEditModel` | global | `xai/grok-imagine-image` | Image-edit model for the picker's `draw` (and `make_meme`'s): the captioned meme goes to xAI's `/v1/images/edits` with the instruction. `xai/` only. About $0.02 per edit, refusals billed too |
| `memeEnabled` | channel | `True` | `False` refuses `@meme` and hides the `make_meme` tool from chat |
| `memeAliases` | global | `tyrone=yallgot biggums=yallgot chappelle=yallgot` | Extra template names, `name=id` or `name=https://image`. An id target adds the name as a keyword on that template; a URL target adds a two-caption template drawn on that image. Hyphens for spaces in the name |

See [Tuning and monitoring](tuning-monitoring.md) for the bridge's security model and the scheduled-task tool.

## Status pages

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `statusPageUrls` | global | `Claude=https://status.claude.com GitHub=https://www.githubstatus.com OpenAI=https://status.openai.com` | Space-separated status pages to poll and announce — both Atlassian Statuspage and incident.io pages work — each written as `Name=url` or as a bare url, which takes its host as its name. Names are 1-32 chars of `[A-Za-z0-9._-]`, case-insensitively unique across this key and `statusQueryablePages`, and are what the model uses to ask about one service. Unusable entries are dropped with a warning and the rest survive; at most 5 are polled |
| `statusQueryablePages` | global | empty | Same `Name=url` grammar as `statusPageUrls`, for pages the bot can be **asked** about but never polls or announces — the long tail like Cloudflare or AWS, where polling forever to answer an occasional question is the wrong trade. Fetched lazily on request and cached for 5 minutes, with a failure backoff. An entry naming the same page as a `statusPageUrls` entry is dropped, whatever its name — that page is already reachable, polled, as the fresher entry. At most 20 |
| `statusAnnounce` | channel | `False` | Announce incidents from every configured `statusPageUrls` page in this channel as they open and again as they resolve. `statusQueryablePages` pages never announce, no matter how this is set |

`check_service_status` is on the model's tool surface whenever either key holds at least one page. An empty `statusPageUrls` no longer removes the tool by itself — only polling and announcing stop; a non-empty `statusQueryablePages` keeps the tool available for on-request lookups.

See [Tuning and monitoring](tuning-monitoring.md#status-pages) for the polling rotation, staleness, and the RSS cutover order.

## Aurora

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `auroraEnabled` | channel | `True` | Give the chat model the `aurora` tool: NOAA aurora probability, solar wind, Kp, darkness, and cloud cover for a named place |
| `auroraAnnounce` | channel | `False` | Announce G3+ storm watches and sustained storm conditions in this channel. Works without `auroraEnabled` |
| `auroraAnnouncePlaces` | channel | empty | Optional space-separated places for this channel's announcements; `_` stands for a space, as in `Sault_Ste._Marie`. Adds cloud cover for them and holds a storm line until one is dark. A place that matches nothing is skipped (warned once in the log) |

The tool reads NOAA SWPC (`services.swpc.noaa.gov`) and Open-Meteo (`geocoding-api.open-meteo.com`, `api.open-meteo.com`). No API keys are needed. The tool and the announcer share one cache: solar wind for 2 minutes, aurora probability for 10 minutes, storm alerts for 5 minutes, the Kp forecast for 3 hours, cloud cover for 1 hour per place (a 5-day forecast), and place lookups for 24 hours (misses for 1 hour). A failed fetch is not cached. See [Aurora](../user/aurora.md).

Announcements go to every channel with `auroraAnnounce` on; set `auroraAnnouncePlaces` to add cloud cover and hold storm alerts until it is dark there. The bot checks every 5 minutes and posts two kinds of line, both fixed templates with no model rewrite:

- **Storm watch**: when NOAA's newest geomagnetic storm watch puts a UTC day at G3 or higher. A re-issued watch at the same level is not announced again; a rise (G3 to G4) is. With places set, the line adds "Clouds night of <first date>": the mean over each place's remaining dark hours in the local night that overlaps that UTC day most, or the next night if that one is already over.
- **Storm conditions**: once per episode of Bz at or below -10 nT for most of 3 hours. With places set, the line waits until at least one of them is dark, and cloud figures are the mean over the next 3 hours; without places it goes out at once with no cloud figures. A place that matches nothing is skipped (warned once in the log); if a lookup service outage stops every place resolving, the storm line waits for the next check.

The first check after a restart, and the first check after a channel turns announcements on, only records what is already in effect; nothing already running is announced. A channel whose send fails is retried on the next check. If no channel announces, nothing is fetched.

## Reminders and scheduled tasks

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `pendingTasksEnabled` | channel | `False` | Advertise `set_reminder` and `schedule_llm_task` to the model, so natural-language scheduling works. Off by default: the schemas and their rules cost roughly 1,100 prompt tokens on every completion in the channel |

The `@remind` command and the firing of reminders and tasks already created work regardless. This key only controls whether the model can *create* reminders and scheduled tasks from plain language. Listing and cancelling are not on the chat tool surface at any setting: users reach both their reminders and their scheduled tasks with `@remind list`, `del` and `clear`.

## Timeouts and retries

When an API call times out, the bot retries in the background for a bounded window.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `timeout` | global | `30` | Timeout for LLM API calls, in seconds |
| `drawTimeout` | global | `120` | Timeout for image generation. `0` uses the global timeout |
| `askExpiry` | global | `60` | Seconds to keep retrying a timed-out `@ask`. `0` disables |
| `codeExpiry` | global | `60` | Seconds to keep retrying a timed-out `@code`. `0` disables |
| `drawExpiry` | global | `60` | Seconds to keep retrying a timed-out `@draw`. `0` disables |

## Draw behaviour

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `drawEnabled` | channel | `True` | `False` refuses `@draw` and hides the `generate_image` tool from chat. The tool is also hidden when `imageModel`'s provider has no key |
| `drawAutoRewriteMax` | channel | `1` | Automatic prompt rewordings when a safety filter blocks a request. `0` disables. Each rewording costs a second billed image call on top of the refused one, and every recovery observed in prod landed on the first attempt, so raising this mostly buys repeat refusals of a prompt that was never going to pass |
| `drawContextMaxAgeSeconds` | channel | `60` | Pass conversation context to draw requests only when the last activity is this recent. `0` always starts fresh |

## Animate behaviour

`@animate` renders on a self-hosted video box (see [Video generation](#video-generation) for the credential). Generation is asynchronous: the command submits a job and returns, and the pending-task poller publishes the clip when it lands, so a bot restart mid-render does not lose the video. The box renders one clip at a time, so the queue is capped (`animateMaxPendingPerUser`, `animateMaxPending`) and `@renders` lets users and admins inspect or clear it.

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `animateEnabled` | channel | `True` | `False` refuses `@animate` and hides the `generate_video` tool from chat. Both are off everywhere until the video box is configured |
| `animateSteps` | channel | `25` | Denoising steps, and the dominant cost knob — latency is roughly linear in it. Measured on the reference box at 1280x704: a four-second clip took 68s at 25 steps and 171s at 50; at 25 steps a seven-second clip took 135s |
| `animateSize` | channel | `1280x704` | Output resolution. Must be a geometry the loaded model supports |
| `animateDuration` | channel | `7` | Clip length in seconds. The whole clip is exclusive GPU time, so raising it slows every queued request behind it. Also watch the file size: a 7s clip measured 8.87 MB, leaving about 1 MB under the uploader's 10 MB ceiling, and a clip over it falls back to local storage instead of `imageUploadUrl` |
| `animateAudio` | channel | `True` | Generate a soundtrack with the video. `False` requests silent video |
| `animateModel` | channel | empty | Model to request. Empty sends no model field, letting a single-model box pick its own — usually right |
| `animateFlowShift` | global | `12` | Flow-matching shift for the video sampler |
| `animateAudioFlowShift` | global | `3` | Flow-matching shift for the audio track. Ignored when `animateAudio` is off |
| `animateTimeout` | global | `60` | Timeout for individual HTTP calls to the video server. **Not** the generation budget — that is `animateExpiry` |
| `animateExpiry` | global | `1800` | Seconds to keep polling a submitted job before reporting it expired. Generous on purpose: a queued job can wait a long time before it starts |
| `animateMaxPendingPerUser` | global | `2` | How many clips one person may have waiting to render at once. A request past this is refused before anything is submitted. Owner and admin are exempt. `0` disables the per-user cap |
| `animateMaxPending` | global | `6` | How many clips may be waiting to render across the whole bot. The box renders one at a time, so this bounds the longest wait anyone is promised. Applies to everyone. `0` disables |
| `animateRenderSeconds` | global | `135` | Seconds one clip takes to render at the configured size, duration and steps. Only used to estimate the wait quoted when a clip is queued; measure it on the box after changing `animateDuration` or `animateSteps` |

Clips are published to `imageUploadUrl` when one is configured — the reference host accepts MP4 on the same `images[]` field and files it as `vid_*.mp4` — and fall back to local storage otherwise, including when a clip exceeds the 10 MB upload ceiling.

## Other settings

| Setting | Scope | Default | Description |
|---------|-------|---------|-------------|
| `maxPromptLength` | global | `10000` | Longest accepted user prompt, in characters |
| `commandPrefixes` | global | `. @` | Space-separated prefixes sanitised in output to prevent IRC command injection. The default covers both the conventional `.` and VibeBot's own `@` |
| `databasePath` | global | empty | Path to the SQLite database. Empty uses `data/LLM.db` |
| `logLevel` | global | `WARNING` | Plugin log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `metaMaxSteps` | global | `12` | Cap on tool-call round trips per assistant turn. Stops runaway tool loops |
| `maxConcurrentLLMCalls` | global | `16` | Cap on simultaneous outbound LLM calls across all surfaces |
| `skipAutoWhoOnJoin` | global | `True` | Skip Limnoria's automatic `WHO` on join when the `account-tag` and `extended-join` capabilities are active |
| `enforceRateLimits` | global | `True` | Enforce rate limits. `False` tracks and logs without blocking |
