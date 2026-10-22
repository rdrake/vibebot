# Command reference

Quick reference for every VibeBot command. For walkthroughs and
examples, see the [user guide](../user/getting-started.md).

## Natural language

Most bot features work through plain language. Mention the bot by name
in a channel, or send it a PM:

```
VibeBot, what's the weather like on Mars?
VibeBot, remember that I deploy on Fridays
VibeBot, what do you remember about me?
VibeBot, remind me in 2 hours to deploy
```

The bot reaches for tools as it answers: web search, page fetches,
image generation, saving a memory, and — where the operator
has configured a status page — a service-status check. Where the
[bridge](bridge-tools.md) is enabled, it can also look up and run a
Limnoria command. It can recite what it remembers about you, because
your stored memories sit in its context on every turn.

Listing memories precisely, editing or deleting them, setting an
instruction, checking usage, and clearing context are command-only.
Natural-language reminders and scheduled tasks need
`pendingTasksEnabled` set for the channel; `@remind` works either way.

Use the commands below when you want direct, predictable behaviour.

## Core AI commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@ask` | `<question>` | Ask the AI a question. Supports follow-ups and image URLs. Requires `llm.ask`. |
| `@draw` | `<prompt>` | Generate an image from text. Requires `llm.draw` and an authenticated account. |
| `@story` | `<brief>` | Generate an illustrated page (tale or explainer) and post a link when ready. Same gate as `@draw`. |

### ask

Ask the AI a question. Supports conversation context for follow-up
questions and vision for image URLs.

```
@ask What is the capital of France?
@ask Describe this: https://example.com/image.jpg
@ask And what about Germany?
```

See [AI commands](../user/ai-commands.md) for full details.

### draw

Generate an image from a text description. Requires an authenticated
account.

```
@draw A sunset over mountains in watercolour style
@draw A cyberpunk cityscape at night
```

### story

Generate an illustrated page from your brief and post a link when the
page is ready. The bot picks one of two modes from your wording: an
in-character illustrated tale, or a concept explainer with labelled
diagrams. No verse mode required. One page per account every five
minutes by default (`verseStorybookCooldownSeconds`), a cooldown shared
with the verse storybook tool.

```
@story an illustrated tale of the crew winning the pub quiz
@story explain how photosynthesis works, with diagrams
```

## Memory commands

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@forget` | `[<channel>]` | Clear volatile conversation context: your own thread, plus the channel's shared recent history when run in that channel. Naming another channel clears only your own thread there. Requires `llm.ask`. |
| `@memories` | `[<nick> \| del <id> [<id>...] \| edit <id> <text> \| clear \| cleanup [<nick>]]` | Manage stored facts. `del` takes several ids at once. The `<nick>` forms — viewing and cleaning up another user's memories — are owner-only. Needs an authenticated account. |
| `@instruct` | `[<instruction> \| clear]` | Set a persistent instruction that rides `@ask`, plain mentions, `@rp` and verse turns, and reminder fires. `@draw` ignores it. Empty shows the current one. Needs an authenticated account. |

```
@memories
@memories del 3
@memories edit 5 corrected fact
@memories clear
@instruct Respond only in haiku
@instruct clear
```

See [memory and instructions](../user/memory.md) for full details.

## Reminders

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@remind` | `[<text> \| list \| del <id> [<id>...] \| clear \| admin <list\|del\|clear> <nick> [<id>...]]` | Natural-language reminders, plus your scheduled tasks in `list`, `del` and `clear`. `del` takes several ids at once. Requires `llm.ask`; the `admin` subcommands are owner-only. |
| `@tz` | `[<zone> \| clear]` | Set the IANA time zone reminders read absolute times in, such as `America/Toronto`. Unset, the bot uses your client's CTCP TIME clock, else UTC. Empty shows the current zone. Needs an authenticated account. |

Reminders that ask the bot to *do* something (look up, check, fetch,
summarise) run as an LLM query at fire time and appear as `[auto]` in
`list`. Recurring action work, such as "every weekday at 9 a.m. check
the build", becomes a scheduled task. Setting one in plain language
needs `pendingTasksEnabled`; listing and cancelling is command-only,
and `@remind list`, `del` and `clear` cover your tasks — marked
`[task]` — alongside your reminders.

```
@remind in 30 minutes check the build
@remind in 2 hours check status of CVE-2026-31431 in Debian
@remind list
@remind del abc1
@remind admin list someone        # owner only
```

See [reminders](../user/reminders.md) for caveats:
reminders count against the `@ask` rate limit, run with no elevated
capabilities at fire time, and recurring chains cap at 50 fires.

## Accounting

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@usage` | `[<nick> \| #channel]` | Show API usage statistics. A nick queried in a channel is scoped to that channel, account-wide by PM. The global overview by PM is admin-only. |
| `@model` | `[<channel>]` | Which model answered the bot's last reply in the channel, and how long ago, next to the chat model the channel is set to use. Forgotten on restart. |

```
@usage
@usage someone
@usage #channel
```

## IRC lookups

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@channels` | `[--min <n>] [<channel> \| <glob>]` | The busiest public channels on the network with user counts and topics. A channel name shows that one; `#linux*` filters; `--min` hides small channels. |
| `@names` | `[<channel>]` | Who is in a channel at this moment, joined or not. Defaults to the current channel. |

```
@channels
@channels #linux
@channels --min 5 #linux*
@names #linux
```

Both answer from a live `LIST` / `NAMES` round trip; `LIST` results are
cached for 60 seconds. Secret and private channels are hidden by the
server, the same as for anyone typing `/list`. For a single user, the
stock `Network` plugin's `@whois <nick>` does the job once it is loaded;
in conversation, "who is eck?" reaches the same `WHOIS` through the
`irc_lookup` tool.

## Verse commands (user)

`@verseopt`, `@verse`, `@look` and `@who` must be typed in a
verse-enabled channel and need the `llm.verse` capability. `@rp` needs
`llm.verse` too, but answers as ordinary chat when the channel has no
verse or you have no avatar. `@avatar` needs neither capability and works
anywhere, including PM, but needs an authenticated account.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@verseopt` | `<in \| out>` | Opt your avatar in or out of this channel's verse. |
| `@rp` | `<text> \| on \| off` | Speak or act as your avatar. Shares the `@ask` rate-limit bucket. |
| `@verse` | | Show your current scene in one line. |
| `@look` | `[<target>]` | Describe your scene or a named entity. |
| `@who` | | List active avatars and their locations. |
| `@avatar` | `[<persona> \| clear]` | Set the persona that shapes your verse avatar. Independent of `@instruct`. Needs an authenticated account. |

### rp

`@rp <text>` takes one in-character turn. `@rp on` makes your plain
messages roleplay turns until `@rp off` or a spell of silence; a leading
`//` slips a single message out of character. Without an avatar the
command still answers, as ordinary chat grounded in canon.

```
@rp Archie kicks the door open and bellows for the lads
@rp on
@rp off
```

Mentioning canon without `@rp` no longer puts the bot in character. It
grounds the reply in canon facts, and, for an avatar holder, answers as
an inline prose tale. See
[the verse](../operator/verse.md#the-canon-layer-and-roleplay-mode).

## Verse commands (editor)

These require the `llm.verse.edit` capability, granted globally.
`@canon` must be run in the verse-enabled channel itself; `@versedit`
takes a leading `#channel`, so it can be run from a PM.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@canon` | `<lock \| unlock \| forget> <name>` | Mark a character as durable canon, or release it. |
| `@versedit` | `[#channel] <verb> <args>` | Edit the verse universe directly. |

`@versedit` verbs: `add`, `pin`, `unpin`, `set`, `name`, `desc`,
`retire`, `restore`, `relate`, `unrelate`, `event`, `editevent`,
`delevent`, and `show`. Entity kinds are `avatar`, `npc`, `place`,
`faction`, and `item`. Refer to entities by `#id` or by name, but name
lookup only finds active entities — so `restore` needs the `#id`, which
`@versedump` and the `retired #<id>` reply both give you.

```
@versedit add npc Headmaster Pringle :: stern keeper of the academy
@versedit pin #12
@versedit set #12 mood grumpy
@versedit show #12
```

## Verse commands (GM)

These require the `llm.verse.gm` capability.

| Command | Arguments | Description |
|---------|-----------|-------------|
| `@versedump` | `[#channel] [--format=json]` | Dump the verse state — entities, relations, aliases, avatar links, the 1000 most recent proposals, and the 200 most recent events — to the bot's pastebin, and reply with the link. |
| `@versepurge` | `[#channel] [<token>]` | Irreversibly wipe a channel's verse. Two-step token confirmation. |
| `@versecompact` | `[<channel>]` | Run retention compaction for the channel now. Defaults to the channel you type it in; name one explicitly from a private message. |

See [verse operations](../operator/verse.md) for what compaction
does and how to operate the verse.
