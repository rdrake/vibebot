# Getting started

Chat with AI models, create images, and set reminders, all from IRC.

## Talking to the bot

Mention the bot by name in a channel, or send it a private message:

```
<you>     VibeBot, what's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), Denali (North America),
          Kilimanjaro (Africa), Elbrus (Europe), Vinson (Antarctica),
          Puncak Jaya (Oceania).

<you>     ok but which of those, VibeBot, can you climb without a guide?
<VibeBot> Kilimanjaro is the most accessible: no technical climbing, just
          a multi-day trek.
```

Starting the line with the bot's nick always works, and so does ending with it: `what's the tallest mountain, VibeBot?` gets an answer, as does the `/me` form `* you prods VibeBot`. A trailing nick needs a space or comma in front of it, so `reboot` is still just a word. Naming the bot mid-sentence, as in the second question above, needs the [NickInMiddle plugin](../plugins/nickinmiddle.md) loaded and enabled for the channel, with a space or comma on each side of the nick. Inside a `/me` line the bot handles mid-sentence naming itself, and that one is not switchable per channel.

For a follow-up, reply to one of the bot's lines with your client's reply feature and skip the nick. The bot answers any reply to something it said in the last 2,000 lines. Your client needs IRCv3 replies (`+draft/reply`), and the bot has to be running since it sent that line, because it forgets which lines are its own when it restarts. Replies from other bots are ignored.

Long answers never flood the channel. Anything that will not fit on one IRC line comes back as a one-line summary plus a link to the full text on the bot's web page.

In plain language the bot answers questions, draws images, writes code, looks things up on the web, and stores what you ask it to remember:

```
<you>     VibeBot, remember that I prefer metric units
<VibeBot> Noted, metric from here on.
```

Reminders and scheduled tasks work this way only where your operator has turned on `supybot.plugins.LLM.pendingTasksEnabled`, which is off by default. The `@remind` command always works.

```
<you>     VibeBot, remind me in 2 hours to check the build
<VibeBot> Reminder set: check the build (in 2 hours).
```

## Using commands

Every command starts with `@`. Commands give you direct, predictable behaviour:

```
<you>     @ask What's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), ...
```

The bot remembers your recent conversation, so follow-ups work naturally:

```
<you>     @ask Which of those can a beginner climb?
<VibeBot> Kilimanjaro is the most accessible...
```

Context is per user and per channel. It expires after a few minutes of inactivity, or you can clear it with `@forget`, which drops your own thread and, when you run it in the channel itself, the channel's shared recent history.

The two paths do not overlap completely. Chat handles questions, images, code, web lookups, and "remember that…". Stored state is managed with commands: `@memories delete 3`, `@instruct clear`, `@forget`, `@usage`, `@remind list`.

## Accounts and permissions

Some commands, such as `@draw` and `@story`, need you to be authenticated with your network account. The bot tells you when a command needs authentication.

!!! note
    AfterNet uses network account authentication, not a NickServ service. Log in to your account through your usual method (for example, SASL) before using account-gated commands.

Your bot operator controls access to each command through Limnoria's capability system. If the bot says you lack permission, ask your operator for access. See the [Limnoria capabilities documentation](https://docs.limnoria.net/use/capabilities.html) for background.

## Rate limits

Each command family carries its own rate limit, and your allowance depends on your status: unregistered, registered, or trusted, the last being a capability your operator grants. `@ask` allows 15 requests a minute at every tier. The image commands are tighter: `@draw` and `@story` need an authenticated account and allow two per five minutes, or five a minute at the trusted tier. The bot replies with the length of the window when you hit a limit. Bot admins and owners are exempt. See [Default limits](../operator/rate-limiting-security.md#default-limits).

## The verse

Channels with the verse enabled host a persistent, collaborative fiction: a shared world where you play as an avatar. Join with `@verseopt in`, give your avatar a voice with `@avatar <persona>`, and check your surroundings with `@verse`, `@look`, and `@who`.

Mention anyone or anything in the world and the bot answers with a tale. Use `@rp <text>` to act as your avatar for one turn, or `@rp on` to stay in character until `@rp off` — or until you go quiet for a while (15 minutes by default).

To speak out of character in a verse channel, start your message with `//` or wrap it in `((...))`.

See [AI commands](ai-commands.md#the-verse) for the verse command list.

## Common commands

| Command | Description | Details |
|---------|-------------|---------|
| `@ask` | Ask the AI a question (supports vision and follow-ups) | [AI commands](ai-commands.md#ask) |
| `@draw` | Generate an image from a text description | [AI commands](ai-commands.md#draw) |
| `@story` | Generate an illustrated story or explainer page | [AI commands](ai-commands.md#story) |
| `@forget` | Clear your conversation context, plus the channel's shared recent history when run in that channel | [Memory](memory.md#conversation-context) |
| `@memories` | Manage stored facts about you | [Memory](memory.md#saved-memories) |
| `@instruct` | Set persistent instructions for `@ask` and mentions | [Memory](memory.md#custom-instructions) |
| `@avatar` | Set your verse avatar's persona | [Memory](memory.md#avatar-persona) |
| `@remind` | Set reminders with natural language | [Reminders](reminders.md#remind) |
| `@usage` | View API usage statistics | [AI commands](ai-commands.md#usage) |
| `@verseopt` | Opt your avatar in or out of the verse | [AI commands](ai-commands.md#the-verse) |
| `@rp` | Speak or act as your verse avatar | [AI commands](ai-commands.md#the-verse) |

That is the everyday set. Verse channels add `@verse`, `@look`, and `@who`; editors get `@canon` and `@versedit`, and GMs get `@versedump`, `@versepurge`, and `@versecompact`. The [command reference](../reference/commands.md) lists all twenty.

## Getting help

- `@help <command>` shows a command's built-in help.
- This guide lives at the address the bot shares when you ask it for help.
