# Getting started

Chat with AI models, create images, and set reminders, all from IRC.

## Talking to the bot

Mention the bot by name in a channel, or send it a private message:

```irc
<you>     VibeBot, what's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), Denali (North America),
          Kilimanjaro (Africa), Elbrus (Europe), Vinson (Antarctica),
          Puncak Jaya (Oceania).

<you>     VibeBot, which of those can you climb without a guide?
<VibeBot> Kilimanjaro is the most accessible: no technical climbing, just
          a multi-day trek.
```

Starting the line with the bot's nick always works, and so does ending with it: `what's the tallest mountain, VibeBot?` gets an answer, as does the `/me` form `* you prods VibeBot`. A trailing nick needs a space or comma in front of it, so `reboot` is still just a word. Naming the bot mid-sentence, as in `which of those, VibeBot, has no glacier?`, needs the [NickInMiddle plugin](../reference/nickinmiddle.md) loaded and enabled for the channel, with a space or comma on each side of the nick. Inside a `/me` line the bot handles mid-sentence naming itself, and that one is not switchable per channel.

For a follow-up, reply to one of the bot's lines with your client's reply feature and skip the nick. The bot answers any reply to something it said in the last 2,000 lines. Your client needs IRCv3 replies (`+draft/reply`), and the bot has to be running since it sent that line, because it forgets which lines are its own when it restarts. Replies from other bots are ignored.

Long answers never flood the channel. A reply of up to three lines goes out inline as one multiline message where the server supports it; anything longer comes back as a one-line summary plus a link to the full text on the bot's web page.

In plain language the bot answers questions, draws images, writes code, looks things up on the web, and stores what you ask it to remember:

```irc
<you>     VibeBot, remember that I prefer metric units
<VibeBot> Noted, metric from here on.
```

Reminders and scheduled tasks work this way only where your operator has turned on plain-language scheduling (`pendingTasksEnabled`), which is off by default. The `@remind` command always works.

```irc
<you>     VibeBot, remind me in 2 hours to check the build
<VibeBot> Reminder set: check the build (in 2 hours).
```

## Using commands

Every command starts with `@`. Commands give you direct, predictable behaviour:

```irc
<you>     @ask What's the tallest mountain on each continent?
<VibeBot> Everest (Asia), Aconcagua (South America), ...
```

The bot remembers your recent conversation, so follow-ups work naturally:

```irc
<you>     @ask Which of those can a beginner climb?
<VibeBot> Kilimanjaro is the most accessible...
```

Context is per user and per channel. It expires after a few minutes of inactivity, or you can clear it with `@forget`, which drops your own thread and, when you run it in the channel itself, the channel's shared recent history.

The two paths do not overlap completely. Chat handles questions, images, code, web lookups, and "remember that…". Stored state is managed with commands: `@memories delete 3`, `@instruct clear`, `@forget`, `@usage`, `@remind list`.

## Accounts and permissions

`@draw`, `@meme`, `@story`, `@memories`, `@instruct`, `@avatar`, `@tz` and scheduled tasks need you logged in to your AfterNET network account, because the bot files your data under that account, not your nick. Log in the way your client normally does, such as SASL, before using them. Plain chat and `@ask` work without an account.

## When the bot says no { #rate-limits }

| The bot says | What it means | What to do |
|--------------|---------------|------------|
| `You must be identified to use this command.` | You aren't logged in to your network account | Log in, then try again |
| `Rate limit exceeded for draw. Try again in 300s.` | You hit that command's limit. The number is the length of the window, not the time left | Wait up to that long |
| `Storybook cooldown is active — try again in a bit.` | One `@story` page per person every five minutes, for everyone | Wait five minutes |
| `You don't have the llm.draw capability…` | Your operator hasn't given you that command | Ask your operator |
| `Blocked by the image filter. Try a different subject.` (or similar) | The image provider refused the picture, and the bot already tried rewording it | Pick a different subject; rewording it yourself rarely helps |
| `The model refused that one outright. Try rewording it.` (or similar) | The chat model declined the request | Rephrase it, or ask something else |

The limits, per person:

| Command | Limit | Trusted users |
|---------|-------|---------------|
| `@ask` and chat | 15 a minute | the same |
| `@draw` | 2 per 5 minutes | 5 a minute |
| `@meme` | 6 a minute | 12 a minute |
| `@story` | 1 per 5 minutes | the same |

Trusted is a status your operator can give you. Bot admins and owners skip every limit except `@story`'s. Scheduled reminders and tasks check your `@ask` limit when they fire but never use it up.

## The verse

Some channels host a shared story world where you play an avatar: join with `@verseopt in`, then act with `@rp`. [The verse](verse.md) covers how to play.

## Common commands

| Command | Description | Details |
|---------|-------------|---------|
| `@ask` | Ask the AI a question (supports vision and follow-ups) | [AI commands](ai-commands.md#ask) |
| `@draw` | Generate an image from a text description | [AI commands](ai-commands.md#draw) |
| `@meme` | Caption a meme template, or let the bot pick one | [AI commands](ai-commands.md#meme) |
| `@story` | Generate an illustrated story or explainer page | [AI commands](ai-commands.md#story) |
| `@forget` | Clear your conversation context, plus the channel's shared recent history when run in that channel | [Memory](memory.md#conversation-context) |
| `@memories` | Manage stored facts about you | [Memory](memory.md#saved-memories) |
| `@instruct` | Set persistent instructions for `@ask` and mentions | [Memory](memory.md#custom-instructions) |
| `@avatar` | Set your verse avatar's persona | [The verse](verse.md#joining-and-leaving) |
| `@remind` | Set reminders with natural language | [Reminders](reminders.md#remind) |
| `@usage` | What you, another user or a channel spent this month | [AI commands](ai-commands.md#usage) |
| `@verseopt` | Opt your avatar in or out of the verse | [The verse](verse.md) |
| `@rp` | Speak or act as your verse avatar | [The verse](verse.md#playing) |

That is the everyday set. Verse channels add `@verse`, `@look`, and `@who`; editors get `@canon` and `@versedit`, and GMs get `@versedump`, `@versepurge`, and `@versecompact`. The [command reference](../reference/commands.md) lists every command.

## Getting help

- `@help <command>` shows a command's built-in help.
- This guide lives at the address the bot shares when you ask it for help.
