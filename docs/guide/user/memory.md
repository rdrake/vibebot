# Memory and instructions

VibeBot has two kinds of memory: **conversation context** (short-term) and **long-term memories** (stored facts). You can also set **custom instructions** that persist across conversations.

!!! tip "What conversation can and can't do"
    Ask "VibeBot, what do you remember about me?" and it will tell you — your memories are already in front of it. Saving a new one works the same way: "VibeBot, remember that I run Debian." Everything else on this page — listing precisely, editing, deleting, clearing context, setting an instruction — is command-only.

## Conversation context

Every `@ask` or `@code` adds to a short-term conversation history, so you can ask follow-up questions without repeating yourself.

- Context is per user, per channel: your conversation in `#general` stays separate from `#dev`.
- Context expires after a few minutes of inactivity (default: 5 minutes).
- The bot can also weave in recent channel messages, so it can follow group conversations. By default that shared window holds only lines addressed to the bot and its own replies — the last 10. An operator can turn off `privacyOptInOnly` to feed it every message in the channel and let the bot read the channel's server history for "what did I miss" questions. Both send that chatter to the LLM provider.

Clear your context and start fresh with:

```
@forget
@forget #otherchannel
```

The two forms are not symmetric. `@forget` typed in a channel clears your own thread there *and* that channel's shared recent history, which everyone's follow-ups draw on. Naming another channel, or sending `@forget` in a private message, clears only your own thread — a channel's shared history can be cleared only from inside that channel.

## Saved memories

The bot picks up durable facts from your conversations, such as your preferred programming language, and recalls them later to give more relevant answers. Unlike conversation context, memories are per user and not per channel: a fact learned in `#dev` is in front of the bot when you talk to it in `#general` or in a private message.

A fact isn't saved the first time it comes up. It becomes a candidate, and the bot promotes it to a saved memory when it recurs in a later conversation. Candidates that never recur expire after about two weeks. This keeps one-off remarks out of your memory list.

### Viewing your memories

```
@memories
```

This lists your stored facts with their IDs:

```
[1] Prefers Python for scripting | [2] Works on network infrastructure | [3] Located in Ontario
```

### Managing memories

Conversation covers one operation, saving:

```
VibeBot, remember that I run Debian on my servers
```

A fact you ask the bot to remember is stored at once — it skips the candidate stage above. Deleting, editing and cleaning up go through the command:

```
@memories delete 3          # delete one (del also works)
@memories delete 1 2 3      # delete several
@memories edit 2 Works on cloud infrastructure
@memories clear             # delete all
@memories cleanup           # AI pass: merge duplicates, drop stale facts
```

### Memories and your account

Your memories are filed under your account once you authenticate, and under your current nick before that. The first time you talk to the bot in a session while authenticated, it moves everything held under that nick across: memories, candidates that haven't been promoted yet, your `@instruct` text, and your `@avatar` persona. Facts the bot learned before you authenticated are not lost.

Two consequences:

- Memories and candidates merge. An instruction or a persona already set on the account wins, and the nick's copy is dropped.
- Facts saved under a nick you never authenticate with stay attached to the nick until you authenticate on it.

`@memories`, `@instruct`, and `@avatar` need an authenticated account, and the bot reads neither your memories nor your instruction while you aren't authenticated. Holding someone's nick doesn't give you their data.

## Custom instructions

Set a persistent instruction that shapes how the bot answers you. On every path but one it rides along as a user-role message fenced in `<user_instruction>` markers rather than as part of the system prompt, so it steers the reply but cannot override the bot's identity or its safety rules.

```
@instruct Always respond in French
@instruct You are Captain Picard. Respond in character.
@instruct           # show your current instruction
@instruct clear     # remove it
```

Setting one is command-only. Asking the bot in conversation to change how it answers shapes that conversation and nothing more; `@instruct` is what makes it stick.

The instruction applies to `@ask`, to plain messages addressed to the bot, to `@rp` and verse turns, to the planning step of `@code`, and to tasks the bot runs for you from a reminder or a schedule — on the scheduled-task path it is folded into the system prompt instead. `@draw` ignores it, and ignores your saved memories too.

## Avatar persona

Your avatar has its own voice in the verse, separate from `@instruct`:

```
@avatar A moss-covered tree spirit who speaks in riddles.
@avatar           # show your current persona
@avatar clear     # remove it
```

The persona shapes how the verse portrays your avatar, and it colours `@story` pages in any channel, verse-enabled or not. It never affects `@ask`. See [The verse](ai-commands.md#the-verse) for the other verse commands.
