# Limnoria crash course

VibeBot is a [Limnoria](https://docs.limnoria.net/) bot, so you run it from IRC with Limnoria's own commands. This page covers the commands you need to change the bot's settings: reading, setting, clearing, per-channel overrides and who is allowed to do each. The full list of VibeBot settings is in [Configuration](configuration.md).

Every example uses the `@` prefix. `VibeBot: config ...` works too.

## Finding your way around

```
@list                       # loaded plugins
@list LLM                   # commands in one plugin
@help config channel        # the exact syntax of any command
@apropos remind             # commands whose NAME contains a word
@whoami                     # which bot account you are logged in as
```

When you are not sure of a command's syntax, run `@help` on it first. The bot replies with the real signature.

## Who can change what

Limnoria checks a **capability** on your bot account before every change.

| Change | Capability needed |
|--------|-------------------|
| A global setting (`@config <key> <value>`) | `owner` |
| A setting for one channel (`@config channel ...`) | `#channel,op` for that channel, or `owner` |
| Load or reload plugins, `@flush` | `owner` |
| Grant capabilities to users | `admin` (only capabilities you hold yourself) |

Reading a setting needs no capability, except for values marked private, such as passwords.

If the bot answers `You don't have the owner capability`, it does not know who you are. Log in by private message, never in a channel:

```
/msg VibeBot identify <account> <password>
```

Accounts set up with a hostmask or a network account are recognised automatically. Check with `@whoami`.

## Setting names

A setting is a dotted path in Limnoria's registry. VibeBot's settings live under `plugins.LLM`, for example `plugins.LLM.assistantModel`. The leading `supybot.` is optional.

```
@config list plugins.LLM            # every LLM setting
@config search model                # settings whose name contains "model"
@config searchhelp prompt           # settings whose description mentions "prompt"
@config help plugins.LLM.bridgeEnabled   # what it does, plus its current value
```

In `@config list` output, a leading `#` marks a setting that can differ per channel, `:` one that can differ per network, and `@` a group with more settings under it.

Names are not case-sensitive, but they must exist: copy them from `@config list` or [Configuration](configuration.md) rather than guessing.

## Reading a value

```
@config plugins.LLM.assistantModel            # the current value
@config default plugins.LLM.assistantModel    # the value shipped with the code
```

Run in a channel, `@config` on a per-channel setting shows both values at once:

```
Global: anthropic/claude-haiku-5-5; #chat @ afternet: xai/grok-4.3
```

To read another channel's value, name it: `@config channel #chat plugins.LLM.assistantModel`.

## Setting a value globally

```
@config plugins.LLM.assistantModel anthropic/claude-haiku-5-5
```

A global value applies to **every channel without its own override**. It does not touch channels that have one. To change just one channel, use the channel form below.

Value types:

- **Booleans** take `True` or `False` (also `on`/`off`).
- **Numbers** are plain digits: `@config plugins.LLM.contextTimeoutMinutes 10`.
- **Text** takes everything after the name, spaces included. Wrap it in double quotes if it contains square brackets, which Limnoria otherwise runs as a nested command.
- **Space-separated lists** take the whole list at once, which replaces the old list: `@config channel #chat plugins.LLM.bridgeAllowedPlugins Seen Time Weather`.

The bot checks the value's type before saving and says `Error: ...` when it is wrong. Any other reply means it saved.

## Channel-specific values

```
@config channel #chat plugins.LLM.assistantModel xai/grok-4.3
```

Run from inside the channel, you can leave the channel out:

```
@config channel plugins.LLM.bridgeEnabled True
```

Several channels at once take a comma-separated list: `@config channel #chat,#dev plugins.LLM.pendingTasksEnabled True`.

Precedence: a channel value beats a network value, which beats the global value. A channel with no value of its own follows the global one, and keeps following it when the global one changes later.

Only settings marked *channel* in [Configuration](configuration.md) accept this form. Others answer `That configuration variable is not a channel-specific configuration variable.`

## Clearing and resetting

There are three different "undo" operations. Pick the right one:

| You want | Command |
|----------|---------|
| A channel to stop overriding and follow the global value again | `@config reset channel #chat plugins.LLM.assistantModel` |
| The global value back to the shipped default | `@config setdefault plugins.LLM.assistantModel` |
| A text or list setting set to empty | `@config plugins.LLM.memoryModel ""` |

Things that catch people out:

- `setdefault` changes the **global** value only. A channel override stays in place; remove it with `reset channel`.
- Setting a channel's value to the global one with `@config channel` is not the same as resetting it. The channel then keeps that value when the global one changes later.
- `""` empties a value; it does not reset it. For settings whose empty value means "fall back", such as `memoryModel` and `verseModel` falling back to `assistantModel`, empty is what you want. For `assistantSystemPrompt`, empty means no personality prompt at all; use `setdefault` or `reset channel` instead to get the shipped prompt back.
- `reset channel` takes the channel before the name, and defaults to the current channel: `@config reset channel plugins.LLM.bridgeEnabled`.

## Making a change stick

On this deployment the bot is restarted on every deploy, and changes made with `@config` have been lost on restart more than once. After any change you want to keep, write the registry to disk:

```
@flush
```

Then read the value back with `@config` to confirm it. A quiet log does not prove a setting took effect.

Editing `bot.conf` by hand is for when the bot is down. Stop the bot first, or the running bot overwrites the file. See [Editing bot.conf safely](operations.md#editing-botconf-safely).

## Settings people ask about most

| Setting | Scope | What it does |
|---------|-------|--------------|
| `assistantModel` | channel | The chat model, for example `anthropic/claude-haiku-5-5` or `xai/grok-4.3`. Only `anthropic/` and `xai/` models work |
| `assistantSystemPrompt` | channel | The bot's personality and house rules for a channel |
| `bridgeEnabled` | channel | Lets the bot run allowlisted Limnoria commands as a tool. `@load` the plugins first; see [Bridge tools](../reference/bridge-tools.md#loading-the-plugins) |
| `pendingTasksEnabled` | channel | Lets the bot set reminders and scheduled tasks from plain language |
| `privacyOptInOnly` | channel | `True` sends only lines addressed to the bot to the provider |
| `contextTimeoutMinutes` | channel | Minutes of quiet before the bot forgets the conversation |
| `verseEnabled` | channel | Turns the shared-fiction verse on |
| `statusAnnounce` | channel | Announces service-status incidents in the channel. Remove any RSS feed for the same page first, or incidents post twice |
| `auroraAnnounce` | channel | Announces aurora storm watches in the channel |

Examples:

```
@config channel #chat plugins.LLM.assistantSystemPrompt You are terse and dry. Never use emoji.
@config reset channel #chat plugins.LLM.assistantSystemPrompt
@config channel #chat plugins.LLM.pendingTasksEnabled True
@config plugins.LLM.assistantModel anthropic/claude-haiku-5-5
@flush
```

After a model change, `@model #chat` shows the model the channel is set to and the one that answered its last reply.

A user's own `@instruct` instruction is not a registry setting; users set and clear it themselves. See [Memory and instructions](../user/memory.md).

## Turning commands off

Disable a command or a whole plugin in one channel (needs `#channel,op`):

```
@channel disable #chat LLM draw     # just @draw
@channel disable #chat LLM          # every LLM command
@channel enable #chat LLM draw
```

To block a command for everyone, or for one person, use capabilities:

```
@defaultcapability add -llm.draw            # nobody gets llm.draw by default
@admin capability add <account> llm.draw    # except this account
@channel capability add #chat <account> -llm.ask   # one user, one channel
@user capabilities <account>                # what an account holds
```

An account's own capability beats the default, and `owner` passes every check. The `llm.*` capabilities and how to lock down the verse are in [Rate limiting and security](rate-limiting-security.md).

## Plugins

```
@load Seen          # load a stock plugin
@reload LLM         # reload a plugin's code
@unload Seen
```

These need `owner`. On this deployment new code arrives by image deploy, and `@reload LLM` does not reload the plugin's inner modules, so leave reloading to the deploy.

## More

- [Configuration](configuration.md): every VibeBot setting, its scope and default
- [Command reference](../reference/commands.md): every VibeBot command
- [Limnoria's configuration guide](https://docs.limnoria.net/use/configuration.html) and [capabilities guide](https://docs.limnoria.net/use/capabilities.html)
