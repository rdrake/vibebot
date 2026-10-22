# Plugins

VibeBot v8 is a [Limnoria](https://docs.limnoria.net/) workspace that
ships two first-party plugins. Load either one from IRC with
`@load <name>`.

| Plugin | What it does |
|--------|--------------|
| [LLM](llm.md) | The main AI surface: `@ask`, `@draw`, `@story`, memory, reminders, scheduled tasks, the verse roleplay engine, and the Limnoria-as-tools bridge. |
| [NickInMiddle](nickinmiddle.md) | A small `inFilter` plugin that moves the bot's nick to the front when it appears mid-sentence, so the LLM plugin's addressing check picks the message up. |

Both plugins live in the same uv workspace (`plugins/<name>/`) and share
linting, type checking, and the 93% test coverage floor.

## Stock plugins first

The project follows one guiding principle: defer to stock Limnoria
plugins wherever possible. The LLM plugin acts as a natural-language
shim over them rather than reimplementing their features. When a user
asks "have you seen alice?" or "what time is it in Tokyo?", the
assistant calls the stock `Seen` or `Time` plugin through the
[tool bridge](../reference/bridge-tools.md) instead of answering from
the model alone.

The bridge stays off until an operator sets `bridgeEnabled` for the
channel. Once on, it exposes a curated read-safe set — Misc, Time,
Math, Utilities, Seen, Web, Later, Note, Karma, QuoteGrabs, RSS, DDG,
Network —
unless `bridgeAllowedPlugins` names a different selection.
