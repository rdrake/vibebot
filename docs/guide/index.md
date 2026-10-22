# VibeBot v8

AI on IRC. Ask questions, create images, and hold conversations with large language models without leaving your IRC client.

```
<alice> @ask What causes the northern lights?
<VibeBot> The northern lights (aurora borealis) happen when charged particles
          from the sun collide with gases in Earth's atmosphere. The particles
          follow magnetic field lines toward the poles, exciting nitrogen and
          oxygen molecules that release photons as they return to their ground
          state. Green comes from oxygen, purple and blue from nitrogen.
<alice> @ask How far south can you see them?
<VibeBot> During strong solar storms, auroras have been visible as far south
          as Texas and Florida (around 30°N). Typically you need to be above
          50°N for regular sightings: northern Canada, Iceland, Scandinavia,
          and Alaska.
```

## Features

- **Natural-language interaction.** Mention the bot by name or send it a PM. Ask questions, search the web, pull a page, draw an image, or tell it something to remember — no command needed. Managing memories, instructions, usage, and conversation context stays command-only.
- **Conversation memory.** Follow up on earlier questions. The bot tracks context per user and channel.
- **Vision.** Drop an image URL into `@ask` and the bot describes or reasons about it.
- **Image generation.** `@draw` creates images from text descriptions.
- **Illustrated stories.** `@story` builds an illustrated story or explainer page and posts the link.
- **[Aurora](user/aurora.md).** Ask whether the northern lights are likely at a place, or have a channel told when NOAA forecasts a strong storm.
- **[Reminders](user/reminders.md) and [scheduled tasks](user/scheduled-tasks.md).** Natural-language reminders, plus recurring LLM runs with the full native tool set.
- **Stored facts.** The bot picks up durable facts from your conversations, with [two-stage promotion](operator/memory-promotion.md) so casual remarks don't stick. Say "remember that I use Arch" and it saves immediately; review or prune the list with `@memories`.
- **[The verse](operator/verse.md).** A per-channel world model with avatars, user-driven roleplay, and a persistent entity graph.
- **[Limnoria bridge](reference/bridge-tools.md).** The LLM can call stock Limnoria plugin commands (Time, Math, Seen, Web, Karma, RSS, and so on) as tools, read-only: write commands are never exposed.
- **Custom instructions.** Shape how the bot responds to you with `@instruct`.
- **Two providers.** Chat runs on Anthropic or xAI models, chosen per channel; images and web search run on xAI.

## Documentation

**[User guide](user/getting-started.md)**: start here. Learn the commands and how to get the most out of the bot.

**[Operator guide](operator/installation.md)**: deploy and configure VibeBot for your network.

**[Command reference](reference/commands.md)**: every command, its syntax, and its permissions.

## Links

- [GitHub](https://github.com/rdrake/vibebot-v8)
- [Limnoria](https://github.com/ProgVal/Limnoria)
