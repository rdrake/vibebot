# VibeBot v8

An AI-powered IRC bot for AfterNet, built on [Limnoria](https://github.com/ProgVal/Limnoria) with multi-provider model support through [LiteLLM](https://github.com/BerriAI/litellm).

The full guide lives at **[rdrake.github.io/vibebot-v8](https://rdrake.github.io/vibebot-v8/)**: a user guide, an operator guide, and a command reference.

## Features

- **Multi-provider AI**: OpenAI, Anthropic, Google Gemini, and xAI Grok behind one LiteLLM interface, with per-channel model overrides
- **Assistant with tools**: `@ask` answers with conversation context, vision for image URLs, web search, and a bounded tool loop
- **Code generation**: `@code` writes code and posts it as an HTTP link
- **Image generation**: `@draw` renders images with xAI grok-imagine (`xai/grok-imagine-image` by default, on `XAI_API_KEY`) or Google Imagen; Vertex AI Imagen works too, but needs `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION`
- **Video generation**: `@animate` renders short clips on a self-hosted video box, reports your place in the queue, and posts the link when the clip lands; `@renders` shows or clears the queue
- **Illustrated pages**: `@story` builds an illustrated story or explainer and posts the link
- **Two memory layers**: volatile conversation context that expires on its own, plus durable per-user facts — picked up automatically once they recur, or saved at once when you say "remember this"
- **Reminders and scheduled tasks**: `@remind` parses natural language and handles recurrence; letting the model create them from `@ask` is opt-in per channel via `pendingTasksEnabled`
- **Verse mode**: an opt-in persistent fiction layer with avatars, a world store, and canon editing
- **Abuse controls**: capability checks, account gating, tiered rate limits, and bounded LLM concurrency
- **NickInMiddle**: a companion plugin that recognises the bot's nick mid-sentence, so "can you, vibebot, help?" works like addressed speech

## Quick start

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

make install        # uv sync
make install-hooks  # prek git hooks
make run            # start Limnoria with bot.conf
```

Set API keys as environment variables, one per provider (see `.env.example`):

```bash
export XAI_API_KEY=...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Commands

| Command | Description |
|---------|-------------|
| `@ask <question>` | Ask the assistant; supports context, vision, and tools |
| `@code <request>` | Generate code, delivered as an HTTP link |
| `@draw <prompt>` | Generate an image (authenticated account required) |
| `@story <brief>` | Generate an illustrated story or explainer page |
| `@forget [channel]` | Clear your conversation context |
| `@memories [subcommand]` | List, edit, or delete your stored facts |
| `@instruct [text \| clear]` | Set a persistent instruction for `@ask` |
| `@remind [text \| list \| del \| clear]` | Set and manage reminders |
| `@usage [nick \| #channel]` | View API usage statistics |

Verse channels add more: `@verseopt`, `@rp`, `@verse`, `@look`, `@who`, `@avatar`, and editor and GM commands. The [command reference](https://rdrake.github.io/vibebot-v8/reference/commands/) covers the full list.

## Configuration

Models and behaviour live in the Limnoria registry under `supybot.plugins.LLM.*`. Most keys support per-channel overrides. API keys are the exception: they come from environment variables, not the registry (see [Quick start](#quick-start)).

```
supybot.plugins.LLM.assistantModel: gemini/gemini-flash-latest
supybot.plugins.LLM.codeModel: gemini/gemini-flash-latest
supybot.plugins.LLM.imageModel: gemini/imagen-4.0-fast-generate-001
```

See the [LiteLLM provider list](https://docs.litellm.ai/docs/providers) for supported models and the [operator configuration guide](https://rdrake.github.io/vibebot-v8/operator/configuration/) for every key, including rate limits, memory, context, and verse settings.

## Production deployment

Production runs the Docker image `ghcr.io/rdrake/vibebot-v8:latest` as a systemd user service named `vibebot`. The service mounts `~/.config/vibebot` at `/config` for `bot.conf`, the env file, and runtime data.

```bash
make install-service   # scaffold the unit, config dir, and env file
```

Follow the printed instructions to copy `bot.conf` into place and enable the service. `ExecStartPre` pulls the latest image, so every restart also updates.

### Updates

New code reaches production two ways:

1. **CI auto-deploy**: after a push to `main` passes CI and the Docker image publishes, the deploy step connects over SSH and restarts the service.
2. **Updater timer** (fallback): `make install-timer` installs `vibebot-updater.timer`, which polls GHCR every 15 minutes and restarts the service when the image digest changes. With CI auto-deploy in place it only matters for out-of-band image pushes or a host that was offline during the push.

```bash
systemctl --user status vibebot-updater.timer      # check timer status
journalctl --user -u vibebot-updater.service -f    # view update logs
make uninstall-timer                               # disable auto-updates
```

## Static assets (reverse proxy)

Generated code and images serve over HTTP. Set the public URL the bot should advertise:

```
@config supybot.servers.http.publicUrl https://example.com
```

The bot then generates URLs such as `https://example.com/llm/filename.py`.

With Limnoria's built-in HTTP server, proxy `/llm` through Nginx:

```nginx
location /llm/ {
    proxy_pass         http://127.0.0.1:8080/llm/;
    proxy_set_header   Host $host;
    proxy_set_header   X-Forwarded-For $remote_addr;
    proxy_read_timeout 30s;

    # Generated artifacts are immutable (filenames are content-hashed).
    add_header Cache-Control "public, max-age=31536000, immutable";
}
```

If `httpRoot` points at a directory the web server owns, serve it directly:

```nginx
location /llm/ {
    alias       /var/www/llm/;
    autoindex   off;
    add_header  Cache-Control "public, max-age=31536000, immutable";
}
```

The same with Caddy:

```caddyfile
example.com {
    handle /llm/* {
        reverse_proxy 127.0.0.1:8080
        header Cache-Control "public, max-age=31536000, immutable"
    }
}
```

## Development

```bash
make test         # pytest with a 93% coverage floor (skips slow tests)
make test-all     # full suite, including slow tests
make lint         # ruff check
make format       # ruff format
make typecheck    # ty
make syntax-check # Python 3.12 to 3.14 compatibility
make check        # lint + format-check + typecheck + syntax-check + test
make preflight    # format, then check: run before calling work done
make docs         # build the MkDocs guide
```

The toolchain: [uv](https://github.com/astral-sh/uv) for packaging, Ruff for lint and format, ty for types, pytest with Hypothesis property tests, prek pre-commit hooks with gitleaks secret scanning, Vale for the docs, and Dependabot for weekly dependency updates.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Repository layout

```
vibebot-v8/
├── plugins/llm/             # Main AI plugin
│   ├── src/llm/             # plugin.py, service.py, assistant.py, config.py,
│   │   └── verse/           # verse world store, avatars, compaction, tooling
│   └── tests/               # unit + Hypothesis property tests
├── plugins/nickinmiddle/    # Mid-sentence nick addressing (inFilter)
├── go/                      # v9 Go rewrite (in progress)
├── docs/guide/              # MkDocs source for the published guide
├── docs/plans/              # Design and implementation plans
├── scripts/                 # Maintenance and compatibility scripts
├── bot.conf                 # Development bot configuration
└── pyproject.toml           # Workspace and tool configuration
```

`plugins/llm/README.md` describes the plugin internals for developers.

## Licence

See the LICENSE file for details.

## Credits

- Built with [Limnoria](https://github.com/ProgVal/Limnoria)
- Powered by [LiteLLM](https://github.com/BerriAI/litellm)
- Developed for AfterNet IRC (irc.afternet.org)
