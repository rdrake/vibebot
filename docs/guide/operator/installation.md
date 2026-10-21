# Installation and deployment

To ask a chat assistant about setup, give it [llms-full.txt](https://rdrake.github.io/vibebot-v8/llms-full.txt): this whole guide as one Markdown file. [llms.txt](https://rdrake.github.io/vibebot-v8/llms.txt) is the page index.

## Prerequisites

You need one of:

- **Docker** (recommended): a systemd deployment needs the binary at `/usr/bin/docker`, which the unit calls directly.
- **Python 3.14 or newer** with [uv](https://docs.astral.sh/uv/): for manual and development installs.

VibeBot is a [Limnoria](https://docs.limnoria.net/) plugin. If you have never used Limnoria before, follow the [Limnoria getting started guide](https://docs.limnoria.net/use/getting_started.html) to create your initial `bot.conf`. You need a working `bot.conf` before proceeding.

## Docker deployment (recommended)

Pull the latest image:

```bash
docker pull ghcr.io/rdrake/vibebot-v8:latest
```

Run the container, mounting your configuration and data directories:

```bash
docker run --rm --name vibebot \
    --user $(id -u):$(id -g) \
    --env-file /path/to/config/env \
    -v /path/to/config:/config \
    -v /var/www/llm:/var/www/llm \
    -w /config \
    ghcr.io/rdrake/vibebot-v8:latest
```

The `/config` volume holds your `bot.conf` and its `conf/`, `data/`, and `logs/` directories. The `/var/www/llm` mount is optional: add it only when you point `httpRoot` there for an external web server to serve. `httpRoot` is empty by default, which keeps the `@draw` and `@story` output inside `/config` under `data/web/llm/` — see [Configuration → Web and HTTP output](configuration.md#web-and-http-output).

`--env-file` is not optional. Provider API keys are read from the environment at call time and have no `bot.conf` equivalent, so without it the bot connects to IRC and every LLM command answers `no API key configured for provider`. Start from `.env.example` in the repository — it lists the four provider keys, the optional ones, and the format rules Docker's `--env-file` parser enforces. See [Configuration → API keys](configuration.md#api-keys).

## systemd service

The repository includes a systemd user service for production deployments. The target lives in the repository's `Makefile`, so clone first even if you are deploying the published image:

```bash
git clone https://github.com/rdrake/vibebot-v8.git
cd vibebot-v8
make install-service
```

This copies the unit file and creates the directory layout:

| Path | Purpose |
|------|---------|
| `~/.config/systemd/user/vibebot.service` | systemd unit |
| `~/.config/vibebot/` | `bot.conf`, the `env` file, and the bot's `conf/`, `data/`, `logs/` and `backup/` directories |

The unit runs the Docker image with `~/.config/vibebot` mounted at `/config`, passes the same directory's `env` file to `docker run --env-file`, and pulls the latest image on every start.
After installing, complete the setup:

1. Copy your `bot.conf` to `~/.config/vibebot/bot.conf`.
2. Edit `~/.config/vibebot/env`. `make install-service` seeds it from `.env.example`, so it arrives full of placeholders: enter the providers you use and delete the lines for the rest, since a leftover `your-xai-key-here` counts as a real key. Keep to the format rules noted in that file — `NAME=value`, no quotes, no trailing comments, no spaces around `=`, LF line endings. Docker's `--env-file` parser rejects a malformed line by exiting the container on start, and `Restart=always` turns that into a silent crashloop.
3. Enable and start the service:

```bash
systemctl --user enable vibebot
systemctl --user start vibebot
loginctl enable-linger $USER   # keeps the service running after logout
```

### Auto-updates

Install the update timer to pull new Docker images automatically (checks every 15 minutes):

```bash
make install-timer
```

Or install both the service and the timer at once:

```bash
make install-deploy
```

See [Operations](operations.md) for how the two update paths work together.

## Manual and development setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/rdrake/vibebot-v8.git
cd vibebot-v8
make install
```

Start the bot:

```bash
make run
```

This runs `uv run limnoria bot.conf` from the repository root. You need a `bot.conf` in the working directory.

For development, also install the pre-commit hooks:

```bash
make install-hooks
```

Useful development targets:

| Target | What it does |
|--------|--------------|
| `make test` | Run the fast test suite with coverage (93% floor) |
| `make test-all` | Run every test, including slow ones |
| `make lint` | Run `ruff check` |
| `make typecheck` | Run `ty check` on both plugins |
| `make check` | Lint, format check, typecheck, syntax check, and tests |
| `make preflight` | Format, then run the full `check` gate |
| `make docs-serve` | Preview this documentation site locally |

## Verifying the installation

For systemd deployments, check the service status:

```bash
systemctl --user status vibebot
```

For Docker, check the running container:

```bash
docker ps | grep vibebot
```

Once the bot connects to IRC, verify the plugin loaded:

```
@list LLM
```

This lists the available commands (`ask`, `draw`, `story`, and so on). If the plugin is not loaded, load it from IRC:

```
@load LLM
```
