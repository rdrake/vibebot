# Installation and deployment

To ask a chat assistant about setup, give it [llms-full.txt](https://rdrake.github.io/vibebot/llms-full.txt): this whole guide as one Markdown file. [llms.txt](https://rdrake.github.io/vibebot/llms.txt) is the page index.

## Prerequisites

You need one of:

- **Docker** (recommended): a systemd deployment needs the binary at `/usr/bin/docker`, which the unit calls directly.
- **Python 3.14 or newer** with [uv](https://docs.astral.sh/uv/): for manual and development installs.

VibeBot is a [Limnoria](https://docs.limnoria.net/) plugin. If you have never used Limnoria before, follow the [Limnoria getting started guide](https://docs.limnoria.net/use/getting_started.html) to create your initial `bot.conf`. You need a working `bot.conf` before proceeding.

## Docker deployment (recommended)

Pull the latest image:

```bash
docker pull ghcr.io/rdrake/vibebot:latest
```

Run the container, mounting your configuration and data directories:

```bash
docker run --rm --name vibebot \
    --user $(id -u):$(id -g) \
    --env-file /path/to/config/env \
    -v /path/to/config:/config \
    -v /var/www/llm:/var/www/llm \
    -w /config \
    ghcr.io/rdrake/vibebot:latest
```

The `/config` volume holds your `bot.conf` and its `conf/`, `data/`, and `logs/` directories. In a hand-typed `docker run` the `/var/www/llm` mount is optional: add it only when you point `httpRoot` there for an external web server to serve. The shipped systemd unit always mounts it (see below). `httpRoot` is empty by default, which keeps the `@draw` and `@story` output inside `/config` under `data/web/llm/` — see [Configuration → Web and HTTP output](configuration.md#web-and-http-output).

`--env-file` is not optional. Provider API keys are read from the environment at call time and have no `bot.conf` equivalent, so without it the bot connects to IRC and every LLM command answers `no API key configured for provider`. Start from `.env.example` in the repository — it lists the two provider keys (`ANTHROPIC_API_KEY` and `XAI_API_KEY`) and the format rules Docker's `--env-file` parser enforces. See [Configuration → API keys](configuration.md#api-keys).

## systemd service

The repository includes a systemd user service for production deployments. The target lives in the repository's `Makefile`, so clone first even if you are deploying the published image:

```bash
git clone https://github.com/rdrake/vibebot.git
cd vibebot
make install-service
```

This copies the unit file, creates `~/.config/vibebot/` and seeds its `env` file. The bot creates `conf/`, `data/`, `logs/` and `backup/` inside it on first start:

| Path | Purpose |
|------|---------|
| `~/.config/systemd/user/vibebot.service` | systemd unit |
| `~/.config/vibebot/` | `bot.conf`, the `env` file, and the bot's `conf/`, `data/`, `logs/` and `backup/` directories |

The unit runs the Docker image with `~/.config/vibebot` mounted at `/config`, passes the same directory's `env` file to `docker run --env-file`, and pulls the latest image on every start. It also always mounts `/var/www/llm`, even when `httpRoot` does not use it; if that directory does not exist on the host, Docker creates it, empty and owned by root. To drop the mount, override `ExecStart` in a drop-in or edit the copied unit.

After installing, complete the setup:

<div class="vb-steps" markdown>

1. **Copy your `bot.conf`** to `~/.config/vibebot/bot.conf`.

2. **Fill in `~/.config/vibebot/env`.** `make install-service` seeds it from `.env.example`, so it arrives full of placeholders: enter the providers you use and delete the lines for the rest, since a leftover `your-xai-key-here` counts as a real key. Keep to the format rules noted in that file: `NAME=value`, no quotes, no trailing comments, no spaces around `=`, LF line endings. Docker's `--env-file` parser rejects a malformed line by exiting the container on start, and `Restart=always` turns that into a silent crashloop. The file holds secrets, so `chmod 600` it.

3. **Enable and start the service**, and turn on lingering so it keeps running after you log out:

    ```bash
    systemctl --user enable vibebot
    systemctl --user start vibebot
    loginctl enable-linger $USER
    ```

4. **Install the update timer** once the bot runs; see [Auto-updates](#auto-updates).

</div>

### First start

Read the log as the bot comes up, and check each of these:

```bash
journalctl --user -u vibebot -f
```

- The unit's `docker pull` finishes and Limnoria logs `Connecting to <server>:<port>.`
- The LLM plugin logs one line naming the key variables it found, never their values:

    ```
    secret redaction: 4 handler(s) filtered, 2 variable(s) covered: ANTHROPIC_API_KEY, XAI_API_KEY
    ```

    It confirms the filter that scrubs key values out of every log line is installed. A key you set but that is missing from the list did not reach the container (or is shorter than 16 characters), and `none` means no key did. The list proves a variable is present, not that its key works: a leftover placeholder is listed too. The line is logged at `INFO` while the plugin loads, so it appears only when `supybot.log.level` and `supybot.log.stdout.level` are both `INFO` or lower, as they are by default, and not again on a `@reload`. Credentials also sit in the container's environment; see [Operations → Credentials in the environment](operations.md#credentials-in-the-environment).
- `@list LLM` in a channel lists `ask`, `draw`, `story` and the rest.
- `@ask hi` gets an answer. Chat runs on `anthropic/claude-haiku-5-5` by default, so this exercises `ANTHROPIC_API_KEY`.
- `@draw a lighthouse` returns a link. Images run on xAI, so this exercises `XAI_API_KEY`. `@draw` needs a logged-in network account, and changing settings later needs the owner login described in [Who can change what](limnoria.md#who-can-change-what).

If any of these fails, see [Troubleshooting](troubleshooting.md).

### Auto-updates

Install the update timer to pull new Docker images automatically (checks every 15 minutes):

```bash
make install-timer
```

!!! warning "The timer starts the bot"

    `make install-timer` enables and starts the timer at once. On its next quarter-hour tick the updater finds no running `vibebot` container, treats that as an outdated image, and runs `systemctl --user restart vibebot`, which starts the bot even if you never enabled it. On a new host, install the timer only after `bot.conf` and `env` are filled in. `make install-deploy` installs the service and the timer together, so use it only on a host that already has both files.

See [Operations](operations.md) for how the two update paths work together.

## Manual and development setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/rdrake/vibebot.git
cd vibebot
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
| `make check` | Lint, format check, typecheck, and the fast test suite |
| `make preflight` | Format, then run the full `check` gate |
| `make docs-serve` | Preview this documentation site locally |

## Verifying the installation

For systemd deployments, check the service status and walk through [First start](#first-start):

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

Anything else that goes wrong is covered in [Troubleshooting](troubleshooting.md).
