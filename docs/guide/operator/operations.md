# Operations and auto-updates

This page covers day-to-day operation of a deployed bot: updates, restarts, logs, configuration changes, and housekeeping.

## How updates reach production

Two independent paths keep the running bot current. Both end in the same place: a restart that pulls the newest image.

**Path 1: CI push deploy.** A push to `main` runs CI. When CI passes, a second workflow builds the Docker image, pushes it to `ghcr.io/rdrake/vibebot-v8:latest`, and then connects to the production host over SSH with a restricted deploy key. The remote key runs one forced command: `systemctl --user restart vibebot`. The service unit pulls the latest image on start, so the restart deploys the new build.

A push that touches only `docs/**`, `*.md`, or `mkdocs.yml` is exempt. CI's `paths-ignore` skips those, and because `docker.yml` triggers on CI success, no image is built and the running bot is never restarted. Documentation still publishes: `pages.yml` has its own trigger on `docs/**` and `mkdocs.yml`. A push that changes only a root-level `*.md` fires nothing at all.

**Path 2: updater timer.** `vibebot-updater.timer` fires every 15 minutes. Its service compares the local image digest against the registry and restarts `vibebot` when they differ — including when the bot is stopped, because the local lookup then returns nothing (see [Editing bot.conf safely](#editing-botconf-safely)). This path catches anything the push deploy missed, such as a host that was offline during the push.

```
git push to main → CI → Docker build → GHCR
                              ↓                ↓
                   SSH restart (immediate)   updater timer (≤15 min)
                              ↓                ↓
                        systemctl --user restart vibebot
                              ↓
                     ExecStartPre pulls :latest
```

The source code never needs to live on the server. The Docker image has everything; the server holds only configuration and data.

## Files involved

| File | Purpose |
|------|---------|
| `vibebot.service` | Runs the Docker container; pulls the latest image on start |
| `vibebot-updater.service` | Compares image digests; restarts the bot when they differ |
| `vibebot-updater.timer` | Triggers the update check every 15 minutes |

Install all three with `make install-deploy`. See [Installation](installation.md) for the full setup walkthrough.

## Quick reference

| Task | Command |
|------|---------|
| Check status | `systemctl --user status vibebot` |
| View logs | `journalctl --user -u vibebot -f` |
| View recent logs | `journalctl --user -u vibebot -n 100` |
| Manual restart | `systemctl --user restart vibebot` |
| Force an update check | `systemctl --user start vibebot-updater` |
| Check timer status | `systemctl --user status vibebot-updater.timer` |
| Stop the bot | `systemctl --user stop vibebot` |
| Start the bot | `systemctl --user start vibebot` |

A stopped bot does not stay stopped: the updater timer restarts it within 15 minutes. Stop `vibebot-updater.timer` as well when you need the bot down for longer than that — see below.

## Editing bot.conf safely

Limnoria flushes its in-memory registry to `bot.conf` on shutdown. If you edit the file while the bot is running, the shutdown flush overwrites your changes.

The updater timer will restart the bot out from under you, so stop it first. `vibebot-updater.service` reads the local digest off the *container* with `docker inspect ... vibebot`, and the container runs `--rm`, so while the bot is stopped that lookup falls back to `none`. It never matches the registry digest, and the timer restarts the bot on its next 15-minute tick — partway through your edit.

Always follow this order:

```bash
systemctl --user stop vibebot-updater.timer
systemctl --user stop vibebot
# edit ~/.config/vibebot/bot.conf
systemctl --user start vibebot
systemctl --user start vibebot-updater.timer
```

For single settings, prefer the `@config` command from IRC instead; it changes the live registry and survives restarts. Reserve direct file edits for bulk changes and values that are awkward to type in IRC.

## Log locations

| Log | Command |
|-----|---------|
| Service logs | `journalctl --user -u vibebot` |
| Update logs | `journalctl --user -u vibebot-updater` |
| Bot message logs | `~/.config/vibebot/logs/messages.log` |

Container output is also available through `docker logs vibebot`.

## Housekeeping

The repository ships two maintenance scripts:

| Script | Purpose |
|--------|---------|
| `scripts/prune_backups.py` | Keep the newest `--keep` of Limnoria's rolling `bot.conf.backup.*` files |
| `scripts/rotate_logs.py` | Size-based rotation of `messages.log`, keeping the last `--keep` copies |

Both default to repository-relative paths, so on a production host give them the real directories:

```bash
python3 scripts/rotate_logs.py --log ~/.config/vibebot/logs/messages.log --keep 7
python3 scripts/prune_backups.py --dir ~/.config/vibebot/backup --keep 20
```

Add `--dry-run` to either one to see what it would touch. Run them from cron or a systemd timer on the host if your deployment accumulates large logs or backups. Unlike the bot itself, these two run on the host rather than in the container. Both are stdlib-only, so copying the single file across and running it with `python3` is enough — no clone, no dependency install.

## Troubleshooting

### Service fails to start

Check that Docker runs, then read the service log:

```bash
docker info
journalctl --user -u vibebot -n 50 --no-pager
docker images | grep vibebot
```

### Timer is not running

```bash
systemctl --user list-timers
systemctl --user enable --now vibebot-updater.timer
```

### Updates are not applying

Force a check and read the updater log:

```bash
systemctl --user start vibebot-updater
journalctl --user -u vibebot-updater -n 20 --no-pager
docker pull ghcr.io/rdrake/vibebot-v8:latest
```

### Service stops after logout

Enable lingering for your user:

```bash
loginctl enable-linger $USER
```

### Configuration changes are not applied

Restart the service after editing configuration, and remember the stop-edit-start order for `bot.conf`:

```bash
systemctl --user restart vibebot
```

## Rolling back a deploy

Rollback is manual. A revert push through CI takes 10-20 minutes end to end, and `docker.yml` only builds after `CI` succeeds across a three-version Python matrix (3.12, 3.13, 3.14) — one flaky test blocks the image from ever publishing. When production needs to go back to a known-good build now, pin the previous image directly rather than waiting on a revert.

Every image build is tagged with the commit SHA (`type=sha` in `docker.yml`, alongside `:latest`), so every build keeps a permanent `sha-<short-sha>` tag and stays pullable after `:latest` moves past it.

1. **Find the previous image tag.** If the bad build has not been restarted onto yet, read the commit off the still-running container:

   ```bash
   docker inspect vibebot --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
   ```

   That prints the full SHA; the tag is `sha-` plus its first seven characters. Do not use `{{.Config.Image}}` — the unit always runs `:latest`, so that only ever prints `:latest`. Once the bad build is live, take the SHA from the GitHub Actions run summary for the last known-good `Build and Push Docker Image` run instead.

2. **Stop the updater first.** `vibebot-updater.service` hardcodes `IMAGE=ghcr.io/rdrake/vibebot-v8:latest` and compares that digest against the running container's every 15 minutes. It cannot undo the pin — the step 3 drop-in wins over the unit's own `Environment=IMAGE` — but a pinned container never matches the `:latest` digest, so the timer restarts `vibebot` on every tick, dropping the bot off IRC each time.

   ```bash
   systemctl --user stop vibebot-updater.timer
   ```

3. **Pin the image with a systemd drop-in.** Overriding `Environment=IMAGE=...` for the `vibebot.service` unit points `ExecStartPre`'s `docker pull ${IMAGE}` and the `docker run` at the pinned tag instead of `:latest`.

   ```bash
   mkdir -p ~/.config/systemd/user/vibebot.service.d
   printf '[Service]\nEnvironment=IMAGE=ghcr.io/rdrake/vibebot-v8:sha-<PREV>\n' \
     > ~/.config/systemd/user/vibebot.service.d/override.conf
   systemctl --user daemon-reload && systemctl --user restart vibebot
   ```

   Replace `<PREV>` with the short SHA found in step 1.

4. **Roll forward again once the fix ships.** Remove the drop-in and restart the updater timer:

   ```bash
   rm ~/.config/systemd/user/vibebot.service.d/override.conf
   systemctl --user daemon-reload && systemctl --user restart vibebot
   systemctl --user start vibebot-updater.timer
   ```

## Credentials in the environment

API keys live in environment variables passed to the container via `--env-file` (see [Configuration → API keys](configuration.md#api-keys)). That moves where the credential is visible on the host:

- `docker inspect vibebot --format '{{.Config.Env}}'` prints every variable, values included, to anyone in the `docker` group.
- `/proc/<pid>/environ` for the container's process exposes the same values to anything with host-level access to that PID.
- Every child process the container spawns inherits the full environment, key values included, whether or not that process needs them.

Registry-stored keys were readable by anyone with owner-level `@config` access; the exposure surface has moved from "Limnoria owner capability" to "docker group and host process access." Scope access to the host and the `docker` group accordingly.

## Uninstalling

Remove the service and timer:

```bash
make uninstall-timer
make uninstall-service
```

Both targets leave `~/.config/vibebot/` alone: `bot.conf`, the `env` file, and the `conf/`, `data/`, `logs/`, and `backup/` subdirectories are preserved. `data/` holds `LLM.db` (conversations, memories, usage), the per-channel verse databases under `data/verse/`, and — while `httpRoot` is empty — the `@draw` and `@story` output under `data/web/llm/`. Delete that directory only when you are sure you no longer want any of it.

If you set `httpRoot`, that output lands there instead, outside `~/.config/vibebot/`; the shipped unit mounts `/var/www/llm` for that case, and removing the config directory leaves it behind.

Installs made before September 2026 also have an empty `~/.local/share/vibebot/{conf,data,logs}`. The container never mounted it, so it is safe to remove.
