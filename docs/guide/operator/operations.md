# Operations and auto-updates

This page covers day-to-day operation of a deployed bot: updates, restarts, logs, configuration changes, backups, and housekeeping. When something is wrong, start at [Troubleshooting](troubleshooting.md).

## How updates reach production

Two independent paths keep the running bot current. Both end in the same place: a restart that pulls the newest image.

<div class="vb-flow" markdown>

<div class="vb-flow__row"><span class="vb-flow__node">git push to <code>main</code></span><span class="vb-flow__arrow">→</span><span class="vb-flow__node">CI<small>lint · tests on 3.14, 3.15</small></span><span class="vb-flow__arrow">→</span><span class="vb-flow__node">Docker build<small>pushed to GHCR as <code>:latest</code> and <code>sha-…</code></small></span></div>
<div class="vb-flow__down">↓</div>
<div class="vb-flow__lanes"><div class="vb-flow__lane"><span class="vb-flow__label">Path 1 · push deploy</span><span class="vb-flow__node">SSH forced command<small>immediately after the build</small></span></div><div class="vb-flow__lane"><span class="vb-flow__label">Path 2 · updater timer</span><span class="vb-flow__node">digest check<small>every 15 minutes</small></span></div></div>
<div class="vb-flow__down">↓</div>
<div class="vb-flow__row"><span class="vb-flow__node vb-flow__node--end">systemctl --user restart vibebot<small>ExecStartPre pulls :latest</small></span></div>

</div>

**Path 1: push deploy.** A push to `main` runs CI. When CI passes, the `Build and Push Docker Image` workflow (`.github/workflows/docker.yml`) builds the image, pushes it to `ghcr.io/rdrake/vibebot`, and connects to the production host over SSH with a restricted deploy key. The key runs one forced command, `systemctl --user restart vibebot`, and the unit pulls the new image as it starts. Setting it up is described [below](#setting-up-push-deploy).

A push that touches only `docs/**`, `*.md`, or `mkdocs.yml` builds nothing. CI's `paths-ignore` skips those, and because `docker.yml` triggers on CI success, no image is built and the running bot is never restarted. Documentation still publishes: `pages.yml` has its own trigger on `docs/**` and `mkdocs.yml`. A push that changes only a root-level `*.md` fires nothing at all.

**Path 2: updater timer.** `vibebot-updater.timer` fires every 15 minutes. Its service compares the image the running container uses against the registry's `:latest` and restarts `vibebot` when they differ, including when the bot is stopped, because there is then no container to compare (see [Editing bot.conf safely](#editing-botconf-safely)). This path catches anything the push deploy missed, such as a host that was offline during the push, or a repository without the deploy secrets.

The source code never needs to live on the server. The Docker image has everything; the server holds only configuration and data.

### Setting up push deploy

The deploy step runs only when all three repository secrets are set. With any of them missing it is skipped, the workflow still succeeds and the image is still published, and the updater timer deploys it within 15 minutes.

| Secret | Value |
|--------|-------|
| `DEPLOY_HOST` | The SSH destination, `user@host`, for the account that runs the bot. The workflow passes no port option, so `sshd` must answer on port 22 |
| `DEPLOY_SSH_KEY` | The private half of a key pair made only for this |
| `DEPLOY_HOST_FINGERPRINT` | Despite the name, a full `known_hosts` line, not a bare fingerprint. The workflow writes it verbatim as its `known_hosts` file, so use the output of `ssh-keyscan -t ed25519 <host>` |

On the host, add the public key to that account's `~/.ssh/authorized_keys` locked to the restart command:

```
command="/usr/bin/systemctl --user restart vibebot",restrict ssh-ed25519 AAAA... vibebot-deploy
```

The forced command ignores whatever the workflow sends, and `restrict` turns off port, agent and X11 forwarding and TTY allocation, so the key can do nothing but restart the bot. The account needs lingering enabled (`loginctl enable-linger`) so its user units run without a login session.

`make deploy` is the manual equivalent from a development checkout: it pushes, waits for CI, then restarts the bot over SSH on `VIBEBOT_HOST`, with a full shell login rather than a forced command.

## Files involved

| File | Purpose |
|------|---------|
| `vibebot.service` | Runs the Docker container; pulls the latest image on start |
| `vibebot-updater.service` | Compares image digests; restarts the bot when they differ |
| `vibebot-updater.timer` | Triggers the update check every 15 minutes |

Install all three with `make install-deploy`. See [Installation](installation.md) for the full setup walkthrough, including why to install the timer last on a new host.

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

A stopped bot does not stay stopped: the updater timer restarts it within 15 minutes. Stop `vibebot-updater.timer` as well when you need the bot down for longer than that; see below.

## Editing bot.conf safely

Limnoria writes its in-memory registry to `bot.conf` on shutdown. If you edit the file while the bot is running, the shutdown write replaces your changes, so restarting after an edit loses it.

The updater timer will restart the bot out from under you, so stop it first. `vibebot-updater.service` reads the local image off the *container* with `docker inspect ... vibebot`, and the container runs `--rm`, so while the bot is stopped that lookup falls back to `none`. It never matches the registry digest, and the timer restarts the bot on its next 15-minute tick, partway through your edit.

Always follow this order:

```bash
systemctl --user stop vibebot-updater.timer
systemctl --user stop vibebot
# edit ~/.config/vibebot/bot.conf
systemctl --user start vibebot
systemctl --user start vibebot-updater.timer
```

For single settings, prefer the `@config` command from IRC; it changes the live registry. Follow it with `@flush`, which writes the registry to `bot.conf` at once: on this deployment a restart has more than once come without a shutdown flush, and an unflushed `@config` change was lost. See the [Limnoria crash course](limnoria.md). Reserve direct file edits for bulk changes and values that are awkward to type in IRC.

## Log locations

| Log | Command |
|-----|---------|
| Service logs | `journalctl --user -u vibebot` (add `-f` to follow) |
| Update logs | `journalctl --user -u vibebot-updater` |
| Bot message logs | `~/.config/vibebot/logs/messages.log` |

`docker logs vibebot` shows the same container output as the service journal, without the unit's own pull and prune lines. What the bot's lines mean is covered under [Tuning and monitoring → Per-call log lines](tuning-monitoring.md#per-call-log-lines).

## Housekeeping

The repository ships two maintenance scripts:

| Script | Purpose |
|--------|---------|
| `scripts/prune_backups.py` | Keep the newest `--keep` of Limnoria's rolling `bot.conf.backup.*` files |
| `scripts/rotate_logs.py` | Size-based rotation of `messages.log`, keeping the last `--keep` copies |

Both default to paths relative to the current directory (`backup/` and `logs/messages.log`), so either run them from `~/.config/vibebot` or give them the real paths:

```bash
python3 scripts/rotate_logs.py --log ~/.config/vibebot/logs/messages.log --keep 7
python3 scripts/prune_backups.py --dir ~/.config/vibebot/backup --keep 20
```

Add `--dry-run` to either one to see what it would touch. Run them from cron or a systemd timer on the host if your deployment accumulates large logs or backups. Unlike the bot itself, these two run on the host rather than in the container. Both are stdlib-only, so copying the single file across and running it with `python3` is enough: no clone, no dependency install.

## Rolling back a deploy

Rollback is manual. A revert push through CI takes 10-20 minutes end to end, and `docker.yml` only builds after `CI` succeeds on both Python versions (3.14, 3.15), so one flaky test blocks the image from ever publishing. When production needs to go back to a known-good build now, pin the previous image directly rather than waiting on a revert.

Every build is pushed with a `sha-<short-sha>` tag alongside `:latest` (`type=sha` in `docker.yml`), and those tags stay pullable from GHCR after `:latest` moves on. The host does not keep the old image: on every start the unit runs `docker image prune -f` right after pulling, which deletes the previous, now untagged, image, so the rollback pulls it back from the registry.

<div class="vb-steps" markdown>

1. **Find the last good tag.** The tag is `sha-` plus the first seven characters of the commit. Take the commit from the last good `Build and Push Docker Image` run in the repository's Actions tab, or list the published tags:

    ```bash
    gh api /users/rdrake/packages/container/vibebot/versions \
      --jq '.[] | select(.metadata.container.tags | length > 0) | .metadata.container.tags'
    ```

    The newest entry carries `latest` and is the bad build; take the `sha-` tag of an entry below it. The [package page](https://github.com/rdrake/vibebot/pkgs/container/vibebot) lists the same tags. To confirm which commit is running now:

    ```bash
    docker inspect vibebot --format '{{index .Config.Labels "org.opencontainers.image.revision"}}'
    ```

    Do not use `{{.Config.Image}}`: the unit always runs `:latest`, so that only ever prints `:latest`.

2. **Stop the updater.** A pinned container never matches the `:latest` digest, so the timer would restart `vibebot` on every 15-minute tick, dropping the bot off IRC each time. It cannot undo the pin itself.

    ```bash
    systemctl --user stop vibebot-updater.timer
    ```

3. **Pin the image with a systemd drop-in.** The drop-in overrides the unit's `Environment=IMAGE=...`, so both the `docker pull` and the `docker run` use the pinned tag.

    ```bash
    mkdir -p ~/.config/systemd/user/vibebot.service.d
    printf '[Service]\nEnvironment=IMAGE=ghcr.io/rdrake/vibebot:sha-<PREV>\n' \
      > ~/.config/systemd/user/vibebot.service.d/override.conf
    systemctl --user daemon-reload && systemctl --user restart vibebot
    ```

    Replace `<PREV>` with the short SHA from step 1.

4. **Roll forward once the fix ships.** Remove the drop-in, restart, and start the timer again:

    ```bash
    rm ~/.config/systemd/user/vibebot.service.d/override.conf
    systemctl --user daemon-reload && systemctl --user restart vibebot
    systemctl --user start vibebot-updater.timer
    ```

</div>

A push deploy while the drop-in is in place still restarts the bot, but onto the pinned tag.

## Credentials in the environment

API keys live in environment variables passed to the container via `--env-file` (see [Configuration → API keys](configuration.md#api-keys)). That moves where the credential is visible on the host:

- `docker inspect vibebot --format '{{.Config.Env}}'` prints every variable, values included, to anyone in the `docker` group.
- `/proc/<pid>/environ` for the container's process exposes the same values to anything with host-level access to that PID.
- Every child process the container spawns inherits the full environment, key values included, whether or not that process needs them.

Registry-stored keys were readable by anyone with owner-level `@config` access; the exposure surface has moved from "Limnoria owner capability" to "docker group and host process access." Scope access to the host and the `docker` group accordingly.

## Backup and restore

Everything the bot keeps lives under `~/.config/vibebot/`:

| Path | Contents |
|------|----------|
| `env` | Provider API keys. Treat any copy as a secret: keep it mode `600` and out of shared backup storage |
| `bot.conf` | The Limnoria registry, including the LLM plugin's settings |
| `conf/` | Limnoria's users, channels, networks and ignores |
| `data/LLM.db` | Conversations, memories, custom instructions, reminders, scheduled tasks, usage |
| `data/verse/` | One `.db` file per verse-enabled channel, and `reactions.jsonl` |
| `data/web/llm/` | `@draw` and `@story` output, while `httpRoot` is empty |
| `data/` (the rest) | Other Limnoria plugins' data |

The SQLite databases run in WAL mode, so recent writes can sit in the `-wal` file beside each database. A plain `cp` of `LLM.db` from a running bot can miss them or catch a half-written page. Take the copy one of two ways:

<div class="vb-vs" markdown>

<div markdown>
**Stop, then copy**

Stop the timer and the bot, copy the whole directory, start them again. The bot is off IRC for the copy, and every file is consistent.

```bash
systemctl --user stop vibebot-updater.timer
systemctl --user stop vibebot
tar -C ~/.config -czf vibebot-$(date +%F).tar.gz vibebot
systemctl --user start vibebot
systemctl --user start vibebot-updater.timer
```
</div>

<div markdown>
**Copy while running**

Use SQLite's online backup for each database, and copy the other files as they are. The bot stays up.

```bash
sqlite3 ~/.config/vibebot/data/LLM.db ".backup '/backup/LLM.db'"
```

Without the `sqlite3` tool on the host, Python's standard library does the same:

```bash
python3 -c "import sqlite3,sys; d=sqlite3.connect(sys.argv[2]); sqlite3.connect(sys.argv[1]).backup(d); d.close()" \
  ~/.config/vibebot/data/LLM.db /backup/LLM.db
```

Repeat for each file in `data/verse/`.
</div>

</div>

To restore, stop the timer and the bot, put the files back under `~/.config/vibebot/` owned by the account that runs the bot, and delete any `-wal` and `-shm` files left beside a database you replaced. Then start the bot and the timer.

## Uninstalling

Remove the service and timer:

```bash
make uninstall-timer
make uninstall-service
```

`make uninstall-service` stops and disables the service and removes the unit file, but not a drop-in. If you ever pinned an image for a [rollback](#rolling-back-a-deploy), remove the drop-in directory too:

```bash
rm -r ~/.config/systemd/user/vibebot.service.d
systemctl --user daemon-reload
```

Both targets leave `~/.config/vibebot/` alone: `bot.conf`, the `env` file, and the `conf/`, `data/`, `logs/`, and `backup/` subdirectories are preserved. `data/` holds `LLM.db` (conversations, memories, usage), the per-channel verse databases under `data/verse/`, and, while `httpRoot` is empty, the `@draw` and `@story` output under `data/web/llm/`. Delete that directory only when you are sure you no longer want any of it; [Backup and restore](#backup-and-restore) covers taking a copy first.

If you set `httpRoot`, that output lands there instead, outside `~/.config/vibebot/`; the shipped unit mounts `/var/www/llm` for that case, and removing the config directory leaves it behind.

Installs made before 29 September 2026 also have an empty `~/.local/share/vibebot/` directory. The container never mounted it, so it is safe to remove.
