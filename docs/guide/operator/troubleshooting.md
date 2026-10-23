# Troubleshooting

Find the symptom in the table, run the check, and follow the link for the fix. Every check reads the same container output: `journalctl --user -u vibebot` and `docker logs vibebot` show the bot's own lines identically, and the journal adds the unit's `docker pull` and `docker image prune` steps around them.

| Symptom | Check | Fix |
|---------|-------|-----|
| Service won't start | `journalctl --user -u vibebot -n 50 --no-pager` | [Service fails to start](#service-fails-to-start) |
| Updates are not applying | `journalctl --user -u vibebot-updater -n 20 --no-pager` | [Updates are not applying](#updates-not-applying) |
| Timer is not running | `systemctl --user list-timers` | [Timer is not running](#timer-not-running) |
| Bot stops after you log out | `loginctl show-user $USER -p Linger` | [Service stops after logout](#stops-after-logout) |
| A `bot.conf` edit or `@config` change was lost | Was the bot running when you edited? | [Configuration change lost](#config-change-lost) |
| Every LLM command answers `no API key configured` | grep `no API key for` | [API key missing or wrong](#api-key) |
| A reply starts with 🔁 | Nothing is wrong | [A reply starts with 🔁](#reworded-icon) |
| The bot says a request was refused | grep `refusal_fallback_served`, `refused on content grounds` | [Refusals](#refusals) |
| No link from `@draw` or `@story` | `httpRoot`, `httpUrlBase`, Limnoria's HTTP server | [No link from @draw or @story](#no-link) |
| An image link 404s or shows the wrong picture | grep `op=image_generation` | [Image link 404s or shows the wrong picture](#fake-image-link) |
| The bot repeats a stock failure line | grep `history_strip` | [Stuck stock failure line](#stuck-failure-line) |
| The bot forgets earlier lines of a conversation | `contextTimeoutMinutes` | [Bot forgets the conversation](#forgets-conversation) |
| The bot never remembers a fact about a user | grep `extract_memories failed`; is the user logged in? | [Bot doesn't remember facts](#no-memories) |
| The bot keeps answering another bot | `botLoopHostSuffixes`, the other bot's `+B` mode | [Bot talks to another bot](#bot-loop) |

## Service and updates

### Service fails to start { #service-fails-to-start }

Check that Docker runs and your user can reach it, then read the service log:

```bash
docker info
journalctl --user -u vibebot -n 50 --no-pager
docker images | grep vibebot
```

The usual causes:

- **A malformed `env` file.** Docker's `--env-file` parser rejects the line and the container exits at once; `Restart=always` retries it every 10 seconds, so `systemctl --user status vibebot` flips between `activating (auto-restart)` and a brief `active`. Fix the file against the rules in [Installation → systemd service](installation.md#systemd-service).
- **No `bot.conf`** in `~/.config/vibebot/`. The container runs `limnoria bot.conf` in that directory and exits without it.
- **Docker is not at `/usr/bin/docker`.** The unit calls that path directly.

### Updates are not applying { #updates-not-applying }

Force a check and read the updater log:

```bash
systemctl --user start vibebot-updater
journalctl --user -u vibebot-updater -n 20 --no-pager
docker pull ghcr.io/rdrake/vibebot:latest
```

The updater prints `Update found: restarting vibebot...` or `No update available`. If the pull fails, the host cannot reach `ghcr.io`. If no new image exists, check that the `Build and Push Docker Image` run for your commit succeeded; a docs-only push builds nothing (see [How updates reach production](operations.md#how-updates-reach-production)). A rollback drop-in left in `~/.config/systemd/user/vibebot.service.d/` also pins the bot to an old image; see [Rolling back a deploy](operations.md#rolling-back-a-deploy).

### Timer is not running { #timer-not-running }

```bash
systemctl --user list-timers
systemctl --user enable --now vibebot-updater.timer
```

### Service stops after logout { #stops-after-logout }

User units stop when your last session ends unless lingering is on:

```bash
loginctl enable-linger $USER
```

### Configuration change lost { #config-change-lost }

Limnoria writes its in-memory registry to `bot.conf` when it shuts down. Edit the file while the bot runs, then restart, and that shutdown write replaces your edit with the old values. Restarting after an edit is exactly what loses it.

Either stop the bot before editing, following the order in [Editing bot.conf safely](operations.md#editing-botconf-safely): stop the bot, edit, start it again. Or change the value live from IRC with `@config` and follow it with `@flush`, which writes the registry to `bot.conf` at once. A `@config` change without `@flush` can be lost on a restart that skips the shutdown flush.

## Keys and refusals

### API key missing or wrong { #api-key }

The reply names the missing variable: `Error: no API key configured for provider 'xai' (set XAI_API_KEY)`. That reply is not logged. In the logs, grep `no API key for`; background calls such as memory extraction log it as `extract_memories failed: no API key for xai (set XAI_API_KEY)`.

Confirm the key reached the container without printing it, by comparing a hash:

```bash
docker exec vibebot python3 -c "import hashlib,os;k=os.environ.get('XAI_API_KEY','');print(len(k), hashlib.sha256(k.encode()).hexdigest()[:12])"
```

A length of `0` means the variable is unset or empty. Compare the hash against one computed the same way from the key you expect.

Check the key matches the model's provider: `anthropic/` models use `ANTHROPIC_API_KEY` and `xai/` models use `XAI_API_KEY`. A placeholder such as `your-xai-key-here` left in `env` counts as a key and fails at the provider. The container reads `env` only when it starts, so restart after changing it. See [Configuration → API keys](configuration.md#api-keys).

### A reply starts with 🔁 { #reworded-icon }

Nothing to fix. A content filter refused the image the user asked for, the bot reworded the prompt, and the second attempt succeeded. The icon tells the user the wording behind the picture changed. In `@usage` the request leaves two `draw:image` rows: the refusal with `status='content_blocked'` and the delivered image with `status='success'`.

### Refusals { #refusals }

A refused chat turn is normally re-run once on `xai/grok-4-1-fast-non-reasoning`, and the user gets that answer; grep `refusal_fallback_served`, whose `trigger=` field is `stop_reason`, `text` or `provider_error`. The re-run is skipped when that is already the turn's model, when `XAI_API_KEY` is unset, or once a tool that isn't read-only has run.

Only then does the user see a refusal, which asks them to reword the request. A refusal is not a fault and nothing needs restarting: successful turns either side of it are normal, because the refused message is not written to history. The provider's category for the refusal stays out of the channel, since these filters produce false positives. For refusals raised as HTTP errors it goes to the log: grep `refused on content grounds`. Anthropic's usual refusal arrives as a normal response and logs neither that line nor `assistant_completion failed`.

Other failures name themselves in the reply: a timeout says it timed out, a missing key names the variable. A generic `Error: Unable to complete` is the one that means read the logs.

## Images and links

### No link from @draw or @story { #no-link }

- **Built-in server** (`httpRoot` empty): confirm Limnoria's HTTP server is enabled and its `publicUrl` is set. Without `publicUrl` the bot falls back to a `localhost` URL that nobody else can open.
- **External server** (`httpRoot` set): confirm the bot can write to `httpRoot`, and that `httpUrlBase` matches what your web server serves at that path.
- Check the logs for file-save errors.

See [Configuration → Web and HTTP output](configuration.md#web-and-http-output).

### Image link 404s or shows the wrong picture { #fake-image-link }

The image was never generated. A model sometimes answers a draw request by writing an image URL instead of calling the `generate_image` tool, either inventing a path or reusing a real link from an earlier turn, which works but shows the wrong picture.

A real image takes roughly 15–20 seconds and always leaves a line:

```bash
docker logs vibebot 2>&1 | grep "op=image_generation"
```

A reply that came back in two or three seconds with no matching line was made up. The bot catches most of these itself: an image URL on a host only the bot publishes to (`httpUrlBase` or `imageUploadUrl`) that the turn did not generate makes it force the tool and retry once, logging:

```
assistant_completion: reply invented image URL <url> without calling generate_image;
forcing the tool and retrying (1/1)
```

That line means the user got a real image. If the retry still skips the tool, the reply is replaced with a plain failure message. Links to other hosts are left alone, so quoting someone else's image still works.

### Stuck stock failure line { #stuck-failure-line }

The model copies its own recent output, so a line such as "Image generation failed." in the history teaches it to answer the next request the same way without calling the tool. The bot strips its own past refusals, failure reports, repeated replies and tool complaints from the history before each turn, and re-rolls a reply once if it blames a tool that never ran:

```
assistant_completion: reply blamed a tool that never ran, nudging and retrying (1/1)
```

If a phrase still sticks, run `@forget` **in the affected channel**. That clears the caller's own thread and the channel's shared recent history. Run from a private message, or with a channel argument from elsewhere, it clears only the caller's thread and leaves the bad line in the shared window.

To measure how often the guards fire:

```bash
docker logs vibebot 2>&1 | grep history_strip
```

```
history_strip model=xai/grok-4-1-fast-reasoning channel=#afternet route=chat assistant_turns=14 removed=2 image_failure=1 tool_complaint=1
```

`removed` out of `assistant_turns` is the rate within one turn. The line is silent on clean turns, so for a rate across the whole log divide by the turn count, `grep -c "op=assistant_step_1"`. The fields are listed under [Per-call log lines](tuning-monitoring.md#per-call-log-lines).

## Memory and conversation

### Bot forgets the conversation { #forgets-conversation }

Conversation context expires after `contextTimeoutMinutes` of quiet in the channel (default `5`), and users clear their own with `@forget`. Raise the timeout for a slower channel:

```
@config channel #yourchan plugins.LLM.contextTimeoutMinutes 15
```

### Bot doesn't remember facts { #no-memories }

Saved memories are a separate system from conversation context, and most "it never remembers" reports are one of these:

- **The user is not logged in to a network account.** Memories are filed under the account the network reports for them, and are extracted only after an `@ask` or nick-addressed reply to a user who has one. `/whois <nick>` shows whether they are logged in.
- **The fact was said once.** Automatic extraction holds a new fact as a candidate until it comes up a second time, and drops it after 14 days without a mention. Asking the bot to "remember that I run Debian" skips this and saves at once.
- **The user has 50 memories.** At the cap extraction stops entirely for that user. `@memories delete <id>` or `@memories cleanup` brings them under it.
- **Extraction is failing.** grep `extract_memories failed`. The usual cause is a missing key for `memoryModel`'s provider (or `assistantModel`'s when `memoryModel` is empty).

[Memory promotion](memory-promotion.md) explains the stages and the cap; [Memory and instructions](../user/memory.md#saved-memories) is what users see.

### Bot talks to another bot { #bot-loop }

The bot stops replying to another bot after 3 consecutive replies, and the count resets after 300 seconds of quiet. It can only cap a bot it recognises: one with the network's `+B` mode, or one whose host ends in a suffix listed in `botLoopHostSuffixes`. Add the other bot's host suffix, or its full host, to that setting. See [Configuration → Bot-to-bot loops](configuration.md#bot-to-bot-loops).
