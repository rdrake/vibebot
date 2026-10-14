# The verse

The verse is a per-channel, persistent world model: a SQLite graph of avatars, places, factions, items, and events that grows out of roleplay in the channel. Opted-in users act through avatars; the bot narrates scenes, records events, and carries canon forward between sessions.

This page is the operator reference: enabling the verse, controlling when it fires, curating canon, and running the maintenance surface.

## Enabling the verse

The verse is off by default. Enable it per channel:

```
@config channel #yourchan plugins.LLM.verseEnabled True
```

Users then opt in individually with `@verseopt in`. The channel switch and the user opt-in are independent: flipping `verseEnabled` to `False` suspends the verse for everyone without discarding avatars or history.

### Capabilities

Three capabilities gate verse access:

| Capability | Who needs it | What it unlocks |
|------------|--------------|-----------------|
| `llm.verse` | Regular users | `@verseopt`, `@rp`, `@verse`, `@look`, `@who`, and verse-routed messages |
| `llm.verse.edit` | Canon editors | `@versedit`, `@canon`, and the model's `verse_edit` tool |
| `llm.verse.gm` | Game moderators (GMs) | `@versedump`, `@versepurge`, `@versecompact` |

!!! warning "These are allow-by-default"

    None of the three is registered default-deny, and `supybot.capabilities.default`
    ships at `True`, so on a fresh install every user already passes all three —
    `llm.verse.gm`, and with it `@versepurge`, included. Add the anti-capabilities
    before you enable the verse: see
    [capability-based access control](rate-limiting-security.md#capability-based-access-control).

Grant them the normal Limnoria way:

```
@admin capability add someone llm.verse
@admin capability add editor llm.verse.edit
```

`@capabilities` is the User plugin's read-only lister, not a grant verb: it takes one argument and errors on two.

The `llm.verse.edit` capability is checked globally against the caller's account. A channel-scoped grant fails closed: it denies rather than escalates. Grant trusted accounts globally.

`@avatar` is the one verse command with no capability check. It only stores a persona string, which does nothing until the user also holds `llm.verse` and opts in.

Without `llm.verse`, a user can still talk to the bot in a verse channel; their messages never enter roleplay. Canon *grounding* is not capability-gated, so naming an entity still pulls the facts block into their reply.

`@rp` has no rate-limit keys of its own; it draws on the `ask` bucket.

## The canon layer and roleplay mode

Canon *retrieval* and roleplay *persona* are two separate things. Retrieval is cheap and happens on any canon mention; the in-character persona is entered explicitly.

### The canon signal

A message carries a canon signal when either of these holds:

- **Entity reference.** The message names a known active entity other than the speaker's own avatar. Matching is whole-word and alias-aware.
- **Trigger keyword.** The message matches `verseTriggerRegex`, a case-insensitive pattern with default `\bverse\b`. Set a channel-specific phrase to give users a deliberate cue. An empty value disables the keyword signal; a malformed pattern is ignored.

A canon signal pulls a compact facts block (roster, relations, recent events) into the turn. On `@ask`, and for anyone without an avatar, that is all it does: the reply stays ordinary chat. For an avatar holder speaking ambiently it also promotes *that single turn* to the verse route, so the answer comes back in the avatar's voice — unless the message asks for a picture. What it never does is arm sticky roleplay; see the table below.

### What a signalled message becomes

The table covers *ambient* messages: the bot addressed by nick, or a bare `vibebot …` line. The `@ask` command is never ambient: it always takes the chat path, canon-grounded when the message signals canon.

| Message | Result |
|---------|--------|
| `@rp <text>` | One in-character roleplay turn. Needs an avatar; without one it degrades to a canon-grounded chat reply |
| Ambient, while `@rp on` is live | Roleplay turn, same as `@rp <text>` |
| Ambient + canon signal + avatar, asking to *illustrate* (`illustrate`, `comic`, `storybook`, `with pictures`) | Illustrated storybook page, when `verseStorybookEnabled` is on and the `@story` cost gates pass (authenticated, `llm.draw`, not on cooldown). With the flag off, the default, this falls through to a prose tale |
| Ambient + canon signal, asking to *draw* (`draw`, `sketch`, `paint`, `picture of`) | Canon-grounded chat turn that draws a single image |
| Ambient + canon signal + avatar, anything else, questions included | Multi-paragraph prose tale in the avatar's voice. No image; as with every multi-line reply, it arrives as a teaser plus a link |
| Ambient + canon signal, no avatar | Ordinary chat reply, grounded in the canon facts |
| Question-shaped message with no canon signal | Straight factual answer. The channel's tall-tale overlay is swapped out so real-world questions get real answers |
| No canon signal | Ordinary chat |

### Sticky roleplay

`@rp on` keeps a caller in character without prefixing every line. Ambient messages become roleplay turns until `@rp off`. The session is keyed by channel plus account, falling back to nick, and expires on a sliding `verseRoleplayStickyTtlSeconds` window (default 900 seconds, `0` never expires). Being per channel, `@rp on` in `#a` does not follow you into `#b`. Every in-character turn refreshes the window, so a session lapses only after silence. Sessions live in memory and are lost on restart.

### Canon written from chat

By default canon only grows during roleplay turns. Set `verseChatRecordEnabled` to let an opted-in avatar's ordinary chat turn call `verse_record` as well; the model is nudged to save only a genuinely new durable fact. Off by default because chat volume dwarfs roleplay volume, and canon pollution is hard to undo.

### Going out of character

Users bypass the verse for a single message by wrapping it in double parentheses or starting it with `//`:

```
((just testing the bot, ignore this))
// what's the weather tomorrow?
```

The message takes the normal chat path and nothing is recorded to the verse. On verse-enabled channels the marker is stripped before the chat model sees the text. The marker also suppresses the ambient prose tale and the storybook, so `// illustrate the lads` costs nothing.

## User commands

| Command | Effect |
|---------|--------|
| `@verseopt in` / `@verseopt out` | Join the verse with a fresh avatar, or retire it. History is preserved; opting back in creates a new avatar |
| `@rp <text>` | One in-character roleplay turn |
| `@rp on` / `@rp off` | Enter or leave sticky roleplay mode |
| `@verse` | Where your avatar is: the place name and its description. Needs an avatar; use `@who` for the roster |
| `@look [target]` | Describe the current location, or a named entity |
| `@who` | Roster of active avatars and their locations |
| `@avatar [persona\|clear]` | Set the persona that shapes your avatar. Independent of `@instruct`; affects only the verse |

`@avatar` stores the persona in the plugin's own database (the `user_avatar_personas` table in `LLM.db`), not in the Limnoria registry, so it never lands in `bot.conf`. One persona per user, shared by every verse channel. When the caller has an active avatar in the channel, the persona is mirrored onto that avatar's summary, so `@look` reflects the change immediately.

## The store

Each channel's verse lives in its own SQLite database under `data/verse/`. The schema has nine tables; the ones an operator meets are:

- **entities**: avatars, non-player characters (NPCs), places, factions, and items, each with a kind, name, summary, and status.
- **attributes**: key-value details attached to entities.
- **relations**: directed, typed links between entities, with optional notes.
- **events**: the timeline. Each event links the entities involved, both as a JSON list and through the `event_actor` join table. Any tool that mutates events must handle both linkages.
- **entity_alias**: alternate names that entity matching resolves.
- **avatar_link**: maps IRC accounts and nicks to avatar entities.

### Canon, pinning, and aging

Entities the model creates during play are tagged auto-created. The daily sweep retires auto-created NPCs that go unmentioned for `verseAutoEntityRetireDays` days (default 14, `0` disables). Retirement is a soft status flip and is reversible.

Two marks exempt an entity from aging and pull it into every verse prompt:

- **Pinned** (`@versedit pin`): part of the always-on canon roster.
- **Author-locked** (`@canon lock`): durable canon claimed by an editor.

The canon roster block in the verse system prompt is capped at `verseRosterMaxChars` characters (default 4000). Pinning is what makes a character appear in every turn; unpinned entities surface contextually through events and relations that mention them.

### Member-driven worldbuilding

Opted-in members can narrate events involving entities other than their own avatar:

> vibebot, stinky dan threw a guff grenade at Andrew

The assistant calls the `verse_record` tool with `actors=["stinky dan", "Andrew"]`. Names that match existing entities link to them (avatar before NPC before item before place, case-insensitive, retired entities skipped). Unmatched names become auto-created NPCs. The exception is an aged-out auto-created NPC of the same name, which is reactivated instead: a character who returns after the retirement sweep keeps its id, attributes, and history rather than spawning a duplicate.

Items and props belong in the recorded summary as prose, not in `actors`. That is an instruction to the model, not a server-side filter: the server only checks the type, drops blanks, and truncates to the cap. A model that lists a weapon as an actor really will get an NPC called `guff grenade`, and `@versedit retire` is the cleanup.

`verseAutoEntityMaxNamesPerCall` (default 8) caps the actors array. Raise it for verses with large casts; values past 16 invite entity flooding.

## Editing canon

### `@versedit`

`@versedit` hand-edits a channel's verse as operator canon. Writes are immediate and audit-logged with an operator source tag. It requires `llm.verse.edit`.

By default the edit applies to the channel where you run the command. A leading `#channel` token overrides that, which also lets you batch edits from a private message without flooding the channel:

```
/query yourbot
@versedit #afternet add npc Gurning Gary :: Supply teacher with a twitchy eye
@versedit #afternet pin Gurning Gary
```

A `<ref>` is either `#<id>` or the name of an *active* entity. Summaries follow a `::` separator.

| Verb | Syntax | Effect |
|------|--------|--------|
| `add` | `add <kind> <name> [:: summary]` | Create an entity. Kinds: `avatar`, `npc`, `place`, `faction`, `item`. Rejects duplicate active names |
| `show` | `show <ref>` | Inspect id, kind, status, summary, and attributes |
| `desc` | `desc <ref> :: <summary>` | Replace the entity summary |
| `name` | `name <ref> <new-name>` | Rename. Rejects duplicate active names |
| `set` | `set <ref> <key> <value>` | Set an attribute |
| `pin` / `unpin` | `pin <ref>` | Add to or remove from the pinned roster |
| `retire` / `restore` | `retire <ref>` | Soft-delete or restore an entity. Name lookup is active-only, so `restore` takes `#<id>`: read it off `@versedump` or off the `retired #<id>` reply |
| `relate` | `relate <ref> <kind> <ref> [:: note]` | Add a directed relation; replies with its id |
| `unrelate` | `unrelate <relation-id>` | Delete a relation |
| `event` | `event <summary> [@id,id,…]` | Add a timeline event, optionally linking entity ids |
| `editevent` | `editevent <event-id> :: <summary>` | Edit an event summary |
| `delevent` | `delevent <event-id>` | Delete an event |

### `@canon`

`@canon lock <name>` marks a character as author-locked canon; `@canon unlock <name>` releases it, letting normal aging apply again. `forget` is an exact alias for `unlock`. The name is matched alias-aware, the same way `@look` matches. Requires `llm.verse.edit`.

### The model's `verse_edit` tool

Editors also get canon edits in-band: when a user holding `llm.verse.edit` triggers a verse turn, the model can call `verse_edit`. The tool is constructive-only: it can add entities, events, relations, and attributes, and update summaries, but it cannot delete, retire, pin, or rewrite history. Destructive verbs stay with `@versedit`. Authorisation is computed per triggering user; for anyone else the tool refuses as a no-op.

## Style exemplars

`verseStyleExemplars` holds a short list of curated lines that show the model what good verse output looks like in this channel. The injector enforces a hard budget: at most 5 lines and 600 characters in total, with sanitisation that drops lines mimicking prompt structure. Curate 3 or 4 short lines.

Populate the list offline with the taste miner, which scans channel logs for lines that one designated taste-setter re-pasted or praised. That nick is hard-coded in `taste_mine.py` as a prefix match, so its `_away` and `|away` variants count too, and there is no flag to point it elsewhere. On a channel without that person the miner returns nothing:

```bash
python -m llm.verse.taste_mine <logfiles...> --verse-dir data/verse --channel "#afternet" --out taste_candidates.md
```

Review the candidates by hand, then set the survivors as a JSON list on the channel key.

## Measurement

Two signals tell you whether verse output lands:

- **Reactions.** With `verseReactionCaptureEnabled` (default on), the bot records IRCv3 emoji reactions to its verse lines into `data/verse/reactions.jsonl`. Capture is measurement-only; the bot never replies to a reaction.
- **Landing-rate report.** The offline `taste_report` CLI counts how often that same taste-setter reacts to verse output, with their own message volume in the same logs as the denominator. Log filenames must carry a `YYYY-MM-DD`; the rest are skipped. Pass `--reactions` to append the explicit thumbs section:

```bash
python -m llm.verse.taste_report <logfiles...> --verse-dir data/verse --channel "#afternet" \
    --reactions data/verse/reactions.jsonl
```

## Storybook

With `verseStorybookEnabled` on, verse turns gain a `verse_storybook` tool that renders an illustrated page and posts a link. An explicit request ("make a storybook of this", "illustrate that fight") forces the tool call, so explicit asks fire reliably.

The `@story <brief>` command generates the same illustrated page on request, in two prompt-inferred modes: a story mode for in-character tales and an explainer mode for accurate illustrated explanations. `@story` requires the `llm.draw` capability and an authenticated account, and it works outside verse mode.

The storybook is reserved for pictures people asked for. A plain canon mention produces a prose tale instead, so an illustrated page only comes from `@story`, the `verse_storybook` tool, or — where `verseStorybookEnabled` is on — an ambient mention that carries an explicit illustrate cue.

Cost controls: `verseStorybookMaxImages` (default 5), `verseStorybookMaxPerTurn` (1), `verseStorybookCooldownSeconds` (300, per account, shared between the tool and `@story`), `verseStorybookMaxChars` (6000), and `verseStorybookImageTimeout` (45 seconds). Illustrations render concurrently; the page is published through the bot's HTTP output.

`verseStorybookDailyImageCap` (30) is registered but **not enforced**: there is no per-account daily image count to check it against. What bounds image cost is `verseStorybookMaxImages` per page and the per-account cooldown between pages; the verse tool path adds `verseStorybookMaxPerTurn` on top, which `@story` does not read.

## GM operations

All three commands require `llm.verse.gm`.

### `@versedump [#channel] [--format=json]`

Dump the full verse state: entities with attributes, relations, the 200 most recent events, avatar links, entity aliases, and up to 1000 rows of the `proposals` audit trail. The dump publishes to the bot's HTTP pastebin and the reply carries only the URL; if the HTTP server is not configured, the JSON falls back inline. Useful for debugging and for a backup before a purge.

### `@versepurge [#channel] [token]`

Permanently delete all verse state for a channel: entities, attributes, relations, events, and avatar links. **This is irreversible**, so it uses two-step confirmation:

1. `@versepurge #chan` issues a one-time 6-character token valid for 60 seconds.
2. `@versepurge #chan <token>` performs the purge.

An expired or wrong token cancels the purge. Tokens are single-use and compared in constant time.

### `@versecompact [#channel]`

Run retention compaction for one channel immediately instead of waiting for the daily timer. Defaults to the channel you type it in; name one explicitly from a private message. Unlike the daily pass it does **not** run the entity aging sweep, so its reply carries no `aged N entities` clause. Reports the outcome; see the reference below.

## Compaction

Once a day at `verseCompactionDailyAt` (global, default `03:00` local time), the plugin walks every verse-enabled channel and replaces the oldest events past `verseEventRetentionDays` (channel, default 30) with a single lore-digest event written by `verseCompactionModel`. The digest is stamped at compaction time and lists up to 32 involved entities.

`verseCompactionMinKeepEvents` (global, default 20) sets a floor: verses with fewer total events are left alone, which keeps small verses from thrashing.

### Drain rate

A single pass compacts at most 200 events, a safety cap that keeps one model call inside the cheap model's context window. Consequences:

- A backlog of 10,000 over-retention events converges in about 50 daily runs.
- A verse producing more than 200 over-retention events per day never converges under the daily cap. Lower `verseEventRetentionDays`, or run `@versecompact` repeatedly; each run drains another 200-event batch.
- Realistic avatar-driven verses produce 1 to 10 events per day, so the cap rarely matters.

Failures log at ERROR with a traceback and never block the timer; the re-arm runs in a `finally`, so the next day's run retries.

### Outcome reference

| State | Meaning |
|-------|---------|
| `compacted` | Old events were summarised into one digest event |
| `skipped_disabled` | `verseEventRetentionDays` is 0 or lower; retention is off |
| `skipped_below_floor` | Total events are under `verseCompactionMinKeepEvents` |
| `skipped_no_events` | Nothing is older than the retention cutoff |

The daily pass's log line also reports aging: `aged N entities (kept M)` counts auto-created NPCs retired by the sweep and those scanned but kept.

## Quality guards

Verse output runs through guards that keep long-form roleplay from degrading:

- **Denial retry.** Some non-reasoning models refuse fictional premises. A detected refusal triggers one retry with a corrective nudge, and past refusals are stripped from the history each turn so they cannot become self-reinforcing.
- **Degradation detection.** One retry when a reply collapses into run-on or looping text: over 90 words per sentence, or under 22% unique words, judged only on replies of 150 words or more. The collapsed turn is stripped from history so it cannot seed the next one. This guard is not verse-specific; it runs on every route.
- **Repeat stripping.** The bot's own near-duplicate replies are dropped from verse history too, so a stuck phrase cannot re-imitate itself.
- **History window.** Verse history is trimmed to the 10 most recent messages, which limits self-imitation drift.

These run automatically; no configuration is required.

## Registry key reference

See the verse and storybook tables in [Configuration](configuration.md) for every key, scope, and default in one place.
