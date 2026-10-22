# Memory promotion

The bot's memory system has two stages. New facts arrive as candidates and become durable memories only after the extractor sees them mentioned again. This filters out one-off statements, model hallucinations, and noisy small talk that would otherwise pile up in the memory store.

A candidate needs 2 mentions before promotion, an unreinforced candidate survives 14 days, and each user holds at most 50 durable memories. These values are fixed; only the extraction model (`memoryModel`) is configurable.

## Why two stages

Single-stage extraction promotes every extracted fact directly to memory. That makes the bot remember "I am tired today" and "my actual job is mechanical engineering" with equal weight. Two-stage extraction holds the first kind in a candidate table where it expires unreferenced, while the second kind gets reinforced on the next mention and promoted.

## How it works

### Promotion

The extractor inserts new facts as candidates at one mention. A second mention reinforces the row and promotes the candidate to a durable memory.

### Candidate expiry

An unreinforced candidate survives 14 days. Every extraction pass for a user deletes that user's candidates whose `last_seen` is older than that, so someone who goes quiet keeps their candidates until they speak again. Nothing else prunes the candidate table.

### Per-user ceiling

Durable memories are capped at 50 per user. The cap is a hard stop, not a rolling window: once a user reaches it, promotion is skipped and the memories already stored stay put. Nothing is evicted to make room.

The cap applies only to the durable side. Candidates do not count against it.

### Cleanup cadence

Cleanup counts promotions, not extractions. Every pass that promotes at least one candidate to a durable memory bumps a per-user counter; every third such pass resets it and runs a cleanup pass. Cleanup asks the model to drop redundant memories and merge overlapping ones — it needs at least two memories to do anything, and it neither expires candidates nor trims a user down to the cap. `@memories cleanup` runs one on demand.

## What users see

Users interact only with durable memories. The candidate stage is invisible to them.

- `@memories` lists their durable memories.
- `@memories del <id>` deletes one.
- `@memories edit <id> <text>` rewrites one.
- `@memories clear` deletes every durable memory for the calling user.
- `@memories cleanup` triggers a one-shot cleanup pass.

Candidates have no user-facing surface. Operators with database access can inspect the `memory_candidates` table directly if a fact looks stuck.

## What writes a memory

The bot writes memories after these events:

- Successful, delivered `@ask` responses from an authenticated user, on that user. Unauthenticated callers never extract. Nick-addressed chat and verse turns share the `@ask` path and count as `@ask`; `@draw` and `@story` never extract.
- Explicit `save_memory` tool calls, when a user asks the bot to remember something.

The candidate path applies only to automatic extraction. Anything the assistant chooses to remember through its own tool skips the candidate stage, so the model can record explicit user preferences in one shot. `save_memory` writes straight to the durable table and does not check the per-user cap, so an explicit "remember this" can carry a user past the ceiling.

## Interaction with other settings

| Setting | Interaction |
|---------|-------------|
| `memoryModel` | Memory extraction calls this model, or `assistantModel` when empty, so that model's provider needs its environment variable set (see [Configuration](configuration.md#api-keys)). Without it, extraction fails on every pass (logged as `extract_memories failed: no API key for <provider> (set <VAR>)`), users see nothing, and no candidates accumulate |

## Operational notes

- Candidate rows carry `first_seen` and `last_seen` timestamps. A user who repeats a fact keeps moving `last_seen` forward, so the TTL fires only on truly stale candidates.
- Reinforcement bumps `mentions` and `last_seen` only; the candidate text is never rewritten. A fact that reappears in different wording still promotes, but it promotes under the wording first recorded.
- A user at the cap stops being extracted altogether — no model call, no new candidates, no reinforcement, no candidate expiry, and no cleanup pass, because cleanup only fires after a promotion. Recovery is `@memories del <id>` or `@memories cleanup`. Facts skipped while the user sat at the cap are not recovered; those have to be said again, and any candidate that aged past 14 days meanwhile is dropped on the next pass.
