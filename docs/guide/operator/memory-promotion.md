# Memory promotion

The bot's memory system has two stages. New facts arrive as candidates and become durable memories only after the extractor sees them mentioned again. This filters out one-off statements, model hallucinations, and noisy small talk that would otherwise pile up in the memory store.

Operators control how many reinforcements a candidate needs before promotion, how long unreinforced candidates survive, and the per-user ceiling on durable memories.

## Why two stages

Single-stage extraction promotes every extracted fact directly to memory. That makes the bot remember "I am tired today" and "my actual job is mechanical engineering" with equal weight. Two-stage extraction holds the first kind in a candidate table where it expires unreferenced, while the second kind gets reinforced on the next mention and promoted.

The threshold is the main knob between a noisy memory store and one that reflects what users keep saying about themselves. Set it low for trusted users on private channels, higher for large public channels where unmoderated extraction builds up noise.

## Configuration

| Setting | Type | Default | Scope |
|---------|------|---------|-------|
| `memoryEnabled` | boolean | `True` | channel |
| `memoryPromotionThreshold` | positive integer | `2` | global |
| `memoryCandidateTTLDays` | non-negative integer | `14` | global |
| `memoryMaxPerUser` | positive integer | `50` | global |
| `memoryCleanupInterval` | non-negative integer | `3` | global |

### Threshold

`memoryPromotionThreshold` sets how many mentions a candidate needs before promotion. The extractor inserts new facts at one mention. A second mention reinforces the row; with the default threshold of `2`, that promotes the candidate to a durable memory.

```
@config plugins.LLM.memoryPromotionThreshold 3
```

A threshold of `1` collapses the system to single-stage and saves every extracted fact immediately. Use this when migrating from older deployments or when the candidate stage feels too cautious for a small group.

### Candidate TTL

`memoryCandidateTTLDays` — TTL is time to live — controls how long an unreinforced candidate survives. Every extraction pass for a user deletes that user's candidates whose `last_seen` is older than the cutoff, so someone who goes quiet keeps their candidates until they speak again.

```
@config plugins.LLM.memoryCandidateTTLDays 30
```

A TTL of `0` disables pruning. Nothing else prunes the candidate table, so it grows for the life of the database — reasonable only on a small, quiet bot where you want a fact mentioned twice a year to still promote.

### Per-user ceiling

`memoryMaxPerUser` caps durable memories per user. The cap is a hard stop, not a rolling window: once a user reaches it, promotion is skipped and the memories already stored stay put. Nothing is evicted to make room.

```
@config plugins.LLM.memoryMaxPerUser 100
```

The cap applies only to the durable side. Candidates do not count against it.

### Cleanup cadence

`memoryCleanupInterval` counts promotions, not extractions. Every pass that promotes at least one candidate to a durable memory bumps a per-user counter; when the counter reaches N it resets and a cleanup pass runs. Cleanup asks the model to drop redundant memories and merge overlapping ones — it needs at least two memories to do anything, and it neither applies the candidate TTL nor trims a user down to `memoryMaxPerUser`.

```
@config plugins.LLM.memoryCleanupInterval 10
```

Setting it to `0` disables periodic cleanup; durable memories then accumulate unmerged until someone runs `@memories cleanup`. The candidate table is unaffected either way — that is `memoryCandidateTTLDays`.

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

- Successful `@ask` responses, on the user who asked. Nick-addressed chat and verse turns share the `@ask` path and count as `@ask`; `@draw` and `@story` never extract.
- Explicit `save_memory` tool calls, when a user asks the bot to remember something.

The candidate path applies only to automatic extraction. Anything the assistant chooses to remember through its own tool skips the candidate stage, so the model can record explicit user preferences in one shot. `save_memory` writes straight to the durable table and does not check `memoryMaxPerUser`, so an explicit "remember this" can carry a user past the ceiling.

## Interaction with other settings

| Setting | Interaction |
|---------|-------------|
| `memoryModel` | Memory extraction calls this model, or `assistantModel` when empty, so that model's provider needs its environment variable set (see [Configuration](configuration.md#api-keys)). Without it, extraction fails silently and no candidates accumulate |
| `memoryEnabled` | Channel-scoped. `False` blocks both candidate insertion and durable promotion for that channel's traffic |

## Operational notes

- Candidate rows carry `first_seen` and `last_seen` timestamps. A user who repeats a fact keeps moving `last_seen` forward, so the TTL fires only on truly stale candidates.
- Reinforcement bumps `mentions` and `last_seen` only; the candidate text is never rewritten. A fact that reappears in different wording still promotes, but it promotes under the wording first recorded.
- Lowering `memoryPromotionThreshold` does not retroactively promote old candidates. Promotion happens on the next reinforcement event.
- A user at `memoryMaxPerUser` stops being extracted altogether — no model call, no new candidates, no reinforcement, no TTL pruning, and no cleanup pass, because cleanup only fires after a promotion. Recovery is `@memories del <id>`, `@memories cleanup`, or a higher cap.
- Raising `memoryMaxPerUser` takes effect on that user's next extraction, since the value is read once per pass. It cannot recover facts skipped while the user sat at the cap; those have to be said again, and any candidate that aged past the TTL meanwhile is dropped on that next pass.
