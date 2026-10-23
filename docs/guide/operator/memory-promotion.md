# Memory promotion

The bot's memory system has two stages. New facts arrive as candidates and become durable memories only after the extractor sees them mentioned again. This filters out one-off statements, model hallucinations, and noisy small talk that would otherwise pile up in the memory store.

A candidate needs 2 mentions before promotion, an unreinforced candidate survives 14 days, and each user holds at most 50 durable memories. These values are fixed; only the extraction model (`memoryModel`) is configurable.

## Why two stages

Single-stage extraction promotes every extracted fact directly to memory. That makes the bot remember "I am tired today" and "my actual job is mechanical engineering" with equal weight. Two-stage extraction holds the first kind in a candidate table where it expires unreferenced, while the second kind gets reinforced on the next mention and promoted.

## How it works

One fact's life, from the first time a user says it:

<div class="vb-tl" markdown>

1. <span class="vb-at">Day 0 · first mention</span>
   **Candidate.** After a delivered reply, the extractor pulls facts out of the exchange and inserts each new one into `memory_candidates` with `mentions = 1` and `first_seen` and `last_seen` set to now.

2. <span class="vb-at">Any later pass · mentioned again</span>
   **Reinforced.** The extractor matches the fact to the existing row, bumps `mentions` and moves `last_seen` forward. The candidate's text is never rewritten: a fact that comes back in different words still counts, under the wording first recorded.

    - <span class="vb-at vb-at--ok">mentions reach 2 · promoted</span>
      The candidate becomes a durable memory, the one users see in `@memories`.
    - <span class="vb-at vb-at--drop">14 days without a mention · expired</span>
      Every extraction pass for a user deletes that user's candidates whose `last_seen` is older than 14 days. Nothing else prunes the table, so a user who goes quiet keeps their candidates until they speak again.

3. <span class="vb-at">Every 3rd promoting pass · cleanup</span>
   **Tidied.** Each pass that promotes at least one candidate bumps a per-user counter; the third resets it and asks the model to drop redundant memories and merge overlapping ones. Cleanup needs at least two memories, and it neither expires candidates nor trims a user to the cap. `@memories cleanup` runs one on demand.

4. <span class="vb-at vb-at--drop">50 durable memories · ceiling</span>
   **Extraction stops.** The cap is a hard stop, not a rolling window: nothing is evicted to make room, and the user is no longer extracted at all — no model call, no new candidates, no reinforcement, no candidate expiry, and no cleanup, because cleanup only follows a promotion. Candidates do not count against the cap.

</div>

## What writes a memory

Two paths write memories, and only the first goes through the candidate stage:

- **Automatic extraction** after a successful, delivered `@ask` reply to an authenticated user, on that user. Nick-addressed chat and verse turns share the `@ask` path and count; `@draw` and `@story` never extract, and unauthenticated callers never do.
- **`save_memory`**, the tool the model calls when a user says "remember this". It writes straight to the durable table so an explicit preference sticks at once, and it does not check the cap, so it can carry a user past 50.

## What users see

Only durable memories, through `@memories` (see [Memory and instructions](../user/memory.md#saved-memories)). Candidates have no user-facing surface; operators with database access can read the `memory_candidates` table if a fact looks stuck.

## When it goes quiet

!!! warning "No key, no memories"

    Extraction runs on `memoryModel`, or `assistantModel` when that is empty, so that model's provider needs its key (see [Configuration → API keys](configuration.md#api-keys)). Without it every pass fails, logged as `extract_memories failed: no API key for <provider> (set <VAR>)`; users see nothing and no candidates accumulate.

A user at the cap recovers once `@memories del <id>`, or a `@memories cleanup` that merges some away, brings them under 50; extraction resumes on their next reply. Facts skipped meanwhile are not recovered and have to be said again, and any candidate that aged past 14 days is dropped on that first pass.
