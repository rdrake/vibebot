# Formal models

TLA+ models of the plugin's concurrent state, and a Lean 4 proof about one
pure function. Each model that documents a bug keeps the pre-fix version next
to the fixed one. `./check.sh` re-runs all of them. It needs Java 11+ for TLC
and [elan](https://github.com/leanprover/elan) for Lean.

| Model | Code | Result |
|---|---|---|
| `TypingHolds.tla` | `typing_holds.py` before the fix | `NoStuckActive` violated: the keepalive sends a snapshotted `active` after the last release's `done`, so the client shows typing after the reply |
| `TypingHoldsSendLock.tla` | `typing_holds.py` as fixed | passes with 3 holders, including `WireMatches` (no send in flight ⇒ client state = refcount) and the liveness property `HeldHeals` |
| `PendingTasks.tla` via `PTSingle` | pending-task poller, one instance | passes with 2 tasks (71,291 states) |
| `PTZombie` | plus the poll `die()` leaves running across an `@reload` | `NoStaleWrite` violated: the old poll's late write overwrites the new instance's result |
| `PTZombieFixed` | writes carry `lease=` (`AND claimed_until = ?`) | passes. A duplicate billed call is still possible: it is in flight before the lease expires |
| `IrcLine.lean` | `_finish_irc_line`, pastebin branch | pre-fix: kernel-checked counterexamples (a 74-byte line for a 40-byte budget; an 11-char label for 2 chars of room). Fixed: `finishFixed_fits` proves the line fits `allowed` bytes for every teaser function, URL, and cap |

## Limits

The models cover what their comments name and nothing else. For example,
`PendingTasks` has no process crashes or ack-delete failures. Those paths are
at-least-once by design. `IrcLine.lean` models Python strings as lists of code
points and assumes a successful save, so a URL is present.
