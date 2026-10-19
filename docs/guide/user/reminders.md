# Reminders

Set reminders in natural language. When the time comes, the bot either echoes your text back or, if you asked it to *do* something, runs the task at fire time.

!!! tip "Natural language"
    Where an operator has enabled it, you can *create* reminders and scheduled tasks by talking to the bot: "VibeBot, remind me in 2 hours to check the build". Listing and cancelling stay on the `@remind` command either way. The channel key is `supybot.plugins.LLM.pendingTasksEnabled`, and it's off by default. `@remind` always works.

## `remind`

**Usage:** `@remind <natural language time and message>`

```
@remind in 30 minutes check the build
@remind at 5pm review the pull request
@remind tomorrow at 9am standup meeting
```

The bot parses times such as "in 30 minutes", "at 5 p.m.", and "tomorrow at 9 a.m." An absolute time with no time zone is read in your time zone, found in this order:

1. The zone you set with `@tz`, such as `@tz America/Toronto`. Recurring reminders keep their wall-clock time across daylight saving changes.
2. Your IRC client's clock, if it answers CTCP TIME. The bot asks only when the request names a clock time or a day, remembers the answer, and asks again after a week. That gives an offset, not a zone, so a recurring reminder drifts an hour at a daylight saving change. A bouncer answers with its own clock.
3. UTC.

For the last two, the confirmation says which one it used. Without an authenticated account the bot skips the first two and uses UTC.

## `tz`

**Usage:** `@tz [<zone> | clear]`

```
@tz America/Toronto
@tz
@tz clear
```

Takes an IANA zone name (Area/City). With no argument it shows your current zone. `@tz` needs an authenticated account.

A reminder has to fire between 10 seconds and 7 days from now, the text after `@remind` is capped at 500 characters, and you can hold 25 pending reminders at a time.

## Action reminders

If your reminder asks the bot to *perform a task* (look something up, check a status, fetch a URL, summarise something), the bot runs the task as a fresh AI query when the timer fires. The query has the bot's full tool surface: web search, URL fetch, code, drawing, memory, and nested reminders.

```
@remind in 2 hours check the status of CVE-2026-31431 in Debian 12 and 13
@remind in 30 minutes check if https://example.com/build is green
@remind tomorrow at 9 a.m. summarize the top 3 HN headlines about Rust
```

Action reminders appear marked **`[auto]`** in `@remind list`, so you can tell them apart from plain echo reminders.

Phrase one as a watch — "every 10 minutes let me know when the mirror has 24.04" — and the bot stays quiet on every fire that finds nothing. Only a positive result reaches the channel, so a watch that never speaks is working, not broken.

"Remind me to" phrasing, where *you* do the thing, stays a plain echo:

```
@remind in 5 minutes remind me to check the build   # echo only
@remind tomorrow at 3 p.m. call Bob                 # echo only
```

The parser leans towards action: any imperative it can carry out — check, search, fetch, draw, summarise — becomes an action reminder. It stays an echo only when the subject is plainly you, or when there is no verb at all. If it misclassifies, rephrase.

## Recurring reminders and scheduled tasks

For repeating work, ask in plain English:

```
VibeBot, every 2 hours check the status of #1234 and ping me if it changed
VibeBot, every weekday at 9 a.m. summarize the overnight CVE feed
```

Anything that needs tools at fire time becomes a [scheduled task](scheduled-tasks.md). Plain echo reminders also recur: `@remind every Friday at 5 p.m. switch laundry over` schedules one reminder that re-arms itself after each fire, up to 50 times, before you have to set it again. It keeps the same id the whole way, so the id `@remind list` showed you still cancels it after it has fired.

None of it can be listed or cancelled by talking to the bot — creating is all chat can do. Everything you own, reminders and scheduled tasks alike, comes back through `@remind list` and `@remind delete`, below.

Things to know:

- **Action reminders count against your `@ask` rate limit.** If you're over the limit when one fires, the bot delivers your original text as a plain reminder with a note, and makes no API call.
- **No elevated capabilities at fire time.** Even if an owner or admin scheduled the action, it runs without those rights.
- **A fired scheduled task cannot schedule another one.** The depth cap is one. A recurring reminder also loses `set_reminder` at fire time, so it cannot double-book against the scheduler's own re-arm.
- **Recurring chains cap at 50 fires.** After that, re-arm the reminder.
- **Scheduled tasks need an authenticated account.** Log in to your network account first.
- **Conversational scheduling is per channel.** The plain-English forms in this section need `pendingTasksEnabled` on that channel; `@remind` does not.

## Listing and cancelling

```
@remind list                # your reminders and tasks, with IDs
@remind delete abc1         # cancel one (del also works)
@remind delete abc1 def2    # cancel several
@remind clear               # cancel everything you own
```

`list` marks action reminders `[auto]` and [scheduled tasks](scheduled-tasks.md) `[task]`. Delete either kind by the id `list` shows it under.

`@remind clear` cancels every reminder and scheduled task you own. It says what went, so `Cleared 2 reminders and 1 scheduled task.` is your receipt.

## Delivery

The bot delivers a reminder to the channel where you set it, or by private message if you set it by PM. If the bot was offline when a reminder came due, it delivers the reminder shortly after it comes back. A one-off reminder that came due more than 24 hours before the bot returned is dropped, not delivered. A recurring reminder is never dropped for being late: it fires once when the bot returns and carries on from there.

## Owner admin

Bot owners can list, delete, and clear other users' reminders and scheduled tasks:

```
@remind admin list someone
@remind admin del someone abc1
@remind admin clear someone
```

Everyone else receives a permission error.
