# Reminders

Set reminders in natural language. When the time comes, the bot either echoes your text back or, if you asked it to *do* something, runs the task at fire time.

!!! tip "Natural language"
    Where an operator has enabled it, you can set, list and cancel reminders and scheduled tasks by talking to the bot: "VibeBot, remind me in 2 hours to check the build". It is off by default; the setting to ask your operator for is plain-language scheduling (`pendingTasksEnabled`). `@remind` always works.

## Four kinds

What you get depends on how you ask. The bot picks the kind; `@remind list` shows which one it picked.

| | Plain echo | Action reminder | Recurring echo | Scheduled task |
|---|---|---|---|---|
| **Set with** | `@remind in 2 hours stretch` | `@remind in 2 hours check …`: anything the bot can do itself | `@remind every Friday at 5pm …` | Asking in conversation: "VibeBot, every weekday at 9 a.m. check …" |
| **Needs** | Nothing | Nothing | Nothing | Natural language on in the channel, and a logged-in account |
| **Fires** | Once | Once, or up to 50 times if it repeats | Up to 50 times | Up to 5 times |
| **At fire time** | Your text, echoed | A fresh AI query with tools | Your text, echoed | A fresh AI query with tools |
| **Over your `@ask` limit** | Unaffected | Your text arrives as a plain reminder, marked `(action skipped — ask rate limit reached)` | Unaffected | Skipped, with `Scheduled task skipped — ask rate limit reached.` |
| **Can set a reminder when it fires** | — | No | — | Yes, but not another task |
| **In `@remind list`** | No marker | `[auto]` | No marker | `[task]` |
| **How many you can hold** | 25 reminders, all three kinds together, across every channel | same 25 | same 25 | 5 tasks in each channel |

A repeating reminder or task doesn't tell you when it has fired for the last time: it just disappears from `@remind list`. A scheduled fire never uses up your `@ask` allowance; it only checks it.

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

A reminder has to fire between 10 seconds and 7 days from now, the text after `@remind` is capped at 500 characters, and you can hold 25 pending reminders at a time, across all channels.

## Action reminders

If your reminder asks the bot to *perform a task* (look something up, check a status, fetch a URL, summarise something), the bot runs the task as a fresh AI query when the timer fires. The query can search the web, fetch URLs, draw, read and edit your memories, check status pages, and list or cancel your pending tasks. It cannot set another reminder or schedule a task, and it cannot check the [aurora](aurora.md) forecast.

```
@remind in 2 hours check the status of CVE-2026-31431 in Debian 12 and 13
@remind in 30 minutes check if https://example.com/build is green
@remind tomorrow at 9 a.m. summarize the top 3 HN headlines about Rust
```

Action reminders appear marked **`[auto]`** in `@remind list`, so you can tell them apart from plain echo reminders.

Phrase one as a watch — "every 10 minutes let me know when the mirror has 24.04" — and the bot stays quiet on every fire that finds nothing. Only a positive result reaches the channel, so a watch that never speaks is working, not broken.

A watch you set with `@remind` is an action reminder, so it runs up to 50 times. Ask for the same watch in conversation and the bot usually makes it a [scheduled task](scheduled-tasks.md), which stops after 5 fires. Either way it stops without a word: if `@remind list` no longer shows it, set it again.

"Remind me to" phrasing, where *you* do the thing, stays a plain echo:

```
@remind in 5 minutes remind me to check the build   # echo only
@remind tomorrow at 3 p.m. call Bob                 # echo only
```

The parser leans towards action: any imperative it can carry out — check, search, fetch, draw, summarise — becomes an action reminder. It stays an echo only when the subject is plainly you, or when there is no verb at all. If it misclassifies, rephrase.

## Recurring reminders and scheduled tasks

For repeating work, ask in plain English:

```irc
<you>     VibeBot, every 2 hours check the status of #1234 and ping me if it
          changed
<you>     VibeBot, every weekday at 9 a.m. summarize the overnight CVE feed
```

Asked this way, anything that needs tools at fire time usually becomes a [scheduled task](scheduled-tasks.md). `@remind every 2 hours check …` makes a repeating action reminder instead. Plain echo reminders also recur: `@remind every Friday at 5 p.m. switch laundry over` schedules one reminder that re-arms itself after each fire, up to 50 times, before you have to set it again. It keeps the same id the whole way, so the id `@remind list` showed you still cancels it after it has fired.

Where an operator has turned on `pendingTasksEnabled`, you can also ask the bot to list or cancel them. `@remind list` and `@remind delete`, below, always work and cover reminders and scheduled tasks alike.

Things to know:

- **Action reminders respect your `@ask` rate limit but don't use it up.** If you're over the limit when one fires, the bot delivers your original text as a plain reminder with a note, and makes no API call.
- **No elevated capabilities at fire time.** Even if an owner or admin scheduled the action, it runs without those rights.
- **A fired scheduled task cannot schedule another one**, though it can set a reminder. A fired action reminder can do neither, so it never double-books against its own chain.
- **Recurring reminder chains cap at 50 fires, scheduled tasks at 5.** After that, set it again.
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

`@remind clear` cancels every reminder and scheduled task you own. It reacts 👍 to your message. On a server without reactions it replies with what went instead, such as `Cleared 2 reminders and 1 scheduled task.`

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
