# Scheduled tasks

The bot can run a future AI task on your behalf. At run time it runs a fresh query as you, with the bot's own tools: search, fetch, draw, memory and reminders. Use a scheduled task for jobs that need *tools* at run time, not a string echoed back.

There's no `@schedule` command. You create scheduled tasks conversationally: ask the bot, and its scheduler tool does the rest.

!!! note
    Conversational scheduling is off by default. An operator turns it on per channel (`pendingTasksEnabled`). Where it's off, the bot has no scheduler tool to reach for, and asking it to schedule something gets you a plain answer instead. `@remind` still works, and tasks already scheduled still fire.

## Task or reminder

<div class="vb-vs" markdown>

<div markdown>

**[Reminder](reminders.md)**

- You already know the exact text you want delivered later.
- You want a poke or a deadline, not a synthesis.
- Set with `@remind`; repeats up to 50 times.

</div>
<div markdown>

**Scheduled task**

- The bot needs to look something up at run time.
- The output should reflect the world as of the run.
- Set by asking in conversation; repeats up to 5 times.

</div>

</div>

An action reminder (`@remind in 2 hours check …`) sits between the two. [Four kinds](reminders.md#four-kinds) compares all of them: limits, caps, and what each can do when it fires.

Good candidates for a task:

- `every Monday at 9am check my open PRs and tell me which are stale`
- `at 6pm summarize the day's #news traffic and post a one-paragraph digest`
- `every 2 hours check the build status and message me if anything's red`

## Scheduling in natural language

Schedule a task the same way you'd ask the bot anything else. Mention it by name, or send a PM:

```irc
<you>     VibeBot, every weekday at 9am check the changelog of <repo> and
          tell me what shipped
<VibeBot> I'll check the changelog every weekday at 9am and tell you what
          shipped.
```

The bot confirms in plain English, describing the schedule rather than the underlying tool call. A one-off works the same way: "VibeBot, in 30 minutes check the weather and tell me whether to bring a jacket".

Scheduling needs an authenticated network account.

## Listing and cancelling

Use `@remind`. It covers your scheduled tasks as well as your reminders:

```
@remind list                    # both kinds; tasks are marked [task]
@remind del <id>                # the id shown by list
@remind clear                   # every reminder and task you own
```

Where conversational scheduling is on, you can also ask the bot to list or cancel them in plain language.

A bot owner can reach anyone's:

```
@remind admin list <nick>
@remind admin del <nick> llm_task_<id>
```

`admin list` prints reminders and tasks together; a task shows as `task:llm_task_<id>` followed by the opening of its prompt. `admin clear <nick>` drops both kinds at once.

## Where the result lands

By default, the result lands where you scheduled it: a task scheduled in `#dev` posts back to `#dev`; a task scheduled by PM posts back by PM.

You can ask for a different target. Cross-channel delivery needs both you and the bot present in the destination channel, with the bridge enabled there. You can't redirect output to another user's private messages, only your own.

```
VibeBot, every morning at 8am summarize yesterday's #ops traffic and post it in #dev
VibeBot, every Monday at 9am check my PRs and message me directly
```

## What a task can do at run time

A fire runs on the reminder-action profile — the one a `@remind` action fire uses, not the chat profile you get from `@ask`. It sees all 20 of the built-in tools, including the bookkeeping ones chat hides, but not the chat-only extras: [aurora](aurora.md), channel and nick lookups, reactions, memes, and the Limnoria bridge. A fire happens with nobody present to type a command, so it keeps the bookkeeping tools too. Capabilities are fixed at `llm.ask` and `llm.draw`, so owner and admin powers are never inherited.

- **Search** the web.
- **Fetch** URLs.
- **Draw** with the image tool.
- **Bookkeeping:** read and edit your memories, read your usage, and list or cancel your other pending tasks.
- **Status:** check the [status pages](service-status.md) the bot watches, wherever any are set.

The Limnoria bridge is not available at fire time — it rides `@ask`, mentions and PMs only. Neither is scheduling: a task can set a follow-up reminder, but it cannot schedule another task.

A run is skipped while you're over your `@ask` rate limit, but runs don't use up that limit. One person may hold at most five active tasks *in a channel*.

## Recurring schedules

Both numeric and calendar recurrences work:

- `every 5 minutes`
- `every 2 hours`
- `every weekday at 9am`
- `every first of the month at noon`

A recurring task fires at most five times and is then deleted — re-arm it if you still want it. Every attempt counts, including a run the rate limit skips. The cap is tighter than the reminder chain's 50 because a task can post into another channel, and nothing tells you that the fifth fire was the last one.

## When things go wrong

- **API key missing or invalid:** the run posts `Error: <reason>` to the delivery target and burns one of the five fires; recurring tasks try again next time.
- **Tool error:** a failed search or fetch can still produce a partial reply explaining what went wrong.
- **Rate limit hit:** the run is dropped with a note in the delivery target — "Scheduled task skipped — ask rate limit reached." Recurring tasks try again next time.
- **Cancelled mid-run:** the current run completes; the next one doesn't fire.
- **Bot restarted:** schedules live in the database, so a restart loses nothing. A fire whose time passed while the bot was down happens a second after it starts up. The rule is the one reminders follow: a one-off task more than 24 hours overdue is dropped, not run; a recurring task fires once however long the bot was away, then carries on.
- **You lost `llm.ask`:** the task is deleted at its next fire, and the bot says so in the delivery target.

An unexpected error is the one silent case: it reaches the log and nothing else. If a recurring task goes quiet, ask an owner to run `@remind admin list <nick>`.

## Privacy

Scheduled tasks run as you. Their output respects your channel context and your `@instruct` text. A task can't read messages from a channel you're not in. A task that posts to a channel speaks over the bot's connection, so others see the bot speaking, not you.

Owners can inspect any user's tasks through `@remind admin list <nick>`.
