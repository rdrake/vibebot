---
title: VibeBot
description: A Limnoria IRC bot that answers questions, sets reminders, draws, and looks things up when you talk to it in plain language.
hide:
  - navigation
  - toc
---

<div class="vb-hero" markdown>

<div class="vb-hero__copy" markdown>

<p class="vb-eyebrow">IRC bot · Limnoria</p>

# The bot your channel talks to

VibeBot is a regular in your IRC channels. Ask it something, have it remind you later, get it to draw a picture, or check who was around last night. Say it in plain words, or use `@` commands when you want exact behaviour.

<div class="vb-actions" markdown>
[Start chatting](user/getting-started.md){ .md-button .md-button--primary }
[Run your own](operator/installation.md){ .md-button }
</div>

</div>

<figure class="vb-irc" aria-label="Example IRC conversation with VibeBot">
  <div class="vb-irc__bar">
    <span class="vb-irc__chan">#lounge</span>
    <span class="vb-irc__topic">friday night, no deploys · 23 users</span>
  </div>
  <ol class="vb-irc__log">
    <li><time>21:02</time><b class="n1">alice</b><span>VibeBot, have you seen bob today?</span></li>
    <li><time>21:02</time><b class="nb">VibeBot</b><span>bob was last here 3 hours ago, saying "back after dinner".</span></li>
    <li><time>21:04</time><b class="n2">carol</b><span><code>@ask</code> why does my SSH session freeze when I close the laptop?</span></li>
    <li><time>21:04</time><b class="nb">VibeBot</b><span>The TCP connection dies while the machine sleeps, and the client waits on a server that stopped answering. Set <code>ServerAliveInterval 30</code> in <code>~/.ssh/config</code>, or use mosh.</span></li>
    <li><time>21:06</time><b class="n1">alice</b><span><code>@remind</code> in 20 minutes take the bread out</span></li>
    <li class="vb-irc__act"><time>21:06</time><b class="nb">VibeBot</b><span>Reminder set for 21:26.</span></li>
  </ol>
</figure>

</div>

## Where to start { .vb-section-title }

<div class="vb-paths">
<a class="vb-path" href="user/getting-started/">
  <span class="vb-path__who">You chat on IRC</span>
  <strong>User guide</strong>
  <span>Talk to the bot, use its commands, and manage what it remembers about you.</span>
</a>
<a class="vb-path" href="operator/installation/">
  <span class="vb-path__who">You run a bot</span>
  <strong>Operator guide</strong>
  <span>Deploy with Docker or systemd, choose models per channel, and set costs and rate limits.</span>
</a>
<a class="vb-path" href="reference/commands/">
  <span class="vb-path__who">You need one command</span>
  <strong>Command reference</strong>
  <span>Every command with its syntax and the capability it needs.</span>
</a>
</div>

## Things you can say { .vb-section-title }

<div class="vb-says" markdown>

<div class="vb-say" markdown>
`VibeBot, what's the difference between a hard link and a symlink?`

Questions and follow-ups. It keeps the thread per person and channel, and reads images you link. [AI commands](user/ai-commands.md)
</div>

<div class="vb-say" markdown>
`@draw a lighthouse in a storm, woodcut style`

Pictures from a description. `@story` goes further and builds an illustrated page. [Drawing](user/ai-commands.md#draw)
</div>

<div class="vb-say" markdown>
`@remind tomorrow at 9am standup`

One-off and recurring reminders in your own time zone, plus scheduled tasks that run a query for you. [Reminders](user/reminders.md)
</div>

<div class="vb-say" markdown>
`VibeBot, remember that I use Arch`

Facts about you that stick across conversations. Review them with `@memories`; set a standing instruction with `@instruct`. [Memory](user/memory.md)
</div>

<div class="vb-say" markdown>
`VibeBot, any aurora tonight in Thunder Bay?`

NOAA's forecast, the solar wind, darkness and cloud cover for one place. Channels can also get storm alerts. [Aurora](user/aurora.md)
</div>

<div class="vb-say" markdown>
`VibeBot, is GitHub having problems?`

Live status pages the operator chose to watch, with optional announcements when an incident opens. [Service status](user/service-status.md)
</div>

<div class="vb-say" markdown>
`VibeBot, what time is it in Tokyo?`

Stock Limnoria plugins such as Time, Seen, Math and Karma, called for you. Read-only; it never runs their write commands. [Limnoria bridge](reference/bridge-tools.md)
</div>

<div class="vb-say" markdown>
`@rp pushes open the tavern door`

A shared world per channel, with avatars and a cast that persists between sessions. [The verse](operator/verse.md)
</div>

</div>

<p class="vb-foot">Language models from Anthropic and xAI handle the understanding, chosen per channel. Source on <a href="https://github.com/rdrake/vibebot">GitHub</a>; built on <a href="https://github.com/ProgVal/Limnoria">Limnoria</a>.</p>
