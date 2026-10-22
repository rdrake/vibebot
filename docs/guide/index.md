---
title: VibeBot
description: A Limnoria IRC bot that answers questions, sets reminders, draws, and looks things up when you talk to it in plain language.
hide:
  - navigation
  - toc
---

<section class="vb-mast" markdown>
<div class="vb-mast__copy" markdown>

# VibeBot

<p class="vb-lede">The bot your IRC channel talks to. Ask it things, have it remind you later, get it to draw something, or check who was around last night. Plain words work, and <code>@</code> commands are there when you want exact behaviour.</p>

<p class="vb-cta"><a class="vb-btn vb-btn--go" href="user/getting-started/">Start chatting</a><a class="vb-btn" href="operator/installation/">Run your own</a></p>

</div>
<div class="vb-whois" aria-label="WHOIS VibeBot">
  <div class="vb-whois__cmd">/whois VibeBot</div>
  <dl>
    <dt>is</dt><dd>an IRC bot built on Limnoria</dd>
    <dt>listens</dt><dd>to its name, a PM, or @ask</dd>
    <dt>models</dt><dd>Anthropic or xAI, picked per channel</dd>
    <dt>tools</dt><dd>web search, Seen, Time, Math, Karma</dd>
    <dt>ships as</dt><dd>ghcr.io/rdrake/vibebot</dd>
    <dt>idle</dt><dd>0 seconds</dd>
  </dl>
  <div class="vb-whois__end">End of /WHOIS list.</div>
</div>
</section>

<section class="vb-wc" aria-label="Example IRC session with VibeBot">
  <div class="vb-wc__title"><strong>#lounge</strong> friday night, no deploys · docs at rdrake.github.io/vibebot</div>
  <div class="vb-wc__body">
    <nav class="vb-wc__bufs" aria-label="Guide sections">
      <a href="user/getting-started/"><i>1 </i>#users</a>
      <span class="is-on"><i>2 </i>#lounge</span>
      <a href="operator/installation/"><i>3 </i>#operators</a>
      <a href="reference/commands/"><i>4 </i>#reference</a>
    </nav>
    <ol class="vb-wc__log">
      <li class="is-evt"><time>21:01</time><b>--&gt;</b><span><em class="c-carol">carol</em> (~carol@example.net) has joined #lounge</span></li>
      <li><time>21:02</time><b class="c-alice">alice</b><span>VibeBot, have you seen bob today?</span></li>
      <li><time>21:02</time><b class="c-bot">VibeBot</b><span>bob was last here 3 hours ago, saying "back after dinner".</span></li>
      <li><time>21:04</time><b class="c-carol">carol</b><span><span class="k">@ask</span> why does my SSH session freeze when I close the laptop?</span></li>
      <li><time>21:04</time><b class="c-bot">VibeBot</b><span>The TCP connection dies while the machine sleeps, and your client keeps waiting on a server that stopped answering. Put <span class="s">ServerAliveInterval 30</span> in <span class="s">~/.ssh/config</span>, or use mosh.</span></li>
      <li><time>21:05</time><b class="c-dave">dave</b><span>VibeBot, what time is it in Tokyo?</span></li>
      <li><time>21:05</time><b class="c-bot">VibeBot</b><span>10:05 Saturday morning in Tokyo.</span></li>
      <li><time>21:06</time><b class="c-alice">alice</b><span><span class="k">@remind</span> in 20 minutes take the bread out</span></li>
    </ol>
    <ul class="vb-wc__nicks" aria-label="Nick list">
      <li class="c-bot">@VibeBot</li>
      <li class="c-alice">alice</li>
      <li class="c-bob">bob</li>
      <li class="c-carol">carol</li>
      <li class="c-dave">dave</li>
    </ul>
  </div>
  <div class="vb-wc__status">[21:06] [alice] 2:#lounge(+nt){5} [Act: 3,4]</div>
  <div class="vb-wc__input" aria-hidden="true"><span>[alice]</span> @help<i></i></div>
</section>

<nav class="vb-join" aria-label="Where to start">
<a href="user/getting-started/"><code>/join #users</code><strong>User guide</strong><span>Talking to the bot, its commands, and what it remembers about you.</span></a>
<a href="operator/installation/"><code>/join #operators</code><strong>Operator guide</strong><span>Docker or systemd, models per channel, costs and rate limits.</span></a>
<a href="reference/commands/"><code>/join #reference</code><strong>Command reference</strong><span>Every command, its syntax, and the capability it needs.</span></a>
</nav>

<div class="vb-say-head" markdown>

## Things you can say { #say }

<p>Address it by name, or start with <code>@</code>.</p>

</div>

<dl class="vb-help" markdown>

<dt>VibeBot, hard link or symlink here?</dt>
<dd markdown>Questions and follow-ups. It keeps the thread per person and channel, and reads images you link. [AI commands](user/ai-commands.md)</dd>

<dt>@draw a lighthouse in a storm, woodcut</dt>
<dd markdown>A picture from a description. `@story` builds a whole illustrated page. [Drawing](user/ai-commands.md#draw)</dd>

<dt>@remind tomorrow at 9am standup</dt>
<dd markdown>One-off and recurring reminders in your own time zone, and scheduled tasks that run a query for you. [Reminders](user/reminders.md)</dd>

<dt>VibeBot, remember that I use Arch</dt>
<dd markdown>Facts that stick across conversations. `@memories` lists them; `@instruct` sets how it answers you. [Memory](user/memory.md)</dd>

<dt>VibeBot, any aurora tonight in Tromsø?</dt>
<dd markdown>NOAA's forecast, the solar wind, darkness and cloud cover for one place, plus storm alerts for a channel. [Aurora](user/aurora.md)</dd>

<dt>VibeBot, is GitHub having problems?</dt>
<dd markdown>The status pages your operator watches, with a heads-up when an incident opens. [Service status](user/service-status.md)</dd>

<dt>@rp pushes open the tavern door</dt>
<dd markdown>A shared world per channel, with avatars and a cast that carries between sessions. [The verse](operator/verse.md)</dd>

</dl>

<p class="vb-colophon">Built on <a href="https://github.com/ProgVal/Limnoria">Limnoria</a>. Source on <a href="https://github.com/rdrake/vibebot">GitHub</a>.</p>
