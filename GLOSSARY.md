# Glossary

Domain terms for vibebot. Code, docs, and reviews use these names.

**Identity**: a person on IRC. Holds the live IRC nick (`raw_nick`) and the authenticated account, or `None` when the person isn't authenticated.

**Storage key** (`Identity.key`): the account when authenticated, otherwise `!` followed by the nick. Rate limits, conversation history, sticky `@rp`, and usage logging use it. The `!` can never appear in a nick or an account name, so an unauthenticated nick never shares a storage key with an account of the same name. Replies and displays use the nick, never the storage key.

**Personal key** (`Identity.personal_key`): the account, or `None` when the person isn't authenticated. Memories, persona, time zone, and `@instruct` are read and written only through it, so an unauthenticated nick can never reach an account's personal data.

**Caller**: the person who sent this message (an Identity) plus where they sent it from: the channel, or a private message. It's resolved once per message, and everything after preflight uses it.

**Reply target**: where a reply to the Caller goes. That's the channel, or the Caller's IRC nick in a private message. It's never the bot's own nick and never an account name.

**Agenda** (`agenda.py`): everything that is set to fire later on its own: reminders and scheduled LLM tasks. They stay in separate tables and are delivered by separate code, but one set of rules decides when each fires, how often a recurring one may repeat, and what a restart does with an overdue one. The plugin runs one wakeup timer, armed for the agenda's next due time.

**Agenda entry**: one reminder or scheduled LLM task. Its database row is the only record of whether it's still wanted: cancelling deletes the row, and a fire that finds its row gone doesn't reschedule. A recurring entry keeps its row, and the id shown by `@remind list`, across fires.

**Chain**: the fires of one recurring agenda entry. `chain_position` counts them; a reminder chain stops after 50 fires and a scheduled-task chain after 5, and the owner has to set it again.
