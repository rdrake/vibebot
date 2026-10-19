# Glossary

Domain terms for vibebot. Code, docs, and reviews use these names.

**Identity**: a person on IRC. Holds the live IRC nick (`raw_nick`) and the authenticated account, or `None` when the person isn't authenticated.

**Storage key** (`Identity.key`): the account when authenticated, otherwise the nick. Rate limits, conversation history, and usage logging use it.

**Personal key** (`Identity.personal_key`): the account, or `None` when the person isn't authenticated. Memories, persona, time zone, and `@instruct` are read and written only through it, so an unauthenticated nick can never reach an account's personal data.

**Caller**: the person who sent this message (an Identity) plus where they sent it from: the channel, or a private message. It's resolved once per message, and everything after preflight uses it.

**Reply target**: where a reply to the Caller goes. That's the channel, or the Caller's IRC nick in a private message. It's never the bot's own nick and never an account name.
