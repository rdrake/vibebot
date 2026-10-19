"""Who sent a message, and where to answer them.

``Identity`` is a person on IRC; ``Caller`` is the person who sent the
message being handled plus where they sent it from. Both are pure value
types: resolving the account from IRC state is the plugin's job
(``LLM._resolve_caller``), so ``service.py`` can carry a ``Caller``
without importing the plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import supybot.ircutils as ircutils

if TYPE_CHECKING:
    from supybot.ircmsgs import IrcMsg


# Marks a storage key that belongs to an unauthenticated nick. '!' is the
# nick!user@host separator, so it can't appear in a nick or an account name.
# Not '~': rfc1459 casemapping folds '^' to '~'.
UNAUTHENTICATED_KEY_PREFIX = "!"


@dataclass(frozen=True)
class Identity:
    """A person on IRC: their live nick and their authenticated account.

    * ``raw_nick`` — what the person is presenting as on IRC right now.
      Use it for replies, displays, and IRC-protocol operations.
    * ``account`` — the authenticated account, or ``None`` when the person
      isn't authenticated.
    * ``key`` (storage key) — the account when authenticated, otherwise
      ``"!" + raw_nick``. Rate limits, conversation history, sticky @rp, and
      usage logging use it. The ``!`` can't appear in an IRC nick or an
      account name, so an unauthenticated nick never shares a key with an
      account of the same name. It's a key, not a name: display
      ``raw_nick`` instead.
    * ``personal_key`` — the account, or ``None``. Memories, avatar persona,
      time zone, @instruct, and background memory extraction are read and
      written only through it, so an unauthenticated nick never reaches an
      account's personal data.

    Two ``Identity`` values refer to the same person when their accounts
    match (case-insensitive) — or, lacking accounts on one or both sides,
    when their raw nicks match (case-insensitive).
    """

    raw_nick: str
    account: str | None

    @property
    def key(self) -> str:
        """Storage key: the account when authenticated, ``"!" + raw_nick`` otherwise."""
        return self.account or UNAUTHENTICATED_KEY_PREFIX + self.raw_nick

    @property
    def personal_key(self) -> str | None:
        """Personal key: the account, or ``None`` when not authenticated.

        ``None`` means "no personal data": skip the read, refuse the write.
        """
        return self.account

    def matches(self, other: Identity) -> bool:
        """True when both identities refer to the same person.

        Account-to-account match wins when both sides have one; falls
        back to raw-nick comparison when either side is unauthenticated.
        Both comparisons are case-insensitive (IRC nicks and account
        names are case-insensitive on AfterNet).
        """
        if self.account and other.account:
            return ircutils.toLower(self.account) == ircutils.toLower(other.account)
        return ircutils.toLower(self.raw_nick) == ircutils.toLower(other.raw_nick)

    def owns(self, owner: Identity) -> bool:
        """True when this caller may act on a row stored under ``owner``.

        A row stored with an account belongs only to a caller authenticated
        as that account; a nick-only row matches by nick. :meth:`matches`
        alone would hand an account-owned row to whoever takes the nick
        while its owner is offline.
        """
        return self.matches(owner) and (bool(self.account) or not owner.account)


@dataclass(frozen=True)
class Caller:
    """The person who sent this message, and where they sent it from.

    ``channel`` is the channel the message arrived on, or ``None`` for a
    private message. Resolved once per message (in preflight); everything
    after that reads it instead of ``msg.args[0]``, which in a private
    message is the bot's own nick.
    """

    identity: Identity
    channel: str | None

    @property
    def is_private(self) -> bool:
        """True for a private message."""
        return self.channel is None

    @property
    def reply_target(self) -> str:
        """Where a reply goes: the channel, or the caller's nick in a PM.

        Never the bot's own nick and never an account name. Also the
        location key for per-place state (history, usage rows, the PM
        duplicate filter), so two people's private messages never share
        one bucket.
        """
        return self.channel or self.identity.raw_nick

    @classmethod
    def from_msg(cls, msg: IrcMsg, account: str | None = None) -> Caller:
        """Build a Caller from ``msg`` and an already-resolved ``account``.

        Pure: it does not consult IRC state. ``LLM._resolve_caller`` looks
        the account up and calls this; tests and msg-only helpers that need
        just the location call it directly.
        """
        target = msg.args[0] if msg.args else ""
        prefix = msg.prefix or ""
        # A server-originated line carries a bare server name, which
        # nickFromHostmask would assert on.
        nick = ircutils.nickFromHostmask(prefix) if ircutils.isUserHostmask(prefix) else prefix
        return cls.from_row(nick, account, target)

    @classmethod
    def from_row(cls, raw_nick: str, account: str | None, where: str | None) -> Caller:
        """Rebuild a Caller from a stored row's nick, account and place.

        ``where`` is a channel, or for a private message a nick: the
        caller's own, or the bot's in rows (reminders, scheduled tasks)
        written before replies were keyed on the reply target. Anything
        that isn't a channel reads as a private message.
        """
        channel = where if where and ircutils.isChannel(where) else None
        return cls(Identity(raw_nick, account), channel)
