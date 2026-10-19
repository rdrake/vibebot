"""Property-based tests for ``Identity`` (matches, key, personal_key) and ``Caller``.

``Identity.matches`` uses ``ircutils.toLower`` (``plugin.py:120-121``),
which implements RFC 1459 casemapping where ``[]\\^`` are the
upper-case partners of ``{}|~``. Python's ``str.lower()`` does not
bridge these characters (``'['.lower() == '['``), so a strategy that
constructs case-equivalent pairs via ``str.upper()`` / ``str.lower()``
cannot exercise the IRC-specific behavior. We build pairs with
``ircutils.toLower`` directly.
"""

from __future__ import annotations

from string import ascii_letters, digits

from hypothesis import given
from hypothesis.strategies import builds, none, one_of, sampled_from, text
from llm.identity import Caller, Identity
from supybot import ircutils

# Alphabet is the RFC 2812 nick charset, including the special characters
# []\\`^{}| so the RFC 1459 casemapping in ircutils.toLower is exercised,
# not just str.lower.
_NICK_ALPHABET = ascii_letters + digits + "[]\\`^{}|_-"
nicks = text(alphabet=_NICK_ALPHABET, min_size=1, max_size=15)
accounts = one_of(none(), nicks)


@given(raw=nicks, account=accounts)
def test_matches_is_reflexive(raw: str, account: str | None) -> None:
    ident = Identity(raw_nick=raw, account=account)
    assert ident.matches(ident)


@given(raw_a=nicks, raw_b=nicks, acct_a=accounts, acct_b=accounts)
def test_matches_is_symmetric(
    raw_a: str, raw_b: str, acct_a: str | None, acct_b: str | None
) -> None:
    a = Identity(raw_nick=raw_a, account=acct_a)
    b = Identity(raw_nick=raw_b, account=acct_b)
    assert a.matches(b) == b.matches(a)


@given(
    pair=builds(lambda r: (r, ircutils.toLower(r)), nicks),
    account=accounts,
)
def test_matches_uses_irc_casemapping_on_raw_nick(
    pair: tuple[str, str],
    account: str | None,  # noqa: ARG001 -- account unused; reserved for future
) -> None:
    """``[, ], \\, ^`` should be case-equivalent to ``{, }, |, ~`` via toLower."""
    raw_upper, raw_lower = pair
    a = Identity(raw_nick=raw_upper, account=None)
    b = Identity(raw_nick=raw_lower, account=None)
    assert a.matches(b)


@given(pair=builds(lambda r: (r, ircutils.toLower(r)), nicks))
def test_matches_uses_irc_casemapping_on_account(pair: tuple[str, str]) -> None:
    acct_upper, acct_lower = pair
    a = Identity(raw_nick="x", account=acct_upper)
    b = Identity(raw_nick="y", account=acct_lower)
    # When both have an account, raw_nick is irrelevant.
    assert a.matches(b)


@given(raw=nicks, acct=nicks)
def test_account_overrides_raw_nick_in_matches(raw: str, acct: str) -> None:
    """When both have the same account, mismatched raw_nicks still match."""
    a = Identity(raw_nick=raw + "_distinct_a", account=acct)
    b = Identity(raw_nick=raw + "_distinct_b", account=acct)
    assert a.matches(b)


@given(raw=nicks, acct=nicks)
def test_key_equals_account_when_present(raw: str, acct: str) -> None:
    """Two identities with the same account share a storage key
    regardless of raw nick."""
    assert (
        Identity(raw_nick=raw, account=acct).key
        == Identity(raw_nick=raw + "_other", account=acct).key
    )


@given(raw=nicks)
def test_key_is_the_prefixed_raw_nick_when_unidentified(raw: str) -> None:
    assert Identity(raw_nick=raw, account=None).key == "!" + raw


@given(raw=nicks, other=nicks, account=nicks)
def test_unauthenticated_key_never_equals_an_authenticated_key(
    raw: str, other: str, account: str
) -> None:
    """No nick, however it's spelled or cased, lands on an account's storage
    key: conversation history, rate limits, sticky @rp and usage stay apart.
    Holds under both str.lower (context) and RFC 1459 toLower (IRC)."""
    squatter = Identity(raw_nick=raw, account=None).key
    owner = Identity(raw_nick=other, account=account).key
    assert squatter != owner
    assert squatter.lower() != owner.lower()
    assert ircutils.toLower(squatter) != ircutils.toLower(owner)


@given(raw=nicks, account=accounts)
def test_personal_key_is_the_account_or_none(raw: str, account: str | None) -> None:
    """The personal key is the account and nothing else: an unauthenticated
    nick never gets one, so it can never reach an account's personal data."""
    ident = Identity(raw_nick=raw, account=account)
    assert ident.personal_key == account
    if ident.personal_key is not None:
        assert ident.personal_key == ident.key


@given(raw=nicks)
def test_unauthenticated_nick_shaped_like_an_account_has_no_personal_key(raw: str) -> None:
    """A nick squatter on someone's account name still has no personal key."""
    squatter = Identity(raw_nick=raw, account=None)
    owner = Identity(raw_nick=raw, account=raw)
    assert squatter.key != owner.key
    assert squatter.personal_key is None


channels = builds(lambda name: f"#{name}", nicks)


@given(raw=nicks, account=accounts, channel=channels)
def test_caller_in_channel_replies_to_the_channel(
    raw: str, account: str | None, channel: str
) -> None:
    caller = Caller(Identity(raw, account), channel)
    assert caller.is_private is False
    assert caller.reply_target == channel


@given(raw=nicks, account=accounts)
def test_caller_in_pm_replies_to_the_raw_nick_never_the_account(
    raw: str, account: str | None
) -> None:
    caller = Caller(Identity(raw, account), None)
    assert caller.is_private is True
    assert caller.reply_target == raw


@given(raw=nicks, account=accounts, where=one_of(channels, nicks, sampled_from(["", "testbot"])))
def test_from_row_reads_any_non_channel_as_a_pm(raw: str, account: str | None, where: str) -> None:
    """Stored rows hold a channel or a nick (the bot's own, in old PM rows);
    either way the reply goes to a channel or to the caller, never the bot."""
    caller = Caller.from_row(raw, account, where)
    if ircutils.isChannel(where):
        assert caller.reply_target == where
    else:
        assert caller.is_private
        assert caller.reply_target == raw
