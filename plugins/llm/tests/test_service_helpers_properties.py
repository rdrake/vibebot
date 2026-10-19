"""Property tests for pure helper functions in ``llm.service``.

These functions have no dependencies on plugin/registry/database state,
so they sit in their own module rather than being attached to an
``LLMService`` instance fixture.

Covers ``_compute_backoff``, ``truncate_to_word_boundary``, ``truncate_to_byte_budget``,
and the byte budget of ``LLM._finish_irc_line``'s link line.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import integers, none, one_of, text
from llm.plugin import LLM
from llm.service import (
    PENDING_INITIAL_BACKOFF_SECONDS,
    PENDING_MAX_BACKOFF_SECONDS,
    LLMService,
    exponential_backoff,
    truncate_to_byte_budget,
    truncate_to_word_boundary,
)

# Cap the attempt count at something well past the saturation point
# (``log2(300/30) ≈ 3.32``) so both the growing and the saturated
# regimes are exercised. ``2**1000`` would be a Python big-int and the
# property still holds, but it slows the test for no extra coverage.
_attempts = integers(min_value=0, max_value=64)


@given(n=_attempts)
def test_backoff_bounded_above(n: int) -> None:
    """Result is never above the configured cap."""
    assert LLMService._compute_backoff(n) <= PENDING_MAX_BACKOFF_SECONDS


@given(n=_attempts)
def test_backoff_bounded_below_by_initial(n: int) -> None:
    """Result is never below the initial backoff (n=0 case is the floor)."""
    assert LLMService._compute_backoff(n) >= PENDING_INITIAL_BACKOFF_SECONDS


@given(n=_attempts)
def test_backoff_monotone_non_decreasing(n: int) -> None:
    """``_compute_backoff(n) <= _compute_backoff(n + 1)``."""
    assert LLMService._compute_backoff(n) <= LLMService._compute_backoff(n + 1)


def test_backoff_at_zero_equals_initial() -> None:
    """At attempt 0, no doubling has happened yet."""
    assert LLMService._compute_backoff(0) == PENDING_INITIAL_BACKOFF_SECONDS


# --- truncate_to_word_boundary -------------------------------------------------
#
# The dangerous edge is the exact-fit boundary: when ``len(text) == max_chars``
# the text already fits and must be returned verbatim. An off-by-one in the
# guard (``<=`` weakened to ``<``) sends that case down the truncation path,
# silently dropping everything after the last interior space.

_lengths = integers(min_value=1, max_value=300)


@given(s=text(min_size=1, max_size=200))
def test_truncate_returns_input_unchanged_when_it_fits(s: str) -> None:
    """Text that already fits is returned verbatim, including at the exact
    ``len(s) == max_chars`` boundary."""
    assert truncate_to_word_boundary(s, len(s)) == s
    assert truncate_to_word_boundary(s, len(s) + 5) == s


@given(s=text(max_size=300), max_chars=_lengths)
def test_truncate_never_exceeds_max_chars(s: str, max_chars: int) -> None:
    """A positive ``max_chars`` is a hard upper bound on the result length."""
    assert len(truncate_to_word_boundary(s, max_chars)) <= max_chars


@given(s=text(max_size=300), max_chars=_lengths)
def test_truncate_never_grows_the_text(s: str, max_chars: int) -> None:
    """Truncation only ever shortens; the result never exceeds the input."""
    assert len(truncate_to_word_boundary(s, max_chars)) <= len(s)


@given(s=text(max_size=200), max_chars=integers(max_value=0))
def test_truncate_nonpositive_max_returns_unchanged(s: str, max_chars: int) -> None:
    """A non-positive ``max_chars`` disables truncation (degenerate guard)."""
    assert truncate_to_word_boundary(s, max_chars) == s


# --- truncate_to_byte_budget ---------------------------------------------------


def _nbytes(s: str) -> int:
    return len(s.encode("utf-8"))


@given(s=text(max_size=300), max_bytes=integers(min_value=-5, max_value=600))
def test_byte_truncate_never_exceeds_budget(s: str, max_bytes: int) -> None:
    """The result always fits: ``max(0, max_bytes)`` is a hard bound, in bytes."""
    assert _nbytes(truncate_to_byte_budget(s, max_bytes)) <= max(0, max_bytes)


@given(s=text(max_size=300), max_bytes=integers(min_value=0, max_value=600))
def test_byte_truncate_is_a_prefix(s: str, max_bytes: int) -> None:
    """Only ever cuts the tail — never splices or re-encodes the text."""
    assert s.startswith(truncate_to_byte_budget(s, max_bytes))


@given(s=text(max_size=200))
def test_byte_truncate_keeps_text_that_fits(s: str) -> None:
    """At the exact-fit boundary the text is returned verbatim."""
    assert truncate_to_byte_budget(s, _nbytes(s)) == s


# --- LLM._finish_irc_line: the pastebin link line fits the wire budget ----------
#
# Mirrors ``finishFixed_fits`` in docs/formal/IrcLine.lean: for ANY teaser
# (``teaser_fn`` can be an LLM summary), whenever the prefix and link suffix fit
# at all, the line fits ``allowed`` bytes. The character budget it replaced let
# a CJK teaser run a 400-byte line to 946 bytes, cutting the URL off.

_URL = "https://bot.hextalk.org/llm/0123456789abcdef.html"


@given(
    teaser=text(min_size=20, max_size=400),
    nick=text(alphabet="abcdefghij_", min_size=1, max_size=12),
    allowed=integers(min_value=0, max_value=250),
    cap=one_of(none(), integers(min_value=1, max_value=300)),
)
def test_link_line_fits_the_byte_budget(teaser: str, nick: str, allowed: int, cap) -> None:
    prefix = f"{nick}: "
    line = LLM._finish_irc_line(
        None,
        "body",
        inline=None,
        allowed=allowed,
        teaser_fn=lambda _c, _m: teaser,
        save_fn=lambda _c: _URL,
        nick_prefix=prefix,
        teaser_cap=cap,
    )
    assert line.startswith(prefix) and line.endswith(_URL)
    if _nbytes(f"{prefix} - Full answer: {_URL}") <= allowed:
        assert _nbytes(line) <= allowed


def test_cjk_teaser_keeps_the_url_on_the_line() -> None:
    """The concrete counterexample: 3-byte characters, 400-byte budget."""
    line = LLM._finish_irc_line(
        None,
        "日本語の説明 " * 200,
        inline=None,
        allowed=400,
        teaser_fn=LLM._fallback_long_reply_teaser,
        save_fn=lambda _c: _URL,
        nick_prefix="rdrake: ",
    )
    assert _nbytes(line) <= 400
    assert line.endswith(_URL)


def test_label_fallback_respects_a_tiny_budget() -> None:
    """A teaser that trims to nothing no longer returns the 11-char label into 2 chars."""
    assert LLM._trim_long_reply_teaser("--- x", 2) == ""


@given(
    n=_attempts, base=integers(min_value=1, max_value=60), cap=integers(min_value=60, max_value=600)
)
def test_shared_ladder_doubles_until_the_cap(n: int, base: int, cap: int) -> None:
    """The one ladder both queue retries climb: doubling from ``base``, held at ``cap``."""
    delay = exponential_backoff(n, base=base, cap=cap)
    assert delay == min(base * 2**n, cap)
    assert exponential_backoff(n + 1, base=base, cap=cap) >= delay


def test_delivery_retry_ladder_is_unchanged() -> None:
    """IRC delivery retries wait 15, 30, 60, then 120 seconds."""
    from llm.plugin import LLM

    ladder = [
        exponential_backoff(n, base=LLM._DELIVERY_BASE_BACKOFF, cap=LLM._DELIVERY_MAX_BACKOFF)
        for n in range(5)
    ]
    assert ladder == [15, 30, 60, 120, 120]
