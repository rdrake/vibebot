"""Daily retention compaction for forest-verse channels.

A pure helper (`compact_verse`) and a thin scheduling driver
(`register_daily_timer` / `cancel_daily_timer`). The helper is the unit
of work; the driver picks the fire time, walks verse-enabled channels,
and invokes the helper. Failures abort the helper but never the timer.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable
from typing import Any, NamedTuple, Protocol

_LOG = logging.getLogger("llm.verse.compaction")
SECONDS_PER_DAY = 86400


class CompactionOutcome(NamedTuple):
    """Result of one ``compact_verse`` call.

    Carrying the counts (rather than just the state string) lets the
    plugin render a single human-readable line covering both compaction
    and aging, instead of two cryptic log records per channel.
    """

    state: str
    """One of ``'compacted'``, ``'skipped_disabled'``,
    ``'skipped_below_floor'``, ``'skipped_no_events'``."""

    total_events: int
    """``COUNT(*) FROM events`` at pass entry. ``0`` when retention is
    disabled (we never query)."""

    kept_in_digest: int
    """``len(union_ids)`` after the ``_MAX_DIGEST_ENTITY_IDS`` truncation
    when ``state == 'compacted'``; ``0`` for any skipped state."""


# Per-pass tunables. Constants — operators tune retention via registry;
# the rest are safety-net caps.
_MAX_EVENTS_PER_PASS = 200
"""Hard cap on how many old events one compaction pass touches. Long
backlogs drain across multiple daily runs; one pass writes one digest
event covering at most this many originals."""

_MAX_SUMMARY_CHARS_PER_EVENT = 240
"""Per-event truncation in the bullet block; longer summaries are
elided with an ellipsis. Stops a single pathological event from
blowing past the cheap model's context."""

_MAX_BULLET_BLOCK_CHARS = 16000
"""Hard cap on the user-message bullet block, after per-event
truncation. Trim from the front (oldest) so the *newest* of the old
events stay in the prompt."""

_MAX_DIGEST_ENTITY_IDS = 32
"""Cap on the lore-digest event's entity_ids array. Beyond this we log
INFO + drop the rest; entity-heavy verses lose some grounding in their
digest event but the events table remains the canonical truth."""


def compact_verse(
    store: Any,
    *,
    retention_days: int,
    min_keep_events: int,
    model: str,
    client: Any,
    log_usage: Callable[..., None],
    now: Callable[[], float],
) -> CompactionOutcome:
    """Compact a single verse. Returns a :class:`CompactionOutcome` whose
    ``state`` is one of:

    - ``'compacted'`` — old events replaced by one digest event
    - ``'skipped_disabled'`` — ``retention_days <= 0``
    - ``'skipped_below_floor'`` — fewer than ``min_keep_events`` total
      events in the store
    - ``'skipped_no_events'`` — no events older than the retention window

    Per-pass behaviour: a single call processes at most
    ``_MAX_EVENTS_PER_PASS`` (200) old events. If the verse has a long
    backlog, additional daily runs drain it incrementally; the digest
    event written by *this* pass covers only the events the LLM
    actually saw.
    """
    if retention_days <= 0:
        return CompactionOutcome("skipped_disabled", 0, 0)

    with store.read_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    if total < min_keep_events:
        return CompactionOutcome("skipped_below_floor", total, 0)

    cutoff_ts = now() - retention_days * SECONDS_PER_DAY
    olds = store.events_older_than(cutoff_ts=cutoff_ts, limit=_MAX_EVENTS_PER_PASS)
    if not olds:
        return CompactionOutcome("skipped_no_events", total, 0)

    # Process the OLDEST batch first. This guarantees forward progress:
    # even if the verse keeps receiving new events past the retention
    # window, the floor on the events-older-than query keeps shrinking.
    batch = olds[:_MAX_EVENTS_PER_PASS]

    def _truncated(s: str) -> str:
        if len(s) <= _MAX_SUMMARY_CHARS_PER_EVENT:
            return s
        return s[: _MAX_SUMMARY_CHARS_PER_EVENT - 1] + "…"

    # Pair each event with its bullet so the trim drops both together.
    # Without this, oldest bullets could be dropped from the prompt
    # while their event ids remained in delete_ids — deleting events
    # the LLM never saw.
    pairs: list[tuple[Any, str]] = [(e, f"- {_truncated(e.summary)}") for e in batch]
    bullets = "\n".join(b for _, b in pairs)
    if len(bullets) > _MAX_BULLET_BLOCK_CHARS:
        # Trim the NEWEST pairs first so the OLDEST-fitting prefix survives and
        # gets compacted + deleted this pass. `batch` is ordered oldest->newest,
        # so pop from the end. Trimming the front (oldest) instead would leave
        # the very oldest events undeleted, and events_older_than would re-serve
        # them at the head of every subsequent pass — they would never compact
        # and the events table would grow without bound under sustained inflow.
        while pairs and len("\n".join(b for _, b in pairs)) > _MAX_BULLET_BLOCK_CHARS:
            pairs.pop()
        bullets = "\n".join(b for _, b in pairs)
        _LOG.info(
            "verse compaction: bullet block trimmed to %i chars over %i-event batch",
            len(bullets),
            len(pairs),
        )
    kept_events = [e for e, _ in pairs]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a chronicler. Summarise the bullet list of past "
                "events into one paragraph (≤120 words). Do not invent "
                "details; only summarise what is in the list."
            ),
        },
        {"role": "user", "content": bullets},
    ]
    content, usage = client.call(op="compact", model=model, messages=messages)
    summary = (content or "").strip() or "A period of unrecorded events passed."

    # delete_ids and entity_ids union are computed only over the events
    # that actually appeared in the prompt — anything trimmed off the
    # front survives for the next pass.
    delete_ids = [e.id for e in kept_events]
    union_ids: list[int] = []
    seen: set[int] = set()
    for ev in kept_events:
        for eid in ev.entity_ids:
            if eid not in seen:
                seen.add(eid)
                union_ids.append(eid)

    if len(union_ids) > _MAX_DIGEST_ENTITY_IDS:
        _LOG.info(
            "verse compaction: digest entity_ids truncated %i → %i "
            "(union over %i events); rest dropped",
            len(union_ids),
            _MAX_DIGEST_ENTITY_IDS,
            len(kept_events),
        )
        union_ids = union_ids[:_MAX_DIGEST_ENTITY_IDS]

    # Stamp the digest at ``now()`` so it sits *newer* than the
    # retention cutoff. If we used ``max(batch.ts)`` (the original
    # plan), the digest would be older than the cutoff on the very
    # next daily pass and ``events_older_than`` would re-summarise it,
    # degrading history each cycle.
    digest_ts = now()
    store.replace_events_with_lore_digest(
        delete_ids=delete_ids,
        summary=summary,
        entity_ids=union_ids,
        ts=digest_ts,
    )
    log_usage(op="compact", model=model, usage=usage)
    return CompactionOutcome("compacted", total, len(union_ids))


def register_daily_timer(
    *,
    schedule_module: Any,
    fire_at_local: str,
    callback: Callable[[], None],
    name: str = "llm_verse_compact",
    now: Callable[[], float] | None = None,
) -> None:
    """Register a single-shot ``schedule.addEvent`` for the next time the
    local clock reaches ``fire_at_local`` (HH:MM). The callback re-arms
    itself at the end of its run; this function is called once at plugin
    load.

    If a timer with ``name`` is already registered, it is cancelled first
    so duplicate registrations cannot crash ``schedule.addEvent``'s
    name-uniqueness check.
    """
    import time as _time

    now_fn: Callable[[], float] = now if now is not None else _time.time
    cancel_daily_timer(schedule_module=schedule_module, name=name)
    fire_at = _next_local_time(fire_at_local, now=now_fn)
    schedule_module.addEvent(callback, fire_at, name=name)


def cancel_daily_timer(*, schedule_module: Any, name: str = "llm_verse_compact") -> None:
    with contextlib.suppress(KeyError):
        schedule_module.removeEvent(name)


def _next_local_time(hhmm: str, *, now: Callable[[], float]) -> float:
    """Return the next epoch second whose local time is ``hhmm``.

    Falls back to one hour from now if ``hhmm`` is malformed.
    """
    try:
        h, m = (int(x) for x in hhmm.split(":", 1))
        if not (0 <= h <= 23 and 0 <= m <= 59):
            raise ValueError
    except ValueError:
        _LOG.warning("compaction fire time malformed (%r); deferring 1h", hhmm)
        return now() + 3600.0
    cur = time.localtime(now())
    candidate = time.mktime(
        (
            cur.tm_year,
            cur.tm_mon,
            cur.tm_mday,
            h,
            m,
            0,
            cur.tm_wday,
            cur.tm_yday,
            cur.tm_isdst,
        )
    )
    if candidate <= now():
        candidate += SECONDS_PER_DAY
    return candidate


class VerseCallUsage(NamedTuple):
    prompt_tokens: int
    completion_tokens: int
    cost: float


class VerseModelClient(Protocol):
    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, VerseCallUsage]: ...


class ProviderVerseClient:
    """Default verse completion client (used by retention compaction).

    Calls ``provider.chat`` synchronously (already on a worker thread by the
    time this runs) and returns the content string plus a ``VerseCallUsage``.
    Errors propagate to the caller.
    """

    def __init__(
        self,
        log: logging.Logger | None = None,
    ) -> None:
        self._log = log or logging.getLogger("llm.verse.compaction")

    def call(
        self, *, op: str, model: str, messages: list[dict[str, str]]
    ) -> tuple[str, VerseCallUsage]:
        from .. import provider

        t0 = time.monotonic()
        reply = provider.chat(model, list(messages), timeout=120)
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        content = reply.text
        pt, ct, cost = reply.usage.input_tokens, reply.usage.output_tokens, reply.cost
        # "compaction:" replaced the deleted loom subsystem's "loom:" op-prefix;
        # historical usage rows were re-keyed in the same deploy.
        # Match service.py:_log_completion_timing's f-string convention.
        # %-args formatting was partially failing under the bot's runtime
        # logger setup (some args substituted, %d ones not) — see #66.
        self._log.warning(
            f"completion_timing op=compaction:{op} model={model} "
            f"elapsed_ms={elapsed_ms:.0f} "
            f"prompt_tokens={pt} completion_tokens={ct} cost={cost:.6f}"
        )
        return content, VerseCallUsage(pt, ct, cost)
