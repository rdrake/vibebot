"""Property-based tests for ``agenda.next_rrule_fire``.

Augments the DST-specific cases in ``test_agenda.py``
(spring-forward / fall-back) with two general-purpose invariants:

1. The next fire is strictly after ``now`` -- never equal, never in
   the past. Catches off-by-one bugs and sign flips.
2. The next fire is within an upper bound implied by the rule's
   frequency. ``MINUTELY/HOURLY/DAILY/WEEKLY`` cap the gap; ``BYDAY``
   filters can stretch a non-WEEKLY rule out to 7 days; ``BYHOUR`` on
   an HOURLY rule stretches to 24 hours. The test uses a generous
   8-day cap that all generated combinations respect.

Per the audit (``docs/reviews/2026-05-04-hypothesis-audit.md`` #11),
RRULE shrinking can produce strings ``rrulestr`` rejects, so the
implementation's fallback to ``None`` on parse failure is intentional;
the test uses ``assume`` to skip those cases rather than fail on them.
"""

from __future__ import annotations

from hypothesis import HealthCheck, assume, given, settings
from hypothesis.strategies import (
    booleans,
    integers,
    lists,
    none,
    one_of,
    sampled_from,
)
from llm.agenda import next_rrule_fire

_FREQS = ["MINUTELY", "HOURLY", "DAILY", "WEEKLY"]
_BYDAYS = ["MO", "TU", "WE", "TH", "FR", "SA", "SU"]

# 8 days: longest plausible gap given our strategy. WEEKLY base is
# 7 days; BYDAY on a non-WEEKLY freq also caps at 7 days. The extra
# day absorbs DST and end-of-month edges without over-loosening the
# bound to the point that runaway iteration would slip through.
_MAX_GAP_SECONDS = 8 * 86400

# A fixed wall-clock anchor far enough from year boundaries that no
# generated rule will hit a year-end DST gotcha. Using a literal makes
# the test deterministic across runs.
_NOW = 1_800_000_000.0  # 2027-01-15 ~08:13 UTC


def _build_rule(
    freq: str,
    byhour: int | None,
    byminute: int | None,
    byday: list[str],
) -> str:
    parts = [f"FREQ={freq}"]
    if byhour is not None:
        parts.append(f"BYHOUR={byhour}")
    if byminute is not None:
        parts.append(f"BYMINUTE={byminute}")
    if byday:
        parts.append(f"BYDAY={','.join(byday)}")
    return ";".join(parts)


@given(
    freq=sampled_from(_FREQS),
    byhour=one_of(none(), integers(min_value=0, max_value=23)),
    byminute=one_of(none(), integers(min_value=0, max_value=59)),
    byday=lists(sampled_from(_BYDAYS), min_size=0, max_size=3, unique=True),
)
@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_next_rrule_fire_is_strictly_forward_and_bounded(
    freq: str,
    byhour: int | None,
    byminute: int | None,
    byday: list[str],
) -> None:
    """Valid RRULE ⇒ ``now < result <= now + 8 days``."""
    rule = _build_rule(freq, byhour, byminute, byday)
    result = next_rrule_fire(rule, _NOW)
    # ``rrulestr`` may reject some shrunken combinations (e.g. an
    # invalid BYDAY for the given freq); skip those.
    assume(result is not None)
    assert result is not None  # mypy / ty narrowing

    assert result > _NOW, f"rule {rule!r} fires at-or-before now ({result} <= {_NOW})"
    gap = result - _NOW
    assert gap <= _MAX_GAP_SECONDS, (
        f"rule {rule!r} fires {gap}s after now (cap {_MAX_GAP_SECONDS}s)"
    )


@given(
    freq=sampled_from(_FREQS),
    byhour=one_of(none(), integers(min_value=0, max_value=23)),
    byminute=one_of(none(), integers(min_value=0, max_value=59)),
    byday=lists(sampled_from(_BYDAYS), min_size=0, max_size=3, unique=True),
    use_first_fire=booleans(),
)
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_next_rrule_fire_is_idempotent_under_self_iteration(
    freq: str,
    byhour: int | None,
    byminute: int | None,
    byday: list[str],
    use_first_fire: bool,
) -> None:
    """Calling ``next_rrule_fire`` with the previous fire as ``now`` advances strictly forward.

    Mirrors the DST reschedule pattern at ``test_reminders.py:2027-2056``
    where the runtime reschedules by feeding the just-fired timestamp
    back in. If this ever produces a non-strictly-forward result, the
    runtime would either double-fire (==) or rewind (<).
    """
    rule = _build_rule(freq, byhour, byminute, byday)
    first = next_rrule_fire(rule, _NOW)
    assume(first is not None)
    assert first is not None
    second_anchor = first if use_first_fire else _NOW + 1.0
    second = next_rrule_fire(rule, second_anchor)
    assume(second is not None)
    assert second is not None
    assert second > second_anchor


def test_malformed_rrule_returns_none() -> None:
    """Pinning the contract that the function swallows parse errors."""
    assert next_rrule_fire("FREQ=NONSENSE;BLAH", _NOW) is None
    assert next_rrule_fire("not-an-rrule", _NOW) is None
