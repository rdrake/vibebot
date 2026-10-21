"""Offline verse reaction signal: capture + report fc42's 👍/👎 to verse lines.

Pure, side-effect-free core (classification, recency attribution, JSONL
(de)serialisation, and the offline report section). The live IRC glue (doTagmsg,
the send-hook) lives in plugin.py and delegates here. See
docs/superpowers/specs/2026-06-27-verse-reaction-signal-design.md.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import NamedTuple

RECENCY_WINDOW_S = 300.0
_EXCERPT_CAP = 120
DEFAULT_ROLLOUT = "2026-06-22"
_RECENT_CAP = 15
_THIN_SAMPLE = 5

_THUMB_UP = "\U0001f44d"
_THUMB_DOWN = "\U0001f44e"
_SKIN_TONES = {"\U0001f3fb", "\U0001f3fc", "\U0001f3fd", "\U0001f3fe", "\U0001f3ff"}
_VARIATION_SELECTOR = "️"


def classify_emoji(emoji: str) -> str:
    """Map a reaction emoji to 'approve' | 'disapprove' | 'other'.

    Strips skin-tone modifiers and the U+FE0F variation selector so 👍🏽 / 👍️ match 👍.
    """
    core = "".join(c for c in (emoji or "") if c not in _SKIN_TONES and c != _VARIATION_SELECTOR)
    if core == _THUMB_UP:
        return "approve"
    if core == _THUMB_DOWN:
        return "disapprove"
    return "other"


# ---------------------------------------------------------------------------
# Task 2: process_reaction + event_to_jsonl
# ---------------------------------------------------------------------------


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def process_reaction(
    *,
    react_emoji: str,
    reactor: str,
    channel: str,
    network: str,
    target_msgid: str | None,
    last_bot_line: dict | None,
    now: float,
    capture_enabled: bool,
    window: float = RECENCY_WINDOW_S,
    excerpt_cap: int = _EXCERPT_CAP,
) -> dict | None:
    """Return a JSONL-ready reaction event, or None to skip.

    Skips when capture is disabled, there is no reaction emoji, there is no
    remembered verse line for the channel, or that line is outside the recency
    window (including clock-skew where now < ts).
    """
    if not capture_enabled or not react_emoji or not last_bot_line:
        return None
    ts = last_bot_line.get("ts")
    if ts is None or now < ts or now - ts > window:
        return None
    text = last_bot_line.get("text") or ""
    return {
        "ts": _iso(now),
        "network": network,
        "channel": channel,
        "reactor": reactor,
        "emoji": react_emoji,
        "sentiment": classify_emoji(react_emoji),
        "was_verse": True,
        "target_msgid": target_msgid,
        "recency_s": round(now - ts, 2),
        "verse_excerpt": text[:excerpt_cap],
    }


def event_to_jsonl(event: dict) -> str:
    """Serialise one reaction event to a single JSON line (no trailing newline)."""
    return json.dumps(event, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Task 3: parse_reaction_lines
# ---------------------------------------------------------------------------


def parse_reaction_lines(lines: Iterable[str]) -> list[dict]:
    """Parse JSONL reaction events, tolerating blank/malformed/non-dict lines."""
    out: list[dict] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        try:
            ev = json.loads(s)
        except ValueError, TypeError:
            continue
        if isinstance(ev, dict) and ev.get("ts"):
            out.append(ev)
    return out


# ---------------------------------------------------------------------------
# Task 4: build_reaction_report + buckets
# ---------------------------------------------------------------------------


class ReactionBucket(NamedTuple):
    label: str
    reactions: int
    approve: int
    disapprove: int
    other: int
    reactors: int


class ReactionReport(NamedTuple):
    buckets: list[ReactionBucket]  # monthly, sorted by label
    pre: ReactionBucket
    post: ReactionBucket
    recent: list[dict]  # latest-first, capped
    rollout: str


def _bucket(label: str, events: list[dict]) -> ReactionBucket:
    approve = sum(1 for e in events if e.get("sentiment") == "approve")
    disapprove = sum(1 for e in events if e.get("sentiment") == "disapprove")
    other = len(events) - approve - disapprove
    reactors = len({e.get("reactor") for e in events})
    return ReactionBucket(label, len(events), approve, disapprove, other, reactors)


def build_reaction_report(
    events: Iterable[dict],
    *,
    rollout: str = DEFAULT_ROLLOUT,
    channel: str | None = None,
) -> ReactionReport:
    """Aggregate reaction events into monthly + pre/post buckets and a recent list.

    Each event is dated by ``ts[:10]``. If ``channel`` is given, only events for
    that channel are counted.
    """
    # IRC channel names are case-insensitive, and events are stored with the
    # server-supplied case (e.g. '#AfterNet') while the report is typically
    # filtered with a lowercased channel — compare case-folded so the filter
    # actually matches captured events.
    want = channel.lower() if channel is not None else None
    evs = [
        e
        for e in events
        if (want is None or str(e.get("channel", "")).lower() == want)
        and isinstance(e.get("ts"), str)
        and len(e["ts"]) >= 10
    ]
    by_month: dict[str, list[dict]] = defaultdict(list)
    pre: list[dict] = []
    post: list[dict] = []
    for e in evs:
        date = e["ts"][:10]
        by_month[date[:7]].append(e)
        (pre if date < rollout else post).append(e)
    buckets = [_bucket(m, by_month[m]) for m in sorted(by_month)]
    recent = sorted(evs, key=lambda e: e["ts"], reverse=True)[:_RECENT_CAP]
    return ReactionReport(buckets, _bucket("pre", pre), _bucket("post", post), recent, rollout)


# ---------------------------------------------------------------------------
# Task 5: render_reaction_section
# ---------------------------------------------------------------------------

_CAVEATS = [
    "",
    "### How to read the reaction signal",
    "",
    "- **Explicit + bidirectional**, but reactions require a reaction-capable "
    "client; absence of a reaction is not disapproval.",
    "- **Recency-attributed.** Each reaction is tied to the bot's most recent verse "
    "line in the channel; a reaction to a non-bot message within the window can be "
    "mis-attributed.",
    "- Thin buckets (`< 5` reactions) are flagged `thin sample`.",
]


def _reaction_rows(rows: list[tuple[str, ReactionBucket]]) -> list[str]:
    out = [
        "| window | reactions | 👍 | 👎 | other | net | reactors | note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for label, b in rows:
        note = "thin sample" if b.reactions < _THIN_SAMPLE else ""
        net = b.approve - b.disapprove
        out.append(
            f"| {label} | {b.reactions} | {b.approve} | {b.disapprove} | "
            f"{b.other} | {net:+d} | {b.reactors} | {note} |"
        )
    out.append("")
    return out


def render_reaction_section(report: ReactionReport) -> str:
    """Render the 'Explicit reactions' markdown section for the landing report."""
    lines = [
        "## Explicit 👍/👎 reactions",
        "",
        f"Rollout boundary: **{report.rollout}**.",
        "",
        "### Headline — pre vs post rollout",
        "",
    ]
    lines += _reaction_rows(
        [
            (f"pre  [.., {report.rollout})", report.pre),
            (f"post [{report.rollout}, ..]", report.post),
        ]
    )
    lines += ["### Monthly trend", ""]
    lines += _reaction_rows([(b.label, b) for b in report.buckets])
    lines += ["### Recent reactions (latest first)", ""]
    if not report.recent:
        lines.append("_none captured yet_")
    for e in report.recent:
        date = e["ts"][:10]
        sentiment = e.get("sentiment", "?")
        reactor = e.get("reactor", "?")
        excerpt = (e.get("verse_excerpt") or "")[:100]
        lines.append(f"- {date} [{sentiment} by {reactor}] {excerpt}")
    lines += _CAVEATS
    return "\n".join(lines)
