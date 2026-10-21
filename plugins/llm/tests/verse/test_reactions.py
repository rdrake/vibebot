import json

from llm.verse.reactions import (
    RECENCY_WINDOW_S,
    ReactionBucket,
    build_reaction_report,
    classify_emoji,
    event_to_jsonl,
    parse_reaction_lines,
    process_reaction,
    render_reaction_section,
)

# ---------------------------------------------------------------------------
# Task 1: classify_emoji
# ---------------------------------------------------------------------------


def test_classify_thumbs_up():
    assert classify_emoji("\U0001f44d") == "approve"


def test_classify_thumbs_down():
    assert classify_emoji("\U0001f44e") == "disapprove"


def test_classify_skin_tone_thumbs_up_still_approve():
    assert classify_emoji("\U0001f44d\U0001f3fd") == "approve"  # 👍🏽


def test_classify_variation_selector_thumbs_up_still_approve():
    assert classify_emoji("\U0001f44d️") == "approve"  # 👍️


def test_classify_other_emoji_is_other():
    assert classify_emoji("❤️") == "other"  # ❤️


# ---------------------------------------------------------------------------
# Task 2: process_reaction + event_to_jsonl
# ---------------------------------------------------------------------------


def _line(ts):
    return {"text": "Methane Max hacked the tannoy", "ts": ts}


def test_process_reaction_approve_within_window():
    ev = process_reaction(
        react_emoji="\U0001f44d",
        reactor="fc42",
        channel="#afnet",
        network="afnet",
        target_msgid="m1",
        last_bot_line=_line(1000.0),
        now=1007.0,
        capture_enabled=True,
    )
    assert ev is not None
    assert ev["sentiment"] == "approve"
    assert ev["reactor"] == "fc42"
    assert ev["was_verse"] is True
    assert ev["recency_s"] == 7.0
    assert ev["ts"] == "1970-01-01T00:16:47Z"  # _iso(1007.0)
    assert ev["verse_excerpt"] == "Methane Max hacked the tannoy"


def test_process_reaction_window_boundary_inclusive():
    assert (
        process_reaction(
            react_emoji="\U0001f44e",
            reactor="fc42",
            channel="#a",
            network="n",
            target_msgid=None,
            last_bot_line=_line(1000.0),
            now=1000.0 + RECENCY_WINDOW_S,
            capture_enabled=True,
        )
        is not None
    )


def test_process_reaction_stale_line_dropped():
    assert (
        process_reaction(
            react_emoji="\U0001f44d",
            reactor="fc42",
            channel="#a",
            network="n",
            target_msgid=None,
            last_bot_line=_line(1000.0),
            now=1000.0 + RECENCY_WINDOW_S + 1,
            capture_enabled=True,
        )
        is None
    )


def test_process_reaction_disabled_dropped():
    assert (
        process_reaction(
            react_emoji="\U0001f44d",
            reactor="fc42",
            channel="#a",
            network="n",
            target_msgid=None,
            last_bot_line=_line(1000.0),
            now=1001.0,
            capture_enabled=False,
        )
        is None
    )


def test_process_reaction_no_line_dropped():
    assert (
        process_reaction(
            react_emoji="\U0001f44d",
            reactor="fc42",
            channel="#a",
            network="n",
            target_msgid=None,
            last_bot_line=None,
            now=1001.0,
            capture_enabled=True,
        )
        is None
    )


def test_process_reaction_clock_skew_dropped():
    assert (
        process_reaction(
            react_emoji="\U0001f44d",
            reactor="fc42",
            channel="#a",
            network="n",
            target_msgid=None,
            last_bot_line=_line(2000.0),
            now=1000.0,
            capture_enabled=True,
        )
        is None
    )


def test_event_to_jsonl_roundtrip_preserves_unicode():
    ev = {"sentiment": "approve", "emoji": "\U0001f44d"}
    assert json.loads(event_to_jsonl(ev)) == ev
    assert "\\u" not in event_to_jsonl(ev)  # ensure_ascii=False


# ---------------------------------------------------------------------------
# Task 3: parse_reaction_lines
# ---------------------------------------------------------------------------


def test_parse_reaction_lines_tolerant():
    lines = [
        '{"ts": "2026-06-25T10:00:00Z", "sentiment": "approve"}',
        "",  # blank skipped
        "   ",  # whitespace skipped
        "not json",  # malformed skipped
        "[1, 2, 3]",  # non-dict skipped
        '{"sentiment": "approve"}',  # no ts skipped
        '{"ts": "2026-06-26T10:00:00Z", "sentiment": "disapprove"}',
    ]
    out = parse_reaction_lines(lines)
    assert len(out) == 2
    assert out[0]["sentiment"] == "approve"
    assert out[1]["sentiment"] == "disapprove"


# ---------------------------------------------------------------------------
# Task 4: build_reaction_report + buckets
# ---------------------------------------------------------------------------


def _ev(date, sentiment, reactor="fc42", channel="#afnet"):
    return {
        "ts": f"{date}T10:00:00Z",
        "sentiment": sentiment,
        "reactor": reactor,
        "channel": channel,
        "verse_excerpt": "x",
    }


def test_build_reaction_report_pre_post_split():
    events = [
        _ev("2026-06-20", "approve"),
        _ev("2026-06-21", "disapprove"),
        _ev("2026-06-23", "approve"),  # post (rollout 2026-06-22)
        _ev("2026-06-24", "approve", reactor="eck"),
    ]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afnet")
    assert rep.pre == ReactionBucket("pre", 2, 1, 1, 0, 1)
    assert rep.post == ReactionBucket("post", 2, 2, 0, 0, 2)


def test_build_reaction_report_channel_filter():
    events = [
        _ev("2026-06-23", "approve", channel="#afnet"),
        _ev("2026-06-23", "approve", channel="#other"),
    ]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afnet")
    assert rep.post.reactions == 1


def test_build_reaction_report_channel_filter_case_insensitive():
    """IRC channels are case-insensitive: events are stored in the server's
    case (#AfterNet), but the report filter is typically lowercased. The match
    must be case-folded or every reaction is silently dropped."""
    events = [
        _ev("2026-06-23", "approve", channel="#AfterNet"),
        _ev("2026-06-23", "approve", channel="#AFTERNET"),
    ]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afternet")
    assert rep.post.reactions == 2


def test_build_reaction_report_monthly_and_recent_order():
    events = [_ev("2026-05-10", "approve"), _ev("2026-06-23", "disapprove")]
    rep = build_reaction_report(events, rollout="2026-06-22", channel="#afnet")
    assert [b.label for b in rep.buckets] == ["2026-05", "2026-06"]
    assert rep.recent[0]["ts"].startswith("2026-06-23")  # latest first


# ---------------------------------------------------------------------------
# Task 5: render_reaction_section
# ---------------------------------------------------------------------------


def test_render_reaction_section_empty_recent():
    rep = build_reaction_report([], rollout="2026-06-22", channel="#afnet")
    out = render_reaction_section(rep)
    assert "_none captured yet_" in out
