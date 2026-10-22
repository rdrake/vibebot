import json
import re as _re

from llm.verse.taste_mine import (
    Candidate,
    Msg,
    classify_praise,
    classify_repaste,
    extract_candidates,
    iter_messages,
    render_review,
)


class _Ent:
    def __init__(self, name):
        self.name = name


class FakeStore:
    def __init__(self, names):
        self._names = names

    def match_entities_in_text(self, text, limit=12):
        low = text.lower()
        return [
            _Ent(n)
            for n in self._names
            if _re.search(r"(?<!\w)" + _re.escape(n.lower()) + r"(?!\w)", low)
        ][:limit]


def test_iter_messages_parses_privmsg_action_skips_rest():
    lines = [
        "2026-06-22T00:01:16  <fc42> a normal message",
        "2026-06-22T00:02:00  * fc42 does a thing",  # CTCP ACTION / /me
        "2026-06-22T00:03:00  *** vibebot has joined #afternet",  # system -> skip
        "2026-06-22T00:04:00  -ChanServ- a notice",  # notice -> skip
        "2026-06-22T00:05:00  <fc42> ",  # empty body -> skip
        "garbage no double-space sep",  # malformed -> skip
        "2026-06-22T00:06:00  <rdrake> hi th�ere",  # garbled char -> kept
    ]
    assert list(iter_messages(lines)) == [
        Msg("fc42", "a normal message"),
        Msg("fc42", "does a thing"),
        Msg("rdrake", "hi th�ere"),
    ]


def test_repaste_long_prose_naming_entity_is_autotrusted():
    store = FakeStore(["stinky lads", "Ripping Robert"])
    text = (
        "the stinky lads marched into the assembly hall and ripping robert let "
        "off a perfectly timed duet that turned the leaves yellow indeed"
    )
    c = classify_repaste(text, store)
    assert c is not None and c.kind == "repaste" and c.needs_review is False  # multiword


def test_repaste_short_lowercase_only_match_flags_review():
    store = FakeStore(["Ghost"])
    text = (
        "i didn't have a ghost of a chance against that lot in the second half today "
        "mate it was a proper disaster from start to finish honestly yeah"
    )
    c = classify_repaste(text, store)
    assert c is not None and c.needs_review is True


def test_repaste_rejects_short_url_and_addressed():
    store = FakeStore(["stinky lads"])
    assert classify_repaste("stinky lads", store) is None  # < 120
    assert classify_repaste("grok " + "the stinky lads are great " * 6, store) is None  # addressed
    assert classify_repaste("look https://x.com/" + "a" * 120, store) is None  # URL


def test_extract_candidates_honours_custom_bot_nicks():
    """bot_nicks parameterizes the addressed-line filter: a rename of the live
    bot must be mineable without editing the miner (--bot-nicks CLI flag)."""
    store = FakeStore(["stinky lads"])
    body = "newbot " + "the stinky lads are great " * 6
    lines = [f"2026-06-22T00:01:16  <fc42> {body}"]
    # Default nicks (grok|vibebot): "newbot ..." is NOT addressed -> mined.
    assert len(extract_candidates(lines, store)) == 1
    # Custom nicks: the same line is a real bot trigger -> excluded.
    assert extract_candidates(lines, store, bot_nicks=("newbot",)) == []


def test_repaste_keeps_name_led_prose():
    # addressed filter is narrowed to grok|vibebot, so name-led prose survives
    store = FakeStore(["stinky lads", "Larry"])
    text = (
        "Larry marched into the assembly hall with the stinky lads and let off a guff "
        "cloud that lingered for several minutes much to everyone's dismay yeah"
    )
    assert classify_repaste(text, store) is not None


def test_praise_inline_keeps_leading_article_starts_at_entity():
    store = FakeStore(["stinky lads"])
    line = (
        "i love it when it said earlier that the stinky lads will either rule "
        "the country or set it on fire"
    )
    c = classify_praise(line, store, prev_line="(some bot line)")
    assert c is not None and c.needs_review is True
    assert c.text.startswith(
        "the stinky lads will either rule"
    )  # 'earlier that' stripped, 'the' kept
    assert "earlier that" not in c.text


def test_praise_bare_attaches_source_line():
    store = FakeStore(["stinky lads"])
    c = classify_praise(
        "haha this is a good one", store, prev_line="the stinky lads stormed the chippy"
    )
    assert c is not None and c.needs_review is True
    assert c.text == "the stinky lads stormed the chippy"


def test_praise_wordlist_is_word_bounded():
    store = FakeStore(["stinky lads"])
    # praise words as a prefix of a longer token must NOT match (\b boundaries)
    assert classify_praise("so goodnight friends", store, prev_line="x") is None
    assert classify_praise("amazingly done", store, prev_line="x") is None


def test_praise_inline_span_without_entity_falls_through_to_prev():
    store = FakeStore(["stinky lads"])
    # inline span ("a cracking goal") names no roster entity -> fall through to prev_line
    c = classify_praise(
        "amazing when it said a cracking goal",
        store,
        prev_line="the stinky lads stormed the chippy",
    )
    assert c is not None and c.needs_review is True
    assert c.text == "the stinky lads stormed the chippy"


def test_non_praise_returns_none():
    store = FakeStore(["stinky lads"])
    assert classify_praise("what time is the match", store, prev_line="x") is None


def test_extract_dedups_orders_and_attributes_prev():
    store = FakeStore(["stinky lads", "Ripping Robert"])
    base = (
        "the stinky lads marched into the assembly hall and ripping robert let "
        "off a perfectly timed duet that turned the leaves yellow indeed"
    )
    lines = [
        f"2026-06-15T19:07:00  <fc42> {base}",
        f"2026-06-15T19:08:00  <fc42> {base}...",  # near-dup
        "2026-06-15T19:09:00  <Larry> the stinky lads stormed the chippy and won big",
        "2026-06-15T19:09:30  <fc42> haha this is a good one",  # praise -> Larry line
        "2026-06-15T19:10:00  <fc42> lol the ref is uzbekistan",  # noise -> dropped
    ]
    cands = extract_candidates(lines, store)
    texts = [c.text for c in cands]
    assert sum(t.startswith("the stinky lads marched") for t in texts) == 1  # deduped
    assert any(c.kind == "praise" and "stormed the chippy" in c.text for c in cands)
    assert not any("uzbekistan" in t for t in texts)


def test_render_review_excludes_denial_from_trusted_json():
    good = Candidate("the lads marched on", "repaste", "raw", needs_review=False)
    denial = Candidate("i'm sorry, i can't help with that", "repaste", "raw", needs_review=False)
    iffy = Candidate("iffy praise line", "praise", "raw", needs_review=True)
    md = render_review([good, denial, iffy])
    assert "DENIAL?" in md  # denial flagged for the human
    trusted = json.loads(md.split("```json")[1].split("```")[0])
    assert trusted == ["the lads marched on"]  # denial + needs_review excluded


# ---------------------------------------------------------------------------
# Focused branch-coverage tests (Task 8)
# ---------------------------------------------------------------------------


def test_repaste_capitalized_entity_is_autotrusted():
    """Line 82: single-word entity name capitalized as whole word → needs_review False."""
    store = FakeStore(["Ghost"])
    # Text must be ≥120 chars and contain "Ghost" capitalized (not just lowercase)
    text = (
        "The Ghost appeared at the end of the long corridor and everyone present "
        "felt a sudden chill as the candles flickered out one by one in sequence"
    )
    assert len(text) >= 120
    c = classify_repaste(text, store)
    assert c is not None
    assert c.needs_review is False  # capitalized whole-word → auto-trusted


def test_nearest_source_skips_url_and_addressed_lines():
    """Lines 155, 157: _nearest_source skips URL/addressed lines; returns "" when none left."""
    store = FakeStore(["stinky lads"])
    # fc42 praises, but all prior non-fc42 lines are URL or bot-addressed → no valid source
    lines = [
        "2026-06-22T00:01:00  <rdrake> https://example.com/some/link",  # URL → skip
        "2026-06-22T00:02:00  <rdrake> grok what do you think about this",  # addressed → skip
        "2026-06-22T00:03:00  <fc42> amazing",  # fc42 praises; no valid source
    ]
    cands = extract_candidates(lines, store)
    # praise with no valid source → candidate dropped (empty text → empty key → filtered)
    assert not any(c.kind == "praise" for c in cands)


def test_nearest_source_skips_url_returns_earlier_valid_line():
    """Line 155: _nearest_source skips a URL line and finds an earlier valid one."""
    store = FakeStore(["stinky lads"])
    # Sequence: valid bot line, then URL-only, then fc42 praises
    lines = [
        "2026-06-22T00:01:00  <Larry> the stinky lads marched into the hall indeed mate",
        "2026-06-22T00:02:00  <rdrake> https://example.com/foo",  # URL → skip
        "2026-06-22T00:03:00  <fc42> amazing",
    ]
    cands = extract_candidates(lines, store)
    praise = [c for c in cands if c.kind == "praise"]
    assert len(praise) == 1
    assert "stinky lads" in praise[0].text


def test_extract_mines_only_the_taste_setter_not_other_nicks():
    """Only fc42's lines are candidate sources (docs/guide/operator/verse.md: a
    channel without the taste-setter yields nothing). The bot's own long verse
    output and other nicks' prose/praise must not be mined, even when they name
    a multiword entity that would otherwise be auto-trusted."""
    store = FakeStore(["stinky lads", "Ripping Robert"])
    prose = (
        "the stinky lads marched into the assembly hall and Ripping Robert let "
        "off a perfectly timed duet that turned the leaves yellow indeed"
    )
    assert len(prose) >= 120
    lines = [
        f"2026-06-22T00:01:00  <vibebot> {prose}",  # bot's own verse output
        f"2026-06-22T00:02:00  <Larry> {prose} again",  # another nick's prose
        "2026-06-22T00:03:00  <Larry> haha this is a good one",  # non-fc42 praise
    ]
    assert extract_candidates(lines, store) == []
