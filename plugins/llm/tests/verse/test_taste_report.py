import re as _re
from pathlib import Path

from llm.verse.taste_report import (
    BucketStats,
    Report,
    Win,
    _main,
    _stat_rows,
    build_report,
    per_100,
    per_day,
    render_report,
)


class _Ent:
    def __init__(self, name):
        self.name = name


class FakeStore:
    """Mirrors the real VerseStore.match_entities_in_text contract (see test_taste_mine)."""

    def __init__(self, names):
        self._names = names

    def match_entities_in_text(self, text, limit=12):
        low = text.lower()
        return [
            _Ent(n)
            for n in self._names
            if _re.search(r"(?<!\w)" + _re.escape(n.lower()) + r"(?!\w)", low)
        ][:limit]


# A long (>=120 char), entity-naming line that classify_repaste accepts as a reaction.
_REP = (
    "the stinky lads marched into the assembly hall and let off a perfectly timed "
    "duet that turned the leaves yellow indeed mate and the whole room erupted"
)


# --------------------------------------------------------------------------- #
# Task 1 — helpers
# --------------------------------------------------------------------------- #


def test_rate_helpers_guard_divide_by_zero():
    """per_100/per_day return None (rendered n/a) when the denominator is zero."""
    assert per_100(0, 0) is None
    assert per_day(5, 0) is None
    assert per_100(3, 200) == 1.5
    assert per_day(6, 3) == 2.0


# --------------------------------------------------------------------------- #
# Task 2 — build_report buckets, pre/post, denominator
# --------------------------------------------------------------------------- #


def test_build_report_empty_input():
    """No logs -> empty buckets/wins and zeroed pre/post."""
    r = build_report([], FakeStore([]))
    assert r.buckets == []
    assert r.wins == []
    assert r.pre == BucketStats("pre", 0, 0, 0)
    assert r.post == BucketStats("post", 0, 0, 0)
    assert r.rollout == "2026-06-22"


def test_build_report_denominator_counts_only_fc42_messages():
    """fc42 privmsg + action count; non-fc42 lines do not."""
    store = FakeStore(["stinky lads"])
    lines = [
        "2026-06-20T10:00:00  <fc42> good morning all",
        "2026-06-20T10:01:00  <rdrake> hello there",  # not fc42 -> excluded
        "2026-06-20T10:02:00  * fc42 waves at the room",  # fc42 action -> counted
    ]
    r = build_report([("2026-06-20", lines)], store)
    assert r.pre.fc42_msgs == 2
    assert r.pre.reactions == 0  # nothing long enough / praising


def test_build_report_splits_pre_post_at_rollout_boundary():
    """date < rollout -> pre; date >= rollout -> post (boundary day is post)."""
    store = FakeStore(["stinky lads"])
    r = build_report(
        [
            ("2026-06-21", [f"2026-06-21T10:00:00  <fc42> {_REP}"]),
            ("2026-06-22", [f"2026-06-22T10:00:00  <fc42> {_REP}"]),  # boundary -> post
        ],
        store,
        rollout="2026-06-22",
    )
    assert r.pre.reactions == 1 and r.pre.fc42_msgs == 1
    assert r.post.reactions == 1 and r.post.fc42_msgs == 1


def test_build_report_buckets_by_month_with_active_days():
    """Months aggregate; active_days counts distinct dates with >=1 fc42 message."""
    store = FakeStore(["stinky lads"])
    logs = [
        ("2026-05-30", [f"2026-05-30T10:00:00  <fc42> {_REP}"]),
        ("2026-06-01", [f"2026-06-01T10:00:00  <fc42> {_REP}"]),
        ("2026-06-02", ["2026-06-02T10:00:00  <fc42> just chatting about the football"]),
    ]
    r = build_report(logs, store, rollout="2026-06-22")
    assert [b.label for b in r.buckets] == ["2026-05", "2026-06"]
    june = next(b for b in r.buckets if b.label == "2026-06")
    assert june.fc42_msgs == 2
    assert june.reactions == 1
    assert june.active_days == 2


def test_build_report_file_with_no_fc42_messages_adds_no_active_day():
    """A file with zero fc42 messages contributes no reactions and no active day (both windows)."""
    store = FakeStore([])
    logs = [
        ("2026-06-05", ["2026-06-05T10:00:00  <rdrake> nobody from fc42 is talking here"]),
        ("2026-06-25", ["2026-06-25T10:00:00  <rdrake> still no fc42 in this one either"]),
    ]
    r = build_report(logs, store, rollout="2026-06-22")
    june = next(b for b in r.buckets if b.label == "2026-06")
    assert june.fc42_msgs == 0
    assert june.active_days == 0
    assert june.reactions == 0
    assert r.pre.active_days == 0  # 06-05 < rollout, zero fc42 msgs
    assert r.post.active_days == 0  # 06-25 >= rollout, zero fc42 msgs


# --------------------------------------------------------------------------- #
# Task 3 — wins (global dedup, recency, cap)
# --------------------------------------------------------------------------- #


def test_wins_dedup_global_but_counts_keep_occurrences():
    """A repeated favorite collapses to one win (latest date) yet still counts per bucket."""
    store = FakeStore(["stinky lads"])
    logs = [
        ("2026-06-01", [f"2026-06-01T10:00:00  <fc42> {_REP}"]),
        ("2026-06-10", [f"2026-06-10T10:00:00  <fc42> {_REP}"]),
    ]
    r = build_report(logs, store)
    assert len(r.wins) == 1
    assert r.wins[0].date == "2026-06-10"  # latest occurrence kept
    june = next(b for b in r.buckets if b.label == "2026-06")
    assert june.reactions == 2  # both occurrences still counted


def test_wins_keep_latest_when_input_out_of_order():
    """Win-replacement guard: a later-iterated EARLIER date does not overwrite the latest."""
    store = FakeStore(["stinky lads"])
    logs = [
        ("2026-06-10", [f"2026-06-10T10:00:00  <fc42> {_REP}"]),
        ("2026-06-01", [f"2026-06-01T10:00:00  <fc42> {_REP}"]),  # earlier, iterated second
    ]
    r = build_report(logs, store)
    assert len(r.wins) == 1
    assert r.wins[0].date == "2026-06-10"  # not overwritten by the earlier date


def test_wins_latest_first_and_capped_at_15():
    """19 distinct wins -> 15 returned, newest first."""
    store = FakeStore(["lads"])
    logs = []
    for i in range(1, 20):
        d = f"2026-06-{i:02d}"
        txt = (
            f"the lads pulled off distinct caper number {i} in the great assembly hall "
            "with much fanfare and merriment that lasted well into the small hours indeed"
        )
        logs.append((d, [f"{d}T10:00:00  <fc42> {txt}"]))
    r = build_report(logs, store)
    assert len(r.wins) == 15
    assert r.wins[0].date == "2026-06-19"
    assert r.wins[-1].date == "2026-06-05"


# --------------------------------------------------------------------------- #
# Task 4 — render
# --------------------------------------------------------------------------- #


def test_render_shows_na_for_zero_denominator():
    """Empty input -> pre/post rates render as n/a and the wins section says none."""
    md = render_report(build_report([], FakeStore([])))
    assert "n/a" in md
    assert "none detected" in md.lower()


def test_stat_rows_thin_and_not_thin_branches():
    """_stat_rows flags small buckets; the OR also flags low-reaction busy months."""
    big = "\n".join(_stat_rows([("big", BucketStats("big", 55, 5, 1))]))
    small = "\n".join(_stat_rows([("small", BucketStats("small", 1, 1, 1))]))
    # busy month, few reactions: caught by the `reactions < N` arm of the OR, not by msgs
    busy = "\n".join(_stat_rows([("busy", BucketStats("busy", 200, 2, 30))]))
    assert "thin sample" not in big  # 55 msgs and 5 reactions -> not thin
    assert "thin sample" in small
    assert "thin sample" in busy  # pins the OR: an `and` mutation would drop this


def test_render_win_truncation_boundary():
    """Win text exactly at the cap renders whole; one char over is truncated (pins <=)."""
    at_cap = "x" * 160
    over_cap = "y" * 161
    rep = Report(
        buckets=[],
        pre=BucketStats("pre", 0, 0, 0),
        post=BucketStats("post", 0, 0, 0),
        wins=[Win("2026-06-22", "repaste", at_cap), Win("2026-06-21", "repaste", over_cap)],
        rollout="2026-06-22",
    )
    md = render_report(rep)
    assert at_cap in md  # 160 chars -> not truncated
    assert "y" * 157 + "..." in md  # 161 chars -> truncated to 157 + ellipsis
    assert over_cap not in md


def test_active_days_counts_distinct_dates_not_files():
    """Two files sharing a date in a month collapse to one active day."""
    store = FakeStore([])
    logs = [
        ("2026-06-15", ["2026-06-15T10:00:00  <fc42> morning chatter from the one file"]),
        ("2026-06-15", ["2026-06-15T18:00:00  <fc42> evening chatter from the other file"]),
    ]
    r = build_report(logs, store)
    june = next(b for b in r.buckets if b.label == "2026-06")
    assert june.fc42_msgs == 2  # both files' messages counted
    assert june.active_days == 1  # same date -> one active day, not two


# --------------------------------------------------------------------------- #
# Task 6 — _main CLI wiring (--reactions flag)
# --------------------------------------------------------------------------- #


def test_main_appends_reaction_section(tmp_path: Path):
    # one empty log file named with a date (no entities -> 0 landing reactions)
    log = tmp_path / "#afnet.2026-06-23.log"
    log.write_text("2026-06-23T10:00:00  <fc42> hello\n", encoding="utf-8")
    reactions = tmp_path / "reactions.jsonl"
    reactions.write_text(
        '{"ts": "2026-06-23T10:01:00Z", "sentiment": "approve", "reactor": "fc42",'
        ' "channel": "#afnet", "verse_excerpt": "Methane Max"}\n',
        encoding="utf-8",
    )
    out = tmp_path / "report.md"
    _main(
        [
            str(log),
            "--verse-dir",
            str(tmp_path),
            "--channel",
            "#afnet",
            "--rollout",
            "2026-06-22",
            "--reactions",
            str(reactions),
            "--out",
            str(out),
        ]
    )
    text = out.read_text(encoding="utf-8")
    assert "# Verse landing-rate report" in text  # landing report still present
    assert "## Explicit 👍/👎 reactions" in text  # reaction section appended


def test_main_without_reactions_has_no_reaction_section(tmp_path: Path):
    log = tmp_path / "#afnet.2026-06-23.log"
    log.write_text("2026-06-23T10:00:00  <fc42> hello\n", encoding="utf-8")
    out = tmp_path / "report.md"
    _main([str(log), "--verse-dir", str(tmp_path), "--channel", "#afnet", "--out", str(out)])
    assert "## Explicit" not in out.read_text(encoding="utf-8")
