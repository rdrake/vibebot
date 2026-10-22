"""Meme templates: the bundled catalog, template resolution, and the renderer.

The bot never picks a template on its own. The user names one ("drake",
"distracted boyfriend"), the catalog resolves that name deterministically, and
:mod:`meme_render` draws the captions locally with Pillow.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from llm import meme, meme_render
from PIL import Image, ImageChops

_NO_IMAGES = Path("/nonexistent/images")


def _boxes(count: int) -> list[dict[str, float]]:
    """``count`` full-width boxes stacked top to bottom."""
    return [
        {"anchor_x": 0.0, "anchor_y": i / count, "scale_x": 1.0, "scale_y": 1.0 / count}
        for i in range(count)
    ]


def _entry(
    template_id: str,
    name: str,
    lines: int,
    keywords: tuple[str, ...] = (),
    example: tuple[str, ...] = (),
    rank: int | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": template_id,
        "name": name,
        "keywords": list(keywords),
        "example": list(example),
        "boxes": _boxes(lines),
    }
    if rank is not None:
        entry["rank"] = rank
    return entry


def _parse(entries: list[dict[str, Any]]) -> list[meme.MemeTemplate]:
    return meme.parse_catalog(entries, _NO_IMAGES)


_TEMPLATES_JSON = [
    _entry("drake", "Drakeposting", 2, example=("left on unread", "left on read")),
    _entry(
        "db",
        "Distracted Boyfriend",
        3,
        keywords=("girlfriend", "cheating"),
        example=("me", "new thing", "old thing"),
    ),
    _entry(
        "fry",
        "Futurama Fry",
        2,
        keywords=("not sure if",),
        example=("not sure if trolling", "or just stupid"),
    ),
    _entry("fine", "This is Fine", 2, keywords=("dog", "fire"), example=("", "this is fine")),
]


@pytest.fixture
def catalog() -> meme.MemeCatalog:
    return meme.MemeCatalog(_parse(_TEMPLATES_JSON))


class TestParseMemeRequest:
    def test_pipe_separated_template_then_lines(self):
        assert meme.parse_meme_request("drake | left on unread | left on read") == (
            "drake",
            ["left on unread", "left on read"],
        )

    def test_template_alone_has_no_lines(self):
        assert meme.parse_meme_request("drake") == ("drake", [])

    def test_blank_is_none(self):
        assert meme.parse_meme_request("   ") is None

    def test_leading_pipe_is_a_blank_template(self):
        assert meme.parse_meme_request("| a | b") is None


class TestParseCatalog:
    def test_reads_the_bundled_shape(self, tmp_path):
        entries = [{**_TEMPLATES_JSON[1], "rank": 7}]
        (db,) = meme.parse_catalog(entries, tmp_path)
        assert db.name == "Distracted Boyfriend"
        assert db.lines == 3 and len(db.boxes) == 3
        assert db.boxes[1] == meme_render.Box(0.0, 1 / 3, 1.0, 1 / 3)
        assert db.keywords == ("girlfriend", "cheating")
        assert db.example == ("me", "new thing", "old thing")
        assert db.image == tmp_path / "db.jpg"
        assert db.rank == 7

    def test_box_fields_beyond_geometry_are_read(self):
        box = {**_boxes(1)[0], "valign": "bottom", "align": "right", "color": "black"}
        (t,) = _parse([{"id": "v", "name": "Valign", "boxes": [box]}])
        assert t.boxes[0].valign == "bottom"
        assert t.boxes[0].align == "right" and t.boxes[0].color == "black"

    def test_missing_rank_sorts_last(self):
        (drake,) = _parse([_TEMPLATES_JSON[0]])
        assert drake.rank == 10_000

    def test_skips_malformed_entries(self):
        templates = _parse(
            [
                {"id": "x"},
                {"name": "no id", "boxes": _boxes(2)},
                {"id": "", "name": "blank id", "boxes": _boxes(2)},
                {"id": 7, "name": "numeric id", "boxes": _boxes(2)},
                "junk",
            ]
        )
        assert templates == []

    def test_entry_without_usable_boxes_is_dropped(self):
        templates = _parse(
            [
                {"id": "none", "name": "No Boxes"},
                {"id": "empty", "name": "Empty Boxes", "boxes": []},
                {"id": "bad", "name": "Bad Box", "boxes": [{"anchor_x": 0.0, "nonsense": 1}]},
            ]
        )
        assert templates == []

    def test_a_bad_box_is_skipped_not_the_template(self):
        entry = {
            "id": "half",
            "name": "Half Good",
            "boxes": [*_boxes(2), {"anchor_x": 0.0}],
        }
        (half,) = _parse([entry])
        assert half.lines == 2

    def test_non_list_is_empty(self):
        assert meme.parse_catalog({"id": "x"}, _NO_IMAGES) == []


class TestBundledCatalog:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self, monkeypatch):
        monkeypatch.setattr(meme, "_catalog_cache", None)

    def test_missing_file_is_none(self, tmp_path):
        assert meme.bundled_catalog(tmp_path / "templates.json") is None

    def test_reads_and_points_images_at_the_images_dir(self, tmp_path):
        path = tmp_path / "templates.json"
        path.write_text(json.dumps(_TEMPLATES_JSON[:1]), encoding="utf-8")

        cat = meme.bundled_catalog(path)

        assert cat is not None
        assert [t.id for t in cat.templates] == ["drake"]
        assert cat.templates[0].image == tmp_path / "images" / "drake.jpg"

    def test_unchanged_file_is_cached(self, tmp_path):
        path = tmp_path / "templates.json"
        path.write_text(json.dumps(_TEMPLATES_JSON), encoding="utf-8")

        assert meme.bundled_catalog(path) is meme.bundled_catalog(path)

    def test_changed_file_is_reread(self, tmp_path):
        path = tmp_path / "templates.json"
        path.write_text(json.dumps(_TEMPLATES_JSON[:1]), encoding="utf-8")
        first = meme.bundled_catalog(path)

        path.write_text(json.dumps(_TEMPLATES_JSON), encoding="utf-8")
        mtime = path.stat().st_mtime + 10
        os.utime(path, (mtime, mtime))
        second = meme.bundled_catalog(path)

        assert first is not None and len(first) == 1
        assert second is not None and len(second) == 4


class TestResolve:
    def test_exact_id(self, catalog):
        assert catalog.resolve("drake").id == "drake"

    def test_id_is_case_insensitive(self, catalog):
        assert catalog.resolve("DRAKE").id == "drake"

    def test_name_substring(self, catalog):
        assert catalog.resolve("distracted boyfriend").id == "db"
        assert catalog.resolve("boyfriend").id == "db"

    def test_keyword(self, catalog):
        assert catalog.resolve("cheating").id == "db"

    def test_articles_and_punctuation_do_not_matter(self, catalog):
        assert catalog.resolve("the this-is-fine").id == "fine"

    def test_no_match_is_none(self, catalog):
        assert catalog.resolve("loss") is None

    def test_ambiguous_substring_is_none(self):
        cat = meme.MemeCatalog(
            _parse([_entry("a", "Grumpy Cat", 2), _entry("b", "Business Cat", 2)])
        )
        assert cat.resolve("cat") is None

    def test_suggest_ranks_closest_first(self, catalog):
        ids = [t.id for t in catalog.suggest("fin", limit=5)]
        assert ids[0] == "fine"


class TestPlanMeme:
    """One entry point for both @meme and the chat tool."""

    def test_happy_path(self, catalog):
        plan = meme.plan_meme(catalog, "drake", ["a", "b"])
        assert plan.error is None
        assert plan.lines == ["a", "b"]
        assert plan.template.id == "drake"

    def test_fewer_lines_are_padded_blank(self, catalog):
        plan = meme.plan_meme(catalog, "fine", ["this is fine"])
        assert plan.lines == ["this is fine", ""]

    def test_captions_are_stripped(self, catalog):
        plan = meme.plan_meme(catalog, "drake", ["  a ", "b  "])
        assert plan.lines == ["a", "b"]

    def test_no_lines_is_an_error(self, catalog):
        plan = meme.plan_meme(catalog, "drake", [])
        assert plan.lines is None
        assert "2 captions" in plan.error
        assert "left on unread" in plan.error

    def test_too_many_lines_names_the_limit(self, catalog):
        plan = meme.plan_meme(catalog, "drake", ["a", "b", "c"])
        assert plan.lines is None
        assert "takes 2 captions" in plan.error
        assert "you gave 3" in plan.error

    def test_tied_template_suggests_the_candidates(self, catalog):
        plan = meme.plan_meme(catalog, "this fry", ["a"])
        assert plan.lines is None and plan.template is None
        assert "this fry" in plan.error
        assert "fine" in plan.error and "fry" in plan.error

    def test_caption_too_long_is_an_error(self, catalog):
        plan = meme.plan_meme(catalog, "drake", ["x" * 300, "b"])
        assert plan.lines is None
        assert "too long" in plan.error


_WONKA_JSON = [
    _entry("wonka", "Condescending Wonka", 2),
    _entry("gb", "Galaxy Brain", 4),
    _entry("sb", "Scumbag Brain", 2),
    _entry("elmo", "Elmo Choosing Cocaine", 5),
    _entry("yallgot", "Y'all Got Any More of Them", 2),
    _entry("pigeon", "Is This a Pigeon?", 3),
    _entry("fine", "This is Fine", 2),
]


@pytest.fixture
def wonka() -> meme.MemeCatalog:
    return meme.MemeCatalog(_parse(_WONKA_JSON))


class TestResolveByTokenOverlap:
    """Templates are named by quote, users name them by character."""

    def test_half_the_words_hitting_one_template_is_enough(self, wonka):
        assert wonka.resolve("willy wonka").id == "wonka"
        assert wonka.resolve("elmo fire").id == "elmo"

    def test_a_tie_is_still_a_miss(self, wonka):
        assert wonka.resolve("expanding brain") is None

    def test_one_word_of_five_is_not_enough(self, wonka):
        assert wonka.resolve("what if i told you them") is None

    def test_more_words_in_common_wins(self, wonka):
        assert wonka.resolve("is this a pigeon").id == "pigeon"
        assert wonka.resolve("this is fine").id == "fine"


class TestSuggestHonesty:
    def test_nothing_in_common_suggests_nothing(self, wonka):
        assert wonka.suggest("tyrone biggums") == []

    def test_partial_overlap_still_suggests(self, wonka):
        assert [t.id for t in wonka.suggest("brain")] == ["gb", "sb"]

    def test_unknown_with_no_suggestions_points_at_list_and_urls(self, wonka):
        plan = meme.plan_meme(wonka, "tyrone", ["a", "b"])
        assert plan.lines is None
        assert "@meme list" in plan.error and "image URL" in plan.error


class TestAliases:
    def test_parse_aliases_reads_name_equals_target(self):
        parsed = meme.parse_aliases(
            ["tyrone=yallgot", "junk", "=x", "y=", "wojak=https://i.example.com/w.png"]
        )
        assert parsed == {"tyrone": "yallgot", "wojak": "https://i.example.com/w.png"}

    def test_id_alias_becomes_a_keyword_on_the_target(self, wonka):
        cat = wonka.with_aliases({"tyrone": "yallgot", "biggums": "yallgot"})
        assert cat.resolve("tyrone").id == "yallgot"
        assert cat.resolve("tyrone biggums").id == "yallgot"
        assert "tyrone" in cat.resolve("yallgot").keywords

    def test_url_alias_becomes_a_two_box_custom_template(self, wonka):
        cat = wonka.with_aliases({"wojak": "https://i.example.com/wojak.png"})
        t = cat.resolve("wojak")
        assert t.lines == 2 and t.background == "https://i.example.com/wojak.png"
        assert t.boxes == meme_render.TOP_BOTTOM
        assert len(cat) == len(wonka) + 1
        plan = meme.plan_meme(cat, "wojak", ["a", "b"])
        assert plan.error is None and plan.lines == ["a", "b"]
        assert plan.template.background == "https://i.example.com/wojak.png"

    def test_alias_to_unknown_id_or_unsafe_url_is_dropped(self, wonka):
        cat = wonka.with_aliases({"a": "nosuch", "b": "http://127.0.0.1/x.png"})
        assert cat.resolve("a") is None and cat.resolve("b") is None
        assert len(cat) == len(wonka)


class TestCustomBackground:
    _BG = "https://i.imgflip.com/1c1uej.jpg"

    def test_url_template_renders_through_custom(self, wonka):
        plan = meme.plan_meme(wonka, self._BG, ["y'all got", "more?"])
        assert plan.error is None
        assert plan.lines == ["y'all got", "more?"]
        assert plan.template.id == "custom"
        assert plan.template.background == self._BG
        assert plan.template.boxes == meme_render.TOP_BOTTOM

    def test_custom_takes_two_captions(self, wonka):
        plan = meme.plan_meme(wonka, self._BG, ["a", "b", "c"])
        assert plan.lines is None and "takes 2 captions" in plan.error

    def test_private_background_is_refused(self, wonka):
        plan = meme.plan_meme(wonka, "http://10.0.0.1/x.png", ["a"])
        assert plan.lines is None and "URL" in plan.error

    def test_parse_request_keeps_the_url_intact(self):
        assert meme.parse_meme_request(f"{self._BG} | a | b") == (self._BG, ["a", "b"])


class TestOptions:
    """Font and layout: the options the local renderer still takes."""

    def test_font_and_top_layout_pass_validation(self, catalog):
        opts = meme.MemeOptions(font="thin", layout_top=True)
        plan = meme.plan_meme(catalog, "drake", ["a", "b"], opts)
        assert plan.error is None and plan.lines == ["a", "b"]

    def test_custom_background_takes_a_font(self, catalog):
        opts = meme.MemeOptions(font="comic")
        plan = meme.plan_meme(catalog, "https://i.example.com/x.png", ["a", "b"], opts)
        assert plan.error is None and plan.template.id == "custom"


class TestTopics:
    """The checked-in keyword column: the catalog's own keywords cover a
    fraction of the templates, so "gym" found nothing until meme_topics
    tagged bd."""

    def test_with_keywords_adds_tags_and_ignores_unknown_ids(self, catalog):
        cat = catalog.with_keywords({"fine": ("burning", "coping"), "nosuch": ("x",)})
        assert cat.resolve("burning").id == "fine"
        assert cat.resolve("coping meme").id == "fine"
        assert cat.resolve("fine").keywords == ("dog", "fire", "burning", "coping")
        assert len(cat) == len(catalog)

    def test_shared_tag_is_ambiguous_not_a_guess(self, catalog):
        cat = catalog.with_keywords({"fine": ("sad",), "fry": ("sad",)})
        assert cat.resolve("sad") is None
        assert {t.id for t in cat.suggest("sad")} == {"fine", "fry"}

    def test_every_topic_id_and_tag_is_well_formed(self):
        from llm import meme_topics

        for template_id, tags in meme_topics.TOPICS.items():
            assert template_id == template_id.lower().strip(), template_id
            assert tags, template_id
            for tag in tags:
                assert meme._normalise(tag), (template_id, tag)

    def test_list_and_suggest_search_the_example_captions(self, catalog):
        assert meme.matches(catalog.resolve("fry"), "trolling")
        assert [t.id for t in catalog.suggest("trolling")] == ["fry"]

    def test_resolve_does_not_read_the_example_captions(self, catalog):
        assert catalog.resolve("trolling") is None

    def test_a_name_word_outranks_topic_tags(self, catalog):
        elmo = meme.MemeTemplate("elmo", "Elmo Choosing Cocaine", 2)
        cat = meme.MemeCatalog([*catalog.templates, elmo]).with_keywords(
            {"fine": ("fire", "burning"), "elmo": ("sesame street",)}
        )
        assert cat.resolve("elmo fire").id == "elmo"

    def test_suggest_ignores_two_letter_words_in_a_longer_query(self):
        cat = meme.MemeCatalog(
            _parse(
                [
                    _entry("money", "Shut Up and Take My Money!", 2),
                    _entry("yuno", "Y U NO Guy", 2),
                ]
            )
        )
        assert cat.suggest("push-up") == []
        assert [t.id for t in cat.suggest("y u no")] == ["yuno"]


class TestCatalogBrief:
    """The picker sees a shortlist, not the whole thousand-row catalog."""

    @pytest.fixture
    def ranked(self) -> meme.MemeCatalog:
        return meme.MemeCatalog(
            _parse(
                [
                    _entry("drake", "Drakeposting", 2, rank=1),
                    _entry("db", "Distracted Boyfriend", 3, rank=2),
                    _entry("fine", "This is Fine", 2, rank=3),
                    _entry("gym", "Gym Bro", 2, keywords=("workout",), rank=500),
                ]
            )
        )

    def test_without_a_request_the_most_popular_are_shown(self, ranked, monkeypatch):
        monkeypatch.setattr(meme, "BRIEF_POPULAR", 2)
        assert [t.id for t in meme.brief_templates(ranked)] == ["drake", "db"]

    def test_request_matches_beyond_the_popular_cut(self, ranked, monkeypatch):
        monkeypatch.setattr(meme, "BRIEF_POPULAR", 2)
        ids = [t.id for t in meme.brief_templates(ranked, "a workout meme")]
        assert ids == ["gym", "drake", "db"]
        assert "gym | Gym Bro | 2" in meme.catalog_brief(ranked, "a workout meme")
        assert "gym |" not in meme.catalog_brief(ranked)

    def test_url_alias_customs_are_always_included(self, ranked, monkeypatch):
        monkeypatch.setattr(meme, "BRIEF_POPULAR", 0)
        cat = ranked.with_aliases({"wojak": "https://i.example.com/wojak.png"})
        assert [t.id for t in meme.brief_templates(cat, "unrelated words")] == ["wojak"]

    def test_no_duplicates_when_a_match_is_also_popular(self, ranked):
        ids = [t.id for t in meme.brief_templates(ranked, "drake")]
        assert ids.count("drake") == 1
        assert len(ids) == len(ranked)

    def test_search_brief_returns_matching_rows_best_first(self, ranked):
        rows = meme.search_brief(ranked, "workout").splitlines()
        assert rows == ["gym | Gym Bro | 2 |  | workout"]

    def test_search_brief_says_so_when_nothing_matches(self, ranked):
        assert meme.search_brief(ranked, "zzzz").startswith("No templates match 'zzzz'")
        assert meme.search_brief(ranked, "  ").startswith("No templates match")

    def test_search_opening_brief_is_short(self, ranked):
        ids = [t.id for t in meme.brief_templates(ranked, popular=1, matches=0)]
        assert ids == ["drake"]

    def test_picture_slot_tag_survives_the_keyword_cut(self, ranked):
        """2026-10-14: imgflip alt names pushed spirit's tag to index 9 of
        an 8-tag row, so the picker stopped adding the costume photo."""
        from llm import meme_topics

        many = tuple(f"alt name {i}" for i in range(12))
        cat = ranked.with_keywords({"fine": many}).with_keywords(
            {"fine": (meme_topics.PICTURE_SLOT,)}
        )
        row = next(r for r in meme.catalog_brief(cat).splitlines() if r.startswith("fine |"))
        assert row.endswith(
            f"| {meme_topics.PICTURE_SLOT}, alt name 0, alt name 1, alt name 2, alt name 3"
        )

    def test_long_name_and_duplicate_tags_stay_out_of_the_brief(self, ranked):
        """imgflip alt names run to sentences; at 8 of them the brief was 63 KB."""
        cat = ranked.with_keywords(
            {
                "fine": (
                    "dog sitting in a burning room drinking coffee",
                    "This Is Fine",
                    "fire",
                    "Fire",
                    "coping",
                )
            }
        )
        row = next(r for r in meme.catalog_brief(cat).splitlines() if r.startswith("fine |"))
        assert row.endswith("| fire, coping")

    def test_a_long_example_is_clipped(self):
        long = ("a caption that goes on and on and on",) * 3
        cat = meme.MemeCatalog(_parse([_entry("wordy", "Wordy", 3, example=long)]))
        example = meme.catalog_brief(cat).split(" | ")[3]
        assert len(example) <= 60 and example.endswith("…")

    def test_bundled_slotted_templates_have_masks_and_the_tag(self):
        from llm import meme_topics

        catalog = meme.bundled_catalog()
        assert catalog is not None
        slotted = [t for t in catalog.templates if t.slots]
        assert {"spirit", "same", "who-would-win", "milk-carton"} <= {t.id for t in slotted}
        for t in slotted:
            assert meme_topics.PICTURE_SLOT in t.keywords, t.id
            for slot in t.slots:
                assert slot.mask.exists(), slot.mask
                x, y, w, h = slot.box
                assert min(x, y) >= 0 and max(x + w, y + h) <= 1.0001, t.id

    def test_a_multi_slot_row_says_how_many_pictures(self, tmp_path):
        Image.new("L", (10, 10), 255).save(tmp_path / "m.png")
        entry = _entry("www", "Who Would Win?", 2)
        entry["slots"] = [{"box": [0, 0, 0.5, 1], "mask": "m.png"}] * 2
        (tmp_path / "images").mkdir()
        row = meme.catalog_brief(meme.MemeCatalog(meme.parse_catalog([entry], tmp_path / "images")))
        assert row.endswith("| blank picture area x2")


class TestSlots:
    """Picture slots: where a generated picture is masked into a template."""

    def test_parse_catalog_reads_slots_and_tags_the_template(self, tmp_path):
        from llm import meme_topics

        entry = _entry("spirit", "Spirit", 1)
        entry["slots"] = [{"box": [0.5, 0.1, 0.4, 0.8], "mask": "spirit-0.png"}, {"box": "bad"}]
        (t,) = meme.parse_catalog([entry], tmp_path / "images")
        assert len(t.slots) == 1
        assert t.slots[0].box == (0.5, 0.1, 0.4, 0.8)
        assert t.slots[0].mask == tmp_path / "masks" / "spirit-0.png"
        assert t.keywords[0] == meme_topics.PICTURE_SLOT

    @pytest.mark.parametrize(
        ("draw", "count", "expected"),
        [
            ("a cat", 2, ["a cat", "a cat"]),
            ("a cat | a dog", 2, ["a cat", "a dog"]),
            ("a cat | a dog | a cow", 2, ["a cat", "a dog"]),
            ("a | b", 4, ["a", "b", "b", "b"]),
        ],
    )
    def test_slot_drawings(self, draw, count, expected):
        assert meme.slot_drawings(draw, count) == expected

    @pytest.mark.parametrize(
        ("size", "ratio"),
        [((500, 500), "1:1"), ((1000, 560), "16:9"), ((300, 410), "2:3"), ((100, 400), "9:16")],
    )
    def test_nearest_aspect_ratio(self, size, ratio):
        assert meme.nearest_aspect_ratio(*size) == ratio

    def test_picture_lands_inside_the_mask_only(self, tmp_path):
        base = Image.new("RGB", (100, 100), "white")
        mask = Image.new("L", (100, 100), 0)
        mask.paste(255, (60, 20, 90, 80))  # the slot's blank, inside a wider box
        mask.save(tmp_path / "m.png")
        slot = meme_render.Slot((0.5, 0.1, 0.45, 0.8), tmp_path / "m.png")

        out = meme_render.insert_pictures(base, (slot,), [Image.new("RGB", (40, 40), "red")])

        assert out.getpixel((75, 50)) == (255, 0, 0)
        assert out.getpixel((75, 21)) == (255, 0, 0)  # the mask's top row: pasted at the box top
        assert out.getpixel((55, 50)) == (255, 255, 255)  # in the box, outside the mask
        assert out.getpixel((10, 10)) == (255, 255, 255)
        assert base.getpixel((75, 50)) == (255, 255, 255)  # input untouched

    def test_corners_are_parsed_only_when_there_are_four(self, tmp_path):
        entry = _entry("same", "Same", 1)
        entry["slots"] = [
            {
                "box": [0, 0, 1, 1],
                "mask": "a.png",
                "corners": [[0.1, 0], [1, 0.1], [0.9, 1], [0, 0.9]],
            },
            {"box": [0, 0, 1, 1], "mask": "b.png", "corners": [[0, 0]]},
        ]
        (t,) = meme.parse_catalog([entry], tmp_path / "images")
        assert t.slots[0].corners == ((0.1, 0.0), (1.0, 0.1), (0.9, 1.0), (0.0, 0.9))
        assert t.slots[1].corners == ()

    def test_a_cornered_slot_warps_the_picture_onto_the_quad(self, tmp_path):
        """A picture split red (left) / blue (right) keeps that split along
        a tilted quad: its left edge lands on the quad's left edge."""
        Image.new("L", (100, 100), 255).save(tmp_path / "m.png")
        corners = ((0.3, 0.1), (0.9, 0.2), (0.7, 0.9), (0.1, 0.8))
        slot = meme_render.Slot((0.1, 0.1, 0.8, 0.8), tmp_path / "m.png", corners)
        picture = Image.new("RGB", (200, 200), "blue")
        picture.paste((255, 0, 0), (0, 0, 100, 200))

        out = meme_render.insert_pictures(Image.new("RGB", (100, 100), "white"), (slot,), [picture])

        assert out.getpixel((25, 60))[0] > 200  # near the left edge: red
        assert out.getpixel((70, 40))[2] > 200  # near the right edge: blue

    def test_the_quad_corners_show_the_picture_corners(self, tmp_path):
        """Each corner of the tilted quad shows the matching corner of the
        picture, and the quad's diagonals cross at the picture's centre (a
        perspective map keeps straight lines straight). The picture encodes
        its own coordinates: red is x, green is y."""
        Image.new("L", (200, 200), 255).save(tmp_path / "m.png")
        # Not a parallelogram: a true perspective, not just an affine skew.
        corners = ((0.3, 0.1), (0.75, 0.15), (0.9, 0.9), (0.1, 0.8))
        slot = meme_render.Slot((0.1, 0.1, 0.8, 0.8), tmp_path / "m.png", corners)
        picture = Image.new("RGB", (256, 256))
        picture.putdata([(x, y, 0) for y in range(256) for x in range(256)])

        out = meme_render.insert_pictures(Image.new("RGB", (200, 200), "white"), (slot,), [picture])

        quad = [(x * 200, y * 200) for x, y in corners]
        middle = (sum(x for x, _ in quad) / 4, sum(y for _, y in quad) / 4)
        expected = [(0, 0), (255, 0), (255, 255), (0, 255)]  # TL, TR, BR, BL
        for (x, y), (u, v) in zip(quad, expected, strict=True):
            # A tenth of the way in from the corner, so the probe is inside.
            probe = (int(x + (middle[0] - x) / 10), int(y + (middle[1] - y) / 10))
            red, green, _ = out.getpixel(probe)
            assert abs(red - u) <= 20 and abs(green - v) <= 20, (probe, (red, green))
        # The diagonals TL-BR and TR-BL cross at the picture's centre.
        (ax, ay), (bx, by), (cx, cy), (dx, dy) = quad
        t = ((ax - bx) * (by - dy) - (ay - by) * (bx - dx)) / (
            (ax - cx) * (by - dy) - (ay - cy) * (bx - dx)
        )
        red, green, _ = out.getpixel((round(ax + t * (cx - ax)), round(ay + t * (cy - ay))))
        assert abs(red - 128) <= 4 and abs(green - 128) <= 4

    def test_warp_coefficients_map_each_quad_corner_to_its_picture_corner(self, monkeypatch):
        """Pillow's PERSPECTIVE maps an output (x, y) to the input
        ((a x + b y + c) / (g x + h y + 1), (d x + e y + f) / (g x + h y + 1)):
        each corner of the quad must land exactly on the picture's corner."""
        seen: list[tuple[float, ...]] = []
        transform = Image.Image.transform

        def spy(self, size, method, data=None, *args, **kwargs):
            seen.append(tuple(data))
            return transform(self, size, method, data, *args, **kwargs)

        monkeypatch.setattr(Image.Image, "transform", spy)
        quad = [(60.0, 20.0), (150.0, 30.0), (180.0, 180.0), (20.0, 160.0)]
        meme_render._warp(Image.new("RGB", (256, 128)), quad, (200, 200))

        (a, b, c, d, e, f, g, h) = seen[-1]
        for (x, y), (u, v) in zip(quad, [(0, 0), (256, 0), (256, 128), (0, 128)], strict=True):
            w = g * x + h * y + 1
            assert (a * x + b * y + c) / w == pytest.approx(u, abs=1e-6)
            assert (d * x + e * y + f) / w == pytest.approx(v, abs=1e-6)

    def test_slot_size_is_the_box_in_pixels(self, tmp_path):
        slot = meme_render.Slot((0.0, 0.0, 0.5, 0.25), tmp_path / "m.png")
        assert meme_render.slot_size(Image.new("RGB", (200, 100)), slot) == (100, 25)

    @staticmethod
    def _framed() -> Image.Image:
        """10x10: a red frame two pixels thick around a green centre."""
        picture = Image.new("RGB", (10, 10), (255, 0, 0))
        picture.paste((0, 255, 0), (2, 2, 8, 8))
        return picture

    def _cover(self, tmp_path, size: tuple[int, int]) -> Image.Image:
        Image.new("L", size, 255).save(tmp_path / "m.png")
        slot = meme_render.Slot((0, 0, 1, 1), tmp_path / "m.png")
        return meme_render.insert_pictures(
            Image.new("RGB", size, "white"), (slot,), [self._framed()]
        )

    @staticmethod
    def _is(pixel: tuple[int, int, int], colour: str) -> bool:
        red, green, blue = pixel
        if colour == "red":
            return red > 200 and green < 60 and blue < 60
        return green > 200 and red < 60 and blue < 60

    def test_a_square_picture_covers_a_tall_slot_without_stretching(self, tmp_path):
        """Scaled 2x to cover 6x20: the whole height shows (frame at top and
        bottom) and the sides are cropped evenly, so the middle row is all
        centre — a stretched picture would show the frame at the sides."""
        out = self._cover(tmp_path, (6, 20))
        assert self._is(out.getpixel((3, 1)), "red")
        assert self._is(out.getpixel((3, 18)), "red")
        for x in range(6):
            assert self._is(out.getpixel((x, 10)), "green"), x

    def test_a_square_picture_covers_a_wide_slot_without_stretching(self, tmp_path):
        out = self._cover(tmp_path, (20, 6))
        assert self._is(out.getpixel((1, 3)), "red")
        assert self._is(out.getpixel((18, 3)), "red")
        for y in range(6):
            assert self._is(out.getpixel((10, y)), "green"), y

    def test_cover_crops_the_overflow_from_the_centre(self, tmp_path):
        """A picture in thirds (red, green, blue) covering a square slot
        shows only its middle third, whichever way it overflows."""
        Image.new("L", (10, 10), 255).save(tmp_path / "m.png")
        slot = meme_render.Slot((0, 0, 1, 1), tmp_path / "m.png")
        tall = Image.new("RGB", (10, 30), (255, 0, 0))
        tall.paste((0, 255, 0), (0, 10, 10, 20))
        tall.paste((0, 0, 255), (0, 20, 10, 30))
        for picture in (tall, tall.transpose(Image.Transpose.TRANSPOSE)):
            out = meme_render.insert_pictures(
                Image.new("RGB", (10, 10), "white"), (slot,), [picture]
            )
            for xy in ((0, 0), (9, 0), (0, 9), (9, 9), (5, 5)):
                assert self._is(out.getpixel(xy), "green"), xy

    def test_a_missing_picture_leaves_its_slot_blank(self, tmp_path):
        Image.new("L", (50, 50), 255).save(tmp_path / "m.png")
        slot = meme_render.Slot((0, 0, 1, 1), tmp_path / "m.png")
        base = Image.new("RGB", (50, 50), "white")
        out = meme_render.insert_pictures(base, (slot,), [None])
        assert ImageChops.difference(out, base).getbbox() is None


class TestPicker:
    """The model's answer is a proposal; the catalog decides whether it stands."""

    def test_catalog_brief_is_one_line_per_template(self, catalog):
        brief = meme.catalog_brief(catalog.with_keywords({"fine": ("coping",)}))
        rows = brief.splitlines()
        assert len(rows) == len(catalog)
        assert "fine | This is Fine | 2 | _ / this is fine | dog, fire, coping" in rows
        assert rows[0].startswith("drake | Drakeposting | 2 | left on unread / left on read")

    def test_valid_pick_resolves_and_keeps_lines(self, catalog):
        choice = meme.parse_pick('{"template": "drake", "lines": ["a", "b"]}', catalog)
        assert choice.template.id == "drake" and choice.lines == ["a", "b"]

    def test_pick_tolerates_fences_and_names(self, catalog):
        text = '```json\n{"template": "Distracted Boyfriend", "lines": ["me", "x", "y"]}\n```'
        assert meme.parse_pick(text, catalog).template.id == "db"

    def test_extra_lines_are_cut_not_refused(self, catalog):
        choice = meme.parse_pick('{"template": "drake", "lines": ["a", "b", "c"]}', catalog)
        assert choice.lines == ["a", "b"]

    def test_null_template_returns_the_reason(self, catalog):
        assert meme.parse_pick('{"template": null, "reason": "No gym meme."}', catalog) == (
            "No gym meme."
        )

    def test_invented_id_is_a_miss(self, catalog):
        result = meme.parse_pick('{"template": "gigachad", "lines": ["a"]}', catalog)
        assert isinstance(result, str) and "gigachad" in result

    @pytest.mark.parametrize(
        "content",
        [None, "", "sure!", '{"template": "drake"}', '{"template": "drake", "lines": []}'],
    )
    def test_unusable_answers_are_misses(self, catalog, content):
        assert isinstance(meme.parse_pick(content, catalog), str)

    def test_pick_carries_an_optional_draw(self, catalog):
        text = '{"template": "drake", "lines": ["a", "b"], "draw": "a tired dad"}'
        assert meme.parse_pick(text, catalog).draw == "a tired dad"
        assert meme.parse_pick('{"template": "drake", "lines": ["a", "b"]}', catalog).draw is None


def _picture(size: tuple[int, int] = (200, 150)) -> Image.Image:
    """A two-tone picture, so drawn text changes pixels whatever its colour."""
    image = Image.new("RGB", size, (40, 90, 160))
    image.paste((200, 120, 40), (0, size[1] // 2, size[0], size[1]))
    return image


def _decode(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


class TestRender:
    """meme_render.render: captions in, PNG bytes out."""

    def test_returns_a_png_the_size_of_the_input(self):
        png = meme_render.render(_picture(), meme_render.TOP_BOTTOM, ["top text", "bottom text"])
        assert png.startswith(b"\x89PNG")
        out = _decode(png)
        assert out.size == (200, 150)
        assert out.tobytes() != _picture().tobytes()

    def test_layout_top_adds_a_quarter_height_band(self):
        png = meme_render.render(
            _picture(), meme_render.TOP_BOTTOM, ["one", "two"], layout_top=True
        )
        out = _decode(png)
        assert out.size == (200, 150 + 150 // 4)
        assert out.getpixel((0, 0)) == (255, 255, 255)

    def test_blank_lines_draw_nothing(self):
        png = meme_render.render(_picture(), meme_render.TOP_BOTTOM, ["", "   "])
        assert _decode(png).tobytes() == _picture().tobytes()

    def test_each_font_renders_and_differs(self):
        outputs = {
            name: meme_render.render(
                _picture(), meme_render.TOP_BOTTOM, ["font check", "WIDE CAPTION"], font=name
            )
            for name in meme.FONTS
        }
        for png in outputs.values():
            assert _decode(png).size == (200, 150)
        assert len({_decode(png).tobytes() for png in outputs.values()}) == len(meme.FONTS)

    def test_load_image_flattens_transparency(self):
        rgba = Image.new("RGBA", (20, 10), (0, 0, 0, 0))
        buf = io.BytesIO()
        rgba.save(buf, "PNG")
        image = meme_render.load_image(buf.getvalue())
        assert image.mode == "RGB" and image.getpixel((0, 0)) == (255, 255, 255)

    def test_load_image_keeps_opaque_pixels(self):
        rgba = Image.new("RGBA", (20, 10), (255, 0, 0, 255))
        buf = io.BytesIO()
        rgba.save(buf, "PNG")
        assert meme_render.load_image(buf.getvalue()).getpixel((0, 0)) == (255, 0, 0)

    def test_layout_top_puts_one_row_per_caption_in_the_band(self):
        """--top: the picture sits whole below the band, and each caption
        gets its own row inside the band (lowercase, descenders and all)."""
        picture = Image.new("RGB", (200, 160), (255, 0, 0))
        out = _decode(
            meme_render.render(
                picture, meme_render.TOP_BOTTOM, ["going up", "and away"], layout_top=True
            )
        )
        band = 160 // 4
        assert out.size == (200, 160 + band)
        assert out.crop((0, band, 200, 160 + band)).tobytes() == picture.tobytes()
        white = Image.new("RGB", (200, band // 2), "white")
        for top in (0, band // 2):
            row = out.crop((0, top, 200, top + band // 2))
            assert ImageChops.difference(row, white).getbbox() is not None, top

    def test_wrapped_captions_stack_inside_the_box(self):
        """A caption wrapped to three rows is drawn as three legible rows:
        taller than one row would be, and every row inside the box."""
        size = (200, 400)
        bbox = _ink(
            meme_render.render(
                _gray(size), (meme_render.Box(0, 0, 1, 1),), ["hello there my good friend"]
            ),
            size,
        )
        assert bbox[1] > 0 and bbox[3] < size[1]
        assert 120 < bbox[3] - bbox[1] < 240

    def test_an_overlong_caption_still_renders(self):
        caption = " ".join(["word"] * 100)
        png = meme_render.render(_picture(), meme_render.TOP_BOTTOM, [caption, caption])
        assert _decode(png).size == (200, 150)


def _gray(size: tuple[int, int]) -> Image.Image:
    return Image.new("RGB", size, (128, 128, 128))


def _ink(png: bytes, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """The bounding box of every pixel the render changed on a gray blank."""
    bbox = ImageChops.difference(_decode(png), _gray(size)).getbbox()
    assert bbox is not None, "nothing was drawn"
    return bbox


def _centre(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


class TestBoxPlacement:
    """Where a caption lands inside its box: valign, align, rotation, outline."""

    def test_valign_top_and_bottom_in_a_tall_box(self):
        size = (200, 400)
        top = _ink(
            meme_render.render(_gray(size), (meme_render.Box(0, 0, 1, 1, valign="top"),), ["hi"]),
            size,
        )
        bottom = _ink(
            meme_render.render(
                _gray(size), (meme_render.Box(0, 0, 1, 1, valign="bottom"),), ["hi"]
            ),
            size,
        )
        assert top[3] < size[1] // 2
        assert bottom[1] > size[1] // 2
        # Pushed to the edge, not past it: a pad (a twentieth of the box)
        # stays clear and the caption is as tall as a middle-aligned one.
        middle = _ink(meme_render.render(_gray(size), (meme_render.Box(0, 0, 1, 1),), ["hi"]), size)
        pad = size[1] / 20
        assert top[1] >= pad - 2
        assert bottom[3] <= size[1] - pad + 2
        assert top[3] - top[1] == bottom[3] - bottom[1] == middle[3] - middle[1]

    def test_valign_defaults_to_middle(self):
        size = (200, 400)
        bbox = _ink(meme_render.render(_gray(size), (meme_render.Box(0, 0, 1, 1),), ["hi"]), size)
        assert bbox[1] < size[1] // 2 < bbox[3]

    def test_align_left_and_right_in_a_wide_box(self):
        size = (400, 100)
        left = _ink(
            meme_render.render(_gray(size), (meme_render.Box(0, 0, 1, 1, align="left"),), ["hi"]),
            size,
        )
        right = _ink(
            meme_render.render(_gray(size), (meme_render.Box(0, 0, 1, 1, align="right"),), ["hi"]),
            size,
        )
        assert left[2] < size[0] // 2
        assert right[0] > size[0] // 2

    def test_rotated_box_turns_about_its_centre(self):
        size = (300, 300)
        box = meme_render.Box(1 / 3, 1 / 3, 1 / 3, 1 / 3)
        flat = _centre(_ink(meme_render.render(_gray(size), (box,), ["XOX"]), size))
        turned = _centre(
            _ink(meme_render.render(_gray(size), (replace(box, angle=30.0),), ["XOX"]), size)
        )
        assert abs(turned[0] - flat[0]) <= 6 and abs(turned[1] - flat[1]) <= 6

    def test_an_angled_box_turns_its_caption(self):
        size = (300, 300)
        box = meme_render.Box(1 / 3, 1 / 3, 1 / 3, 1 / 3)
        flat = _ink(meme_render.render(_gray(size), (box,), ["XOX"]), size)
        turned = _ink(meme_render.render(_gray(size), (replace(box, angle=30.0),), ["XOX"]), size)
        # Turned 30 degrees, a wide word stands taller than it does flat.
        assert turned[3] - turned[1] > (flat[3] - flat[1]) * 1.5

    def test_a_wide_angled_box_turns_whole_about_its_own_centre(self):
        """An off-centre, non-square box: the turned caption is not clipped
        to the box's flat outline and stays centred on the box."""
        size = (400, 300)
        box = meme_render.Box(0.1, 0.3, 0.8, 0.2, angle=20.0)  # centre (200, 120), 60 px tall
        bbox = _ink(meme_render.render(_gray(size), (box,), ["hello there my friend"]), size)
        assert bbox[3] - bbox[1] > 60
        centre = _centre(bbox)
        assert abs(centre[0] - 200) <= 6 and abs(centre[1] - 120) <= 6

    def test_centred_caption_in_a_wide_band(self):
        size = (400, 300)
        bbox = _ink(meme_render.render(_gray(size), (meme_render.Box(0, 0, 1, 0.2),), ["hi"]), size)
        assert abs(_centre(bbox)[0] - 200) <= 6

    def test_black_text_gets_a_white_outline(self):
        size = (200, 150)
        png = meme_render.render(
            _gray(size), (meme_render.Box(0, 0, 1, 1, color="black"),), ["outline"]
        )
        (r_lo, r_hi), _g, _b = _decode(png).getextrema()
        assert r_lo < 20 and r_hi > 235

    def test_stroke_colours(self):
        assert meme_render._stroke("black", 3) == (3, "white")
        assert meme_render._stroke("white", 3) == (3, "black")
        assert meme_render._stroke("#ff0000", 3) == (1, "#000000")


class TestCaptionText:
    """Caption casing, row splits and font fitting (memegen's text rules)."""

    @pytest.mark.parametrize(
        ("style", "text", "lines", "expected"),
        [
            # upper: the Box default, the classic all-caps meme.
            ("upper", "top text", ["top text"], "TOP TEXT"),
            # none: the user's casing, verbatim (--top bands, label text).
            ("none", "keep Me", ["keep Me"], "keep Me"),
            ("none", "i know", ["i know"], "i know"),
            ("none", "abcd", ["abcd"], "abcd"),
            # default: an all-lowercase meme gets capitalised; "I", never "i".
            ("default", "i know", ["i know", "ok"], "I know"),
            ("default", "hello there", ["hello there", "bye"], "Hello there"),
            ("default", "so i said", ["so i said"], "So I said"),
            # ...but mixed case stays as typed, sentence or not.
            ("default", "I use NixOS", ["I use NixOS"], "I use NixOS"),
            ("default", "Go NixOS.", ["Go NixOS."], "Go NixOS."),
            ("default", "hello there", ["hello there", "BYE"], "hello there"),
            # mock: sPoNgEbOb alternation.
            ("mock", "abcd", ["abcd"], "aBcD"),
            ("mock", "abcdef", ["abcdef"], "aBcDeF"),
        ],
    )
    def test_stylize(self, style, text, lines, expected):
        assert meme_render._stylize(style, text, lines) == expected

    def test_split_2_breaks_at_the_central_space(self):
        assert meme_render._split_2("hello world") == "hello\nworld"

    def test_split_3_makes_three_rows_of_about_a_third_each(self):
        out = meme_render._split_3("aa bb cc dd ee ff")
        assert "\n\n" not in out
        assert [row.strip() for row in out.split("\n")] == ["aa bb", "cc dd", "ee ff"]

    def test_fit_font_uses_the_full_size_when_the_text_fits(self):
        assert meme_render._fit_font("thick", "hi", (400, 100), 50).size == 50

    def test_fit_font_shrinks_a_long_caption_to_the_box_width(self):
        text = "when you finally fix the bug and three more appear"
        font = meme_render._fit_font("thick", text, (200, 30), 30)
        assert meme_render.MINIMUM_FONT_SIZE < font.size < 30
        assert meme_render._text_size(text, font)[0] <= 200

    def test_fit_font_bounds_width_even_in_a_tall_box(self):
        text = "when you finally fix the bug"
        font = meme_render._fit_font("thick", text, (100, 400), 60)
        assert meme_render._text_size(text, font)[0] <= 100 * 34 / 35

    def test_fit_font_in_a_wide_short_band_is_height_limited(self):
        """A 400x60 band (TOP_BOTTOM's shape): the height sets the size."""
        font = meme_render._fit_font("thick", "HELLO WORLD", (400, 60), 60)
        left, top, right, bottom = font.getbbox("HELLO WORLD")  # the ink
        assert font.size > 30
        assert right - left <= 400 and bottom - top <= 60

    def test_wrap_keeps_a_short_caption_on_one_row(self):
        assert meme_render._wrap("thick", "one does not", (600, 100), 60) == "one does not"

    def test_wrap_keeps_one_row_when_it_is_as_big_as_two(self):
        # One row only just fits at full size; two or three rows would be
        # no bigger, so the caption stays one row.
        assert meme_render._wrap("thick", "nice try", (196, 400), 60) == "nice try"

    def test_wrap_uses_three_rows_for_a_long_caption(self):
        text = "when you finally fix the bug and then three more appear right away"
        assert meme_render._wrap("thick", text, (400, 150), 60).count("\n") == 2

    def test_wrap_uses_two_rows_when_three_would_be_too_narrow(self):
        text = "hello there my good friend"
        assert meme_render._wrap("thick", text, (60, 30), 40).count("\n") == 1

    def test_wrap_keeps_one_row_when_wrapped_text_would_be_too_narrow(self):
        # Two rows fit bigger, but would fill under 60% of the width.
        assert meme_render._wrap("thick", "hello world", (240, 400), 60) == "hello world"

    def test_wrap_splits_an_overlong_caption_in_two(self):
        text = " ".join(["word"] * 100)
        out = meme_render._wrap("thick", text, (200, 30), 16)
        assert isinstance(out, str) and out.count("\n") == 1
