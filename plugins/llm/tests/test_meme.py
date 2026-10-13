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

    def test_fonts_are_the_bundled_three(self):
        assert meme.FONTS == ("thick", "thin", "comic")

    def test_font_and_top_layout_pass_validation(self, catalog):
        opts = meme.MemeOptions(font="thin", layout_top=True)
        plan = meme.plan_meme(catalog, "drake", ["a", "b"], opts)
        assert plan.error is None and plan.lines == ["a", "b"]

    def test_unknown_font_lists_the_fonts(self, catalog):
        plan = meme.plan_meme(catalog, "drake", ["a", "b"], meme.MemeOptions(font="impact"))
        assert plan.lines is None
        assert plan.error == "Fonts: thick, thin, comic."

    def test_custom_background_takes_a_font(self, catalog):
        opts = meme.MemeOptions(font="comic")
        plan = meme.plan_meme(catalog, "https://i.example.com/x.png", ["a", "b"], opts)
        assert plan.error is None and plan.template.id == "custom"

    def test_describe_short(self, catalog):
        assert meme.describe_short(catalog.resolve("drake")) == "drake (Drakeposting, 2)"


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
