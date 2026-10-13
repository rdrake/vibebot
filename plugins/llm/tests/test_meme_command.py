"""@meme: the user names the template and the captions, nothing is planned.

No LLM turn, no image model, no cost row. The command resolves the name
through the bundled catalog, draws the captions locally with Pillow, hosts
the PNG through the existing save-image path, and posts the link.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from llm import meme, meme_render
from llm.service import MemePick
from PIL import Image

from .conftest import make_registry_side_effect

_HOSTED = "https://paste.boxlabs.uk/img/img_abc.png"
_BG = "https://i.imgflip.com/1c1uej.jpg"


def _boxes(count: int) -> list[dict[str, float]]:
    return [
        {"anchor_x": 0.0, "anchor_y": i / count, "scale_x": 1.0, "scale_y": 1.0 / count}
        for i in range(count)
    ]


_ENTRIES: list[dict[str, Any]] = [
    {
        "id": "drake",
        "name": "Drakeposting",
        "keywords": [],
        "example": ["left on unread", "left on read"],
        "boxes": _boxes(2),
    },
    {
        "id": "db",
        "name": "Distracted Boyfriend",
        "keywords": ["girlfriend"],
        "example": ["me", "new", "old"],
        "boxes": _boxes(3),
    },
]


def _png_bytes(size: tuple[int, int] = (120, 90)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (30, 60, 90)).save(buf, "PNG")
    return buf.getvalue()


def write_catalog(entries: list[dict[str, Any]], image_dir: Path) -> meme.MemeCatalog:
    """Parse ``entries`` and write a real 200x150 JPEG blank for each."""
    templates = meme.parse_catalog(entries, image_dir)
    for t in templates:
        Image.new("RGB", (200, 150), (90, 140, 60)).save(image_dir / f"{t.id}.jpg", "JPEG")
    return meme.MemeCatalog(templates)


@pytest.fixture
def meme_plugin(plugin_env, mocker, tmp_path):
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    plugin.registryValue.side_effect = make_registry_side_effect({"memeEnabled": True})
    mocker.patch.object(meme, "bundled_catalog", return_value=write_catalog(_ENTRIES, tmp_path))
    plugin.llm_service._save_image_bytes.return_value = _HOSTED
    plugin.llm_service._download_image_bytes.return_value = (_png_bytes(), "png")
    # The picker declines unless a test says otherwise, so a resolver miss
    # still reads as the did-you-mean list.
    plugin.llm_service.meme_pick.return_value = MemePick(
        '{"template": null, "reason": "Nothing fits."}', "test-model"
    )
    return plugin, mock_irc, mock_msg


@pytest.fixture
def rendered(mocker):
    """Spy on the real renderer: which boxes, which lines, which options."""
    return mocker.spy(meme_render, "render")


def _saved(plugin) -> bytes:
    data, ext = plugin.llm_service._save_image_bytes.call_args.args
    assert ext == "png"
    return data


class TestMemeCommand:
    def test_refused_when_memes_are_off(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect({"memeEnabled": False})

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert mock_irc.reply.call_args.args[0] == "Memes are turned off in this channel."
        plugin.llm_service._save_image_bytes.assert_not_called()

    def test_posts_the_hosted_render(self, meme_plugin, rendered) -> None:
        plugin, mock_irc, _ = meme_plugin

        plugin.meme(mock_irc, meme_plugin[2], ["drake | left on unread | left on read"])

        assert rendered.call_args.args[2] == ["left on unread", "left on read"]
        assert len(rendered.call_args.args[1]) == 2
        png = _saved(plugin)
        assert png.startswith(b"\x89PNG")
        assert Image.open(io.BytesIO(png)).size == (200, 150)
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_unknown_template_suggests_and_renders_nothing(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake boyfriend | a | b"])

        plugin.llm_service._save_image_bytes.assert_not_called()
        err = mock_irc.error.call_args.args[0]
        assert "drake boyfriend" in err and "db" in err

    def test_missing_captions_show_the_example(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake"])

        err = mock_irc.error.call_args.args[0]
        assert "2 captions" in err and "left on unread" in err

    def test_save_failure_is_reported(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service._save_image_bytes.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        mock_irc.reply.assert_not_called()
        assert mock_irc.error.call_args.args[0] == "Could not save that meme."

    def test_unreadable_blank_is_a_draw_error(self, meme_plugin, tmp_path) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        (tmp_path / "drake.jpg").write_bytes(b"not a jpeg")

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert mock_irc.error.call_args.args[0] == "Could not draw that meme."
        plugin.llm_service._save_image_bytes.assert_not_called()

    def test_list_filters_by_word(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["list boyfriend"])

        reply = mock_irc.reply.call_args.args[0]
        assert "db" in reply and "Distracted Boyfriend" in reply and "3" in reply
        assert "drake" not in reply

    def test_list_with_no_match_says_so(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["list exercise"])

        reply = mock_irc.reply.call_args.args[0]
        assert "No template matches 'exercise'" in reply and "image URL" in reply

    def test_catalog_unavailable_is_an_error(self, meme_plugin, mocker) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        mocker.patch.object(meme, "bundled_catalog", return_value=None)

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert "unavailable" in mock_irc.error.call_args.args[0]

    def test_requires_an_account(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        mock_irc.state.nickToAccount.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        plugin.llm_service._save_image_bytes.assert_not_called()


class TestPickerInCommand:
    """No template named → the picker chooses, the reply shows the choice."""

    def test_unnamed_request_is_picked_and_rendered(self, meme_plugin, rendered) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "drake", "lines": ["waiting for ci", "pushing anyway"]}',
            "test-model",
            prompt_tokens=5000,
            completion_tokens=20,
            cost=0.001,
        )

        plugin.meme(mock_irc, mock_msg, ["waiting for CI"])

        request = plugin.llm_service.meme_pick.call_args.args[0]
        assert request == "waiting for CI"
        assert (
            "drake | Drakeposting | 2"
            in plugin.llm_service.meme_pick.call_args.kwargs["catalog_brief"]
        )
        assert rendered.call_args.args[2] == ["waiting for ci", "pushing anyway"]
        assert _saved(plugin).startswith(b"\x89PNG")
        assert (
            mock_irc.reply.call_args.args[0]
            == f"{_HOSTED} — drake | waiting for ci | pushing anyway"
        )
        models = [c.args[3] for c in plugin.db.log_usage.call_args_list]
        assert models == ["test-model", "meme-render"]
        assert plugin.db.log_usage.call_args_list[0].args[6] == 0.001

    def test_miss_with_captions_hands_them_to_the_picker(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["how did you get so strong | i do one push-up"])

        assert plugin.llm_service.meme_pick.call_args.args[0] == (
            "how did you get so strong | i do one push-up"
        )
        error = mock_irc.error.call_args.args[0]
        assert error.startswith("Nothing fits. No meme template called 'how did you get so strong'")

    def test_sentence_that_fuzzy_matches_a_name_is_a_request(self, meme_plugin) -> None:
        """Six of seven words overlap Drakeposting's keywords; still a request."""
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "drake", "lines": ["left on unread", "flamethrowers"]}', "test-model"
        )

        plugin.meme(mock_irc, mock_msg, ["left on unread or left on read flamethrowers"])

        assert plugin.llm_service.meme_pick.call_args.args[0] == (
            "left on unread or left on read flamethrowers"
        )
        assert mock_irc.reply.call_args.args[0].endswith("drake | left on unread | flamethrowers")

    def test_fuzzy_name_with_captions_skips_the_picker(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["distracted | me | new | old"])

        plugin.llm_service.meme_pick.assert_not_called()
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_named_template_never_asks_the_picker(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])
        plugin.meme(mock_irc, mock_msg, ["drake"])

        plugin.llm_service.meme_pick.assert_not_called()

    def test_invented_id_never_renders(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "gigachad", "lines": ["a", "b"]}', "test-model"
        )

        plugin.meme(mock_irc, mock_msg, ["gym"])

        plugin.llm_service._save_image_bytes.assert_not_called()
        assert "gigachad" in mock_irc.error.call_args.args[0]

    def test_picker_failure_is_reported_and_booked(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            None, "test-model", error="The meme picker is not answering."
        )

        plugin.meme(mock_irc, mock_msg, ["gym"])

        assert "not answering" in mock_irc.error.call_args.args[0]
        assert plugin.db.log_usage.call_args.kwargs["status"] == "error"


class TestMakeMemeTool:
    """The chat tool: grok transcribes the name the user said, code resolves it."""

    def test_handler_returns_the_hosted_url_as_message(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        schemas, handlers = plugin._build_meme_tool(meme_plugin[2])

        payload = json.loads(
            handlers["make_meme"]({"template": "drake", "lines": ["a", "b"]}).content
        )

        assert payload == {"status": "ok", "message": _HOSTED}
        assert schemas[0]["function"]["name"] == "make_meme"

    def test_handler_error_carries_the_suggestions(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool(meme_plugin[2])

        payload = json.loads(
            handlers["make_meme"]({"template": "drake boyfriend", "lines": ["a"]}).content
        )

        assert "drake" in payload["error"] and "db" in payload["error"]
        plugin.llm_service._save_image_bytes.assert_not_called()

    def test_brief_goes_to_the_picker(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool(meme_plugin[2])
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "drake", "lines": ["a", "b"]}', "test-model"
        )

        payload = json.loads(handlers["make_meme"]({"brief": "a meme about ci"}).content)

        assert plugin.llm_service.meme_pick.call_args.args[0] == "a meme about ci"
        assert payload == {"status": "ok", "message": f"{_HOSTED} — drake | a | b"}

    def test_brief_miss_has_no_did_you_mean(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool(meme_plugin[2])

        payload = json.loads(handlers["make_meme"]({"brief": "a meme about ci"}).content)

        assert payload == {"error": "Nothing fits."}

    def test_nothing_at_all_is_an_error(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool(meme_plugin[2])

        payload = json.loads(handlers["make_meme"]({}).content)

        assert "error" in payload
        plugin.llm_service.meme_pick.assert_not_called()

    def test_handler_tolerates_junk_arguments(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool(meme_plugin[2])

        payload = json.loads(handlers["make_meme"]({"template": 7, "lines": "a|b"}).content)

        assert "error" in payload

    def test_schema_tells_the_model_not_to_choose(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        schemas, _ = plugin._build_meme_tool(meme_plugin[2])
        desc = schemas[0]["function"]["parameters"]["properties"]["template"]["description"]
        assert "as the user" in desc


class TestChatWiring:
    def _result(self):
        from llm.service import AssistantResult

        return AssistantResult(
            content="ok",
            grounding_used=False,
            prompt_tokens=1,
            completion_tokens=1,
            cost=0.0,
            model="m",
        )

    def test_ask_advertises_make_meme_when_enabled(self, meme_plugin) -> None:
        plugin, irc, msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeEnabled": True, "ircLookupEnabled": False, "auroraEnabled": False}
        )
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._result()

        plugin.ask(irc, msg, ["make a drake meme"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        names = [t["function"]["name"] for t in kwargs["extra_tools"]]
        assert names == ["make_meme"]
        assert "make_meme" in kwargs["extra_handlers"]

    def test_ask_omits_make_meme_when_disabled(self, meme_plugin) -> None:
        plugin, irc, msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeEnabled": False, "ircLookupEnabled": False, "auroraEnabled": False}
        )
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_request.side_effect = None
        plugin.llm_service.assistant_request.return_value = self._result()

        plugin.ask(irc, msg, ["hello"])

        assert plugin.llm_service.assistant_request.call_args.kwargs["extra_tools"] is None


class TestCustomAndAliases:
    def test_image_url_template_downloads_and_renders(self, meme_plugin, rendered) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, [f"{_BG} | top | bottom"])

        plugin.llm_service._download_image_bytes.assert_called_once_with(_BG)
        assert rendered.call_args.args[1] == meme_render.TOP_BOTTOM
        assert rendered.call_args.args[2] == ["top", "bottom"]
        png = _saved(plugin)
        assert Image.open(io.BytesIO(png)).size == (120, 90)
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_image_url_download_failure_is_reported(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service._download_image_bytes.return_value = None

        plugin.meme(mock_irc, mock_msg, [f"{_BG} | top | bottom"])

        assert mock_irc.error.call_args.args[0] == "Could not fetch that image."
        plugin.llm_service._save_image_bytes.assert_not_called()
        assert plugin.db.log_usage.call_args.kwargs["status"] == "error"

    def test_url_alias_downloads_its_image(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeEnabled": True, "memeAliases": [f"wojak={_BG}"]}
        )

        plugin.meme(mock_irc, mock_msg, ["wojak | a | b"])

        plugin.llm_service._download_image_bytes.assert_called_once_with(_BG)
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_bundled_template_downloads_nothing(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        plugin.llm_service._download_image_bytes.assert_not_called()

    def test_registry_aliases_reach_the_resolver(self, meme_plugin, rendered) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeEnabled": True, "memeAliases": ["hotline=drake"]}
        )

        plugin.meme(mock_irc, mock_msg, ["hotline bling | a | b"])

        assert rendered.call_args.args[2] == ["a", "b"]
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_tool_accepts_an_image_url_as_template(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool(meme_plugin[2])

        payload = json.loads(handlers["make_meme"]({"template": _BG, "lines": ["a", "b"]}).content)

        assert payload["status"] == "ok"


class TestMemeUsageRow:
    """Free, but attributable: one $0 row per render, like draw's per image."""

    def test_success_writes_a_zero_cost_row_under_meme_render(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        args, kwargs = plugin.db.log_usage.call_args
        assert args == ("testnick", "#test", "meme", "meme-render", 0, 0, 0.0)
        assert kwargs["prompt"] == "drake | a | b"
        assert kwargs["status"] == "success"

    def test_save_failure_is_an_error_row(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service._save_image_bytes.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert plugin.db.log_usage.call_args.kwargs["status"] == "error"

    def test_resolver_miss_books_the_picker_but_not_a_render(self, meme_plugin) -> None:
        """The picker ran and cost tokens; nothing was rendered."""
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake boyfriend | a | b"])

        assert plugin.db.log_usage.call_count == 1
        assert plugin.db.log_usage.call_args.args[3] == "test-model"
        plugin.llm_service._save_image_bytes.assert_not_called()

    def test_tool_path_writes_the_same_row(self, meme_plugin) -> None:
        plugin, _, mock_msg = meme_plugin
        _, handlers = plugin._build_meme_tool(mock_msg)

        handlers["make_meme"]({"template": "drake", "lines": ["a", "b"]})

        args, _ = plugin.db.log_usage.call_args
        assert args[2:4] == ("meme", "meme-render")


class TestMemeOptions:
    def test_font_and_top_flags_reach_the_renderer(self, meme_plugin, rendered) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["--font", "comic", "--top", "drake | a | b"])

        assert rendered.call_args.kwargs == {"font": "comic", "layout_top": True}
        png = _saved(plugin)
        assert Image.open(io.BytesIO(png)).size == (200, 150 + 150 // 4)
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_unknown_font_lists_the_fonts(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["--font", "bogus", "drake | a | b"])

        assert mock_irc.error.call_args.args[0] == "Fonts: thick, thin, comic."
        plugin.llm_service._save_image_bytes.assert_not_called()

    def test_list_shows_id_name_and_count(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["list drake"])

        assert "drake (Drakeposting, 2)" in mock_irc.reply.call_args.args[0]
