"""@meme --draw: the captioned meme goes through xAI's image edit.

The captions are already pixels when the edit runs, so the text survives
whatever the model does to the picture, and a refusal costs the picture, not
the meme. Every outcome is booked: xAI bills refusals too.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
from typing import TYPE_CHECKING

import pytest
from llm import meme
from llm.service import ImageResult, LLMService

from .conftest import make_registry_side_effect
from .test_meme_command import write_catalog

if TYPE_CHECKING:
    pass

CAPTIONED = "https://paste.boxlabs.uk/img/img_captioned.png"
EDITED = "https://paste.boxlabs.uk/img/img_edited.png"
PNG = b"\x89PNG\r\n\x1a\nedited"


@pytest.fixture
def service(make_service, monkeypatch) -> LLMService:  # type: ignore[no-untyped-def]
    svc, _plugin = make_service(memeEditModel="xai/grok-imagine-image", drawTimeout=30)
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    svc._save_image_bytes = lambda data, ext="png": EDITED if data == PNG else None  # type: ignore[method-assign]
    return svc


def _response(payload: dict) -> io.BytesIO:
    body = io.BytesIO(json.dumps(payload).encode())
    body.__enter__ = lambda self=body: self  # type: ignore[attr-defined]
    body.__exit__ = lambda self=body, *a: None  # type: ignore[attr-defined]
    return body


class TestImageEdit:
    def test_success_hosts_the_result_and_prices_from_ticks(self, service, mocker) -> None:
        opened = mocker.patch(
            "urllib.request.urlopen",
            return_value=_response(
                {
                    "data": [{"b64_json": base64.b64encode(PNG).decode()}],
                    "usage": {"cost_in_usd_ticks": 220000000},
                }
            ),
        )

        result = service.image_edit(CAPTIONED, "a tired dad in a lawn chair", channel="#c")

        assert result.url == EDITED and result.error is None
        assert result.cost == pytest.approx(0.022)
        request = opened.call_args.args[0]
        body = json.loads(request.data)
        assert request.full_url == "https://api.x.ai/v1/images/edits"
        assert body["model"] == "grok-imagine-image"
        assert body["image"] == {"url": CAPTIONED}
        assert "a tired dad in a lawn chair" in body["prompt"]
        assert "Keep every letter" in body["prompt"]
        assert request.get_header("Authorization") == "Bearer xai-test"

    def test_refusal_is_an_error_and_still_billed(self, service, mocker) -> None:
        err = urllib.error.HTTPError(
            "https://api.x.ai/v1/images/edits",
            400,
            "Bad Request",
            {},  # type: ignore[arg-type]
            io.BytesIO(
                b'{"error": "content moderation", "usage": {"cost_in_usd_ticks": 200000000}}'
            ),
        )
        mocker.patch("urllib.request.urlopen", side_effect=err)

        result = service.image_edit(CAPTIONED, "something refused")

        assert result.url is None and "refused" in result.error
        assert result.cost == pytest.approx(0.02)

    def test_network_failure_is_free(self, service, mocker) -> None:
        mocker.patch("urllib.request.urlopen", side_effect=OSError("boom"))

        result = service.image_edit(CAPTIONED, "x")

        assert result.error == "The image model failed." and result.cost == 0.0

    def test_non_xai_model_is_refused_before_any_call(self, make_service, mocker) -> None:
        svc, _ = make_service(memeEditModel="gemini/gemini-2.5-flash-image")
        opened = mocker.patch("urllib.request.urlopen")

        result = svc.image_edit(CAPTIONED, "x")

        assert "xai/" in result.error
        opened.assert_not_called()


def _boxes(count: int) -> list[dict[str, float]]:
    return [
        {"anchor_x": 0.0, "anchor_y": i / count, "scale_x": 1.0, "scale_y": 1.0 / count}
        for i in range(count)
    ]


_ENTRIES = [
    {"id": "spirit", "name": "Fake Spirit Halloween Costume", "boxes": _boxes(5)},
    {"id": "drake", "name": "Drakeposting", "boxes": _boxes(2)},
]


@pytest.fixture
def meme_plugin(plugin_env, mocker, tmp_path):
    from llm.service import MemePick

    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    plugin.registryValue.side_effect = make_registry_side_effect({"memeEnabled": True})
    mocker.patch.object(meme, "bundled_catalog", return_value=write_catalog(_ENTRIES, tmp_path))
    plugin.llm_service._save_image_bytes.return_value = CAPTIONED
    plugin.llm_service.image_edit.return_value = ImageResult(
        content=EDITED, cost=0.022, model="xai/grok-imagine-image", url=EDITED
    )
    plugin.llm_service.meme_pick.return_value = MemePick(
        '{"template": null, "reason": "Nothing fits."}', "test-model"
    )
    return plugin, mock_irc, mock_msg


class TestPictureFromThePicker:
    """No flag: the picker writes the edit instruction when the request
    describes the picture, and the meme goes through the editor after the
    captions are on it."""

    def test_picker_draw_edits_the_captioned_meme(self, meme_plugin) -> None:
        from llm.service import MemePick

        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "spirit", "lines": ["My Dad", "Includes:", "- Nothing"], '
            '"draw": "a tired dad asleep in a lawn chair"}',
            "test-model",
        )

        plugin.meme(mock_irc, mock_msg, ["spirit halloween costume of my dad"])

        assert plugin.llm_service.image_edit.call_args.args == (
            CAPTIONED,
            "a tired dad asleep in a lawn chair",
        )
        assert mock_irc.reply.call_args.args[0].startswith(EDITED)
        rows = [(c.args[3], c.args[6]) for c in plugin.db.log_usage.call_args_list]
        assert rows == [
            ("test-model", 0.0),
            ("meme-render", 0.0),
            ("xai/grok-imagine-image", 0.022),
        ]
        assert plugin.llm_service._save_image_bytes.call_args.args[0].startswith(b"\x89PNG")

    def test_refused_edit_keeps_the_captioned_meme(self, meme_plugin) -> None:
        from llm.service import MemePick

        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "drake", "lines": ["a", "b"], "draw": "x"}', "test-model"
        )
        plugin.llm_service.image_edit.return_value = ImageResult(
            content="",
            cost=0.02,
            model="xai/grok-imagine-image",
            error="The image model refused that picture.",
        )

        plugin.meme(mock_irc, mock_msg, ["drake but x"])

        reply = mock_irc.reply.call_args.args[0]
        assert reply.startswith(CAPTIONED) and "picture not added" in reply and "refused" in reply
        assert plugin.db.log_usage.call_args.kwargs["status"] == "error"

    def test_no_draw_from_the_picker_means_no_edit(self, meme_plugin) -> None:
        from llm.service import MemePick

        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "drake", "lines": ["a", "b"]}', "test-model"
        )

        plugin.meme(mock_irc, mock_msg, ["waiting for ci"])

        plugin.llm_service.image_edit.assert_not_called()

    def test_named_template_with_captions_never_edits(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        plugin.llm_service.meme_pick.assert_not_called()
        plugin.llm_service.image_edit.assert_not_called()

    def test_tool_draw_parameter(self, meme_plugin) -> None:
        plugin, _, mock_msg = meme_plugin
        _, handlers = plugin._build_meme_tool(mock_msg)

        payload = json.loads(
            handlers["make_meme"](
                {"template": "drake", "lines": ["a", "b"], "draw": "make drake a sysadmin"}
            ).content
        )

        assert plugin.llm_service.image_edit.call_args.args == (CAPTIONED, "make drake a sysadmin")
        assert payload == {"status": "ok", "message": EDITED}

    def test_tool_draw_with_a_brief_reaches_the_picker(self, meme_plugin) -> None:
        plugin, _, mock_msg = meme_plugin
        _, handlers = plugin._build_meme_tool(mock_msg)

        handlers["make_meme"]({"brief": "spirit costume", "draw": "an Irish republican"})

        request = plugin.llm_service.meme_pick.call_args.args[0]
        assert request == "spirit costume\nPicture: an Irish republican"


class TestSharedGrounding:
    """The picker gets the canon block and subject dossier @draw's planner gets."""

    def test_picker_request_carries_the_dossier(self, meme_plugin, mocker) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        mocker.patch.object(plugin, "_subject_dossier_for", return_value="- Poilievre — glasses")
        mocker.patch.object(plugin, "_verse_context_for", return_value="CANON: the stinky lads")

        plugin.meme(mock_irc, mock_msg, ["spirit costume of Poilievre"])

        kwargs = plugin.llm_service.meme_pick.call_args.kwargs
        assert kwargs["context"] == "CANON: the stinky lads\n\n- Poilievre — glasses"
        plugin._subject_dossier_for.assert_called_once()
        assert plugin._subject_dossier_for.call_args.args[2] == "spirit costume of Poilievre"

    def test_named_template_with_captions_researches_nothing(self, meme_plugin, mocker) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        dossier = mocker.patch.object(plugin, "_subject_dossier_for", return_value="- x — y")
        typing = plugin.llm_service._begin_typing

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        dossier.assert_not_called()
        typing.assert_not_called()

    def test_typing_is_held_while_the_picker_runs(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["waiting for ci"])

        plugin.llm_service._begin_typing.assert_called_once_with(mock_irc, mock_msg)
        plugin.llm_service._begin_typing.return_value.assert_called_once()


def _picture_png(colour: str) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def slot_plugin(plugin_env, mocker, tmp_path):
    """A two-slot template (left and right halves) with real masks."""
    from llm.service import MemePick, MemePicture
    from PIL import Image

    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    plugin.registryValue.side_effect = make_registry_side_effect({"memeEnabled": True})
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for i, box in enumerate(((0, 0, 100, 150), (100, 0, 200, 150))):
        mask = Image.new("L", (200, 150), 0)
        mask.paste(255, box)
        mask.save(masks / f"www-{i}.png")
    entry = {
        "id": "www",
        "name": "Who Would Win?",
        "boxes": _boxes(2),
        "slots": [
            {"box": [0, 0, 0.5, 1], "mask": "www-0.png"},
            {"box": [0.5, 0, 0.5, 1], "mask": "www-1.png"},
        ],
    }
    mocker.patch.object(meme, "bundled_catalog", return_value=write_catalog([entry], images))
    plugin.llm_service._save_image_bytes.return_value = CAPTIONED
    plugin.llm_service.image_generate.side_effect = [
        MemePicture(_picture_png("red"), 0.02, "xai/grok-imagine-image"),
        MemePicture(_picture_png("blue"), 0.02, "xai/grok-imagine-image"),
    ]
    plugin.llm_service.meme_pick.return_value = MemePick(
        '{"template": "www", "lines": ["cat", "dog"], "draw": "a cat | a dog"}', "test-model"
    )
    return plugin, mock_irc, mock_msg


def _saved_image(plugin):
    from PIL import Image

    return Image.open(io.BytesIO(plugin.llm_service._save_image_bytes.call_args.args[0]))


class TestSlotPictures:
    """Templates with picture slots: generated pictures go in through the
    masks, then the captions; the whole-image edit is not used."""

    def test_each_slot_gets_its_own_picture(self, slot_plugin) -> None:
        plugin, mock_irc, mock_msg = slot_plugin

        plugin.meme(mock_irc, mock_msg, ["who would win, a cat or a dog"])

        drawn = [c.args[0] for c in plugin.llm_service.image_generate.call_args_list]
        assert sorted(drawn) == ["a cat", "a dog"]
        ratios = {
            c.kwargs["aspect_ratio"] for c in plugin.llm_service.image_generate.call_args_list
        }
        assert ratios == {"2:3"}  # each half is 100x150
        plugin.llm_service.image_edit.assert_not_called()
        saved = _saved_image(plugin).convert("RGB")
        assert saved.getpixel((50, 75))[0] > 200  # red on the left
        assert saved.getpixel((150, 75))[2] > 200  # blue on the right
        assert mock_irc.reply.call_args.args[0].startswith(CAPTIONED)
        rows = [(c.args[3], c.args[6]) for c in plugin.db.log_usage.call_args_list]
        assert rows == [
            ("test-model", 0.0),
            ("xai/grok-imagine-image", 0.02),
            ("xai/grok-imagine-image", 0.02),
            ("meme-render", 0.0),
        ]

    def test_a_failed_picture_leaves_its_slot_blank_and_says_so(self, slot_plugin) -> None:
        from llm.service import MemePicture

        plugin, mock_irc, mock_msg = slot_plugin
        plugin.llm_service.image_generate.side_effect = [
            MemePicture(_picture_png("red"), 0.02, "xai/grok-imagine-image"),
            MemePicture(
                None, 0.02, "xai/grok-imagine-image", "The image model refused that picture."
            ),
        ]

        plugin.meme(mock_irc, mock_msg, ["a cat versus a dog"])

        reply = mock_irc.reply.call_args.args[0]
        assert reply.startswith(CAPTIONED)
        assert "1 of 2 pictures not added: The image model refused that picture." in reply
        statuses = [c.kwargs["status"] for c in plugin.db.log_usage.call_args_list[1:3]]
        assert sorted(statuses) == ["error", "success"]

    def test_no_draw_means_no_pictures(self, slot_plugin) -> None:
        plugin, mock_irc, mock_msg = slot_plugin

        plugin.meme(mock_irc, mock_msg, ["www | cats | dogs"])

        plugin.llm_service.image_generate.assert_not_called()
        plugin.llm_service.image_edit.assert_not_called()


class TestImageGenerate:
    def test_posts_to_generations_with_the_aspect_ratio(self, service, mocker) -> None:
        opened = mocker.patch(
            "urllib.request.urlopen",
            return_value=_response(
                {
                    "data": [{"b64_json": base64.b64encode(PNG).decode()}],
                    "usage": {"cost_in_usd_ticks": 200000000},
                }
            ),
        )

        result = service.image_generate("a tired dad in a lawn chair", aspect_ratio="2:3")

        assert result.data == PNG and result.error is None
        assert result.cost == pytest.approx(0.02)
        request = opened.call_args.args[0]
        body = json.loads(request.data)
        assert request.full_url == "https://api.x.ai/v1/images/generations"
        assert body["aspect_ratio"] == "2:3"
        assert body["prompt"].startswith("a tired dad in a lawn chair.")
        assert "No text" in body["prompt"]

    def test_non_xai_model_is_refused(self, make_service, mocker) -> None:
        svc, _ = make_service(memeEditModel="gemini/gemini-2.5-flash-image")
        opened = mocker.patch("urllib.request.urlopen")

        result = svc.image_generate("x")

        assert result.data is None and result.error
        opened.assert_not_called()
