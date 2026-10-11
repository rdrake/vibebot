"""@meme: the user names the template and the captions, nothing is planned.

No LLM turn, no image model, no cost row. The command resolves the name
through the catalog, asks memegen for the PNG through the existing
rehost-a-provider-URL path, and posts the link.
"""

from __future__ import annotations

import json

import pytest
from llm import meme
from llm.service import MemePick

from .conftest import make_registry_side_effect

_TEMPLATES = meme.parse_templates(
    [
        {
            "id": "drake",
            "name": "Drakeposting",
            "lines": 2,
            "keywords": [],
            "example": {"text": ["left on unread", "left on read"]},
        },
        {
            "id": "db",
            "name": "Distracted Boyfriend",
            "lines": 3,
            "keywords": ["girlfriend"],
            "example": {"text": ["me", "new", "old"]},
        },
    ]
)

_HOSTED = "https://paste.boxlabs.uk/img/img_abc.png"


@pytest.fixture
def meme_plugin(plugin_env, mocker):
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    plugin.registryValue.side_effect = make_registry_side_effect({"memeEnabled": True})
    mocker.patch.object(
        meme.CachedCatalog, "get", return_value=meme.MemeCatalog(_TEMPLATES), autospec=True
    )
    plugin.llm_service._download_and_save_image.return_value = _HOSTED
    # The picker declines unless a test says otherwise, so a resolver miss
    # still reads as the did-you-mean list.
    plugin.llm_service.meme_pick.return_value = MemePick(
        '{"template": null, "reason": "Nothing fits."}', "test-model"
    )
    # No network in tests: memegen's canonical spelling is the one we built.
    mocker.patch.object(meme, "canonical_url", side_effect=lambda url, **_kw: url)
    return plugin, mock_irc, mock_msg


class TestMemeCommand:
    def test_refused_when_memes_are_off(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect({"memeEnabled": False})

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert mock_irc.reply.call_args.args[0] == "Memes are turned off in this channel."
        plugin.llm_service._download_and_save_image.assert_not_called()

    def test_posts_the_rehosted_image(self, meme_plugin) -> None:
        plugin, mock_irc, _ = meme_plugin

        plugin.meme(mock_irc, meme_plugin[2], ["drake | left on unread | left on read"])

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched == "https://api.memegen.link/images/drake/left_on_unread/left_on_read.png"
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_unknown_template_suggests_and_fetches_nothing(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake boyfriend | a | b"])

        plugin.llm_service._download_and_save_image.assert_not_called()
        err = mock_irc.error.call_args.args[0]
        assert "drake boyfriend" in err and "db" in err

    def test_missing_captions_show_the_example(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake"])

        err = mock_irc.error.call_args.args[0]
        assert "2 captions" in err and "left on unread" in err

    def test_fetch_failure_is_reported(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service._download_and_save_image.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        mock_irc.reply.assert_not_called()
        assert "memegen" in mock_irc.error.call_args.args[0]

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
        meme.CachedCatalog.get.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert "unavailable" in mock_irc.error.call_args.args[0]

    def test_requires_an_account(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        mock_irc.state.nickToAccount.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        plugin.llm_service._download_and_save_image.assert_not_called()


class TestPickerInCommand:
    """No template named → the picker chooses, the reply shows the choice."""

    def test_unnamed_request_is_picked_and_rendered(self, meme_plugin) -> None:
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
        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched == "https://api.memegen.link/images/drake/waiting_for_ci/pushing_anyway.png"
        assert (
            mock_irc.reply.call_args.args[0]
            == f"{_HOSTED} — drake | waiting for ci | pushing anyway"
        )
        models = [c.args[3] for c in plugin.db.log_usage.call_args_list]
        assert models == ["test-model", "memegen"]
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

    def test_invented_id_never_reaches_memegen(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service.meme_pick.return_value = MemePick(
            '{"template": "gigachad", "lines": ["a", "b"]}', "test-model"
        )

        plugin.meme(mock_irc, mock_msg, ["gym"])

        plugin.llm_service._download_and_save_image.assert_not_called()
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
        plugin.llm_service._download_and_save_image.assert_not_called()

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
    def test_image_url_template_fetches_the_custom_render(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["https://i.imgflip.com/1c1uej.jpg | top | bottom"])

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched.startswith(
            "https://api.memegen.link/images/custom/top/bottom.png?background="
        )
        assert mock_irc.reply.call_args.args[0] == _HOSTED

    def test_registry_aliases_reach_the_resolver(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"memeEnabled": True, "memeAliases": ["hotline=drake"]}
        )

        plugin.meme(mock_irc, mock_msg, ["hotline bling | a | b"])

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched == "https://api.memegen.link/images/drake/a/b.png"

    def test_tool_accepts_an_image_url_as_template(self, meme_plugin) -> None:
        plugin, _, _ = meme_plugin
        _, handlers = plugin._build_meme_tool(meme_plugin[2])

        payload = json.loads(
            handlers["make_meme"](
                {"template": "https://i.imgflip.com/1c1uej.jpg", "lines": ["a", "b"]}
            ).content
        )

        assert payload["status"] == "ok"


class TestMemeUsageRow:
    """Free, but attributable: one $0 row per memegen fetch, like draw's per image."""

    def test_success_writes_a_zero_cost_row_under_memegen(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        args, kwargs = plugin.db.log_usage.call_args
        assert args == ("testnick", "#test", "meme", "memegen", 0, 0, 0.0)
        assert kwargs["prompt"] == "drake | a | b"
        assert kwargs["status"] == "success"

    def test_fetch_failure_is_an_error_row(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        plugin.llm_service._download_and_save_image.return_value = None

        plugin.meme(mock_irc, mock_msg, ["drake | a | b"])

        assert plugin.db.log_usage.call_args.kwargs["status"] == "error"

    def test_resolver_miss_books_the_picker_but_not_memegen(self, meme_plugin) -> None:
        """The picker ran and cost tokens; memegen was never asked."""
        plugin, mock_irc, mock_msg = meme_plugin

        plugin.meme(mock_irc, mock_msg, ["drake boyfriend | a | b"])

        assert plugin.db.log_usage.call_count == 1
        assert plugin.db.log_usage.call_args.args[3] == "test-model"
        plugin.llm_service._download_and_save_image.assert_not_called()

    def test_tool_path_writes_the_same_row(self, meme_plugin) -> None:
        plugin, _, mock_msg = meme_plugin
        _, handlers = plugin._build_meme_tool(mock_msg)

        handlers["make_meme"]({"template": "drake", "lines": ["a", "b"]})

        args, _ = plugin.db.log_usage.call_args
        assert args[2:4] == ("meme", "memegen")


class TestMemeOptions:
    def test_gif_flag_fetches_the_animated_gif(self, meme_plugin, mocker) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        animated = meme.parse_templates(
            [
                {
                    "id": "fine",
                    "name": "This is Fine",
                    "lines": 2,
                    "keywords": [],
                    "styles": ["animated"],
                }
            ]
        )
        meme.CachedCatalog.get.return_value = meme.MemeCatalog(animated)

        plugin.meme(mock_irc, mock_msg, ["--gif", "fine | | this is fine"])

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched == "https://api.memegen.link/images/fine/_/this_is_fine.gif?style=animated"

    def test_style_font_top_flags(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        styled = meme.parse_templates(
            [{"id": "doge", "name": "Doge", "lines": 2, "keywords": [], "styles": ["bark"]}]
        )
        meme.CachedCatalog.get.return_value = meme.MemeCatalog(styled)

        plugin.meme(
            mock_irc, mock_msg, ["--style", "bark", "--font", "impact", "--top", "doge | a | b"]
        )

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert (
            fetched
            == "https://api.memegen.link/images/doge/a/b.png?style=bark&font=impact&layout=top"
        )

    def test_list_shows_gif_and_styles(self, meme_plugin) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        styled = meme.parse_templates(
            [
                {
                    "id": "doge",
                    "name": "Doge",
                    "lines": 2,
                    "keywords": [],
                    "styles": ["bark", "animated"],
                }
            ]
        )
        meme.CachedCatalog.get.return_value = meme.MemeCatalog(styled)

        plugin.meme(mock_irc, mock_msg, ["list doge"])

        assert "doge (Doge, 2, gif, styles: bark)" in mock_irc.reply.call_args.args[0]

    def test_tool_takes_animated_and_style(self, meme_plugin) -> None:
        plugin, _, mock_msg = meme_plugin
        styled = meme.parse_templates(
            [
                {
                    "id": "fine",
                    "name": "This is Fine",
                    "lines": 2,
                    "keywords": [],
                    "styles": ["animated"],
                }
            ]
        )
        meme.CachedCatalog.get.return_value = meme.MemeCatalog(styled)
        schemas, handlers = plugin._build_meme_tool(mock_msg)

        handlers["make_meme"]({"template": "fine", "lines": ["", "ok"], "animated": True})

        fetched = plugin.llm_service._download_and_save_image.call_args.args[0]
        assert fetched.endswith(".gif?style=animated")
        props = schemas[0]["function"]["parameters"]["properties"]
        assert "animated" in props and "style" in props


class TestCanonicalRedirectInCommand:
    def test_fetch_uses_memegen_canonical_spelling(self, meme_plugin, mocker) -> None:
        plugin, mock_irc, mock_msg = meme_plugin
        canon = "https://api.memegen.link/images/drake/a----b/c.png"
        mocker.patch.object(meme, "canonical_url", return_value=canon)

        plugin.meme(mock_irc, mock_msg, ["drake | a - b | c"])

        assert plugin.llm_service._download_and_save_image.call_args.args[0] == canon
