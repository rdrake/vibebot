"""@animate with an image URL in the line — the command and chat entry points.

A reference image does NOT skip the planner. MiniMax-H3 rewards prompts written
like a shot script — "more detail means more adherence" (ps, #afternet
2026-08-21) — and a two-word ask renders poorly whether or not a picture is
attached. So the reference path runs the same planner turn as text-only
@animate, with two additions: the planner SEES the picture, and it is told the
first frame is already fixed so its job is the motion, not the subject.

The two entry points both matter. "@animate <url> ..." is the command;
"vibebot animate <url> ..." never reaches it and arrives as chat, which is the
trap documented in service.py's EXPLICIT_VIDEO notes — so the tool callback
has to pick the reference out of the user's message too.
"""

from __future__ import annotations

import pytest
from llm.identity import Caller

_IMG = "https://pics.example.com/cat.png"


def _assistant_result(content: str = "Queued that up, it is rendering now."):
    from llm.service import AssistantResult, Spend

    return AssistantResult(
        content=content,
        grounding_used=False,
        spend=Spend(prompt_tokens=120, completion_tokens=40, cost=0.0012),
        model="gemini/gemini-flash-latest",
    )


def _video_result(content: str = "Rendering your video — I'll post the link here."):
    from llm.service import VideoResult

    return VideoResult(content=content, job_id="video_gen_1", queued=True, model="")


@pytest.fixture
def animate_plugin(plugin_env, mocker):
    """A plugin whose account check passes and whose video path is stubbed."""
    from llm.service import ReferenceImage

    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    plugin.llm_service.assistant_request.side_effect = None
    plugin.llm_service.assistant_request.return_value = _assistant_result()
    plugin.llm_service.video_generation.return_value = _video_result()
    # No split_reference_url stub: the plugin calls the real module-level
    # parser, so these tests exercise the same URL handling production does.
    plugin.llm_service.fetch_reference_image.return_value = ReferenceImage(
        data=b"\xff\xd8\xffcat", extension="jpg"
    )
    plugin.llm_service.reference_vision_url.return_value = "data:image/jpeg;base64,Y2F0"
    mocker.patch.object(plugin, "_verse_context_for", return_value=None)
    return plugin, mock_irc, mock_msg


class TestAnimateCommandWithReference:
    """The picture fixes the subject; the planner still writes the script."""

    def test_reference_still_plans(self, animate_plugin) -> None:
        """GIVEN a URL in the prompt WHEN @animate runs THEN the planner turn happens.

        A bare "make it dance" is exactly the prompt shape H3 renders worst.
        """
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "the", "cat", "dance"])

        plugin.llm_service.assistant_request.assert_called_once()
        ctx = plugin.llm_service.assistant_request.call_args.kwargs["request_context"]
        assert ctx.profile == "animate"

    def test_planner_sees_the_picture(self, animate_plugin) -> None:
        """GIVEN a reference WHEN the planner runs THEN the image rides the request.

        A planner writing blind can only guess what is in frame, and any guess
        it gets wrong is detail the video model has to reconcile against the
        actual first frame.
        """
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "the", "cat", "dance"])

        images = plugin.llm_service.assistant_request.call_args.kwargs["images"]
        assert images == ["data:image/jpeg;base64,Y2F0"]

    def test_planner_is_told_the_first_frame_is_fixed(self, animate_plugin) -> None:
        """GIVEN a reference WHEN the planner runs THEN its overlay says so."""
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "the", "cat", "dance"])

        overlay = plugin.llm_service.assistant_request.call_args.kwargs["system_prompt"]
        assert overlay and "first frame" in overlay.lower()

    def test_url_is_not_in_the_planners_prompt(self, animate_plugin) -> None:
        """GIVEN a URL WHEN the planner runs THEN it plans on the words alone."""
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "the", "cat", "dance"])

        assert plugin.llm_service.assistant_request.call_args.args[0] == "make the cat dance"

    def test_submission_carries_the_reference(self, animate_plugin) -> None:
        """GIVEN a reference WHEN the planner calls the tool THEN the file is attached."""
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "the", "cat", "dance"])

        animate_fn = plugin.llm_service.assistant_request.call_args.kwargs["animate_fn"]
        animate_fn("A ginger cat on a kitchen counter rises onto its hind legs...")

        call = plugin.llm_service.video_generation.call_args
        assert call.kwargs["reference"].data == b"\xff\xd8\xffcat"
        assert call.args[0].startswith("A ginger cat")

    def test_canon_layers_with_the_reference_block(self, animate_plugin, mocker) -> None:
        """GIVEN canon and a reference WHEN it plans THEN the overlay carries both."""
        plugin, mock_irc, mock_msg = animate_plugin
        mocker.patch.object(
            plugin, "_verse_context_for", return_value="Established characters:\n- Archie: windbag"
        )

        plugin.animate(mock_irc, mock_msg, [_IMG, "the", "stinky", "lads", "run"])

        overlay = plugin.llm_service.assistant_request.call_args.kwargs["system_prompt"]
        assert "Archie: windbag" in overlay
        assert "first frame" in overlay.lower()

    def test_unfetchable_reference_spends_nothing(self, animate_plugin) -> None:
        """GIVEN a reference that will not fetch WHEN @animate runs THEN it says so.

        Rendering the text alone would silently ignore the picture the user
        chose, which reads as the bot ignoring them.
        """
        plugin, mock_irc, mock_msg = animate_plugin
        plugin.llm_service.fetch_reference_image.return_value = None

        plugin.animate(mock_irc, mock_msg, [_IMG, "make", "it", "dance"])

        plugin.llm_service.video_generation.assert_not_called()
        plugin.llm_service.assistant_request.assert_not_called()
        said = " ".join(
            str(c) for c in mock_irc.error.call_args_list + mock_irc.reply.call_args_list
        )
        assert "image" in said.lower()

    def test_plain_prompt_sends_no_image(self, animate_plugin) -> None:
        """GIVEN no URL WHEN @animate runs THEN the planner path is untouched."""
        plugin, mock_irc, mock_msg = animate_plugin

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        kwargs = plugin.llm_service.assistant_request.call_args.kwargs
        assert not kwargs.get("images")
        assert kwargs.get("system_prompt") is None
        plugin.llm_service.fetch_reference_image.assert_not_called()


class TestChatVideoWithReference:
    """ "vibebot animate <url> ..." is chat, and must still use the picture."""

    def test_tool_callback_uses_the_message_image(self, animate_plugin) -> None:
        """GIVEN a URL in the user's line WHEN the tool fires THEN it conditions the clip."""
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.args = ("#test", f"vibebot animate {_IMG} make the cat dance")

        plugin._animate_for_assistant(
            mock_irc,
            mock_msg,
            "a cat dancing on a table",
            caller=Caller.from_msg(mock_msg, "test_account"),
        )

        assert plugin.llm_service.video_generation.call_args.kwargs["reference"] is not None

    def test_url_never_reaches_the_video_prompt(self, animate_plugin) -> None:
        """GIVEN a model prompt containing a URL WHEN the tool fires THEN it is stripped.

        The video model renders stray text on screen, so a URL left in the
        prompt becomes part of the picture.
        """
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.args = ("#test", f"vibebot animate {_IMG} make the cat dance")

        plugin._animate_for_assistant(
            mock_irc,
            mock_msg,
            f"{_IMG} a cat dancing",
            caller=Caller.from_msg(mock_msg, "test_account"),
        )

        assert "http" not in plugin.llm_service.video_generation.call_args.args[0]

    def test_plain_chat_request_is_unchanged(self, animate_plugin) -> None:
        """GIVEN no URL anywhere WHEN the tool fires THEN nothing is fetched."""
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.args = ("#test", "vibebot make me a video of a corgi")

        plugin._animate_for_assistant(
            mock_irc,
            mock_msg,
            "a corgi riding a unicorn",
            caller=Caller.from_msg(mock_msg, "test_account"),
        )

        plugin.llm_service.fetch_reference_image.assert_not_called()
        assert plugin.llm_service.video_generation.call_args.kwargs.get("reference") is None


class TestDeliveryNick:
    """The clip is addressed to a person in a channel, so use their nick.

    #afternet 2026-08-21: Larry asked for a clip and the delivery line came
    back "lstrong2k: your video is ready!" — the account, which is not what
    anyone in the channel calls them and not what their client highlights on.
    Every other stashed task type already stores msg.nick (_msg_stash_context);
    animate was the outlier because the command threads the account-resolved
    PreflightResult.nick all the way down.
    """

    def test_stashed_nick_is_the_irc_nick(self, animate_plugin) -> None:
        """GIVEN an account that differs from the nick WHEN queued THEN the nick is stored."""
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.nick = "Larry"
        mock_msg.prefix = "Larry!user@host"
        mock_irc.state.nickToAccount.return_value = "lstrong2k"

        plugin.animate(mock_irc, mock_msg, ["teletubbies", "on", "a", "smoke", "break"])
        animate_fn = plugin.llm_service.assistant_request.call_args.kwargs["animate_fn"]
        animate_fn("Four costumed figures loiter behind a studio wall...")

        kwargs = plugin.llm_service.video_generation.call_args.kwargs
        assert kwargs["nick"] == "Larry"

    def test_account_is_still_stored_for_identity(self, animate_plugin) -> None:
        """GIVEN a resolved account WHEN queued THEN it rides along separately.

        The display nick must not cost the account: ownership, rate limits and
        delivery-time logging all read the account column.
        """
        plugin, mock_irc, mock_msg = animate_plugin
        mock_msg.nick = "Larry"
        mock_msg.prefix = "Larry!user@host"
        mock_irc.state.nickToAccount.return_value = "lstrong2k"

        plugin.animate(mock_irc, mock_msg, ["teletubbies", "on", "a", "smoke", "break"])
        animate_fn = plugin.llm_service.assistant_request.call_args.kwargs["animate_fn"]
        animate_fn("Four costumed figures loiter behind a studio wall...")

        assert plugin.llm_service.video_generation.call_args.kwargs["account"] == "lstrong2k"
