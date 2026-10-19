"""Tests for @animate — text-to-video against a self-hosted vLLM box.

Animate is the only generation path where the media does not exist when the
command returns. A 4s clip measured 67.7s at 25 steps and 171s at 50 on the
reference box (2026-08-20), so submission and delivery are split: the command
stashes a job id in ``pending_tasks`` and the existing pending-task machinery
polls until the clip lands. Most of what is worth testing here is that seam —
that a submission is either stashed or honestly reported as untracked, and
that polling distinguishes "not yet" from "never".

The other half is the promise made to the user. ``generate_video`` returns an
acknowledgement and never a URL, because a tool that hands back a link to a
video that does not exist yet is exactly the shape of the image-fabrication
bug guarded in test_image_fabrication_guard.py.
"""

from __future__ import annotations

import json
import threading
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest
from llm.identity import Caller, Identity
from llm.service import VideoResult

from .conftest import (
    make_completion_response,
    make_registry_side_effect,
    make_tool_call,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from llm.service import LLMService

_URL = "http://video.example.com:14205"
# Deliberately low-entropy and self-describing: a realistic-looking token here
# trips the gitleaks pre-commit hook, and nothing in these tests inspects the
# value beyond "is it set".
_KEY = "not-a-real-token-for-tests"


@pytest.fixture
def animate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A configured animate deployment: URL in the registry, key in the env."""
    monkeypatch.setenv("ANIMATE_API_KEY", _KEY)


def _service(make_service: Callable[..., tuple[LLMService, Mock]], **overrides: Any):
    overrides.setdefault("animateApiUrl", _URL)
    return make_service(**overrides)


def _queued_result() -> VideoResult:
    """What a successful submission looks like coming back from the service."""
    return VideoResult(content="Queued at position 1 (0 ahead).", job_id="job-1", queued=True)


class TestAnimateAvailability:
    """Both halves of the credential are required, and they fail independently."""

    def test_unavailable_without_url(self, make_service, animate_env) -> None:
        """GIVEN a key but no URL WHEN checked THEN animate is unavailable."""
        service, _ = make_service(animateApiUrl="")
        assert service.animate_available() is False

    def test_unavailable_without_key(self, make_service, monkeypatch) -> None:
        """GIVEN a URL but no key WHEN checked THEN animate is unavailable."""
        monkeypatch.delenv("ANIMATE_API_KEY", raising=False)
        service, _ = _service(make_service)
        assert service.animate_available() is False

    def test_available_with_both(self, make_service, animate_env) -> None:
        """GIVEN URL and key WHEN checked THEN animate is available."""
        service, _ = _service(make_service)
        assert service.animate_available() is True

    def test_unsafe_url_disables_rather_than_dials(self, make_service, animate_env) -> None:
        """GIVEN a non-http URL WHEN checked THEN the feature is off.

        A typo in the registry must not become a request to something that is
        not a video server; this request carries a bearer token.
        """
        service, _ = make_service(animateApiUrl="file:///etc/passwd")
        assert service.animate_available() is False
        assert service._animate_base_url() == ""

    def test_trailing_slash_is_normalised(self, make_service, animate_env) -> None:
        """GIVEN a URL with a trailing slash WHEN read THEN paths do not double up."""
        service, _ = make_service(animateApiUrl=_URL + "/")
        assert service._animate_base_url() == _URL


class TestAnimateForm:
    """The submission form carries the config, and the audio switch is real."""

    def test_audio_on_requests_t2va(self, make_service, animate_env) -> None:
        """GIVEN animateAudio on WHEN a form is built THEN the task is t2va."""
        service, _ = _service(make_service, animateAudio=True)
        extra = json.loads(service._animate_form("a forest", None)["extra_params"])
        assert extra["task"] == "t2va"
        assert extra["audio_flow_shift"] == 3

    def test_audio_off_requests_t2v_and_drops_audio_knob(self, make_service, animate_env) -> None:
        """GIVEN animateAudio off WHEN a form is built THEN no audio is requested."""
        service, _ = _service(make_service, animateAudio=False)
        extra = json.loads(service._animate_form("a forest", None)["extra_params"])
        assert extra["task"] == "t2v"
        assert "audio_flow_shift" not in extra

    def test_steps_and_size_come_from_config(self, make_service, animate_env) -> None:
        """GIVEN step/size config WHEN a form is built THEN it carries them."""
        service, _ = _service(make_service, animateSteps=25, animateSize="640x352")
        form = service._animate_form("a forest", None)
        assert form["num_inference_steps"] == "25"
        assert form["size"] == "640x352"

    def test_empty_model_is_omitted(self, make_service, animate_env) -> None:
        """GIVEN no configured model WHEN a form is built THEN no model field is sent.

        A single-model box picks its own; sending "" would be a request for a
        model literally named empty string.
        """
        service, _ = _service(make_service, animateModel="")
        assert "model" not in service._animate_form("a forest", None)

    def test_configured_model_is_sent(self, make_service, animate_env) -> None:
        """GIVEN a configured model WHEN a form is built THEN it is sent."""
        service, _ = _service(make_service, animateModel="/work/models/H3")
        assert service._animate_form("a forest", None)["model"] == "/work/models/H3"

    def test_duration_rides_extra_params(self, make_service, animate_env) -> None:
        """GIVEN a duration WHEN a form is built THEN it is in extra_params."""
        service, _ = _service(make_service, animateDuration=6)
        extra = json.loads(service._animate_form("a forest", None)["extra_params"])
        assert extra["duration"] == 6


class TestVideoGenerationSubmit:
    """Submission returns immediately; what matters is what it promises."""

    def test_unconfigured_returns_error_without_calling_out(
        self, make_service, monkeypatch, mocker
    ) -> None:
        """GIVEN no config WHEN submitting THEN it errors and makes no request."""
        monkeypatch.delenv("ANIMATE_API_KEY", raising=False)
        service, _ = make_service(animateApiUrl="")
        request = mocker.patch.object(service, "_animate_request")

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert result.queued is False
        request.assert_not_called()

    def test_successful_submit_stashes_and_promises_delivery(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a job id WHEN submitting THEN it stashes and promises a later link."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "video_gen_1"}))
        stash = mocker.patch.object(service, "_stash_timeout", return_value=True)

        result = service.video_generation(
            "a forest",
            nick="alice",
            reply_target="#chan",
            is_channel=True,
            account="alice",
        )

        assert result.queued is True
        assert result.job_id == "video_gen_1"
        assert result.error is None
        # The acknowledgement must not contain a URL — there is nothing to link.
        assert "http" not in result.content

        kwargs = stash.call_args.kwargs
        assert kwargs["task_type"] == "animate"
        assert kwargs["reply_target"] == "#chan"
        assert kwargs["request_data"]["job_id"] == "video_gen_1"

    def test_rejected_submit_reports_the_server_reason(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a 400 WHEN submitting THEN the server's reason reaches the user."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            return_value=(400, {"error": {"message": "unsupported size"}}),
        )
        stash = mocker.patch.object(service, "_stash_timeout")

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert "unsupported size" in result.content
        stash.assert_not_called()

    def test_unreachable_server_does_not_raise(self, make_service, animate_env, mocker) -> None:
        """GIVEN a network error WHEN submitting THEN it returns an error result."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", side_effect=OSError("no route"))

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert result.queued is False

    def test_failed_stash_does_not_promise_delivery(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN stashing fails WHEN submitting THEN it says so rather than promising.

        The job is running on the box either way; the lie worth avoiding is
        telling someone a clip is coming when nothing will ever collect it.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "video_gen_1"}))
        mocker.patch.object(service, "_stash_timeout", return_value=False)

        result = service.video_generation("a forest", reply_target="#chan")

        assert result.error is not None
        assert result.job_id == "video_gen_1"
        assert result.queued is False

    def test_no_reply_target_skips_stashing(self, make_service, animate_env, mocker) -> None:
        """GIVEN no delivery target WHEN submitting THEN nothing is stashed.

        A stashed row with an empty target emits a malformed PRIVMSG on every
        delivery attempt — the trap _stash_timeout documents for completions.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "video_gen_1"}))
        stash = mocker.patch.object(service, "_stash_timeout")

        result = service.video_generation("a forest", reply_target="")

        assert result.queued is True
        stash.assert_not_called()


def _task(**overrides: Any) -> Mock:
    task = Mock()
    task.task_type = "animate"
    task.nick = "alice"
    task.reply_target = "#chan"
    task.is_channel = True
    task.prompt_preview = "a forest"
    task.model = ""
    for key, value in overrides.items():
        setattr(task, key, value)
    return task


class TestRetryVideo:
    """Polling must tell "not yet" apart from "never", and never resubmit."""

    def test_still_rendering_raises_for_backoff(self, make_service, animate_env, mocker) -> None:
        """GIVEN an in-progress job WHEN polled THEN it raises to be retried."""
        import litellm

        service, _ = _service(make_service)
        mocker.patch.object(
            service, "_animate_request", return_value=(200, {"status": "in_progress"})
        )

        with pytest.raises(litellm.Timeout):
            service._retry_video(_task(), {"job_id": "video_gen_1"})

    def test_completed_job_is_downloaded_and_published(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a completed job WHEN polled THEN the clip is published."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            side_effect=[(200, {"status": "completed"}), (200, b"\x00\x00\x00 ftypmp42")],
        )
        save = mocker.patch.object(
            service, "_save_video_bytes", return_value="https://paste.example/img/vid_a.mp4"
        )

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "completed"
        assert result.content == "https://paste.example/img/vid_a.mp4"
        assert save.call_args.args[0] == b"\x00\x00\x00 ftypmp42"

    def test_missing_job_id_is_terminal(self, make_service, animate_env) -> None:
        """GIVEN malformed request data WHEN polled THEN it fails terminally."""
        service, _ = _service(make_service)
        result = service._retry_video(_task(), {})
        assert result.status == "failed_terminal"

    def test_vanished_job_is_terminal(self, make_service, animate_env, mocker) -> None:
        """GIVEN a 404 WHEN polled THEN it fails terminally rather than retrying.

        The server restarted or reaped the job. More polling cannot bring it
        back, so retrying until expiry only delays telling the user.
        """
        service, _ = _service(make_service)
        mocker.patch.object(
            service, "_animate_request", return_value=(404, {"error": {"message": "not found"}})
        )

        result = service._retry_video(_task(), {"job_id": "gone"})

        assert result.status == "failed_terminal"

    def test_failed_job_reports_its_reason(self, make_service, animate_env, mocker) -> None:
        """GIVEN a failed job WHEN polled THEN the server's reason is carried out."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            return_value=(200, {"status": "failed", "error": {"message": "OOM on device"}}),
        )

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"
        assert "OOM on device" in result.reason

    def test_transient_server_error_raises_rather_than_failing(
        self, make_service, animate_env, mocker
    ) -> None:
        """GIVEN a 503 WHEN polled THEN it raises so the job is retried.

        A bounced box must not lose a job that is very likely still running.
        """
        import litellm

        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(503, {"detail": "busy"}))

        with pytest.raises(litellm.Timeout):
            service._retry_video(_task(), {"job_id": "video_gen_1"})

    def test_download_failure_is_terminal(self, make_service, animate_env, mocker) -> None:
        """GIVEN a completed job with an empty download WHEN polled THEN it fails."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            side_effect=[(200, {"status": "completed"}), (200, b"")],
        )

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"

    def test_publish_failure_is_terminal(self, make_service, animate_env, mocker) -> None:
        """GIVEN a clip that cannot be published WHEN polled THEN it fails."""
        service, _ = _service(make_service)
        mocker.patch.object(
            service,
            "_animate_request",
            side_effect=[(200, {"status": "completed"}), (200, b"mp4bytes")],
        )
        mocker.patch.object(service, "_save_video_bytes", return_value=None)

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"

    def test_deconfigured_animate_is_terminal(self, make_service, monkeypatch) -> None:
        """GIVEN animate turned off WHEN a stashed job is polled THEN it fails cleanly."""
        monkeypatch.delenv("ANIMATE_API_KEY", raising=False)
        service, _ = make_service(animateApiUrl="")

        result = service._retry_video(_task(), {"job_id": "video_gen_1"})

        assert result.status == "failed_terminal"


class TestPollCadence:
    """Animate polls a running job; it does not retry a failing one."""

    def test_animate_backoff_is_flat(self, make_service) -> None:
        """GIVEN repeated animate polls WHEN backoff is computed THEN it stays flat.

        Every poll before the clip lands raises Timeout, so an exponential
        curve here is applied to the expected case rather than a fault. In
        prod on 2026-08-21 that put a 135s render on a 30/60/120 ladder and
        delivered it at 210s — over a minute of dead air after the video was
        already sitting on the box. The value is also the poll cadence, since
        the plugin arms its next wakeup from it, so it is what the user waits
        past the render finishing.
        """
        service, _ = _service(make_service)
        delays = [service._compute_backoff(n, "animate") for n in range(5)]
        assert delays == [10, 10, 10, 10, 10]

    def test_other_task_types_keep_exponential_backoff(self, make_service) -> None:
        """GIVEN a draw retry WHEN backoff is computed THEN it still doubles.

        There a repeat means the provider call keeps failing, which is exactly
        when backing off is the right move.
        """
        service, _ = _service(make_service)
        delays = [service._compute_backoff(n, "draw") for n in range(4)]
        assert delays == [30, 60, 120, 240]

    def test_backoff_is_capped_for_retry_paths(self, make_service) -> None:
        """GIVEN many failed attempts WHEN backoff is computed THEN it caps."""
        service, _ = _service(make_service)
        assert service._compute_backoff(20, "ask") == 300


class TestThreadedDelivery:
    """The clip arrives as an IRCv3 reply to the request, like @draw's image."""

    def test_msgid_is_stashed_with_the_job(self, make_service, animate_env, mocker) -> None:
        """GIVEN a msgid WHEN submitting THEN it rides along in request_data.

        Delivery happens minutes later in another process lifetime, so the
        msgid has to be durable — holding it in memory would lose the thread
        across the restart the design otherwise survives.
        """
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "video_gen_1"}))
        stash = mocker.patch.object(service, "_stash_timeout", return_value=True)

        service.video_generation("a forest", reply_target="#chan", reply_msgid="abc123")

        assert stash.call_args.kwargs["request_data"]["reply_msgid"] == "abc123"

    def test_privmsg_carries_the_reply_tag(self) -> None:
        """GIVEN a msgid WHEN the message is built THEN +draft/reply is attached."""
        from llm.plugin import LLM

        out = LLM._safe_privmsg("#chan", "https://paste.example/img/vid_a.mp4", "abc123")

        assert out.server_tags.get("+draft/reply") == "abc123"
        assert out.args[1] == "https://paste.example/img/vid_a.mp4"

    def test_privmsg_without_msgid_is_untagged(self) -> None:
        """GIVEN no msgid WHEN the message is built THEN it is a plain PRIVMSG.

        The tag improves how the line renders; it is never a precondition for
        sending one, so a server without message-tags still gets the clip.
        """
        from llm.plugin import LLM

        out = LLM._safe_privmsg("#chan", "https://paste.example/img/vid_a.mp4")

        assert not (out.server_tags or {}).get("+draft/reply")
        assert out.args[1] == "https://paste.example/img/vid_a.mp4"

    def test_reply_tag_does_not_defeat_injection_safety(self) -> None:
        """GIVEN a body with CRLF WHEN tagged THEN it is still neutralised."""
        from llm.plugin import LLM

        out = LLM._safe_privmsg("#chan", "ok\r\nQUIT :bye", "abc123")

        assert "\r\n" not in out.args[1]
        assert out.server_tags.get("+draft/reply") == "abc123"


class TestVideoUpload:
    """Video rides the image uploader, but the two are not interchangeable."""

    def test_mp4_upload_accepts_an_mp4_reply(self, make_service, mocker) -> None:
        """GIVEN an mp4 upload WHEN the host returns vid_*.mp4 THEN the URL is used."""
        service, _ = make_service(imageUploadUrl="https://paste.example/img/")
        mocker.patch.object(
            service,
            "_upload_image_bytes",
            return_value="https://paste.example/img/vid_a.mp4",
        )
        assert service._save_video_bytes(b"mp4bytes") == "https://paste.example/img/vid_a.mp4"

    def test_image_upload_rejects_a_video_reply(self, make_service, mocker) -> None:
        """GIVEN a png upload WHEN the host returns an .mp4 THEN it is not trusted.

        Images may legitimately come back under a different image extension
        (the host recompresses), but a video URL is never a valid answer to an
        image POST.
        """
        service, plugin = make_service(imageUploadUrl="https://paste.example/img/")
        payload = json.dumps(
            {"results": [{"success": True, "filePath": "/img/vid_a.mp4"}]}
        ).encode()

        opener = mocker.MagicMock()
        response = opener.open.return_value.__enter__.return_value
        response.read.return_value = payload
        mocker.patch("urllib.request.build_opener", return_value=opener)

        assert service._upload_image_bytes(b"\x89PNG", "png") is None

    def test_local_fallback_when_upload_is_off(self, make_service, tmp_path, mocker) -> None:
        """GIVEN no upload host WHEN publishing THEN the clip is written locally."""
        service, _ = make_service(
            imageUploadUrl="", httpRoot=str(tmp_path), httpUrlBase="https://bot.example/llm"
        )
        mocker.patch.object(service, "_upload_image_bytes", return_value=None)

        url = service._save_video_bytes(b"mp4bytes")

        assert url is not None
        assert url.endswith(".mp4")
        written = list(tmp_path.glob("vid_*.mp4"))
        assert len(written) == 1
        assert written[0].read_bytes() == b"mp4bytes"


class TestGenerateVideoTool:
    """The tool queues work; it must never hand back a link."""

    def _assistant(self, animate_fn):
        from llm.assistant import AssistantToolExecutor

        return AssistantToolExecutor(
            db=Mock(),
            context=Mock(),
            nick="alice",
            channel="#chan",
            animate_fn=animate_fn,
        )

    def test_tool_returns_the_acknowledgement(self) -> None:
        """GIVEN a queued job WHEN the tool runs THEN it returns the acknowledgement."""
        from llm.assistant import ToolCallbackResult

        assistant = self._assistant(lambda p: ToolCallbackResult(True, "Rendering your video."))
        out = assistant._tool_generate_video({"prompt": "a forest"})
        assert "Rendering your video." in out
        assert "http" not in out

    def test_tool_reports_failure(self) -> None:
        """GIVEN a rejected submission WHEN the tool runs THEN it reports the error."""
        from llm.assistant import ToolCallbackResult

        assistant = self._assistant(lambda p: ToolCallbackResult(False, "server said no"))
        out = assistant._tool_generate_video({"prompt": "a forest"})
        assert "server said no" in out

    def test_tool_requires_a_prompt(self) -> None:
        """GIVEN an empty prompt WHEN the tool runs THEN it errors without submitting."""
        called = []
        assistant = self._assistant(lambda p: called.append(p))
        assistant._tool_generate_video({"prompt": "   "})
        assert called == []

    def test_tool_unavailable_without_callback(self) -> None:
        """GIVEN no animate callback WHEN the tool runs THEN it says so."""
        assistant = self._assistant(None)
        out = assistant._tool_generate_video({"prompt": "a forest"})
        assert "not available" in out.lower()

    def test_tool_is_asked_for_only(self) -> None:
        """GIVEN the tool spec WHEN checked THEN it is not on unattended routes.

        A video is ~70s of exclusive GPU time, so it stays on the two routes
        where a human just asked for one — chat and the @animate planner — and
        off the ones where the bot generates unprompted: verse narrates
        ambiently, and remind_action fires with nobody present.
        """
        from llm.assistant import get_tools_for_profile
        from llm.profile import (
            PROFILE_ANIMATE,
            PROFILE_CHAT,
            PROFILE_REMIND_ACTION,
            PROFILE_VERSE,
        )

        def names(profile: str) -> set[str]:
            return {t["function"]["name"] for t in get_tools_for_profile(profile)}

        assert "generate_video" in names(PROFILE_CHAT)
        assert "generate_video" in names(PROFILE_ANIMATE)
        assert "generate_video" not in names(PROFILE_VERSE)
        assert "generate_video" not in names(PROFILE_REMIND_ACTION)


class TestAnimatePlanner:
    """@animate plans through the assistant, so canon can reach the prompt.

    The video box takes its prompt literally: it has never heard of the cast,
    and a bare name renders as nothing or as text on screen. A planner turn in
    front of the submission is what lets the lore block become a description,
    the same way @draw's planner turns canon into an image prompt.
    """

    def _result(self, mocker, content: str = "Rendering your clip."):
        from llm.service import AssistantResult, Spend

        return AssistantResult(
            content=content,
            grounding_used=False,
            spend=Spend(prompt_tokens=120, completion_tokens=40, cost=0.0012),
            model="gpt-4",
        )

    def _wire(self, plugin, mocker, *, canon: str | None = None):
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = self._result(mocker)
        mocker.patch.object(plugin, "_verse_context_for", return_value=canon)

    def test_animate_routes_through_assistant_completion(self, plugin_env, mocker) -> None:
        """GIVEN @animate WHEN it runs THEN it dispatches on the animate profile."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker)

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["route_profile"] == "animate"
        assert kwargs["entry_route"] == "animate"

    def test_canon_reference_layers_the_lore_block(self, plugin_env, mocker) -> None:
        """@animate of canon puts the lore in the overlay slot, like @draw."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker, canon="Established characters:\n- Archie: windbag")

        plugin.animate(mock_irc, mock_msg, ["the", "stinky", "lads"])

        sp = plugin.llm_service.assistant_completion.call_args.kwargs["system_prompt"]
        assert sp is not None and "Archie: windbag" in sp

    def test_no_grounding_without_canon(self, plugin_env, mocker) -> None:
        """No canon reference → system_prompt stays None (default animate prompt)."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker, canon=None)

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        assert plugin.llm_service.assistant_completion.call_args.kwargs["system_prompt"] is None

    def test_video_tool_is_wired_to_the_requester(self, plugin_env, mocker) -> None:
        """The planner's only generator is generate_video, bound to this caller.

        The clip is delivered minutes later, so the callback has to carry who
        asked and where — a submission stashed against nobody is a render with
        nowhere to go.
        """
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker)
        forwarded = mocker.patch.object(plugin, "_animate_for_assistant")

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs.get("draw_fn") is None
        kwargs["animate_fn"]("three men wade into grey surf")
        # The resolved Caller: the stashed job is keyed on the account and
        # delivered to the reply target, greeting the IRC nick.
        caller = forwarded.call_args.kwargs["caller"]
        assert caller == Caller(Identity("testnick", "test_account"), "#test")

    def test_planner_spend_is_booked(self, plugin_env, mocker) -> None:
        """The planner turn is the only thing @animate spends.

        The video box is self-hosted and reports no token accounting, so if
        this row does not carry the planner's usage, nothing does.
        """
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker)
        logged = mocker.patch.object(plugin, "_store_context_and_log_usage")

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        assert logged.call_args.args[1] == "animate"
        assert logged.call_args.args[4].cost == 0.0012

    def test_rendered_prompt_is_booked_with_the_ask(self, plugin_env, mocker) -> None:
        """The row carries the planner's script, not just what the user typed.

        The video box renders the planner's rewrite; the pending_tasks row that
        holds it is deleted on delivery, so the usage row is the only place the
        rendered wording can survive the render.
        """
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker)
        mocker.patch.object(plugin, "_animate_for_assistant")
        logged = mocker.patch.object(plugin, "_store_context_and_log_usage")

        def _plan(*args, **kwargs):
            kwargs["animate_fn"]("Three men wade into grey surf at dawn, handheld.")
            return self._result(mocker)

        plugin.llm_service.assistant_completion.side_effect = _plan

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        assert logged.call_args.args[2] == "a pine forest"
        assert (
            logged.call_args.kwargs["rendered_prompt"]
            == "Three men wade into grey surf at dawn, handheld."
        )

    def test_two_submissions_in_one_turn_are_both_booked(self, plugin_env, mocker) -> None:
        """A planner that calls the tool twice must not lose the first script."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker)
        mocker.patch.object(plugin, "_animate_for_assistant")
        logged = mocker.patch.object(plugin, "_store_context_and_log_usage")

        def _plan(*args, **kwargs):
            kwargs["animate_fn"]("first script")
            kwargs["animate_fn"]("second script")
            return self._result(mocker)

        plugin.llm_service.assistant_completion.side_effect = _plan

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        assert logged.call_args.kwargs["rendered_prompt"] == "first script | second script"

    def test_no_submission_leaves_the_rendered_prompt_empty(self, plugin_env, mocker) -> None:
        """A planner turn that never calls the tool rendered nothing."""
        plugin, mock_irc, mock_msg = plugin_env
        mock_irc.state.nickToAccount.return_value = "test_account"
        self._wire(plugin, mocker)
        logged = mocker.patch.object(plugin, "_store_context_and_log_usage")

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        assert logged.call_args.kwargs["rendered_prompt"] == ""

    def test_animate_requires_account(self, plugin_env, mocker) -> None:
        """Unauthenticated → no planner turn and no clip."""
        plugin, mock_irc, mock_msg = plugin_env
        self._wire(plugin, mocker)

        plugin.animate(mock_irc, mock_msg, ["a", "pine", "forest"])

        mock_irc.error.assert_called_once()
        plugin.llm_service.assistant_completion.assert_not_called()


class TestJobMarkerPoisoning:
    """The bot must never answer with a job id it copied out of its history.

    Earlier revisions stored "[Video job: <id>]" as the assistant's side of an
    @animate turn — a bookkeeping string, in the slot the model reads as "how I
    answer requests like this". #afternet, 2026-08-21: two identical requests
    four minutes apart both answered with the SAME id, which two real
    submissions cannot share. No video was queued either time.
    """

    _MARKER = "[Video job: video_gen_0b5b4e3a1e6e48b3a9e5f1c0d2b4a6e7]"

    def test_marker_only_reply_is_detected(self) -> None:
        """The copied-marker reply, exactly as it reached the channel."""
        from llm.service import _is_job_marker_reply

        assert _is_job_marker_reply(self._MARKER)
        assert _is_job_marker_reply(f"  {self._MARKER}  ")
        assert _is_job_marker_reply("[Video job: rejected]")
        assert _is_job_marker_reply("[Generated image: https://example.com/img_a.png]")

    def test_a_real_reply_is_left_alone(self) -> None:
        """Narrow on purpose: only a reply that is NOTHING but markers counts.

        A reply that says something to the user and happens to name the job is
        chatty, not broken, and rewriting it would be the guard overreaching.
        """
        from llm.service import _is_job_marker_reply

        assert not _is_job_marker_reply("Rendering your clip — I'll post it here.")
        assert not _is_job_marker_reply(f"Rendering now. {self._MARKER}")
        assert not _is_job_marker_reply("")

    def test_markers_are_stripped_from_history(self) -> None:
        """Stripped at read time, so the poisoned rows need no DB surgery."""
        from llm.context import Role
        from llm.service import _strip_job_markers

        history = [
            {"role": Role.USER, "content": "animate a stinky lad in a theatre"},
            {"role": Role.ASSISTANT, "content": self._MARKER},
            {"role": Role.USER, "content": "animate a stinky lad in a theatre"},
        ]

        kept = _strip_job_markers(history)

        assert [m["content"] for m in kept] == [
            "animate a stinky lad in a theatre",
            "animate a stinky lad in a theatre",
        ]

    def test_user_turns_are_never_stripped(self) -> None:
        """A user quoting a marker back is a premise, not the model's own output."""
        from llm.context import Role
        from llm.service import _strip_job_markers

        history = [{"role": Role.USER, "content": self._MARKER}]
        assert _strip_job_markers(history) == history

    def test_strip_runs_on_every_route(self) -> None:
        """Registered in the shared strip table, not wired into one path.

        The poisoned turns sit in both the personal thread and the shared
        channel window, and every route reads one or both.
        """
        from llm.service import _EVERY_ROUTE_STRIPS

        assert "job_marker" in dict(_EVERY_ROUTE_STRIPS)

    def test_guard_catches_a_marker_that_survives_to_the_reply(self) -> None:
        """Belt to the strip's braces: a marker must not reach the channel.

        The nudge has to send the model back to the tool, not just tell it to
        reword — nothing was queued, so a politely-worded version of the same
        reply is still a lie about a clip that is not rendering.
        """
        from llm.profile import PROFILE_CHAT
        from llm.service import REPLY_GUARDS, _ReplyGuardContext

        guard = REPLY_GUARDS["job_marker"]
        ctx = _ReplyGuardContext(
            content=self._MARKER,
            prompt="animate a stinky lad in a theatre",
            route_profile=PROFILE_CHAT,
            any_tool_ran=False,
            prior_replies=(),
        )

        assert guard.detect(ctx)
        assert "generate_video" in guard.nudge


class TestAnimateForcesTheTool:
    """On the @animate route, step 0 has no choice about calling generate_video.

    The user ran the command; deciding whether to make a video is not the
    planner's call. Without this it can answer in text — which on 2026-08-21
    meant copying a "[Video job: …]" marker out of history and acknowledging a
    clip nobody queued.
    """

    @pytest.fixture
    def service(self, make_service, animate_env):  # type: ignore[no-untyped-def]
        svc, _plugin = _service(make_service, assistantModel="gpt-4")
        return svc

    def _run(self, service, mocker, profile: str, prompt: str = "the stinky lads at the beach"):
        """One assistant_completion on ``profile``, returning step 0's kwargs."""
        from llm.assistant import ToolCallbackResult

        tool_call = make_tool_call(
            "generate_video", {"prompt": "three men wade into surf"}, call_id="call_vid"
        )
        completion = mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[
                make_completion_response(None, tool_calls=[tool_call]),
                make_completion_response("Rendering your clip."),
            ],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.assistant_completion(
            prompt=prompt,
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=profile,
            account="tester",
            capabilities=frozenset({"llm.animate"}),
            animate_fn=lambda p: ToolCallbackResult(True, "Rendering your video."),
        )
        return completion.call_args_list[0].kwargs

    def test_animate_route_forces_generate_video(self, service, mocker) -> None:
        """GIVEN the animate profile WHEN step 0 runs THEN the tool is forced."""
        kwargs = self._run(service, mocker, "animate")
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "generate_video"},
        }

    def test_explicit_chat_request_is_forced(self, service, mocker) -> None:
        """The case that was actually broken.

        "vibebot animate X" is not a command — inFilter suppresses Limnoria's
        dispatcher for anything without the prefix char — so it lands here, on
        the chat route, where the model answered by inventing a link instead of
        calling the tool. An explicit ask leaves it no choice.
        """
        kwargs = self._run(
            service,
            mocker,
            "chat",
            prompt="animate baby prince harry riding a corgi around the throne",
        )
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "generate_video"},
        }

    def test_chat_without_an_explicit_ask_is_not_forced(self, service, mocker) -> None:
        """Chat keeps its judgement everywhere else: mentioning the lads, or a
        video someone else made, is not a request for 70 seconds of GPU time."""
        assert "tool_choice" not in self._run(service, mocker, "chat")
        assert "tool_choice" not in self._run(
            service, mocker, "chat", prompt="did you see that video of the cat"
        )

    def test_trigger_is_tight_enough_to_be_safe(self) -> None:
        """A false positive costs a real render, so pin both directions."""
        from llm.service import EXPLICIT_VIDEO_RE

        for asks in (
            "animate year 7 stinky lad having norovirus in a crowded theatre",
            "video of the lads at the beach",
            "make me a video of a corgi",
            "can you generate a short clip of the throne room",
            "render an animation of the lads",
        ):
            assert EXPLICIT_VIDEO_RE.search(asks), asks

        for chats in (
            "did you see that video of the cat",
            "what is a good animation studio",
            "the video was great",
            "draw a corgi",
            "tell me about video codecs",
        ):
            assert not EXPLICIT_VIDEO_RE.search(chats), chats

    def test_unconfigured_box_forces_nothing(self, make_service, mocker) -> None:
        """Forcing a tool that was excluded from the list is a provider error.

        The box being unwired drops generate_video from profile_tools, so the
        gate has to be tool presence, not the profile alone.
        """
        from llm.assistant import ToolCallbackResult

        service, _plugin = make_service(assistantModel="gpt-4")  # no animateApiUrl
        completion = mocker.patch(
            "llm.service.litellm.completion",
            side_effect=[make_completion_response("I can't make videos right now.")],
        )
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        service.assistant_completion(
            prompt="the stinky lads at the beach",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile="animate",
            animate_fn=lambda p: ToolCallbackResult(True, "queued"),
        )

        assert "tool_choice" not in completion.call_args_list[0].kwargs


class TestNoInventedLinks:
    """No link may leave the @animate route, whatever host it names.

    The image guard next door ignores URLs on hosts that are not ours — in
    chat, somebody else's picture is none of our business. Here the only tool
    is generate_video, which never returns a URL, so a link in the reply is
    invented. #afternet, 2026-08-21:

        <rdrake> vibebot animate baby prince harry riding a corgi
        <vibebot> https://files.oaiusercontent.com/file-VlKq…harry-corgi.gif

    Four seconds after the request, for a render that takes 135, on a host the
    bot has never published to.
    """

    _FAKE = (
        "https://files.oaiusercontent.com/file-VlKqN8pL3jXvR2mW9bTqYh"
        "?se=2026-08-22T00%3A00%3A00Z&rscd=attachment%3B%20filename%3D%22harry-corgi.gif%22"
    )

    @pytest.fixture
    def service(self, make_service, animate_env):  # type: ignore[no-untyped-def]
        svc, _plugin = _service(make_service, assistantModel="gpt-4")
        return svc

    def _run(self, service, mocker, reply: str, *, profile: str = "animate", tool: bool = True):
        """One completion on ``profile`` whose final text is ``reply``."""
        from llm.assistant import ToolCallbackResult

        responses = []
        if tool:
            responses.append(
                make_completion_response(
                    None,
                    tool_calls=[
                        make_tool_call("generate_video", {"prompt": "a corgi"}, call_id="call_vid")
                    ],
                )
            )
        responses.append(make_completion_response(reply))
        mocker.patch("llm.service.litellm.completion", side_effect=responses)
        mocker.patch("llm.service.litellm.completion_cost", return_value=0.0)

        return service.assistant_completion(
            prompt="baby prince harry riding a corgi around the throne",
            nick="testuser",
            channel="#test",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            route_profile=profile,
            account="tester",
            capabilities=frozenset({"llm.animate"}),
            animate_fn=lambda p: ToolCallbackResult(True, "Rendering your video."),
        )

    def test_bare_invented_link_becomes_the_real_outcome(self, service, mocker) -> None:
        """The transcript case: the reply was the link and nothing else."""
        from llm.service import _VIDEO_FABRICATION_FALLBACK

        result = self._run(service, mocker, self._FAKE)

        assert "http" not in result.content
        assert result.content == _VIDEO_FABRICATION_FALLBACK

    def test_link_is_stripped_from_an_otherwise_fine_reply(self, service, mocker) -> None:
        """A real sentence keeps its words; only the invented link goes."""
        result = self._run(service, mocker, f"Here's your clip! {self._FAKE} Enjoy.")

        assert "http" not in result.content
        assert result.content == "Here's your clip! Enjoy."

    def test_nothing_queued_says_so(self, service, mocker) -> None:
        """A link invented with no tool call behind it means no clip is coming.

        Saying "it's rendering" here would be the same lie in nicer words.
        """
        from llm.service import _VIDEO_FABRICATION_NOTHING_QUEUED

        result = self._run(service, mocker, self._FAKE, tool=False)

        assert result.content == _VIDEO_FABRICATION_NOTHING_QUEUED

    def test_chat_route_keeps_its_links(self, service, mocker) -> None:
        """Deliberately narrow: chat can legitimately cite somebody's URL.

        The invariant that makes stripping safe — the only tool returns no
        URL — is a property of the animate route, not of the bot.
        """
        result = self._run(service, mocker, f"Found it: {self._FAKE}", profile="chat")

        assert self._FAKE in result.content

    def test_strip_tidies_the_whitespace_it_leaves(self) -> None:
        """A stripped link must not leave a double space or a ragged tail."""
        from llm.service import _strip_urls

        assert _strip_urls("Here you go:  https://x.test/a.gif  done") == "Here you go: done"
        assert _strip_urls("https://x.test/a.gif") == ""


class TestTypingRefreshPass:
    """One pass of the render-typing refresher.

    State is derived from pending_tasks every pass rather than tracked, so
    there is no release to get wrong — see docs/plans/2026-08-21-animate-ux.md.
    """

    def _irc(self, mocker, network: str = "afternet", channels=("#chan",)):
        irc = mocker.MagicMock()
        irc.network = network
        irc.state.channels = dict.fromkeys(channels, mocker.MagicMock())
        return irc

    def test_active_target_is_typed(self, plugin_env, mocker) -> None:
        """GIVEN a rendering clip WHEN a pass runs THEN the target gets active."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        plugin._typing_refresh_pass()

        plugin.llm_service.send_typing_indicator.assert_called_once_with(irc, "#chan", "active")
        assert plugin.llm_service.typing.holds("afternet", "#chan")

    def test_target_dropping_out_gets_one_done(self, plugin_env, mocker) -> None:
        """The clip landed: exactly one done, and not repeated next pass."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])

        plugin.db.active_animate_targets.return_value = ["#chan"]
        plugin._typing_refresh_pass()
        plugin.llm_service.send_typing_indicator.reset_mock()

        plugin.db.active_animate_targets.return_value = []
        plugin._typing_refresh_pass()
        plugin.llm_service.send_typing_indicator.assert_called_once_with(irc, "#chan", "done")
        assert not plugin.llm_service.typing.holds("afternet", "#chan")

        plugin.llm_service.send_typing_indicator.reset_mock()
        plugin._typing_refresh_pass()
        plugin.llm_service.send_typing_indicator.assert_not_called()

    def test_channel_the_bot_has_left_is_skipped(self, plugin_env, mocker) -> None:
        """No membership, no typing — mirrors the delivery path's resolution."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker, channels=("#other",))
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        plugin._typing_refresh_pass()

        plugin.llm_service.send_typing_indicator.assert_not_called()
        assert not plugin.llm_service.typing.holds("afternet", "#chan")

    def test_pm_target_uses_the_first_connection(self, plugin_env, mocker) -> None:
        """A PM has no channel membership to check."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker, channels=())
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["alice"]

        plugin._typing_refresh_pass()

        plugin.llm_service.send_typing_indicator.assert_called_once_with(irc, "alice", "active")

    def test_holds_are_keyed_by_network(self, plugin_env, mocker) -> None:
        """The key is (network, target), and resolution stops at the first hit.

        Keying on the bare target would merge #chan on one network with #chan
        on another; stopping at the first connection carrying it matches how
        the delivery path picks a connection.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        a = self._irc(mocker, network="afternet")
        b = self._irc(mocker, network="other")
        mocker.patch("llm.plugin.world.ircs", [a, b])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        plugin._typing_refresh_pass()

        # Resolution stops at the first connection carrying the target, same
        # as the delivery path, so only one network is held.
        assert plugin.llm_service.typing.active_targets() == {("afternet", "#chan")}

    def test_db_failure_does_not_raise(self, plugin_env, mocker) -> None:
        """A read failure must not kill the refresher thread."""
        plugin, _mock_irc, _mock_msg = plugin_env
        mocker.patch("llm.plugin.world.ircs", [])
        plugin.db.active_animate_targets.side_effect = RuntimeError("db is gone")

        plugin._typing_refresh_pass()  # must not raise

    def test_membership_check_walks_to_the_carrying_connection(self, plugin_env, mocker) -> None:
        """A `continue` must fall through to the next connection, not stop early.

        A `continue`-to-`break` mutation would pass every other test here
        since the carrying connection is always first or the only one; this
        puts the non-carrying connection first to catch that mutant.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        a = self._irc(mocker, network="a", channels=())
        b = self._irc(mocker, network="b", channels=("#chan",))
        mocker.patch("llm.plugin.world.ircs", [a, b])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        plugin._typing_refresh_pass()

        plugin.llm_service.send_typing_indicator.assert_called_once_with(b, "#chan", "active")
        assert plugin.llm_service.typing.active_targets() == {("b", "#chan")}

    def test_active_send_failure_does_not_raise_and_keeps_the_hold(
        self, plugin_env, mocker
    ) -> None:
        """A failed active send must not raise; the hold survives it.

        The registry holds a target because a send was *attempted*, not
        because it was confirmed delivered — send_typing_indicator no-ops
        without raising when the server lacks message-tags. Keeping the hold
        means the target still gets its one done when the clip lands.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]
        plugin.llm_service.send_typing_indicator.side_effect = RuntimeError("queue closed")

        plugin._typing_refresh_pass()  # must not raise

        assert plugin.llm_service.typing.holds("afternet", "#chan")

    def test_one_targets_resolution_failure_does_not_abort_the_pass(
        self, plugin_env, mocker
    ) -> None:
        """A membership check that raises must not stop a later, healthy target."""

        class _RaisingChannels:
            def __contains__(self, _item: object) -> bool:
                raise RuntimeError("state is half-initialized")

        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        irc.state.channels = _RaisingChannels()
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan", "alice"]

        plugin._typing_refresh_pass()  # must not raise

        plugin.llm_service.send_typing_indicator.assert_called_once_with(irc, "alice", "active")
        assert plugin.llm_service.typing.holds("afternet", "alice")
        assert not plugin.llm_service.typing.holds("afternet", "#chan")

    def test_a_hold_on_another_network_does_not_suppress_done(self, plugin_env, mocker) -> None:
        """#chan on afternet and #chan elsewhere are different rooms.

        Holds are keyed by (network, target) precisely so a render on one
        network cannot swallow a planner turn's +typing=done on another.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        other = self._irc(mocker, network="other")
        plugin.llm_service.typing.set_group("render", {("other", "#chan"): other})

        assert plugin.llm_service.typing.holds("other", "#chan")
        assert not plugin.llm_service.typing.holds("afternet", "#chan")

        # A planner turn on afternet ends: its done goes out, and the render
        # on the other network keeps typing.
        afternet = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"), prefix="alice!u@host")
        plugin.llm_service.typing.hold(afternet, msg.args[0])()

        plugin.llm_service.send_typing_indicator.assert_any_call(afternet, "#chan", "done")
        assert plugin.llm_service.typing.holds("other", "#chan")

    def test_a_pass_that_found_work_reports_it(self, plugin_env, mocker) -> None:
        """The return value is the database's answer, not the send's."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        assert plugin._typing_refresh_pass() is True

        plugin.db.active_animate_targets.return_value = []
        assert plugin._typing_refresh_pass() is False

    def test_an_unresolvable_target_still_reports_work(self, plugin_env, mocker) -> None:
        """Nothing could be typed, but something is still rendering."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker, channels=())
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]

        assert plugin._typing_refresh_pass() is True
        plugin.llm_service.send_typing_indicator.assert_not_called()

    def test_persistent_read_failure_clears_the_holds_and_parks(self, plugin_env, mocker) -> None:
        """A dead connection must not strand the set and spin a 4s log cycle.

        A zombie refresher (die()'s join timed out, then db.close() ran) sees
        RuntimeError("LLMDatabase is closed") on every pass. Returning early
        without recomputing left the "render" hold group populated forever,
        which suppressed the +typing=done of every ordinary chat reply on
        that target.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]
        plugin._typing_refresh_pass()
        assert plugin.llm_service.typing.holds("afternet", "#chan")

        plugin.db.active_animate_targets.side_effect = RuntimeError("LLMDatabase is closed")

        # A blip is worth retrying through...
        assert plugin._typing_refresh_pass() is True
        assert plugin._typing_refresh_pass() is True
        # ...but a persistent failure gives up and lets go of the set.
        assert plugin._typing_refresh_pass() is False
        assert not plugin.llm_service.typing.holds("afternet", "#chan")

    def test_a_recovered_read_resets_the_failure_budget(self, plugin_env, mocker) -> None:
        """Two failures then a good read must not leave the loop one strike from parking."""
        plugin, _mock_irc, _mock_msg = plugin_env
        irc = self._irc(mocker)
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.side_effect = RuntimeError("db is busy")
        plugin._typing_refresh_pass()
        plugin._typing_refresh_pass()

        plugin.db.active_animate_targets.side_effect = None
        plugin.db.active_animate_targets.return_value = ["#chan"]
        assert plugin._typing_refresh_pass() is True

        plugin.db.active_animate_targets.side_effect = RuntimeError("db is busy")
        assert plugin._typing_refresh_pass() is True
        assert plugin._typing_refresh_pass() is True
        assert plugin._typing_refresh_pass() is False


class TestTypingRefresherLifecycle:
    """The thread idles instead of polling, and dies with the plugin."""

    def _quiesce_refresher(self, plugin) -> None:
        """Stop the __init__-started refresher so a test can drive the loop itself.

        The plugin starts a live daemon thread on the same two events these
        tests manipulate. Left running, it races the test for the wake flag
        and calls _typing_refresh_pass on its own schedule, so a test that
        counts passes or asserts on the wake flag is measuring two drivers.
        """
        plugin._render_typing_stop.set()
        plugin._render_typing_wake.set()
        plugin._render_typing_thread.join(timeout=2.0)
        assert not plugin._render_typing_thread.is_alive()
        plugin._render_typing_stop.clear()
        plugin._render_typing_wake.clear()

    def test_submission_wakes_the_refresher(self, plugin_env, mocker) -> None:
        """GIVEN a queued clip WHEN the tool callback runs THEN the loop wakes."""
        from llm.service import VideoResult

        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.video_generation.return_value = VideoResult(
            content="Rendering your video.", job_id="job-1", queued=True
        )
        self._quiesce_refresher(plugin)

        plugin._animate_for_assistant(
            mock_irc, mock_msg, "a corgi", caller=Caller.from_msg(mock_msg, "acct")
        )

        assert plugin._render_typing_wake.is_set()

    def test_rejected_submission_does_not_wake(self, plugin_env, mocker) -> None:
        """Nothing is rendering, so nothing should type."""
        from llm.service import VideoResult

        plugin, mock_irc, mock_msg = plugin_env
        plugin.llm_service.video_generation.return_value = VideoResult(
            content="Error: video server rejected the request.", error="rejected"
        )
        self._quiesce_refresher(plugin)

        plugin._animate_for_assistant(
            mock_irc, mock_msg, "a corgi", caller=Caller.from_msg(mock_msg, "acct")
        )

        assert not plugin._render_typing_wake.is_set()

    def test_loop_runs_passes_until_nothing_is_rendering(self, plugin_env, mocker) -> None:
        """Woken → pass; a pass reporting no work → back to blocking, wake flag cleared.

        Control flow only — the pass is mocked, so this says nothing about
        where the loop gets its answer from; see
        test_loop_keeps_going_when_the_target_cannot_be_resolved_yet.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        self._quiesce_refresher(plugin)
        calls = []

        def fake_pass() -> bool:
            calls.append(1)
            # Two passes with work, then nothing left to render.
            return len(calls) < 2

        mocker.patch.object(plugin, "_typing_refresh_pass", side_effect=fake_pass)
        plugin._RENDER_TYPING_INTERVAL = 0.01
        plugin._render_typing_wake.set()

        plugin._render_typing_loop(max_cycles=1)

        assert len(calls) == 2
        assert not plugin._render_typing_wake.is_set()

    def test_a_wake_arriving_during_the_final_pass_is_not_lost(self, plugin_env, mocker) -> None:
        """A submission landing mid-pass must not be erased by that pass's own clear.

        Simulates a worker thread writing a new pending row and calling
        _render_typing_wake.set() while the refresher's final pass (the one
        that empties the "render" hold group) is still in flight. If the wake
        flag is cleared *after* the pass runs, that set() is stomped and the
        new clip renders with no typing indicator until an unrelated later
        submission wakes the loop.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        self._quiesce_refresher(plugin)

        def fake_pass() -> bool:
            # A submission lands mid-pass, after the SELECT found nothing
            # left, before this pass finishes.
            plugin._render_typing_wake.set()
            return False

        mocker.patch.object(plugin, "_typing_refresh_pass", side_effect=fake_pass)
        plugin._RENDER_TYPING_INTERVAL = 0.01
        plugin._render_typing_wake.set()

        plugin._render_typing_loop(max_cycles=1)

        assert plugin._render_typing_wake.is_set()

    def test_loop_keeps_going_when_the_target_cannot_be_resolved_yet(
        self, plugin_env, mocker
    ) -> None:
        """Cold start: the database says a clip is rendering before the bot has joined.

        This is the auto-deploy path. __init__ presets the wake event, so the
        first pass runs about a second in, while state.channels is still
        empty — nothing can be typed. Parking on that would leave a real
        two-minute render in silence, which is the exact failure this feature
        exists to remove. The database is what says whether anything is
        rendering; whether a send resolved is a separate question.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        self._quiesce_refresher(plugin)
        irc = mocker.MagicMock()
        irc.network = "afternet"
        irc.state.channels = {}
        mocker.patch("llm.plugin.world.ircs", [irc])
        plugin.db.active_animate_targets.return_value = ["#chan"]
        plugin._RENDER_TYPING_INTERVAL = 0.01

        passes: list[int] = []
        real_pass = plugin._typing_refresh_pass

        def counting_pass() -> bool:
            passes.append(1)
            result = real_pass()
            if len(passes) >= 3:
                # Enough to prove the loop did not park; let it exit.
                plugin._render_typing_stop.set()
            return result

        mocker.patch.object(plugin, "_typing_refresh_pass", side_effect=counting_pass)
        plugin._render_typing_wake.set()

        plugin._render_typing_loop(max_cycles=1)

        assert len(passes) == 3
        plugin.llm_service.send_typing_indicator.assert_not_called()

    def test_die_warns_when_the_refresher_outlives_the_join(self, plugin_env, mocker) -> None:
        """A thread still alive after die() is a zombie against a closed database.

        die() cannot force it down, but an unlogged zombie is a traceback
        every four seconds with nothing naming its cause.
        """
        plugin, _mock_irc, _mock_msg = plugin_env
        self._quiesce_refresher(plugin)
        plugin._render_typing_thread = mocker.MagicMock()
        plugin._render_typing_thread.is_alive.return_value = True
        log = mocker.patch.object(plugin, "log")

        plugin.die()

        assert log.warning.called
        assert "typing" in " ".join(str(call.args[0]) for call in log.warning.call_args_list)

    def test_die_stops_the_thread(self, plugin_env, mocker) -> None:
        """An orphaned daemon thread would keep typing into a dead plugin."""
        plugin, _mock_irc, _mock_msg = plugin_env
        assert plugin._render_typing_thread.is_alive()
        assert plugin._render_typing_thread.daemon

        plugin.die()

        assert plugin._render_typing_stop.is_set()
        plugin._render_typing_thread.join(timeout=2.0)
        assert not plugin._render_typing_thread.is_alive()


class TestPlannerAndRenderShareOneIndicator:
    """The planner's release must not cancel the render's indicator.

    The planner turn ends seconds after the render refresher starts on the
    same target. Both hold the same refcounted indicator, so the done waits
    for whichever lets go last — without that it shows as a visible flicker.
    """

    def _irc(self, mocker):
        irc = mocker.MagicMock()
        irc.network = "afternet"
        irc.state.capabilities_ack = {"message-tags"}
        return irc

    def _states(self, irc) -> list[str]:
        """The +typing states actually put on the wire for ``irc``."""
        return [call.args[0].server_tags["+typing"] for call in irc.queueMsg.call_args_list]

    def test_done_waits_until_the_render_lets_go(self, make_service, mocker) -> None:
        service, _plugin = make_service()
        irc = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"), prefix="alice!u@host")

        stop = service._begin_typing(irc, msg)
        service.typing.set_group("render", {("afternet", "#chan"): irc})
        stop()

        assert "done" not in self._states(irc), "the render still holds it"

        service.typing.set_group("render", {})

        assert self._states(irc) == ["active", "done"]

    def test_done_is_sent_when_nothing_else_holds_the_target(self, make_service, mocker) -> None:
        service, _plugin = make_service()
        irc = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"), prefix="alice!u@host")

        service._begin_typing(irc, msg)()

        assert self._states(irc) == ["active", "done"]

    def test_a_second_command_on_the_channel_does_not_restart_the_indicator(
        self, make_service, mocker
    ) -> None:
        """Two planner turns on one channel: one active, one done."""
        service, _plugin = make_service()
        irc = self._irc(mocker)
        msg = mocker.MagicMock(args=("#chan", "hi"), prefix="alice!u@host")

        first = service._begin_typing(irc, msg)
        second = service._begin_typing(irc, msg)
        first()
        second()

        assert self._states(irc) == ["active", "done"]


class TestAnimateDeliveryLine:
    """The delivered clip says who asked and for what, and never loses the URL."""

    _URL = "https://paste.boxlabs.uk/img/vid_6a8830b21af1d.mp4"

    def test_line_carries_nick_prompt_and_url(self, plugin_env) -> None:
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery(
            "rdrake", "a corgi riding a unicorn", self._URL, "#chan"
        )

        assert line == f'rdrake: your video is ready! "a corgi riding a unicorn" → {self._URL}'

    def test_long_prompt_never_costs_the_url(self, plugin_env) -> None:
        """The URL is the one indispensable part; it is budgeted first."""
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery("rdrake", "corgi " * 200, self._URL, "#chan")

        assert line.endswith(self._URL)
        assert len(line.encode("utf-8")) <= 400

    def test_a_large_configured_mores_length_is_still_capped(self, plugin_env) -> None:
        """mores.length is not the wire limit, and this path bypasses the wrapper.

        A deployed bot.conf with, say, 470 plus a 60-100 byte PRIVMSG prefix
        blows past 512 and the server truncates — taking the URL, the one
        part this formatter exists to protect. The cap is local to the
        formatter; _reply_mores_length's other callers keep the configured
        value.
        """
        import supybot.conf as conf

        plugin, _mock_irc, _mock_msg = plugin_env
        chan_value = conf.supybot.reply.mores.length.get("#chan")
        original = chan_value()
        chan_value.setValue(470)
        try:
            line = plugin._format_animate_delivery("rdrake", "corgi " * 200, self._URL, "#chan")
        finally:
            chan_value.setValue(original)

        assert line.endswith(self._URL)
        assert len(line.encode("utf-8")) <= 400

    def test_multibyte_prompt_is_budgeted_in_bytes(self, plugin_env) -> None:
        """100 characters of emoji is 400 bytes — a character count is not a budget."""
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery("rdrake", "\U0001f600" * 100, self._URL, "#chan")

        assert line.endswith(self._URL)
        assert len(line.encode("utf-8")) <= 400

    def test_prompt_is_dropped_when_it_cannot_fit(self, plugin_env) -> None:
        """Better a bare-but-attributed line than a truncated link."""
        plugin, _mock_irc, _mock_msg = plugin_env
        long_url = "https://paste.boxlabs.uk/img/" + ("x" * 340) + ".mp4"

        line = plugin._format_animate_delivery(
            "rdrake", "a corgi riding a unicorn", long_url, "#chan"
        )

        assert line == f"rdrake: your video is ready! → {long_url}"

    def test_formatting_codes_are_stripped_from_the_echo(self, plugin_env) -> None:
        """A requester does not get to colour the bot's output."""
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery(
            "rdrake", "\x02bold\x03,4red\x0f corgi", self._URL, "#chan"
        )

        assert "\x02" not in line and "\x03" not in line
        assert "boldred corgi" in line

    def test_blank_prompt_returns_the_bare_line(self, plugin_env) -> None:
        """An empty or whitespace-only prompt has nothing worth quoting."""
        plugin, _mock_irc, _mock_msg = plugin_env

        line = plugin._format_animate_delivery("rdrake", "   ", self._URL, "#chan")

        assert line == f"rdrake: your video is ready! → {self._URL}"

    def test_pm_target_does_not_pick_up_a_channel_override(self, plugin_env) -> None:
        """A PM target is not a channel, so a channel-scoped override on some
        other channel must not leak into it.

        ``getSpecific`` (underneath ``conf.get``, and now ``_reply_mores_length``)
        silently drops a non-channel ``channel=`` argument rather than raising,
        so this can't be pinned by asserting ``conf.get`` was skipped — it is
        always called now. What matters is the *value* it resolves to for a
        PM target, so this sets a tiny override on an unrelated channel and
        checks it has no effect on the nick-targeted call.
        """
        import supybot.conf as conf

        plugin, _mock_irc, _mock_msg = plugin_env
        chan_value = conf.supybot.reply.mores.length.get("#chan")
        original = chan_value()
        chan_value.setValue(40)  # too small to fit the prompt at all
        try:
            pm_line = plugin._format_animate_delivery(
                "rdrake", "a corgi riding a unicorn", self._URL, "rdrake"
            )
            chan_line = plugin._format_animate_delivery(
                "rdrake", "a corgi riding a unicorn", self._URL, "#chan"
            )
        finally:
            chan_value.setValue(original)

        # PM ignores the #chan-scoped override entirely and keeps the prompt...
        assert pm_line == f'rdrake: your video is ready! "a corgi riding a unicorn" → {self._URL}'
        # ...while the channel path is genuinely constrained by it.
        assert chan_line == f"rdrake: your video is ready! → {self._URL}"


class TestAnimateAdmission:
    """Two caps, checked before anything reaches the box.

    Nine clips arrived in eight minutes on 2026-08-31 and two expired
    unrendered. The per-user cap stops one person stacking requests seconds
    apart; the global one bounds the longest wait anyone is promised, because
    the box renders one clip at a time.

    The refusals are tool results, not replies: code states the fact, the
    model phrases it under the channel's prompt.
    """

    def _submit(self, plugin, irc, msg, *, account: str | None = "alice"):
        caller = Caller.from_msg(msg, account)
        return plugin._submit_video(irc, msg, "a corgi", caller=caller)

    def test_per_user_cap_refuses_third_clip(self, plugin_env, mocker) -> None:
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="registered")
        mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=2)
        mocker.patch.object(plugin.db, "count_pending_animate", return_value=2)
        submit = mocker.patch.object(plugin.llm_service, "video_generation")

        result = self._submit(plugin, irc, msg)

        assert result.error
        assert "2" in result.content
        assert "limit 2" in result.content
        submit.assert_not_called()

    def test_per_user_cap_counts_the_asker_not_the_bot(self, plugin_env, mocker) -> None:
        """The cap is per identity: it must ask the database about this user."""
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="registered")
        mine = mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=2)
        mocker.patch.object(plugin.db, "count_pending_animate", return_value=0)
        mocker.patch.object(plugin.llm_service, "video_generation")

        self._submit(plugin, irc, msg, account="alice")

        # display_nick comes from the message, not the account-resolved nick.
        assert mine.call_args.kwargs == {"account": "alice", "nick": msg.nick}

    def test_owner_skips_per_user_cap_but_not_global(self, plugin_env, mocker) -> None:
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="owner")
        per_user = mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=5)
        count_all = mocker.patch.object(plugin.db, "count_pending_animate", return_value=6)
        submit = mocker.patch.object(plugin.llm_service, "video_generation")

        result = self._submit(plugin, irc, msg, account="rdrake")

        assert result.error
        assert "full" in result.content
        submit.assert_not_called()
        per_user.assert_not_called()
        count_all.assert_called_once()

    def test_admin_also_skips_the_per_user_cap(self, plugin_env, mocker) -> None:
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="admin")
        mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=5)
        mocker.patch.object(plugin.db, "count_pending_animate", return_value=0)
        submit = mocker.patch.object(
            plugin.llm_service, "video_generation", return_value=_queued_result()
        )

        result = self._submit(plugin, irc, msg, account="rdrake")

        assert not result.error
        submit.assert_called_once()

    def test_trusted_is_not_exempt_from_the_per_user_cap(self, plugin_env, mocker) -> None:
        """Exemption stops at admin: trusted still queues behind the cap."""
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="trusted")
        mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=2)
        mocker.patch.object(plugin.db, "count_pending_animate", return_value=0)
        submit = mocker.patch.object(plugin.llm_service, "video_generation")

        result = self._submit(plugin, irc, msg)

        assert result.error
        submit.assert_not_called()

    def test_concurrent_requests_cannot_both_take_the_last_slot(self, plugin_env, mocker) -> None:
        """The TLC counterexample (docs/formal/AnimateAdmission.tla): the row
        that makes a clip count is inserted only after the POST to the box, so
        two requests inside that window both counted 1 < 2 and both queued."""
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="registered")
        rows = [1]  # alice already has one clip; the per-user limit is 2
        mocker.patch.object(
            plugin.db, "count_pending_animate_for", side_effect=lambda *a, **k: rows[0]
        )
        mocker.patch.object(plugin.db, "count_pending_animate", side_effect=lambda *a: rows[0])

        def slow_submit(*_a, **_k):
            time.sleep(0.2)  # the POST to the box
            rows[0] += 1  # save_pending_task
            return _queued_result()

        mocker.patch.object(plugin.llm_service, "video_generation", side_effect=slow_submit)

        results: list = []
        threads = [
            threading.Thread(target=lambda: results.append(self._submit(plugin, irc, msg)))
            for _ in range(2)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)

        assert rows[0] == 2, "the per-user cap of 2 was exceeded"
        assert sorted(bool(r.error) for r in results) == [False, True]

    def test_zero_disables_a_cap(self, plugin_env, mocker) -> None:
        """0 means off — both counts are well past the shipped defaults."""
        plugin, irc, msg = plugin_env
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"animateMaxPendingPerUser": 0, "animateMaxPending": 0}
        )
        mocker.patch.object(plugin, "_resolve_tier", return_value="registered")
        per_user = mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=99)
        count_all = mocker.patch.object(plugin.db, "count_pending_animate", return_value=99)
        submit = mocker.patch.object(
            plugin.llm_service, "video_generation", return_value=_queued_result()
        )

        result = self._submit(plugin, irc, msg)

        assert not result.error
        submit.assert_called_once()
        # Disabled means not even queried.
        per_user.assert_not_called()
        count_all.assert_not_called()

    def test_admitted_request_reaches_video_generation(self, plugin_env, mocker) -> None:
        """Under both caps, the request goes through untouched and typing wakes."""
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="registered")
        mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=1)
        mocker.patch.object(plugin.db, "count_pending_animate", return_value=5)
        submit = mocker.patch.object(
            plugin.llm_service, "video_generation", return_value=_queued_result()
        )
        plugin._render_typing_wake.clear()

        result = self._submit(plugin, irc, msg)

        assert result.queued
        assert submit.call_args.args[0] == "a corgi"
        assert plugin._render_typing_wake.is_set()

    def test_refusal_reaches_the_tool_as_a_failed_callback_result(self, plugin_env, mocker) -> None:
        """GIVEN a rejected submission WHEN the tool callback runs THEN ok=False, text kept.

        The refusal is the whole point of the cap, and the model has to see it
        to phrase it. A callback that reported ok=True would have the model
        announce a clip that was never queued.
        """
        plugin, irc, msg = plugin_env
        refusal = "Not queued: you already have 2 clips rendering (limit 2)."
        mocker.patch.object(
            plugin,
            "_submit_video",
            return_value=VideoResult(content=refusal, error=refusal),
        )

        result = plugin._animate_for_assistant(
            irc, msg, "a corgi", caller=Caller.from_msg(msg, "alice")
        )

        assert result.ok is False
        assert result.message == refusal


class TestPrivateMessageTarget:
    """A clip asked for in PM has to come back to the person who asked.

    ``msg.args[0]`` is the PRIVMSG recipient, which in a query is the BOT's own
    nick. Stashing that sends the finished clip to the bot minutes later: the
    job renders, @renders lists it, and the requester sees nothing at all.
    """

    def _pm(self, msg):
        msg.args = ("testbot", "@animate a corgi")
        msg.channel = None
        return msg

    def test_submit_stashes_the_requester_not_the_bot(self, plugin_env, mocker) -> None:
        plugin, irc, msg = plugin_env
        self._pm(msg)
        mocker.patch.object(plugin, "_resolve_tier", return_value="registered")
        mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=0)
        mocker.patch.object(plugin.db, "count_pending_animate", return_value=0)
        submit = mocker.patch.object(plugin.llm_service, "video_generation")

        plugin._submit_video(irc, msg, "a corgi", caller=Caller.from_msg(msg, None))

        assert submit.call_args.kwargs["reply_target"] == "testnick"
        assert submit.call_args.kwargs["is_channel"] is False

    def test_channel_target_is_untouched(self, plugin_env, mocker) -> None:
        plugin, irc, msg = plugin_env
        mocker.patch.object(plugin, "_resolve_tier", return_value="registered")
        mocker.patch.object(plugin.db, "count_pending_animate_for", return_value=0)
        mocker.patch.object(plugin.db, "count_pending_animate", return_value=0)
        submit = mocker.patch.object(plugin.llm_service, "video_generation")

        plugin._submit_video(irc, msg, "a corgi", caller=Caller.from_msg(msg, None))

        assert submit.call_args.kwargs["reply_target"] == "#test"
        assert submit.call_args.kwargs["is_channel"] is True

    def test_stash_context_helper_agrees(self, mocker) -> None:
        """Every other stashed task type (draw, code, ask) goes through here."""
        from llm.service import _msg_stash_context

        msg = mocker.MagicMock()
        msg.nick = "testnick"
        msg.args = ("testbot", "hello")
        msg.server_tags = {}

        nick, reply_target, is_channel, _account = _msg_stash_context(msg)

        assert (nick, reply_target, is_channel) == ("testnick", "testnick", False)


class TestQueuePositionAck:
    """The ack tells you where you stand, because nobody could see the queue.

    Depth is read before the row is stashed, so a clip is never counted as
    ahead of itself, and the estimate assumes every clip ahead takes the full
    render time — the box serialises, so that is the honest upper bound.
    """

    def _submit(self, service, mocker, *, ahead: int):
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "job-1"}))
        mocker.patch.object(service, "_stash_timeout", return_value=True)
        service.plugin.db.count_pending_animate.return_value = ahead
        return service.video_generation(
            "a corgi", nick="alice", reply_target="#chan", is_channel=True
        )

    def test_ack_reports_position_ahead_and_wait(self, make_service, animate_env, mocker) -> None:
        service, _ = _service(make_service)

        result = self._submit(service, mocker, ahead=2)

        assert not result.error
        assert result.queued
        assert "position 3" in result.content
        assert "2 ahead" in result.content
        assert "6m 45s" in result.content
        assert "http" not in result.content

    def test_first_in_queue_says_so(self, make_service, animate_env, mocker) -> None:
        service, _ = _service(make_service)

        result = self._submit(service, mocker, ahead=0)

        assert "position 1 (0 ahead)" in result.content
        assert "2m 15s" in result.content

    def test_depth_is_read_before_the_row_is_stashed(
        self, make_service, animate_env, mocker
    ) -> None:
        """Counting after the stash would make every clip one place worse."""
        service, _ = _service(make_service)
        calls: list[str] = []
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "job-1"}))
        mocker.patch.object(
            service, "_stash_timeout", side_effect=lambda **_kw: calls.append("stash") or True
        )
        service.plugin.db.count_pending_animate.side_effect = lambda _now: (
            calls.append("count") or 1
        )

        service.video_generation("a corgi", nick="alice", reply_target="#chan", is_channel=True)

        assert calls == ["count", "stash"]

    def test_render_seconds_comes_from_the_registry(
        self, make_service, animate_env, mocker
    ) -> None:
        service, _ = _service(make_service, animateRenderSeconds=300)

        result = self._submit(service, mocker, ahead=1)

        assert "10m" in result.content

    def test_a_database_blip_still_queues_the_clip(self, make_service, animate_env, mocker) -> None:
        """The count is decoration; losing it must not fail a live submission."""
        service, _ = _service(make_service)
        mocker.patch.object(service, "_animate_request", return_value=(200, {"id": "job-1"}))
        mocker.patch.object(service, "_stash_timeout", return_value=True)
        service.plugin.db.count_pending_animate.side_effect = RuntimeError("database is locked")

        result = service.video_generation(
            "a corgi", nick="alice", reply_target="#chan", is_channel=True
        )

        assert result.queued
        assert not result.error
        assert "position 1 (0 ahead)" in result.content
