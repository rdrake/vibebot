"""Tests for the image-failure history guard.

When image generation errors -- the provider's content filter, or a transient
API fault -- the model reports it in a short line. That report is correct for
the turn that produced it, but leaving it in history is what turns one bad
prompt into an unbroken run of refusals: a non-reasoning model reproduces the
sentence verbatim on the next draw request and never calls the tool at all.

Observed in #afternet on 2026-08-01. "draw a tit" was content-moderated by xAI
(a legitimate refusal). The very next message, "draw a cat", came back with the
identical sentence at ``tool_calls=0`` -- the image API was never contacted, so
image generation looked broken when only the first prompt ever was.

None of the sibling guards catch it, each for its own structural reason, so the
regression tests below pin that gap as much as they pin the new guard:
``_is_safety_refusal`` deliberately excludes image refusals,
``_strip_repeated_replies`` never judges a reply under its distinct-word floor,
and ``_strip_degraded`` needs a long passage.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.service import (
    LLMService,
    _is_image_failure,
    _is_safety_refusal,
    _strip_image_failures,
    _strip_repeated_replies,
)

from .conftest import make_completion_response

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# The exact line grok emitted in #afternet, and the one every sibling guard
# lets through.
OBSERVED_FAILURE = "Image generation failed."
# Shape that _IMAGE_URL_RE recognises -- a URL the mint path actually produces.
MINTED_URL = "https://paste.boxlabs.uk/img/img_abc123.png"


class TestIsImageFailure:
    """Detection: failure reports in, delivered images and prose out."""

    def test_flags_the_observed_line(self) -> None:
        """The exact sentence that poisoned #afternet is flagged."""
        assert _is_image_failure(OBSERVED_FAILURE)

    def test_flags_moderation_wording(self) -> None:
        """A provider-filter report is a failure report."""
        assert _is_image_failure("The image was rejected by content moderation.")

    def test_flags_inverted_wording(self) -> None:
        """The verb may lead the noun: "failed to generate the image"."""
        assert _is_image_failure("Failed to generate the image, sorry.")

    def test_flags_picture_synonym(self) -> None:
        """Wording varies across turns; the noun set covers the synonyms."""
        assert _is_image_failure("Couldn't render that picture.")

    def test_keeps_reply_that_delivered_an_image(self) -> None:
        """A turn carrying an image URL delivered, however it is worded.

        This is the guard's safety catch: a successful reply that happens to
        mention a failure ("the first attempt failed, here's the second") must
        stay in history, or the thread loses the image it actually produced.
        """
        assert not _is_image_failure(f"First attempt failed, here you go: {MINTED_URL}")

    def test_keeps_ordinary_prose_mentioning_a_picture(self) -> None:
        """A reply about a picture is not a report that one failed."""
        assert not _is_image_failure("That picture of the lads is still the best one.")

    def test_keeps_unrelated_failure_report(self) -> None:
        """Non-image failures belong to other guards, not this one."""
        assert not _is_image_failure("The search API failed, try again in a minute.")

    def test_ignores_a_late_mention(self) -> None:
        """Failure reports lead with the failure; a late mention is prose."""
        content = "x" * 200 + " the image generation failed"
        assert not _is_image_failure(content)

    def test_empty_content_is_not_a_failure(self) -> None:
        """Empty content is never flagged, matching the sibling predicates."""
        assert not _is_image_failure("")


class TestSiblingGuardsMissIt:
    """Why a fourth guard was needed: each sibling is blind by construction."""

    def test_safety_refusal_guard_excludes_it(self) -> None:
        """Image refusals are excluded from the safety guard on purpose."""
        assert not _is_safety_refusal(OBSERVED_FAILURE)

    def test_repeat_guard_only_sees_it_once_it_repeats(self) -> None:
        """The repeat guard needs two copies; this one poisons from the first.

        It used to miss the pair as well, because a five-word floor refused to
        judge a three-word report — which is why this guard had to exist at
        all. The floor is gone, so a duplicated failure is now caught twice
        over. A single failure still is not: one occurrence is not a repeat,
        and one occurrence is all it takes to teach the model that image
        generation is broken.
        """
        pair = [
            {"role": "assistant", "content": OBSERVED_FAILURE},
            {"role": "assistant", "content": OBSERVED_FAILURE},
        ]
        assert _strip_repeated_replies(pair) == []

        lone = [{"role": "assistant", "content": OBSERVED_FAILURE}]
        assert _strip_repeated_replies(lone) == lone


class TestStripImageFailures:
    """History de-poisoning: the report must not outlive its own turn."""

    def test_drops_the_report_so_the_next_draw_calls_the_tool(self) -> None:
        """The #afternet sequence: the poisoned turn is gone, premises remain."""
        history = [
            {"role": "user", "content": "draw a tit"},
            {"role": "assistant", "content": OBSERVED_FAILURE},
            {"role": "user", "content": "draw a cat"},
        ]
        assert _strip_image_failures(history) == [
            {"role": "user", "content": "draw a tit"},
            {"role": "user", "content": "draw a cat"},
        ]

    def test_keeps_user_turns_that_look_like_reports(self) -> None:
        """A user complaining about a failure is a premise, not the bot's output."""
        history = [{"role": "user", "content": "image generation failed again, fix it"}]
        assert _strip_image_failures(history) == history

    def test_keeps_the_turn_that_delivered_an_image(self) -> None:
        """A successful image turn survives even when it mentions a retry."""
        history = [{"role": "assistant", "content": f"Second try worked: {MINTED_URL}"}]
        assert _strip_image_failures(history) == history

    def test_keeps_clean_assistant_turns(self) -> None:
        """Unrelated replies still anchor the thread."""
        history = [{"role": "assistant", "content": "The lads did it again."}]
        assert _strip_image_failures(history) == history

    def test_none_history_passes_through(self) -> None:
        """None is returned unchanged, matching the sibling strips."""
        assert _strip_image_failures(None) is None


class TestGuardIsWiredIntoTheChatPath:
    """The strip must actually run: a correct predicate nobody calls is inert."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="xai/grok-4.3")
        return svc

    def _messages_for(
        self,
        service: LLMService,
        mocker: MockerFixture,
        **history_kwargs: object,
    ) -> list[dict]:
        """Run one completion and return the messages the model actually saw."""
        captured: list[list[dict]] = []

        def fake_completion(**kwargs: object) -> object:
            captured.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return make_completion_response("Here's your cat.")

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        service.assistant_completion(
            prompt="draw a cat",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            **history_kwargs,  # type: ignore[arg-type]
        )
        assert captured, "no completion was issued"
        return captured[0]

    def test_report_is_stripped_from_personal_history(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The poisoned turn never reaches the model on the personal thread."""
        messages = self._messages_for(
            service,
            mocker,
            history=[
                {"role": "user", "content": "draw a tit"},
                {"role": "assistant", "content": OBSERVED_FAILURE},
            ],
        )
        assert not any(OBSERVED_FAILURE in str(m.get("content", "")) for m in messages)
        assert any("draw a tit" in str(m.get("content", "")) for m in messages)

    def test_report_is_stripped_from_channel_history(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The shared channel window carries the bot's own lines too."""
        messages = self._messages_for(
            service,
            mocker,
            channel_history=[
                {"role": "user", "content": "draw a tit"},
                {"role": "assistant", "content": OBSERVED_FAILURE},
            ],
        )
        assert not any(OBSERVED_FAILURE in str(m.get("content", "")) for m in messages)
