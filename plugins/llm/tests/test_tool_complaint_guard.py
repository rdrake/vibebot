"""Tests for the tool-complaint history guard.

The image-failure guard (``_is_image_failure``) fixed one sentence. It is keyed
on an image noun sitting next to a failure verb, and the model promptly drifted
out of that vocabulary while keeping the behaviour: over five hours in
#afternet on 2026-08-01 the same self-imitation loop restated itself as

    20:07  Image generation failed.            <- the only line the sibling catches
    20:36  The image tool's broken.
    21:59  Tool spat back nothing but silence this time.
    22:13  Tool refused. 420.
    22:17  Tool still giving 420.
    22:29  Tool's still choking on the request.

By the end the bot answered "vibebot give Jordan some bacon" -- a request that
needs no tool at all -- with a tool complaint at ``tool_calls=0``. Chasing the
wording is an arms race, so this guard is keyed on the SHAPE instead: a short
reply, leading with a complaint about the machinery, is a failure report. It is
stripped from history every turn (the report is only ever true of the turn that
produced it), and retried when the invocation never called a tool at all --
because then there is provably nothing for it to be reporting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.service import (
    LLMService,
    _is_tool_complaint,
    _strip_tool_complaints,
)

from .conftest import make_completion_response

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

# Every bot line from the #afternet spiral, in the order it was emitted.
OBSERVED_COMPLAINTS = [
    "Image gen fucked up.",
    "The image tool's broken. All requests return the same broken link.",
    "The tool's fucked. Keep spitting the same broken link no matter what I ask for.",
    "Tool spat back nothing but silence this time.",
    "Tool refused. 420.",
    "Then quit giving me tasks the tool rejects.",
    "Tool still giving 420.",
    "Tool's still choking on the request.",
]
# The line that ended the spiral, answering a request that needed no tool.
OBSERVED_FINAL = "Tool's still choking on the request."
MINTED_URL = "https://paste.boxlabs.uk/img/img_abc123.png"


class TestIsToolComplaint:
    """Detection is keyed on shape, not on a vocabulary that keeps moving."""

    @pytest.mark.parametrize("line", OBSERVED_COMPLAINTS)
    def test_flags_every_observed_line(self, line: str) -> None:
        """All eight phrasings of the same failure report are caught."""
        assert _is_tool_complaint(line)

    def test_keeps_reply_that_delivered_an_image(self) -> None:
        """A turn carrying an image URL delivered, however it is worded."""
        assert not _is_tool_complaint(f"Tool choked the first time, here you go: {MINTED_URL}")

    def test_keeps_ordinary_prose(self) -> None:
        """A reply with no complaint in it is not a failure report."""
        assert not _is_tool_complaint("The lads did it again, bacon everywhere.")

    def test_keeps_honest_capability_answer(self) -> None:
        """ "I don't have that" is an honest gap, not a machinery complaint.

        Same rationale as the safety guard's narrowness: retrying an admitted
        gap pushes the model toward inventing an answer instead, which is the
        worse failure.
        """
        assert not _is_tool_complaint("I don't have that information.")

    @pytest.mark.parametrize(
        "line",
        [
            # Honest gap that names a tool as its SUBJECT ("unable" + "tool").
            "Unable to get real-time results for Nefarious 2 — it's a torrent "
            "indexer tool, last I checked v2.5.",
            # Substantive answer; "dead easy" is not a dead tool.
            "Aye, dead easy – pipe Gemini's draft insult into Grok's prompt for "
            "that final tweak, via API chaining.",
            # Substantive answer; "Nothing specific" is not an empty tool result.
            "Nothing specific — no standard nodeTML exists. Node.js handles HTML "
            "via tools like node-html-parser.",
        ],
    )
    def test_keeps_real_answers_that_name_a_tool(self, line: str) -> None:
        """Regression: real #afternet lines an earlier draft wrongly flagged.

        Each is a genuine answer that happens to pair a tool noun with a word
        that reads as failure out of context. They are why "dead", "nothing"
        and "unable" are not in ``_TOOL_FAILURE_RE``.
        """
        assert not _is_tool_complaint(line)

    def test_keeps_substantive_prose_about_a_broken_third_party(self) -> None:
        """A real answer that discusses a broken API is not a canned report.

        The word cap is what separates them: a failure report is a one-liner,
        an answer is not.
        """
        content = (
            "Twitter's API has been broken for third-party clients since Musk "
            "killed free access in 2023, which is why every decent client died "
            "off that year. The paid tiers start at a hundred dollars a month "
            "and the free one is write-only, so hobby projects simply stopped "
            "being viable and the ecosystem never recovered from it."
        )
        assert not _is_tool_complaint(content)

    def test_ignores_a_late_mention(self) -> None:
        """Failure reports lead with the failure; a late mention is prose."""
        assert not _is_tool_complaint("x" * 200 + " the tool is broken")

    def test_empty_content_is_not_a_complaint(self) -> None:
        """Empty content is never flagged, matching the sibling predicates."""
        assert not _is_tool_complaint("")


class TestStripToolComplaints:
    """History de-poisoning: the report must not outlive its own turn."""

    def test_drops_the_spiral_and_keeps_the_premises(self) -> None:
        """The whole run of complaints goes; the user's turns anchor the thread."""
        history: list[dict[str, str]] = [{"role": "user", "content": "draw a tit"}]
        for line in OBSERVED_COMPLAINTS:
            history.append({"role": "assistant", "content": line})
        history.append({"role": "user", "content": "vibebot give Jordan some bacon"})

        assert _strip_tool_complaints(history) == [
            {"role": "user", "content": "draw a tit"},
            {"role": "user", "content": "vibebot give Jordan some bacon"},
        ]

    def test_keeps_user_turns_that_look_like_reports(self) -> None:
        """A user complaining about the tool is a premise, not the bot's output."""
        history = [{"role": "user", "content": "your tool is fucked, fix it"}]
        assert _strip_tool_complaints(history) == history

    def test_keeps_clean_assistant_turns(self) -> None:
        """Unrelated replies still anchor the thread."""
        history = [{"role": "assistant", "content": "The lads did it again."}]
        assert _strip_tool_complaints(history) == history

    def test_none_history_passes_through(self) -> None:
        """None is returned unchanged, matching the sibling strips."""
        assert _strip_tool_complaints(None) is None


class TestGuardIsWiredIntoTheChatPath:
    """The strip must actually run: a correct predicate nobody calls is inert."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
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
            return make_completion_response("Here's your bacon, Jordan.")

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        service.assistant_completion(
            prompt="give Jordan some bacon",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            **history_kwargs,  # type: ignore[arg-type]
        )
        assert captured, "no completion was issued"
        return captured[0]

    def test_complaint_is_stripped_from_personal_history(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The poisoned turn never reaches the model on the personal thread."""
        messages = self._messages_for(
            service,
            mocker,
            history=[
                {"role": "user", "content": "draw a tit"},
                {"role": "assistant", "content": OBSERVED_FINAL},
            ],
        )
        assert not any(OBSERVED_FINAL in str(m.get("content", "")) for m in messages)
        assert any("draw a tit" in str(m.get("content", "")) for m in messages)

    def test_complaint_is_stripped_from_channel_history(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The shared channel window carries the bot's own lines too."""
        messages = self._messages_for(
            service,
            mocker,
            channel_history=[
                {"role": "user", "content": "draw a tit"},
                {"role": "assistant", "content": OBSERVED_FINAL},
            ],
        )
        assert not any(OBSERVED_FINAL in str(m.get("content", "")) for m in messages)


class TestRetryWhenNoToolRan:
    """A complaint on a turn that never called a tool is provably invented."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def test_invented_complaint_is_nudged_and_retried(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The #afternet endgame: complaint with no tool call, so retry it."""
        replies = iter([OBSERVED_FINAL, "Here's your bacon, Jordan."])
        seen: list[list[dict]] = []

        def fake_completion(**kwargs: object) -> object:
            seen.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return make_completion_response(next(replies))

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="give Jordan some bacon",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert len(seen) == 2, "the invented complaint was not retried"
        assert "Here's your bacon, Jordan." in str(result)
        assert OBSERVED_FINAL not in str(result)


class TestHonestReportAfterARealFailure:
    """Once a tool has run, a report that it failed may well be true."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def test_a_failed_search_reported_honestly_is_delivered(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        from llm.assistant import ToolResult

        from .conftest import make_tool_call

        report = "The search tool failed, try again in a minute."
        assert _is_tool_complaint(report)
        replies = iter(
            [
                make_completion_response(
                    None, tool_calls=[make_tool_call("search_web", {"query": "bacon"})]
                ),
                make_completion_response(report),
            ]
        )
        seen: list[object] = []

        def fake_completion(**kwargs: object) -> object:
            seen.append(kwargs)
            return next(replies)

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="search for bacon news",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            search_fn=lambda _q: ToolResult(content='{"error": "upstream 503"}'),
        )

        assert len(seen) == 2, "an honest report was retried as if invented"
        assert result.content == report
