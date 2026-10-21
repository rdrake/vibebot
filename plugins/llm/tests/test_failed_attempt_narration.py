"""Tests for the failed-attempt narration guard.

#afternet on 2026-08-15. ``vibebot draw bunga bunga party`` came back with::

    Second image ready, first failed.
    https://paste.boxlabs.uk/img/img_6a81079496b1b.jpg
    First image failed. Second one ready.

One picture was asked for and one was delivered. The model issued two
generate_image calls in the same assistant message, the provider refused one,
and the retry it performed on its own behalf became the headline — twice, in a
reply short enough that the failure was the only thing visible on the channel
line.

Two fixes, at two altitudes:

* the generate_image short-circuit now fires when EVERY call in the step was a
  draw and AT LEAST ONE delivered, so the narrating step never runs (and the
  turn saves a completion); and
* :func:`_strip_failed_attempt_narration` catches the residue — a step that
  mixed a draw with another tool, where the post-tool text is still needed.

The reply also has to stop poisoning the next turn. ``_is_image_failure``
spares anything carrying an image URL, by design, so this line survived into
history and seeded the same narration on the following draw. Delivering the URL
alone removes the exemplar as well as the noise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.assistant import ToolCallbackResult, ToolResult
from llm.service import LLMService, _strip_failed_attempt_narration

from .conftest import make_completion_response, make_tool_call

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

MINTED = "https://paste.boxlabs.uk/img/img_6a81079496b1b.jpg"
MINTED_2 = "https://paste.boxlabs.uk/img/img_6a6e571adf508.jpg"
# The reply as it actually shipped, newlines and repetition included.
OBSERVED = f"Second image ready, first failed.\n{MINTED}\nFirst image failed. Second one ready."


class TestStripDetection:
    """What counts as narrating a failed attempt, and what does not."""

    def test_strips_the_observed_reply_to_the_image(self) -> None:
        """The #afternet line reduces to the picture the user asked for."""
        assert _strip_failed_attempt_narration(OBSERVED, {MINTED}, True) == MINTED

    def test_keeps_every_delivered_image(self) -> None:
        """A partial failure across three draws still ships both successes."""
        content = f"Two ready, the third attempt failed. {MINTED} {MINTED_2}"
        assert _strip_failed_attempt_narration(content, {MINTED, MINTED_2}, True) == (
            f"{MINTED} {MINTED_2}"
        )

    def test_paraphrase_does_not_escape_it(self) -> None:
        """Wording drifts; the evidence gate is what the guard stands on.

        The tool-complaint guard's history is the argument: vocabulary-keyed
        detection lost to paraphrase within hours. None of these say "image"
        or "failed".
        """
        for line in (
            "One got blocked, the other one's up.",
            "First draw was a dud.",
            "The other one came back censored.",
            "Couldn't get the first one through.",
        ):
            assert _strip_failed_attempt_narration(f"{line}\n{MINTED}", {MINTED}, True) == MINTED

    def test_plain_delivery_is_untouched(self) -> None:
        """A clean draw keeps whatever sentence the model wrote."""
        content = f"Here's your bunga bunga party. {MINTED}"
        assert _strip_failed_attempt_narration(content, {MINTED}, True) == content

    def test_inert_when_no_draw_actually_failed(self) -> None:
        """Without the evidence the guard does nothing, whatever the words.

        Ordinary prose about a picture of something going wrong reads exactly
        like bookkeeping. Only the tool loop can tell them apart, so only the
        tool loop gets to arm this.
        """
        content = f"Here's your drawing of a failed rocket launch.\n{MINTED}"
        assert _strip_failed_attempt_narration(content, {MINTED}, False) == content

    def test_failure_report_with_nothing_delivered_is_untouched(self) -> None:
        """When the turn produced no image, the honest failure must survive.

        This is the case the guard must never touch: there is no picture to
        fall back on, so replacing the text would leave the user with silence.
        """
        content = "First attempt failed and so did the retry."
        assert _strip_failed_attempt_narration(content, set(), True) == content

    def test_unminted_url_is_not_treated_as_delivery(self) -> None:
        """A URL this turn did not mint cannot rescue a failure report.

        The fabricated-image guard owns that case; overlapping with it here
        would let a stale link out under a rewritten sentence.
        """
        stale = "https://paste.boxlabs.uk/img/img_deadbeef1234.jpg"
        content = f"First one failed, here's the earlier one. {stale}"
        assert _strip_failed_attempt_narration(content, {MINTED}, True) == content

    def test_empty_content_is_safe(self) -> None:
        assert _strip_failed_attempt_narration("", {MINTED}, True) == ""


class TestParallelDrawShortCircuit:
    """The narrating step must not run at all when a draw delivered."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(
            assistantModel="xai/grok-4.3", httpUrlBase="https://irc.rdrake.org/llm"
        )
        return svc

    def _run(
        self,
        service: LLMService,
        mocker: MockerFixture,
        responses: list[object],
        draw_results: list[ToolCallbackResult],
    ) -> tuple[object, list[dict]]:
        calls: list[dict] = []

        def fake_completion(**kwargs: object) -> object:
            calls.append(kwargs)  # type: ignore[arg-type]
            return responses[len(calls) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        pending = list(draw_results)

        def draw_fn(_prompt: str, **_kw: object) -> ToolCallbackResult:
            return pending.pop(0)

        result = service.assistant_completion(
            prompt="draw bunga bunga party",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=draw_fn,
        )
        return result, calls

    def test_partial_failure_delivers_the_image_and_nothing_else(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The reported bug: two parallel draws, one refused, one delivered."""
        responses = [
            make_completion_response(
                None,
                tool_calls=[
                    make_tool_call("generate_image", {"prompt": "bunga bunga party"}, call_id="a"),
                    make_tool_call("generate_image", {"prompt": "bunga bunga party"}, call_id="b"),
                ],
            ),
            # Would have produced "Second image ready, first failed." — the
            # short-circuit must make this call unreachable.
            make_completion_response("Second image ready, first failed."),
        ]
        result, calls = self._run(
            service,
            mocker,
            responses,
            [
                ToolCallbackResult(False, "Image generation blocked by safety filters."),
                ToolCallbackResult(True, MINTED),
            ],
        )

        assert result.content == MINTED
        assert "fail" not in (result.content or "").lower()
        assert len(calls) == 1

    def test_both_draws_delivering_returns_both_urls(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """A deliberate two-image turn still short-circuits, with both URLs."""
        responses = [
            make_completion_response(
                None,
                tool_calls=[
                    make_tool_call("generate_image", {"prompt": "one"}, call_id="a"),
                    make_tool_call("generate_image", {"prompt": "two"}, call_id="b"),
                ],
            ),
        ]
        result, calls = self._run(
            service,
            mocker,
            responses,
            [ToolCallbackResult(True, MINTED), ToolCallbackResult(True, MINTED_2)],
        )

        assert result.content == f"{MINTED} {MINTED_2}"
        assert len(calls) == 1

    def test_total_failure_still_reports_honestly(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """No image to show means no short-circuit and no suppression.

        The guard hides bookkeeping, not outages: when every draw failed the
        model's own report is the only thing the user can act on.
        """
        responses = [
            make_completion_response(
                None,
                tool_calls=[
                    make_tool_call("generate_image", {"prompt": "one"}, call_id="a"),
                    make_tool_call("generate_image", {"prompt": "two"}, call_id="b"),
                ],
            ),
            make_completion_response("Both draws were refused by the provider."),
        ]
        result, calls = self._run(
            service,
            mocker,
            responses,
            [
                ToolCallbackResult(False, "blocked"),
                ToolCallbackResult(False, "blocked"),
            ],
        )

        assert result.content == "Both draws were refused by the provider."
        assert len(calls) == 2

    def test_mixed_tool_step_falls_through_to_the_strip(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The residue case: a draw alongside another tool still needs text.

        The short-circuit cannot fire here — the search result has to be
        summarised — so the narration reaches the reply and the guard is what
        removes it.
        """
        responses = [
            make_completion_response(
                None,
                tool_calls=[
                    make_tool_call("search_web", {"query": "bunga bunga"}, call_id="a"),
                    make_tool_call("generate_image", {"prompt": "one"}, call_id="b"),
                    make_tool_call("generate_image", {"prompt": "two"}, call_id="c"),
                ],
            ),
            make_completion_response(f"Second image ready, first failed. {MINTED}"),
        ]
        calls: list[dict] = []

        def fake_completion(**kwargs: object) -> object:
            calls.append(kwargs)  # type: ignore[arg-type]
            return responses[len(calls) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        pending = [
            ToolCallbackResult(False, "Image generation blocked by safety filters."),
            ToolCallbackResult(True, MINTED),
        ]

        result = service.assistant_completion(
            prompt="what is bunga bunga, and draw it",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=lambda _p, **_kw: pending.pop(0),
            search_fn=lambda _q: ToolResult(content='{"results": "a party"}'),
        )

        assert result.content == MINTED
        assert len(calls) == 2
