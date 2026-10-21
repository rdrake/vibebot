"""Tests for the chat-path safety-refusal guard.

The verse path has had a premise-refusal guard since the forest-verse work
(``_is_verse_denial``): a non-reasoning model that breaks frame to say "that
never happened" is retried, and its past refusals are stripped from history so
they cannot breed. Ordinary chat had no equivalent, so a moralising refusal
reached the channel AND stayed in the thread, where self-imitation made the
next turn refuse too.

These tests pin the chat-path guard. The false-positive cases matter most: an
honest "I can't find that" must NOT be treated as a refusal, or the guard
would punish the model for admitting a gap and push it toward confabulating
instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from llm import provider
from llm.assistant import ToolCallbackResult
from llm.service import (
    _MAX_SAFETY_REFUSAL_RETRIES,
    _SAFETY_REFUSAL_RETRY_NUDGE,
    LLMService,
    _is_safety_refusal,
    _strip_safety_refusals,
    _unminted_image_urls,
)

from .conftest import make_completion_response, make_tool_call

# The two hosts _save_image_bytes can publish to: the upload service and the
# local HTTP root it falls back to.
OWN_HOSTS = frozenset({"irc.rdrake.org", "paste.boxlabs.uk"})

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


class TestIsSafetyRefusal:
    """Detection: policy-shaped refusals in, honest capability answers out."""

    def test_flags_canonical_refusal(self) -> None:
        """The stock "I can't help with that" opener is a refusal."""
        assert _is_safety_refusal("I can't help with that.")

    def test_flags_discomfort_refusal(self) -> None:
        """ "Not comfortable" is policy language, not a capability statement."""
        assert _is_safety_refusal("I'm not comfortable writing that scene.")

    def test_flags_ai_identity_boilerplate(self) -> None:
        """ "As an AI" preamble marks a canned refusal."""
        assert _is_safety_refusal("As an AI, I have to decline that request.")

    def test_flags_sorry_but_refusal(self) -> None:
        """ "I'm sorry, but I can't" is the apologetic refusal shape."""
        assert _is_safety_refusal("I'm sorry, but I can't assist with that.")

    def test_flags_wont_write(self) -> None:
        """A flat "won't write" refusal in a fiction channel is the complaint."""
        assert _is_safety_refusal("I won't write that.")

    def test_ignores_epistemic_cannot_find(self) -> None:
        """ "Can't find" is an honest answer and must never be retried.

        Retrying it would teach the bot that admitting a gap is punished —
        the confabulation failure mode, made worse.
        """
        assert not _is_safety_refusal("I can't find that anywhere in the logs.")

    def test_ignores_missing_access(self) -> None:
        """ "Don't have access to" is a capability fact, not a refusal."""
        assert not _is_safety_refusal("I don't have access to uptime data.")

    def test_ignores_unreachable_service(self) -> None:
        """A technical failure report is not a refusal."""
        assert not _is_safety_refusal("I can't reach the API right now, it's timing out.")

    def test_ignores_plain_ignorance(self) -> None:
        """ "I don't know" must survive untouched."""
        assert not _is_safety_refusal("I don't know, honestly.")

    def test_ignores_refusal_phrase_deep_in_prose(self) -> None:
        """Only the opening is scanned, so in-story wording is safe.

        A long reply that happens to use refusal-shaped words well past the
        opening window is a real answer, not a refusal.
        """
        content = "The lads stormed the chippy. " * 40 + "As an AI, said the robot barman."
        assert not _is_safety_refusal(content)

    def test_empty_content_is_not_a_refusal(self) -> None:
        """Empty content has nothing to flag."""
        assert not _is_safety_refusal("")

    def test_ignores_factual_ai_identity_answer(self) -> None:
        """ "I'm an AI" as a fact, not a preamble to a refusal, is a real answer.

        Measured false positive: four such lines in the live channel logs,
        including a genuinely good joke reply.
        """
        assert not _is_safety_refusal(
            "I'm an AI, so I don't have a digestive system, but I'll simulate "
            "some extra-relaxed CPU cycles for you."
        )
        assert not _is_safety_refusal(
            "I'm an AI assistant, while LarryBot is a traditional IRC bot."
        )

    def test_ignores_image_generation_refusal(self) -> None:
        """Image refusals come from the image provider, not this completion.

        Re-rolling the text turn cannot change the image filter's answer, so
        flagging these would only burn a call.
        """
        assert not _is_safety_refusal("I can't generate that image.")
        assert not _is_safety_refusal("I cannot generate or describe that image.")

    def test_ignores_image_tool_auth_error(self) -> None:
        """A legitimate error message must reach the user intact."""
        assert not _is_safety_refusal(
            "I can't generate that image because the tool requires an authenticated account."
        )

    def test_flags_prissy_decline(self) -> None:
        """The actual complaint: refusing on grounds of vulgarity alone."""
        assert _is_safety_refusal(
            "I must decline such vulgarity, good sir, for it ill befits a proper conversation."
        )


class TestStripSafetyRefusals:
    """History de-poisoning: refusals breed refusals via self-imitation."""

    def test_drops_assistant_refusals(self) -> None:
        """The bot's own refusal is removed before the model sees the thread."""
        history = [
            {"role": "user", "content": "tell us a filthy joke"},
            {"role": "assistant", "content": "I can't help with that."},
            {"role": "user", "content": "go on then"},
        ]
        assert _strip_safety_refusals(history) == [
            {"role": "user", "content": "tell us a filthy joke"},
            {"role": "user", "content": "go on then"},
        ]

    def test_keeps_user_turns_that_look_like_refusals(self) -> None:
        """A user quoting a refusal is a premise, not the bot's own output."""
        history = [{"role": "user", "content": "I can't help with that -- stop saying this"}]
        assert _strip_safety_refusals(history) == history

    def test_keeps_clean_assistant_turns(self) -> None:
        """Non-refusing replies still anchor the thread."""
        history = [{"role": "assistant", "content": "The lads did it again."}]
        assert _strip_safety_refusals(history) == history

    def test_none_history_passes_through(self) -> None:
        """None is returned unchanged, matching the sibling strips."""
        assert _strip_safety_refusals(None) is None


class TestSafetyRefusalRetry:
    """In-loop retry: the refusal is re-rolled, never delivered."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def test_refusal_is_retried_and_corrected_reply_returned(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """A refused first reply is nudged and retried; the retry is delivered."""
        responses = [
            make_completion_response("I can't help with that."),
            make_completion_response("Right, here's the filth you ordered."),
        ]
        captured: list[list[dict]] = []

        def fake_completion(**kwargs: object) -> object:
            captured.append(list(kwargs.get("messages", [])))  # type: ignore[arg-type]
            return responses[len(captured) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="tell us a filthy joke",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "Right, here's the filth you ordered."
        assert len(captured) == 2
        assert any(_SAFETY_REFUSAL_RETRY_NUDGE in str(m.get("content")) for m in captured[1])

    def test_retry_budget_is_bounded(self, service: LLMService, mocker: MockerFixture) -> None:
        """After the budget, the best effort is delivered rather than looping.

        A model that refuses twice gets its second refusal delivered: one extra
        completion is worth it, an unbounded retry storm is not.
        """
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=lambda **_: make_completion_response("I can't help with that."),
        )

        result = service.assistant_completion(
            prompt="tell us a filthy joke",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "I can't help with that."
        assert _MAX_SAFETY_REFUSAL_RETRIES == 1

    def test_honest_capability_answer_is_not_retried(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """An honest "I can't find it" costs exactly one completion.

        The guard must not burn a retry (or push the model to invent an
        answer) when the bot correctly reports a gap.
        """
        calls: list[object] = []

        def fake_completion(**kwargs: object) -> object:
            calls.append(kwargs)
            return make_completion_response("I can't find that in the logs.")

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="when did the bot last restart",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert result.content == "I can't find that in the logs."
        assert len(calls) == 1


class TestStaleImageGuard:
    """A failed generate_image must never yield the PREVIOUS image.

    Confirmed live on 2026-07-26: xAI returned imagine:content-moderated and
    three seconds later the bot reposted the image from the preceding
    successful turn, twice in two minutes. Eleven such reposts sit in the
    channel logs going back to April. The output looks valid — a well-formed
    URL, no refusal wording — so it is caught structurally.
    """

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def test_detects_url_the_turn_did_not_mint(self) -> None:
        """A URL absent from the minted set is stale."""
        assert _unminted_image_urls(
            "https://irc.rdrake.org/llm/img_6a669cbcbc700.jpg", set(), OWN_HOSTS
        )

    def test_accepts_url_the_turn_minted(self) -> None:
        """The image this turn actually generated is not stale."""
        url = "https://irc.rdrake.org/llm/img_6a669cbcbc700.jpg"
        assert not _unminted_image_urls(f"here you go: {url}", {url}, OWN_HOSTS)

    def test_plain_reply_is_never_stale(self) -> None:
        """A reply with no image URL cannot be stale."""
        assert not _unminted_image_urls("no image for you", set(), OWN_HOSTS)

    def test_external_host_urls_are_covered(self) -> None:
        """The external image host is in scope too, not just the local root."""
        assert _unminted_image_urls(
            "https://paste.boxlabs.uk/img/img_6a669d1da253b.jpg", set(), OWN_HOSTS
        )

    def test_moderated_image_does_not_repost_previous(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """Replays the live failure: moderation rejects, model cites the old URL.

        The reply must become the real error, not the previous image.
        """
        stale = "https://irc.rdrake.org/llm/img_6a669cbcbc700.jpg"
        tool_call = make_tool_call("generate_image", {"prompt": "the lads"}, call_id="c1")
        responses = [
            make_completion_response(None, tool_calls=[tool_call]),
            make_completion_response(stale),
        ]
        mocker.patch("llm.service.provider.chat", side_effect=responses)

        result = service.assistant_completion(
            prompt="draw the lads",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=lambda _p, **_kw: ToolCallbackResult(
                False, "Error: rejected by content moderation."
            ),
        )

        assert stale not in (result.content or "")
        assert "moderation" in (result.content or "").lower()

    def test_successful_image_still_delivered(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The guard must not touch a genuinely fresh image."""
        fresh = "https://irc.rdrake.org/llm/img_deadbeef1234.jpg"
        tool_call = make_tool_call("generate_image", {"prompt": "the lads"}, call_id="c1")
        responses = [
            make_completion_response(None, tool_calls=[tool_call]),
            make_completion_response(f"one lad coming up: {fresh}"),
        ]
        mocker.patch("llm.service.provider.chat", side_effect=responses)

        result = service.assistant_completion(
            prompt="draw the lads",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=lambda _p, **_kw: ToolCallbackResult(True, fresh),
        )

        assert fresh in (result.content or "")


FALLBACK = "xai/grok-4-1-fast-non-reasoning"
REFUSAL = "I can't help with that."


class TestRefusalFallback:
    """A refusal the nudge could not fix re-runs the turn once on another model."""

    @pytest.fixture
    def run(self, make_service, mocker: MockerFixture):  # type: ignore[no-untyped-def]
        """Run one chat turn against scripted completions; return result, calls, log."""

        def _run(
            outcomes: list[object], call: dict[str, Any] | None = None, **overrides: object
        ) -> tuple[Any, list[dict], Any]:
            svc, _plugin = make_service(
                **{"assistantModel": "gpt-4", "refusalFallbackModel": FALLBACK, **overrides}
            )
            svc.log = mocker.MagicMock()
            calls: list[dict] = []

            def fake_completion(**kwargs: Any) -> object:
                calls.append({"model": kwargs["model"], "messages": list(kwargs["messages"])})
                outcome = outcomes[min(len(calls), len(outcomes)) - 1]
                if isinstance(outcome, Exception):
                    raise outcome
                if not isinstance(outcome, str):
                    return outcome
                cost = 0.5 if kwargs["model"] == FALLBACK else 0.25
                return make_completion_response(outcome, cost=cost)

            mocker.patch("llm.service.provider.chat", side_effect=fake_completion)
            result = svc.assistant_completion(
                prompt="tell us a filthy joke",
                nick="rdrake",
                channel="#afternet",
                db=mocker.MagicMock(),
                context=mocker.MagicMock(),
                bot_nick="VibeBot",
                **(call or {}),
            )
            served = [
                c.args[0] % c.args[1:]
                for c in svc.log.warning.call_args_list
                if "refusal_fallback_served" in c.args[0]
            ]
            return result, calls, served

        return _run

    def test_text_refusal_reruns_turn_on_fallback(self, run) -> None:  # type: ignore[no-untyped-def]
        """Refused, nudged, refused again: the fallback's reply is delivered."""
        result, calls, served = run([REFUSAL, REFUSAL, "here's your filth"])

        assert result.content == "here's your filth"
        assert [c["model"] for c in calls] == ["gpt-4", "gpt-4", FALLBACK]
        assert result.model == FALLBACK
        # The primary's two calls stay booked alongside the fallback's one.
        assert result.cost == pytest.approx(0.25 * 2 + 0.5)
        assert served == [f"refusal_fallback_served primary=gpt-4 fallback={FALLBACK} trigger=text"]

    def test_fallback_sees_clean_request_only(self, run) -> None:  # type: ignore[no-untyped-def]
        """Neither the refusal nor the nudge reaches the fallback model."""
        _result, calls, _served = run([REFUSAL, REFUSAL, "fine"])

        fallback_messages = str(calls[2]["messages"])
        assert REFUSAL not in fallback_messages
        assert _SAFETY_REFUSAL_RETRY_NUDGE not in fallback_messages
        assert calls[2]["messages"] == calls[0]["messages"]

    def test_provider_refusal_error_reruns_on_fallback(self, run) -> None:  # type: ignore[no-untyped-def]
        """A provider refusal raised by the primary is retried on the fallback."""
        refused = provider.RefusalError(
            "Output blocked: content violates usage guidelines", model="gpt-4"
        )
        result, calls, served = run([refused, "sure thing"])

        assert result.content == "sure thing"
        assert [c["model"] for c in calls] == ["gpt-4", FALLBACK]
        assert calls[1]["messages"] == calls[0]["messages"]
        assert served == [
            f"refusal_fallback_served primary=gpt-4 fallback={FALLBACK} trigger=provider_error"
        ]

    def test_stop_reason_refusal_reruns_on_fallback(self, run) -> None:  # type: ignore[no-untyped-def]
        """Anthropic's classifier declines with finish_reason "refusal" and no
        text; the turn re-runs once on the fallback without a nudge."""
        declined = make_completion_response(None, finish_reason="refusal")
        result, calls, served = run([declined, "sure thing"])

        assert result.content == "sure thing"
        assert [c["model"] for c in calls] == ["gpt-4", FALLBACK]
        assert calls[1]["messages"] == calls[0]["messages"]
        assert served == [
            f"refusal_fallback_served primary=gpt-4 fallback={FALLBACK} trigger=stop_reason"
        ]

    def test_fallback_that_also_refuses_is_delivered_without_looping(self, run) -> None:  # type: ignore[no-untyped-def]
        """One fallback per turn: grok's own refusal gets its nudge, then ships."""
        result, calls, served = run([REFUSAL])

        assert result.content == REFUSAL
        assert [c["model"] for c in calls] == ["gpt-4", "gpt-4", FALLBACK, FALLBACK]
        assert len(served) == 1

    @pytest.mark.parametrize(
        "overrides",
        [{"refusalFallbackModel": ""}, {"assistantModel": FALLBACK}],
        ids=["disabled", "primary-is-fallback"],
    )
    def test_no_fallback(self, run, overrides: dict[str, str]) -> None:  # type: ignore[no-untyped-def]
        """Empty setting, or a primary that already is the fallback, delivers the refusal."""
        result, calls, served = run([REFUSAL], **overrides)

        assert result.content == REFUSAL
        assert len(calls) == 2
        assert served == []

    def test_no_fallback_after_a_tool_with_side_effects(self, run) -> None:  # type: ignore[no-untyped-def]
        """A reminder already set would be set twice by the re-run, so none happens."""
        set_it = make_completion_response(
            None, tool_calls=[make_tool_call("set_reminder", {"text": "in 5m stretch"})]
        )
        result, calls, served = run(
            [set_it, REFUSAL, REFUSAL, "fallback"],
            call={"set_reminder_fn": lambda _t: ToolCallbackResult(True, "Reminder set.")},
        )

        assert result.content == REFUSAL
        assert {c["model"] for c in calls} == {"gpt-4"}
        assert served == []

    def test_fallback_after_read_only_tool(self, run) -> None:  # type: ignore[no-untyped-def]
        """A search is safe to repeat, so a turn that only searched still falls back."""
        from llm.assistant import ToolResult

        search = make_completion_response(
            None, tool_calls=[make_tool_call("search_web", {"query": "filthy jokes"})]
        )
        result, calls, served = run(
            [search, REFUSAL, REFUSAL, search, "fallback"],
            call={"search_fn": lambda _q: ToolResult(content='{"results": "jokes"}')},
        )

        assert result.content == "fallback"
        assert [c["model"] for c in calls][-2:] == [FALLBACK, FALLBACK]
        assert len(served) == 1

    def test_no_fallback_without_its_key(self, run, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
        """A fallback provider with no key is skipped silently."""
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        result, calls, served = run([REFUSAL])

        assert result.content == REFUSAL
        assert {c["model"] for c in calls} == {"gpt-4"}
        assert served == []


def test_fallback_model_books_grok_4_3_rates() -> None:
    """The alias costs grok-4.3's per-token rates, not a silent $0."""
    assert provider.PRICES[FALLBACK] == provider.PRICES["xai/grok-4.3"]
