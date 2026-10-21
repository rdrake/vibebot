"""Tests for image-spend accounting.

A draw's price comes back on the provider's ``Picture`` (xAI's own ticks, or
``provider.IMAGE_PRICES`` when there are none). Two ways that price was being
lost before it reached the usage table:

* **The tool boundary.** `_draw_for_assistant` knew the cost and had nowhere to
  put it, so nothing recorded it at all. From 2026-04-11, when @draw was
  converted to run through assistant_completion and every path to an image started
  going through that callback, until 2026-08-16, every image the bot drew was
  booked at $0.00. The prod usage table shows it plainly: draw rows carry
  `xai/grok-imagine-image-pro` (since renamed -quality) at $0.069 average through 2026-04-11, then switch
  to *chat* model names at ~$0.001 — the text cost of the turn with the image
  cost missing. The fix is a second usage row written by the callback itself,
  under the image model, because a row names exactly one model and the turn
  used two.

* **Refused generations.** A moderated refusal still bills; xAI says so in the
  error (`'usage': {'cost_in_usd_ticks': 200000000}`). Nothing costed those at
  all, on any path. With draws refused better than half the time, that is the
  larger of the two leaks per call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm import provider
from llm.assistant import ToolCallbackResult
from llm.service import LLMService

from .conftest import make_completion_response, make_tool_call

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

MINTED = "https://paste.boxlabs.uk/img/img_6a81079496b1b.jpg"
IMAGE_MODEL = "xai/grok-imagine-image"
PRICE = provider.IMAGE_PRICES[IMAGE_MODEL]
PNG = b"\x89PNG\r\n\x1a\nimage"
SAVED = "https://example.com/llm/img_a.png"

XAI_MODERATION_ERROR = (
    "{'code': 'imagine:content-moderated', 'error': 'Generated image rejected by "
    "content moderation.', 'usage': {'cost_in_usd_ticks': 200000000}}"
)


def _refusal() -> provider.ContentPolicyViolationError:
    """xAI's billed moderation refusal, as provider.draw raises it."""
    return provider.ContentPolicyViolationError(XAI_MODERATION_ERROR, billed=True)


def _picture() -> provider.Picture:
    return provider.Picture(data=PNG, cost=PRICE)


class TestUnknownModelPricing:
    """`provider.image_price`: the table is the price, and a miss must be audible."""

    def test_known_model_is_priced_from_the_table(self) -> None:
        assert provider.image_price(IMAGE_MODEL) == PRICE

    def test_unknown_model_warns_instead_of_silently_costing_zero(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A model swap must not quietly zero the books again.

        Returning 0.0 is the only honest answer without a price, but doing it
        silently is what made four months of draw spend unreadable.
        """
        provider._unpriced.discard("xai/grok-imagine-image-ultra")
        with caplog.at_level("WARNING"):
            assert provider.image_price("xai/grok-imagine-image-ultra") == 0.0
        assert "xai/grok-imagine-image-ultra" in caplog.text

    def test_unknown_model_warns_once_not_per_call(self, caplog: pytest.LogCaptureFixture) -> None:
        """Per-call warnings on a busy channel are scrolled past, not read."""
        provider._unpriced.discard("xai/some-unpriced-model")
        with caplog.at_level("WARNING"):
            for _ in range(5):
                provider.image_price("xai/some-unpriced-model")
        assert caplog.text.count("xai/some-unpriced-model") == 1


class TestBilledRefusals:
    """A refusal the provider charged for has to reach the books."""

    def test_moderation_refusal_is_costed(self, make_service) -> None:  # type: ignore[no-untyped-def]
        service, _ = make_service()
        assert service._billed_failure_cost(_refusal(), IMAGE_MODEL) == PRICE

    def test_refusal_with_no_usage_block_is_free(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """Not every provider bills for a block, so don't invent a charge.

        The usage block is read as a yes/no signal that the call was charged;
        without one the honest answer is zero.
        """
        service, _ = make_service()
        error = provider.ContentPolicyViolationError("blocked by safety filter")
        assert service._billed_failure_cost(error, IMAGE_MODEL) == 0.0

    def test_refused_generation_bills_through_image_generation(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """End to end: a refused draw returns an error AND a non-zero cost."""
        service, _ = make_service(
            imageModel=IMAGE_MODEL,
            assistantModel="xai/grok-4.3",
            drawAutoRewriteMax=0,
            httpUrlBase="https://example.com/llm",
        )
        mocker.patch("llm.service.provider.draw", side_effect=_refusal())

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert result.cost == PRICE

    def test_every_attempt_in_the_rewrite_loop_is_billed(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """A refusal plus a refused rewrite is two charges, not one."""
        service, _ = make_service(
            imageModel=IMAGE_MODEL,
            assistantModel="xai/grok-4.3",
            drawAutoRewriteMax=1,
            httpUrlBase="https://example.com/llm",
        )
        mocker.patch("llm.service.provider.draw", side_effect=[_refusal(), _refusal()])
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert result.cost == pytest.approx(PRICE * 2)


class TestSpendIsAttributedToTheModelThatSpentIt:
    """A usage row names one model, so image spend needs its own row.

    Folding it into the caller's total makes `GROUP BY model` read as though
    the chat model bought the pictures. The draw callback writes its own row
    instead — see `LLM._draw_for_assistant` and the tests in test_assistant.py.
    """

    def test_callback_result_carries_no_usage(self) -> None:
        """Deliberate: cost travels via the leaf's own row, not up the stack.

        If these fields ever come back, the same spend lands in two rows. The
        rule is about SPEND, not about field count -- a signal like ``reworded``
        costs nothing and double-books nothing, so it is allowed.
        """
        assert "cost" not in ToolCallbackResult._fields
        assert "prompt_tokens" not in ToolCallbackResult._fields
        assert "completion_tokens" not in ToolCallbackResult._fields

    def test_draw_tool_returns_a_bare_string(self) -> None:
        """No ToolResult, so the executor accumulates nothing for a draw."""
        from llm.assistant import AssistantToolExecutor

        executor = object.__new__(AssistantToolExecutor)
        executor._draw_fn = lambda _p, **_kw: ToolCallbackResult(True, MINTED)

        result = AssistantToolExecutor._tool_generate_image(executor, {"prompt": "a party"})

        assert isinstance(result, str)
        assert MINTED in result

    def test_assistant_turn_reports_only_its_text_cost(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """The turn's own row must not absorb the image bill.

        `AssistantResult.cost` is what `_store_context_and_log_usage` writes
        under the CHAT model. The image is billed separately, so this number
        stays text-only — otherwise the two rows sum to double the real spend.
        """
        service, _ = make_service(
            assistantModel="xai/grok-4.3", httpUrlBase="https://irc.rdrake.org/llm"
        )
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response(
                None, tool_calls=[make_tool_call("generate_image", {"prompt": "a party"})]
            ),
        )

        result = service.assistant_completion(
            prompt="draw bunga bunga party",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=lambda _p, **_kw: ToolCallbackResult(True, MINTED),
        )

        assert result.content == MINTED
        assert result.cost == pytest.approx(0.0)


def test_service_price_lookup_is_used_for_successful_generations(
    make_service, mocker: MockerFixture
) -> None:
    """A delivered image carries the price the provider put on its picture."""
    service, _ = make_service(
        imageModel=IMAGE_MODEL,
        drawAutoRewriteMax=0,
        httpUrlBase="https://example.com/llm",
    )
    mocker.patch("llm.service.provider.draw", return_value=_picture())
    mocker.patch.object(service, "_save_image_bytes", return_value=SAVED)

    result = service.image_generation("a party")

    assert result.error is None
    assert result.cost == PRICE


class TestLLMServiceHelpersExist:
    """Guards the names the rest of this file leans on."""

    def test_helpers_are_public_enough_to_test(self) -> None:
        assert hasattr(LLMService, "_billed_failure_cost")


class TestRefusedAttemptsSurviveTheRewriteLoop:
    """A refusal a rewrite recovered from still happened, and still billed.

    Before this, the loop swallowed it: `image_generation` returned one success
    carrying the *summed* cost, so the usage table booked a two-call turn as a
    single $0.04 image and the refused prompt — the only text that is known to
    have tripped the filter — was never written down. `blocked_attempts` carries
    those calls out so the accounting layer can file one row per provider call.

    The invariant, at every return: the refusals in `blocked_attempts` are the
    ones NOT represented by the returned result. When the result is itself a
    content block, the last refusal is that result, so it stays out.
    """

    def _service(self, make_service, max_rewrites: int):  # type: ignore[no-untyped-def]
        service, _ = make_service(
            imageModel=IMAGE_MODEL,
            assistantModel="xai/grok-4.3",
            drawAutoRewriteMax=max_rewrites,
            httpUrlBase="https://example.com/llm",
        )
        return service

    def test_recovered_refusal_is_carried_out_with_its_prompt_and_price(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """The happy path this whole change exists for: refused, rewritten, delivered."""
        service = self._service(make_service, 1)
        mocker.patch("llm.service.provider.draw", side_effect=[_refusal(), _picture()])
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )
        mocker.patch.object(service, "_save_image_bytes", return_value=SAVED)

        result = service.image_generation("bunga bunga party")

        assert result.error is None
        assert len(result.blocked_attempts) == 1
        blocked = result.blocked_attempts[0]
        assert blocked.prompt == "bunga bunga party"
        assert blocked.cost == PRICE
        assert "content moderation" in blocked.reason

    def test_exhausted_loop_leaves_the_last_refusal_to_the_final_row(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """Two refused calls are two rows, and the result IS the second one.

        Carrying both out would double-book the last refusal: once as a blocked
        attempt, once as the error the caller already writes a row for.
        """
        service = self._service(make_service, 1)
        mocker.patch("llm.service.provider.draw", side_effect=[_refusal(), _refusal()])
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert len(result.blocked_attempts) == 1
        assert result.cost == pytest.approx(PRICE * 2)

    def test_rewrites_disabled_carries_nothing(self, make_service, mocker: MockerFixture) -> None:
        """One call, one row, and the caller already writes it. No change here."""
        service = self._service(make_service, 0)
        mocker.patch("llm.service.provider.draw", side_effect=_refusal())

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert result.blocked_attempts == ()
        assert result.cost == PRICE

    def test_a_refusal_the_provider_did_not_bill_is_still_recorded(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """A block can come back as no picture and no charge.

        The prompt is the evidence, not the money — a free refusal still has to
        leave a row behind, or a block that costs nothing stays invisible.
        """
        service = self._service(make_service, 1)
        mocker.patch(
            "llm.service.provider.draw",
            side_effect=[provider.Picture(data=None, cost=0.0), _picture()],
        )
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )
        mocker.patch.object(service, "_save_image_bytes", return_value=SAVED)

        result = service.image_generation("bunga bunga party")

        assert result.error is None
        assert len(result.blocked_attempts) == 1
        assert result.blocked_attempts[0].cost == 0.0
        assert result.blocked_attempts[0].prompt == "bunga bunga party"

    def test_a_non_content_failure_mid_loop_keeps_the_earlier_refusal(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """The final row is a timeout, so the refusal before it is nobody else's."""
        service = self._service(make_service, 1)
        mocker.patch(
            "llm.service.provider.draw",
            side_effect=[
                _refusal(),
                provider.AuthenticationError("invalid key"),
            ],
        )
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("something milder"),
        )

        result = service.image_generation("bunga bunga party")

        assert result.error is not None
        assert len(result.blocked_attempts) == 1
        assert result.blocked_attempts[0].cost == PRICE


class TestRewordedImagesAreMarked:
    """A delivered image that took a reworded prompt says so, like 🌐 does.

    The picture may not be quite what was asked for -- the rewrite keeps the
    subject but is free to change the wording -- and silently handing back a
    slightly different image is how the bot looks like it ignored the request.
    The signal rides the same rails as `grounding_used`: callback -> executor ->
    AssistantResult -> one icon on the reply.
    """

    def test_the_callback_reports_a_reword(self, mocker: MockerFixture) -> None:
        from llm.service import BlockedAttempt, ImageResult

        from .test_assistant_helpers import make_draw_plugin

        plugin = make_draw_plugin(mocker)
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            cost=0.0403,
            rewritten_prompt="a cat beside a bonfire",
            blocked_attempts=(BlockedAttempt("a cat on fire", "content moderation", 0.02),),
        )
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mocker.MagicMock(), msg, "a cat on fire")

        assert result.reworded is True

    def test_a_first_try_image_is_not_marked(self, mocker: MockerFixture) -> None:
        from llm.service import ImageResult

        from .test_assistant_helpers import make_draw_plugin

        plugin = make_draw_plugin(mocker)
        plugin.llm_service.image_generation.return_value = ImageResult(
            content="https://img.example/cat.png",
            model="xai/grok-imagine-image",
            cost=0.02,
        )
        msg = mocker.MagicMock()
        msg.prefix = "user!ident@host"
        msg.args = ["#test"]

        result = plugin._draw_for_assistant(mocker.MagicMock(), msg, "a cat")

        assert result.reworded is False

    def test_the_executor_latches_it(self) -> None:
        """One reworded image in a turn marks the turn, like grounding does."""
        from llm.assistant import AssistantToolExecutor, ToolCallbackResult

        executor = object.__new__(AssistantToolExecutor)
        executor.image_reworded = False
        executor._draw_fn = lambda _p, **_kw: ToolCallbackResult(True, MINTED, reworded=True)

        AssistantToolExecutor._tool_generate_image(executor, {"prompt": "a cat"})

        assert executor.image_reworded is True

    def test_a_reworded_draw_turn_marks_the_result(
        self, make_service, mocker: MockerFixture
    ) -> None:
        """End to end: the short-circuit that returns the bare URL carries it."""
        from llm.assistant import ToolCallbackResult

        service, _ = make_service(
            assistantModel="xai/grok-4.3", httpUrlBase="https://irc.rdrake.org/llm"
        )
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response(
                None, tool_calls=[make_tool_call("generate_image", {"prompt": "a cat"})]
            ),
        )

        result = service.assistant_completion(
            prompt="draw a cat on fire",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=lambda _p, **_kw: ToolCallbackResult(True, MINTED, reworded=True),
        )

        assert result.content == MINTED
        assert result.image_reworded is True
