"""Tests for the fabricated-image-URL guard.

The stale-image guard caught the model REPOSTING a real image URL out of
history after generate_image failed. This covers the sibling failure it could
not see: the model skipping the tool entirely and INVENTING a URL.

Observed in #afternet on 2026-08-01, in the same minute as a genuine success:

    20:28:58  draw a wizard         -> paste.boxlabs.uk/img/img_6a6e571adf508.jpg   (20s, real)
    20:29:49  draw rdrake ...       -> irc.rdrake.org/llm/image/<slug>.png          (2s, invented)

The invented URLs 404 -- no such directory, no such files -- and there was no
op=image_generation in the log for either. Every structural check passed them
through, because the old detector only recognised the mint filename shape
(``img_<hex>.<ext>``) and a fabrication does not look minted: findall returned
nothing, so the reply read as clean. Worse, the guard was gated on the image
tool having FAILED, which is exactly what does not happen when the model never
calls it.

So detection matches any image URL and filters by host, and the guard runs
unconditionally. A fabrication is recoverable, so it forces the tool and
retries rather than apologising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llm.assistant import ToolCallbackResult
from llm.service import (
    _IMAGE_FABRICATION_FALLBACK,
    _MAX_IMAGE_FABRICATION_RETRIES,
    LLMService,
    _unminted_image_urls,
)

from .conftest import make_completion_response, make_tool_call

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

OWN_HOSTS = frozenset({"irc.rdrake.org", "paste.boxlabs.uk"})
# Shape the model invented: our host, but a path and filename scheme that has
# never existed.
FABRICATED = "https://irc.rdrake.org/llm/image/9c8e7f4b-rdrake-dismantling-vibebot.png"
# Shape _save_image_bytes actually returns.
REAL_MINTED = "https://paste.boxlabs.uk/img/img_6a6e571adf508.jpg"


class TestUnmintedDetection:
    """The detector must see invented URLs, not just mint-shaped ones."""

    def test_flags_fabricated_url_on_our_host(self) -> None:
        """The exact URL from #afternet is caught."""
        assert _unminted_image_urls(FABRICATED, set(), OWN_HOSTS) == [FABRICATED]

    def test_old_detector_shape_would_have_missed_it(self) -> None:
        """Pins the root cause: the fabrication carries no mint filename.

        With an empty host set the detector falls back to mint-shape matching
        only -- which is what the guard used to do, and why this reply shipped.
        """
        assert _unminted_image_urls(FABRICATED, set(), frozenset()) == []

    def test_accepts_the_url_this_turn_minted(self) -> None:
        """A real generated image is not a fabrication."""
        assert not _unminted_image_urls(f"here you go: {REAL_MINTED}", {REAL_MINTED}, OWN_HOSTS)

    def test_flags_mint_shaped_url_regardless_of_host(self) -> None:
        """The original host-independent check survives the widening.

        An unset httpUrlBase leaves the host set empty; the guard must degrade
        to its old behaviour rather than to matching nothing.
        """
        stale = "https://elsewhere.example/llm/img_6a669cbcbc700.jpg"
        assert _unminted_image_urls(stale, set(), frozenset()) == [stale]

    def test_ignores_another_bot_image(self) -> None:
        """Someone else's image link is legitimate and must pass through."""
        other = "https://www.larrystrong.com/img-gen/gi_20260801_picard.jpg"
        assert not _unminted_image_urls(f"look: {other}", set(), OWN_HOSTS)

    def test_plain_reply_has_nothing_to_flag(self) -> None:
        """A reply with no image URL is never a fabrication."""
        assert not _unminted_image_urls("no image for you", set(), OWN_HOSTS)

    def test_empty_content_is_safe(self) -> None:
        """Empty content short-circuits without touching the regexes."""
        assert _unminted_image_urls("", set(), OWN_HOSTS) == []


class TestFabricationForcesTheTool:
    """Recovery: the user asked for a picture, so produce one."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(
            assistantModel="xai/grok-4.3", httpUrlBase="https://irc.rdrake.org/llm"
        )
        return svc

    def test_invented_url_is_retried_with_the_tool_forced(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The #afternet sequence, recovered: fabrication -> forced call -> real image."""
        responses = [
            # Step 1: the model writes a URL without calling anything.
            make_completion_response(FABRICATED),
            # Step 2: forced to call generate_image.
            make_completion_response(
                None, tool_calls=[make_tool_call("generate_image", {"prompt": "rdrake"})]
            ),
            make_completion_response(f"Here you go: {REAL_MINTED}"),
        ]
        captured_kwargs: list[dict] = []

        def fake_completion(**kwargs: object) -> object:
            captured_kwargs.append(kwargs)
            return responses[len(captured_kwargs) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="draw rdrake dismantling you",
            nick="ibutsu",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="ibutsu",
            draw_fn=lambda _p, **_kw: ToolCallbackResult(True, REAL_MINTED),
        )

        assert FABRICATED not in (result.content or "")
        assert REAL_MINTED in (result.content or "")
        # The retry must actually force the tool, not merely ask again.
        assert captured_kwargs[1].get("tool_choice") == {
            "type": "function",
            "function": {"name": "generate_image"},
        }

    def test_unrecovered_fabrication_becomes_an_honest_failure(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """A model that keeps inventing gets replaced, not published.

        An invented link is worse than an admitted failure: it 404s, and
        nothing about the reply looks wrong to the user.
        """
        mocker.patch(
            "llm.service.provider.chat",
            side_effect=lambda **_: make_completion_response(FABRICATED),
        )

        result = service.assistant_completion(
            prompt="draw rdrake dismantling you",
            nick="ibutsu",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
        )

        assert FABRICATED not in (result.content or "")
        assert result.content == _IMAGE_FABRICATION_FALLBACK
        assert _MAX_IMAGE_FABRICATION_RETRIES == 1

    def test_genuine_image_reply_is_untouched(
        self, service: LLMService, mocker: MockerFixture
    ) -> None:
        """The guard must not interfere with a real generated image."""
        responses = [
            make_completion_response(
                None, tool_calls=[make_tool_call("generate_image", {"prompt": "a wizard"})]
            ),
            make_completion_response(REAL_MINTED),
        ]
        calls: list[object] = []

        def fake_completion(**kwargs: object) -> object:
            calls.append(kwargs)
            return responses[len(calls) - 1]

        mocker.patch("llm.service.provider.chat", side_effect=fake_completion)

        result = service.assistant_completion(
            prompt="draw a wizard",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            capabilities=frozenset({"llm.ask", "llm.draw"}),
            account="rdrake",
            draw_fn=lambda _p, **_kw: ToolCallbackResult(True, REAL_MINTED),
        )

        assert REAL_MINTED in (result.content or "")
        # A successful draw short-circuits after generate_image, so it costs
        # exactly one completion. Pinning that catches the guard forcing a
        # needless extra step (and another image charge) on a clean turn.
        assert len(calls) == 1
