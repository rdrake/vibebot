"""Tests for the history-strip instrument.

The strips were the unmeasured half of the guard stack. Every retry logs a
line; the strips ran silently on every turn, so the only visible evidence of
self-imitation was the fraction that survived as far as the retry path. On
2026-08-01 a five-hour run of poisoned replies in #afternet produced exactly
ONE guard-fire line, which made the load look negligible when it was not — and
a six-month log analysis could measure retries per model but not strips.

``history_strip`` closes that gap. These tests pin the two things that make it
trustworthy: that counting cannot change what the model sees, and that the
line only appears when something was actually removed.
"""

from __future__ import annotations

import logging

import pytest
from llm.service import LLMService, _counted_strip

from .conftest import make_completion_response

LOGGER = "supybot.plugins.LLM.service"
POISON = "Tool's still choking on the request."


class TestCountedStrip:
    """Bookkeeping only — the returned history must be the strip's own output."""

    def test_returns_exactly_what_the_strip_returned(self) -> None:
        """Identity for a strip that drops nothing, including the same object."""
        history = [{"role": "user", "content": "hi"}]
        ledger: dict[str, int] = {}
        assert _counted_strip("k", lambda h: h, history, ledger) is history
        assert ledger == {}

    def test_records_the_number_of_turns_dropped(self) -> None:
        """Count is before-minus-after, not a boolean."""
        history = [{"role": "assistant", "content": str(i)} for i in range(5)]
        ledger: dict[str, int] = {}
        result = _counted_strip("k", lambda h: h[:2], history, ledger)
        assert result == history[:2]
        assert ledger == {"k": 3}

    def test_accumulates_across_both_windows(self) -> None:
        """The same key is applied to history and channel_history in turn."""
        ledger: dict[str, int] = {}
        _counted_strip("k", lambda h: h[:1], [{"a": "1"}, {"a": "2"}], ledger)
        _counted_strip("k", lambda h: h[:1], [{"a": "3"}, {"a": "4"}], ledger)
        assert ledger == {"k": 2}

    def test_handles_none_history(self) -> None:
        """None windows (verse drops channel_history) must not be counted."""
        ledger: dict[str, int] = {}
        assert _counted_strip("k", lambda h: h, None, ledger) is None
        assert ledger == {}

    def test_never_counts_a_strip_that_grew_history(self) -> None:
        """Defensive: a strip cannot add turns, and must not log -1 if it did."""
        ledger: dict[str, int] = {}
        _counted_strip("k", lambda h: [*h, {"a": "extra"}], [{"a": "1"}], ledger)
        assert ledger == {}


class TestHistoryStripLine:
    """The log line: silent when clean, structured when not."""

    @pytest.fixture
    def service(self, make_service) -> LLMService:  # type: ignore[no-untyped-def]
        svc, _plugin = make_service(assistantModel="gpt-4")
        return svc

    def _run(self, service: LLMService, mocker, **kwargs) -> None:  # type: ignore[no-untyped-def]
        mocker.patch(
            "llm.service.provider.chat",
            return_value=make_completion_response("Here's your bacon."),
        )
        service.assistant_completion(
            prompt="give Jordan some bacon",
            nick="rdrake",
            channel="#afternet",
            db=mocker.MagicMock(),
            context=mocker.MagicMock(),
            bot_nick="VibeBot",
            **kwargs,
        )

    def test_silent_when_nothing_was_stripped(
        self, service: LLMService, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A clean thread must not emit a row — the denominator comes from
        the matching completion_timing line instead."""
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._run(
                service,
                mocker,
                history=[
                    {"role": "user", "content": "morning"},
                    {"role": "assistant", "content": "Morning, rdrake."},
                ],
            )
        assert "history_strip" not in caplog.text

    def test_reports_what_it_removed(
        self, service: LLMService, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A poisoned thread emits one line naming the guard and the count."""
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._run(
                service,
                mocker,
                history=[
                    {"role": "user", "content": "draw a tit"},
                    {"role": "assistant", "content": POISON},
                    {"role": "user", "content": "give Jordan some bacon"},
                ],
            )
        lines = [ln for ln in caplog.text.splitlines() if "history_strip" in ln]
        assert len(lines) == 1, caplog.text
        line = lines[0]
        assert "tool_complaint=1" in line
        assert "removed=1" in line
        assert "channel=#afternet" in line
        assert "route=chat" in line

    def test_denominator_counts_assistant_turns_before_stripping(
        self, service: LLMService, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        """assistant_turns is the pre-strip count, or the rate is unusable.

        Two poisoned replies plus one clean one: the denominator must be 3,
        not the 1 that survives.
        """
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._run(
                service,
                mocker,
                history=[
                    {"role": "assistant", "content": POISON},
                    {"role": "assistant", "content": "The tool's fucked."},
                    {"role": "assistant", "content": "Bacon it is."},
                ],
            )
        line = next(ln for ln in caplog.text.splitlines() if "history_strip" in ln)
        assert "assistant_turns=3" in line
        assert "removed=2" in line

    def test_counts_the_channel_window_too(
        self, service: LLMService, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Both windows carry the bot's own lines, so both are instrumented."""
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._run(
                service,
                mocker,
                history=[{"role": "assistant", "content": POISON}],
                channel_history=[{"role": "assistant", "content": POISON}],
            )
        line = next(ln for ln in caplog.text.splitlines() if "history_strip" in ln)
        assert "tool_complaint=2" in line

    def test_line_survives_the_supybot_format_bug(
        self, service: LLMService, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Regression: %-args get mangled by supybot's logger (see 1b64332).

        The guard-fire lines logged ``model=1 channel=1`` for months because
        ``%d`` was dropped and the remaining args shifted left. This line is
        built as an f-string, so every field must render its real value.
        """
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            self._run(
                service,
                mocker,
                history=[{"role": "assistant", "content": POISON}],
            )
        line = next(ln for ln in caplog.text.splitlines() if "history_strip" in ln)
        assert "model=gpt-4" in line
        assert "%d" not in line and "%s" not in line and "%i" not in line
        assert "model=1" not in line and "channel=1" not in line
