"""The subject-research pre-stage in front of the @draw and @animate planners.

An image or video generator knows nothing by name. The planners are already
told to keep the user's word and put a description beside it, but for a real
person or a real event nothing supplies the description — the verse canon block
covers what the channel invented, not what the world already contains. This
stage supplies it.

Everything worth testing here is the parse. The researcher is a general model
answering in prose by default, and whatever it returns is about to be pasted
into another model's system prompt, so the filter is by SHAPE: only the
dash-prefixed lines the prompt asked for survive. A refusal, a preamble or a
trailing caveat is dropped for having no dash, which is the same failure
test_image_failure_guard.py guards one stage later — a model reading its own
refusal back and parroting it forever.

The second shape rule is the source tag. A grounded researcher asked about a
name that does not exist ("Stinky Lads", 2026-09-19) does not answer NONE — it
guesses from the words and the planner pastes the guess beside the name. So a
line must end in "(source: ...)" to survive, and the tag is stripped before
injection: the planner needs the appearance, not the citation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .conftest import make_completion_response

if TYPE_CHECKING:
    pass


def _answering(mocker, service, text: str, **usage: Any):
    """Stub the provider call so the researcher returns ``text``."""
    return mocker.patch(
        "llm.service.provider.search",
        return_value=make_completion_response(
            text,
            prompt_tokens=usage.get("prompt_tokens", 40),
            completion_tokens=usage.get("completion_tokens", 60),
        ),
    )


class TestDossierParsing:
    """Only the lines the prompt asked for reach the planner."""

    def test_keeps_the_list_lines(self, make_service, mocker) -> None:
        """GIVEN a well-formed list WHEN researched THEN every line survives."""
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "- Winston Churchill — heavyset man in his late sixties, bald crown, "
            "jowled face, three-piece pinstripe suit and a cigar (source: Wikipedia)\n"
            "- the Blitz — London after dark in 1940, smoke and searchlights "
            "(source: Imperial War Museums)",
        )

        result = service.subject_dossier("churchill during the blitz", channel="#test")

        assert result.text.count("\n") == 1
        assert "Winston Churchill" in result.text
        assert "the Blitz" in result.text

    def test_drops_preamble_and_caveats(self, make_service, mocker) -> None:
        """GIVEN prose wrapped around the list WHEN parsed THEN only the list survives.

        The researcher is not a formatter. It opens with "Here's what I found"
        and closes with a note about accuracy unless stopped, and both would
        otherwise be read by the planner as instructions.
        """
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "Here are the subjects I could identify:\n"
            "- Amelia Earhart — slim woman in her thirties, cropped wavy hair, "
            "leather flying jacket (source: Smithsonian)\n"
            "\nNote that appearance details vary between sources.",
        )

        result = service.subject_dossier("amelia earhart taking off", channel="#test")

        assert result.text == (
            "- Amelia Earhart — slim woman in her thirties, cropped wavy hair, "
            "leather flying jacket"
        )

    def test_a_refusal_yields_nothing(self, make_service, mocker) -> None:
        """GIVEN the researcher declines WHEN parsed THEN nothing is injected.

        A refusal pasted into the planner's system prompt is a model reading a
        "no" and agreeing with it. Dropping it means the request renders with
        less detail, which is the correct failure.
        """
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "I'm not able to provide detailed physical descriptions of real "
            "individuals. Let me know if I can help another way.",
        )

        result = service.subject_dossier("a real person", channel="#test")

        assert result.text == ""

    def test_none_answer_yields_nothing(self, make_service, mocker) -> None:
        """GIVEN a request naming nobody real WHEN researched THEN no block."""
        service, _ = make_service()
        _answering(mocker, service, "NONE")

        result = service.subject_dossier(
            "a slow aerial shot over a pine forest at sunrise", channel="#test"
        )

        assert result.text == ""

    def test_bullet_characters_are_normalised(self, make_service, mocker) -> None:
        """GIVEN asterisk or bullet markers WHEN parsed THEN lines become dashes."""
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "* Elvis Presley — pompadour, white jumpsuit (source: Britannica)\n"
            "• Graceland — white-columned Memphis mansion (source: graceland.com)",
        )

        result = service.subject_dossier("elvis at graceland", channel="#test")

        assert result.text.splitlines() == [
            "- Elvis Presley — pompadour, white jumpsuit",
            "- Graceland — white-columned Memphis mansion",
        ]

    def test_line_count_is_capped(self, make_service, mocker) -> None:
        """GIVEN more subjects than the cap WHEN parsed THEN the cap holds.

        The block rides in the planner's system prompt on every request; an
        unbounded one is an unbounded prompt.
        """
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "\n".join(f"- Subject {i} — a description (source: site)" for i in range(20)),
        )

        result = service.subject_dossier("a crowd scene", channel="#test")

        assert len(result.text.splitlines()) == 8

    def test_character_budget_is_capped(self, make_service, mocker) -> None:
        """GIVEN one enormous line WHEN parsed THEN the budget is not exceeded."""
        service, _ = make_service()
        _answering(mocker, service, f"- Subject — {'x' * 4000} (source: site)")

        result = service.subject_dossier("a subject", channel="#test")

        assert len(result.text) <= 1500

    def test_unsourced_lines_are_dropped(self, make_service, mocker) -> None:
        """GIVEN a line with no source tag WHEN parsed THEN it never reaches the planner.

        The live failure: "Stinky Lads" names nobody, the researcher invented
        "young Australian men in their twenties, tanned, in caps", and three
        of them were drawn. A guess has nowhere to cite.
        """
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "- Stinky Lads — young Australian men in their twenties, athletic, "
            "t-shirts and caps\n"
            "- Hong Kong — dense neon-lit streets, 1980s taxis, harbour skyline "
            "(source: Wikipedia)",
        )

        result = service.subject_dossier("Stinky Lads do Hong Kong", channel="#test")

        assert result.text == ("- Hong Kong — dense neon-lit streets, 1980s taxis, harbour skyline")

    def test_source_tag_is_stripped(self, make_service, mocker) -> None:
        """GIVEN a sourced line WHEN parsed THEN the citation is not injected."""
        service, _ = make_service()
        _answering(
            mocker,
            service,
            "- Churchill — heavyset, bald, cigar  (Source: en.wikipedia.org/wiki/Winston_Churchill)",
        )

        result = service.subject_dossier("churchill", channel="#test")

        assert result.text == "- Churchill — heavyset, bald, cigar"


class TestDossierModelSelection:
    """Two keys, most specific first."""

    def test_prefers_subject_research_model(self, make_service, mocker) -> None:
        """GIVEN both keys set WHEN researched THEN the research key wins."""
        service, _ = make_service(
            subjectResearchModel="xai/grok-4.3",
            searchModel="xai/grok-4-1-fast-reasoning",
            assistantModel="anthropic/claude-haiku-5-5",
        )
        call = _answering(mocker, service, "- A — b")

        result = service.subject_dossier("x", channel="#test")

        assert call.call_args.args[0] == "xai/grok-4.3"
        assert result.model == "xai/grok-4.3"

    def test_falls_back_to_search_model(self, make_service, mocker) -> None:
        """GIVEN no research key WHEN researched THEN searchModel is used."""
        service, _ = make_service(
            subjectResearchModel="",
            searchModel="xai/grok-4-1-fast-reasoning",
            assistantModel="anthropic/claude-haiku-5-5",
        )
        call = _answering(mocker, service, "- A — b")

        service.subject_dossier("x", channel="#test")

        assert call.call_args.args[0] == "xai/grok-4-1-fast-reasoning"


class TestDossierFailureAndSpend:
    """A failure costs detail, never the turn — and a call that happened is booked."""

    def test_provider_failure_yields_nothing(self, make_service, mocker) -> None:
        """GIVEN the provider raises WHEN researched THEN text is empty, no raise.

        _grounded_completion catches and returns a JSON error string, which has
        no dash-prefixed line and so parses to nothing on its own. This asserts
        that arrangement holds rather than trusting it.
        """
        service, _ = make_service()
        mocker.patch("llm.service.provider.search", side_effect=RuntimeError("boom"))

        result = service.subject_dossier("churchill", channel="#test")

        assert result.text == ""
        assert result.cost == 0.0

    def test_usage_rides_back_for_booking(self, make_service, mocker) -> None:
        """GIVEN a successful call WHEN researched THEN tokens come back with it.

        The caller books its own usage row; it can only do that if the numbers
        survive the return.
        """
        service, _ = make_service()
        _answering(mocker, service, "- A — b", prompt_tokens=123, completion_tokens=45)

        result = service.subject_dossier("x", channel="#test")

        assert result.prompt_tokens == 123
        assert result.completion_tokens == 45

    def test_braces_in_the_request_survive(self, make_service, mocker) -> None:
        """GIVEN braces in the user's words WHEN researched THEN no format error.

        An IRC user can type '{'. Assembling the research prompt with
        str.format would raise KeyError on it and take the whole command down.
        """
        service, _ = make_service()
        call = _answering(mocker, service, "NONE")

        service.subject_dossier("draw {this} and {that}", channel="#test")

        sent = call.call_args.args[1]
        assert "{this}" in sent
