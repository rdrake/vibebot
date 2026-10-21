"""The subject dossier reaching the @draw planner.

The parse is tested in test_subject_dossier.py. What is tested here is the
seam: that the block lands in the planner's system prompt, that the kill switch
kills it, that a research failure costs the request its detail and not its
image, and that a call which happened is booked. The last one is not
paranoia — draw spend read $0.00 for four months behind a missing row.
"""

from __future__ import annotations

import pytest

from .conftest import make_registry_side_effect


def _assistant_result(content: str = "Here you go."):
    from llm.service import AssistantResult, Spend

    return AssistantResult(
        content=content,
        grounding_used=False,
        spend=Spend(prompt_tokens=120, completion_tokens=40, cost=0.0012),
        model="xai/grok-4-1-fast-non-reasoning",
    )


def _dossier(text: str, **usage):
    from llm.service import SubjectDossier

    return SubjectDossier(
        text=text,
        model=usage.get("model", "gemini/gemini-3-flash-preview"),
        prompt_tokens=usage.get("prompt_tokens", 90),
        completion_tokens=usage.get("completion_tokens", 60),
        cost=usage.get("cost", 0.0004),
    )


_FACTS = "- Winston Churchill — heavyset man in his late sixties, cigar, pinstripe suit"


@pytest.fixture
def researching_plugin(plugin_env, mocker):
    """A plugin with research enabled and the generation path stubbed."""
    plugin, mock_irc, mock_msg = plugin_env
    mock_irc.state.nickToAccount.return_value = "test_account"
    plugin.registryValue.side_effect = make_registry_side_effect({"subjectResearchEnabled": True})
    plugin.llm_service.assistant_completion.side_effect = None
    plugin.llm_service.assistant_completion.return_value = _assistant_result()
    plugin.llm_service.subject_dossier.return_value = _dossier(_FACTS)
    mocker.patch.object(plugin, "_verse_context_for", return_value=None)
    return plugin, mock_irc, mock_msg


def _system_prompt(plugin) -> str | None:
    return plugin.llm_service.assistant_completion.call_args.kwargs["system_prompt"]


class TestResearchSeam:
    """What the research pre-stage does to the request it sits in front of."""

    def test_the_request_is_what_gets_researched(self, researching_plugin) -> None:
        """GIVEN a request WHEN researched THEN the user's own words are the query."""
        plugin, mock_irc, mock_msg = researching_plugin

        plugin.draw(mock_irc, mock_msg, ["churchill", "on", "a", "balcony"])

        plugin.llm_service.subject_dossier.assert_called_once()
        assert plugin.llm_service.subject_dossier.call_args.args[0] == "churchill on a balcony"

    def test_nothing_to_research_leaves_the_prompt_alone(self, researching_plugin) -> None:
        """GIVEN a pure-fiction request WHEN researched THEN no block is added.

        "A pine forest at sunrise" names nobody, and paying a
        block's worth of prompt to say so would be worse than not asking.
        """
        plugin, mock_irc, mock_msg = researching_plugin
        plugin.llm_service.subject_dossier.return_value = _dossier("")

        plugin.draw(mock_irc, mock_msg, ["a", "pine", "forest", "at", "sunrise"])

        assert _system_prompt(plugin) is None

    def test_canon_and_dossier_coexist(self, researching_plugin, mocker) -> None:
        """GIVEN both canon and real subjects WHEN @draw runs THEN both are present.

        "@draw the stinky lads meet churchill" needs the channel's invented
        cast AND the real man. The block says canon wins where they overlap;
        that is a claim for the model to act on, so all this asserts is that
        both bodies of fact arrive and canon is stated first.
        """
        plugin, mock_irc, mock_msg = researching_plugin
        mocker.patch.object(plugin, "_verse_context_for", return_value="CANON: the stinky lads")

        plugin.draw(mock_irc, mock_msg, ["the", "stinky", "lads", "meet", "churchill"])

        prompt = _system_prompt(plugin)
        assert prompt is not None
        assert prompt.index("CANON: the stinky lads") < prompt.index(_FACTS)

    def test_spend_is_booked_under_its_own_model(self, researching_plugin) -> None:
        """GIVEN a research call WHEN it completes THEN a dossier usage row is written.

        Its own row, not the planner's: the researcher is gemini and the
        planner is grok in production, and one row cannot hold two models.
        """
        plugin, mock_irc, mock_msg = researching_plugin

        plugin.draw(mock_irc, mock_msg, ["churchill", "on", "a", "balcony"])

        rows = [c for c in plugin.db.log_usage.call_args_list if c.args[2] == "dossier"]
        assert len(rows) == 1
        assert rows[0].args[3] == "gemini/gemini-3-flash-preview"
        assert rows[0].args[4] == 90
        assert rows[0].args[5] == 60

    def test_a_failed_call_books_nothing(self, researching_plugin) -> None:
        """GIVEN research that cost nothing WHEN it returns THEN no row is written.

        A zero-token row records that an exception happened, which the
        exception log already records, and pollutes the spend table doing it.
        """
        plugin, mock_irc, mock_msg = researching_plugin
        plugin.llm_service.subject_dossier.return_value = _dossier(
            "", prompt_tokens=0, completion_tokens=0, cost=0.0
        )

        plugin.draw(mock_irc, mock_msg, ["churchill", "on", "a", "balcony"])

        assert [c for c in plugin.db.log_usage.call_args_list if c.args[2] == "dossier"] == []


class TestDrawResearch:
    """@draw pays the research in wall clock the user is watching."""

    def test_facts_reach_the_planner(self, researching_plugin) -> None:
        """GIVEN a real subject WHEN @draw runs THEN the facts are in the prompt."""
        plugin, mock_irc, mock_msg = researching_plugin

        plugin.draw(mock_irc, mock_msg, ["churchill", "on", "a", "balcony"])

        prompt = plugin.llm_service.assistant_completion.call_args.kwargs["system_prompt"]
        assert prompt is not None
        assert _FACTS in prompt

    def test_typing_starts_before_the_research(self, researching_plugin, mocker) -> None:
        """GIVEN research runs WHEN @draw starts THEN typing is already showing.

        The research is seconds the user spends staring at nothing otherwise.
        There is nothing to hide it behind, so the indicator has to be up
        before the call, which means the overlay assembly happens
        inside the try block rather than above it.
        """
        plugin, mock_irc, mock_msg = researching_plugin
        order: list[str] = []
        plugin.llm_service._begin_typing.side_effect = lambda irc, msg: (
            order.append("typing"),
            lambda: None,
        )[1]
        plugin.llm_service.subject_dossier.side_effect = lambda *a, **k: (
            order.append("research"),
            _dossier(_FACTS),
        )[1]

        plugin.draw(mock_irc, mock_msg, ["churchill", "on", "a", "balcony"])

        assert order == ["typing", "research"]

    def test_disabled_means_no_call(self, researching_plugin) -> None:
        """GIVEN the kill switch off WHEN @draw runs THEN no research happens."""
        plugin, mock_irc, mock_msg = researching_plugin
        plugin.registryValue.side_effect = make_registry_side_effect(
            {"subjectResearchEnabled": False}
        )

        plugin.draw(mock_irc, mock_msg, ["churchill", "on", "a", "balcony"])

        plugin.llm_service.subject_dossier.assert_not_called()
        assert plugin.llm_service.assistant_completion.call_args.kwargs["system_prompt"] is None

    def test_research_failure_still_draws(self, researching_plugin) -> None:
        """GIVEN the research raises WHEN @draw runs THEN the image is still planned."""
        plugin, mock_irc, mock_msg = researching_plugin
        plugin.llm_service.subject_dossier.side_effect = RuntimeError("provider down")

        plugin.draw(mock_irc, mock_msg, ["churchill", "on", "a", "balcony"])

        plugin.llm_service.assistant_completion.assert_called_once()
        assert plugin.llm_service.assistant_completion.call_args.kwargs["system_prompt"] is None
