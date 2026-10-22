"""meme_pick: one JSON completion, or a short search_templates loop.

With a ``search`` callable the picker opens on a short brief and may look up
the rest of the catalog; the loop is capped so a model that keeps searching
still has to answer, and every step's usage is booked.
"""

from __future__ import annotations

import pytest
from llm.prompts import MEME_SEARCH_NOTE

from .conftest import make_completion_response, make_tool_call

_ANSWER = '{"template": "drake", "lines": ["a", "b"]}'


@pytest.fixture
def service(make_service, mocker):
    service, _plugin = make_service()
    mocker.patch.object(service, "_missing_key_error", return_value=None)
    return service


def _search_call(query: str, call_id: str = "call_0"):
    return make_tool_call("search_templates", {"query": query}, call_id=call_id)


class TestWithoutSearch:
    def test_one_json_completion_and_no_tools(self, service, mocker) -> None:
        completion = mocker.patch.object(
            service, "_timed_completion", return_value=make_completion_response(_ANSWER)
        )

        pick = service.meme_pick("waiting for CI", catalog_brief="drake | ...", channel="#t")

        assert pick.content == _ANSWER and pick.error is None
        kwargs = completion.call_args.kwargs
        assert kwargs["json_reply"] is True
        assert "tools" not in kwargs
        assert MEME_SEARCH_NOTE not in kwargs["messages"][0]["content"]

    def test_the_shared_grounding_rides_under_the_request(self, service, mocker) -> None:
        """The canon block and subject dossier reach the picker."""
        completion = mocker.patch.object(
            service, "_timed_completion", return_value=make_completion_response(_ANSWER)
        )

        service.meme_pick(
            "bob at work", catalog_brief="drake | ...", channel="#t", context="CANON: Bob is a cat"
        )

        user_turn = completion.call_args.kwargs["messages"][1]["content"]
        assert user_turn == "bob at work\n\nCANON: Bob is a cat"


class TestSearchLoop:
    def test_a_search_feeds_its_rows_back_and_usage_is_summed(self, service, mocker) -> None:
        completion = mocker.patch.object(
            service,
            "_timed_completion",
            side_effect=[
                make_completion_response(None, tool_calls=[_search_call("waiting")]),
                make_completion_response(_ANSWER),
            ],
        )
        searched: list[str] = []

        def search(query: str) -> str:
            searched.append(query)
            return "skeleton | Waiting Skeleton | 2 | ..."

        pick = service.meme_pick(
            "waiting for CI", catalog_brief="drake | ...", channel="#t", search=search
        )

        assert pick.content == _ANSWER
        assert searched == ["waiting"]
        assert pick.prompt_tokens == 20 and pick.completion_tokens == 40
        first, second = completion.call_args_list
        assert MEME_SEARCH_NOTE in first.kwargs["messages"][0]["content"]
        assert first.kwargs["tools"][0]["function"]["name"] == "search_templates"
        assert "json_reply" not in first.kwargs
        # Searching is allowed until the cap: the first step does not switch it off.
        assert "tool_choice" not in first.kwargs
        tool_turn = second.kwargs["messages"][-1]
        assert tool_turn == {
            "role": "tool",
            "tool_call_id": "call_0",
            "content": "skeleton | Waiting Skeleton | 2 | ...",
        }
        assert second.kwargs["messages"][-2]["tool_calls"][0]["function"]["name"] == (
            "search_templates"
        )

    def test_an_answer_without_searching_is_one_call(self, service, mocker) -> None:
        completion = mocker.patch.object(
            service, "_timed_completion", return_value=make_completion_response(_ANSWER)
        )

        pick = service.meme_pick(
            "drake", catalog_brief="drake | ...", channel="#t", search=lambda q: ""
        )

        assert pick.content == _ANSWER
        assert completion.call_count == 1

    def test_searches_are_capped_then_the_tool_is_switched_off(self, service, mocker) -> None:
        completion = mocker.patch.object(
            service,
            "_timed_completion",
            side_effect=[
                make_completion_response(None, tool_calls=[_search_call("one", "c1")]),
                make_completion_response(None, tool_calls=[_search_call("two", "c2")]),
                make_completion_response(_ANSWER),
            ],
        )
        searched: list[str] = []

        pick = service.meme_pick(
            "x", catalog_brief="", channel="#t", search=lambda q: searched.append(q) or "rows"
        )

        assert pick.content == _ANSWER
        assert searched == ["one", "two"]
        assert completion.call_count == 3
        last = completion.call_args_list[-1].kwargs
        assert last["tool_choice"] == "none"
        # Anthropic needs the tool defined while its calls sit in the history.
        assert last["tools"][0]["function"]["name"] == "search_templates"

    def test_parallel_calls_past_the_cap_are_refused_not_run(self, service, mocker) -> None:
        mocker.patch.object(
            service,
            "_timed_completion",
            side_effect=[
                make_completion_response(
                    None,
                    tool_calls=[
                        _search_call("a", "c1"),
                        _search_call("b", "c2"),
                        _search_call("c", "c3"),
                    ],
                ),
                make_completion_response(_ANSWER),
            ],
        )
        searched: list[str] = []

        service.meme_pick(
            "x", catalog_brief="", channel="#t", search=lambda q: searched.append(q) or "rows"
        )

        assert searched == ["a", "b"]

    def test_malformed_arguments_search_for_nothing(self, service, mocker) -> None:
        mocker.patch.object(
            service,
            "_timed_completion",
            side_effect=[
                make_completion_response(
                    None, tool_calls=[make_tool_call("search_templates", "{not json")]
                ),
                make_completion_response(_ANSWER),
            ],
        )
        searched: list[str] = []

        pick = service.meme_pick(
            "x", catalog_brief="", channel="#t", search=lambda q: searched.append(q) or "rows"
        )

        assert searched == [""] and pick.content == _ANSWER

    def test_a_failed_step_keeps_the_usage_so_far(self, service, mocker) -> None:
        mocker.patch.object(
            service,
            "_timed_completion",
            side_effect=[
                make_completion_response(None, tool_calls=[_search_call("waiting")]),
                RuntimeError("upstream 529"),
            ],
        )

        pick = service.meme_pick("x", catalog_brief="", channel="#t", search=lambda q: "rows")

        assert pick.content is None and pick.error
        assert pick.prompt_tokens == 10
