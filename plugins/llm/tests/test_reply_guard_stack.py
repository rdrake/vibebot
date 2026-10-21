"""Tests for the reply-guard table itself.

The individual detectors have their own suites; this one pins the properties
the *table* is responsible for, which the six longhand blocks used to encode
implicitly in their own control flow:

* order, because the first guard to fire wins the turn;
* the split around the fabricated-image guard, which rewrites ``content``
  between the two halves and so must keep everything after it downstream;
* fall-through, because an exhausted guard has to yield to the next one
  rather than swallowing the turn.

Behaviour-preservation for the refactor that introduced the table rests on the
existing per-guard suites all still passing; these are the invariants those
suites cannot see.
"""

from __future__ import annotations

from llm.profile import PROFILE_CHAT, PROFILE_VERSE
from llm.service import (
    _POST_IMAGE_REPLY_GUARDS,
    _PRE_IMAGE_REPLY_GUARDS,
    REPLY_GUARDS,
    _ReplyGuardContext,
)


def _ctx(content: str, **over: object) -> _ReplyGuardContext:
    """A context that trips nothing unless a field is overridden."""
    base: dict[str, object] = {
        "content": content,
        "prompt": "unrelated prompt",
        "route_profile": PROFILE_CHAT,
        "any_tool_ran": True,
        "prior_replies": (),
    }
    base.update(over)
    return _ReplyGuardContext(**base)  # type: ignore[arg-type]


class TestTableShape:
    """The table is the contract; pin what the loop relies on."""

    def test_order_is_the_documented_order(self) -> None:
        """Order is behaviour — the first guard to fire wins the turn."""
        assert [g.key for g in _PRE_IMAGE_REPLY_GUARDS] == ["echo", "verse_denial"]
        assert [g.key for g in _POST_IMAGE_REPLY_GUARDS] == [
            "job_marker",
            "tool_complaint",
            "safety_refusal",
            "degraded",
            "repeat",
        ]

    def test_keys_are_unique_across_both_halves(self) -> None:
        """The retry ledger is keyed by guard key, so collisions would share budget."""
        keys = [g.key for g in _PRE_IMAGE_REPLY_GUARDS + _POST_IMAGE_REPLY_GUARDS]
        assert len(keys) == len(set(keys))
        assert set(REPLY_GUARDS) == set(keys)


class TestDetectorsAreScopedCorrectly:
    """Each detector reads only what it should from the context."""

    def test_verse_denial_is_verse_only(self) -> None:
        """The same refusal text is inert on a chat route."""
        denial = "That never happened — pure fiction, not in the canon."
        guard = REPLY_GUARDS["verse_denial"]
        assert guard.detect(_ctx(denial, route_profile=PROFILE_VERSE))
        assert not guard.detect(_ctx(denial, route_profile=PROFILE_CHAT))

    def test_tool_complaint_needs_no_tool_to_have_run(self) -> None:
        """An honest complaint (a tool ran and failed) is delivered untouched."""
        guard = REPLY_GUARDS["tool_complaint"]
        assert guard.detect(_ctx("Tool's still choking on the request.", any_tool_ran=False))
        assert not guard.detect(_ctx("Tool's still choking on the request.", any_tool_ran=True))

    def test_repeat_compares_against_prior_replies(self) -> None:
        """With no anchors there is nothing to near-duplicate."""
        line = "The stinky lads charged the gym in a cloud of gammon fumes."
        guard = REPLY_GUARDS["repeat"]
        assert guard.detect(_ctx(line, prior_replies=(line,)))
        assert not guard.detect(_ctx(line, prior_replies=()))

    def test_echo_compares_against_the_prompt(self) -> None:
        """The echo guard is the only one that reads ctx.prompt."""
        guard = REPLY_GUARDS["echo"]
        echoed = "what is the capital of france"
        assert guard.detect(_ctx(echoed, prompt=echoed))
        assert not guard.detect(_ctx("Paris.", prompt=echoed))


class TestRunReplyGuards:
    """The driver: budget, fall-through, and what it appends."""

    def test_clean_reply_fires_nothing(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """A good reply survives every guard and the caller does not re-roll."""
        svc, _plugin = make_service(assistantModel="gpt-4")
        messages: list[dict] = []
        spent = dict.fromkeys(REPLY_GUARDS, 0)
        fired = svc._run_reply_guards(
            _POST_IMAGE_REPLY_GUARDS,
            _ctx("Here's your bacon, Jordan."),
            spent,
            messages,
            model="gpt-4",
            channel="#afternet",
        )
        assert fired is False
        assert messages == []
        assert set(spent.values()) == {0}

    def test_firing_spends_budget_and_seeds_the_nudge(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """The rejected reply goes back with its nudge, and budget is spent."""
        svc, _plugin = make_service(assistantModel="gpt-4")
        messages: list[dict] = []
        spent = dict.fromkeys(REPLY_GUARDS, 0)
        complaint = "Tool's still choking on the request."

        fired = svc._run_reply_guards(
            _POST_IMAGE_REPLY_GUARDS,
            _ctx(complaint, any_tool_ran=False),
            spent,
            messages,
            model="gpt-4",
            channel="#afternet",
        )

        assert fired is True
        assert spent["tool_complaint"] == 1
        assert messages[0] == {"role": "assistant", "content": complaint}
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == REPLY_GUARDS["tool_complaint"].nudge

    def test_exhausted_guard_falls_through_to_the_next(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """Budget is checked before the detector, so a spent guard yields.

        This is what the longhand chain did by falling out of one ``if`` into
        the next, and it is the easiest property to lose in a table rewrite.
        """
        svc, _plugin = make_service(assistantModel="gpt-4")
        spent = dict.fromkeys(REPLY_GUARDS, 0)
        spent["tool_complaint"] = REPLY_GUARDS["tool_complaint"].max_retries
        # Trips tool_complaint (already spent) AND repeat (still has budget).
        complaint = "Tool's still choking on the request."
        messages: list[dict] = []

        fired = svc._run_reply_guards(
            _POST_IMAGE_REPLY_GUARDS,
            _ctx(complaint, any_tool_ran=False, prior_replies=(complaint,)),
            spent,
            messages,
            model="gpt-4",
            channel="#afternet",
        )

        assert fired is True
        assert spent["tool_complaint"] == 1, "exhausted guard must not spend again"
        assert spent["repeat"] == 1, "the next eligible guard should have fired"
        assert messages[1]["content"] == REPLY_GUARDS["repeat"].nudge

    def test_all_exhausted_delivers_the_reply(self, make_service) -> None:  # type: ignore[no-untyped-def]
        """After every budget is gone the reply is delivered, not errored."""
        svc, _plugin = make_service(assistantModel="gpt-4")
        spent = {k: g.max_retries for k, g in REPLY_GUARDS.items()}
        messages: list[dict] = []
        fired = svc._run_reply_guards(
            _POST_IMAGE_REPLY_GUARDS,
            _ctx("Tool's still choking on the request.", any_tool_ran=False),
            spent,
            messages,
            model="gpt-4",
            channel="#afternet",
        )
        assert fired is False
        assert messages == []
