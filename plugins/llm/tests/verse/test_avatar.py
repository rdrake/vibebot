"""Tests for verb whitelist (VERB_TABLE) in avatar module."""

from __future__ import annotations

import json
import logging
import time

import pytest
from llm.verse.avatar import (
    ActResult,
    build_verse_system_prompt,
    dispatch_verse_tool_call,
    is_ooc,
    make_verse_denial_handlers,
    make_verse_extra_handlers,
    make_verse_tool_specs,
    strip_ooc,
    verse_act,
    verse_look,
    verse_move,
    verse_recall,
)
from llm.verse.store import VerseStore

from .conftest import link_avatar

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: pytest.TempPathFactory) -> VerseStore:
    return VerseStore(tmp_path, "#test")


def _opt_in(store: VerseStore, nick: str = "alice") -> int:
    """Opt in a user and return their avatar entity_id."""
    result = store.opt_in_avatar(nick=nick, account=None, instruct_text="A wanderer.")
    return result.entity_id


# ---------------------------------------------------------------------------
# TestVerseAct
# ---------------------------------------------------------------------------


class TestVerseAct:
    def test_event_only_verb_records_event(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        result = verse_act(store, alice_id, "speak")

        assert isinstance(result, ActResult)
        events = store.recent_events(limit=10)
        assert len(events) == 1
        ev = events[0]
        assert ev.source == "avatar"
        assert "speak" in ev.summary
        assert alice_id in ev.entity_ids
        assert result.scene_shift_text.startswith("You speak")

    def test_event_only_verb_with_target(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        result = verse_act(store, alice_id, "whisper", target="bob")

        events = store.recent_events(limit=10)
        assert len(events) == 1
        assert "bob" in events[0].summary
        assert result.scene_shift_text == "You whisper bob."

    def test_sibilant_verb_pluralizes_with_es(self, store: VerseStore) -> None:
        """'search' -> 'searches' in the third-person event summary, not 'searchs'."""
        alice_id = _opt_in(store, nick="alice")
        verse_act(store, alice_id, "search", target="the bushes")
        events = store.recent_events(limit=10)
        assert any(e.summary.startswith("alice searches") for e in events)

    def test_move_verb_success_updates_location(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        riverside_id = store.add_entity("place", "Riverside", "A bend in the river.")
        result = verse_act(store, alice_id, "move", target="Riverside")

        # Location attribute updated
        assert store.get_attribute(alice_id, "location") == "Riverside"
        # Event written with both ids
        events = store.recent_events(limit=10)
        move_events = [e for e in events if "Riverside" in e.summary]
        assert len(move_events) == 1
        assert alice_id in move_events[0].entity_ids
        assert riverside_id in move_events[0].entity_ids
        # scene_shift
        assert result.scene_shift_text == "You move to Riverside."

    def test_move_verb_target_not_found_writes_event_no_side_effect(
        self, store: VerseStore
    ) -> None:
        alice_id = _opt_in(store)
        original_location = store.get_attribute(alice_id, "location")

        result = verse_act(store, alice_id, "move", target="Nowhere")

        # Event IS written
        events = store.recent_events(limit=10)
        assert any("Nowhere" in e.summary for e in events)
        # Location UNCHANGED
        assert store.get_attribute(alice_id, "location") == original_location
        # Exact scene_shift text
        assert result.scene_shift_text == "You can't find that place."

    def test_move_to_retired_place_is_not_resolved(self, store: VerseStore) -> None:
        """A retired place is not a valid move target: verse_act must treat it
        as not-found rather than relocating the avatar into a dead entity."""
        alice_id = _opt_in(store)
        gone_id = store.add_entity("place", "GhostTown", "Abandoned.")
        store.set_status(gone_id, "retired")
        original_location = store.get_attribute(alice_id, "location")

        result = verse_act(store, alice_id, "move", target="GhostTown")

        assert result.scene_shift_text == "You can't find that place."
        assert store.get_attribute(alice_id, "location") == original_location

    def test_move_verb_via_avatar_target_resolves_to_their_location(
        self, store: VerseStore
    ) -> None:
        alice_id = _opt_in(store, nick="alice")
        bob_id = _opt_in(store, nick="bob")

        # Move alice to Cliff
        cliff_id = store.add_entity("place", "Cliff", "A high ledge.")
        verse_act(store, alice_id, "move", target="Cliff")
        assert store.get_attribute(alice_id, "location") == "Cliff"

        # Bob follows alice (avatar target resolves to alice's location)
        result = verse_act(store, bob_id, "move", target="alice")
        assert store.get_attribute(bob_id, "location") == "Cliff"
        events = store.recent_events(limit=20)
        bob_move = [e for e in events if bob_id in e.entity_ids and cliff_id in e.entity_ids]
        assert len(bob_move) == 1
        assert result.scene_shift_text == "You move to Cliff."

    def test_item_verb_target_not_found_writes_event_no_relation(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        result = verse_act(store, alice_id, "give", target="Phantom Sword")

        # Event written
        events = store.recent_events(limit=10)
        assert any("Phantom Sword" in e.summary for e in events)
        # No relation added
        assert store.list_relations(from_id=alice_id) == []
        # scene_shift acknowledges absence
        assert "Phantom Sword" in result.scene_shift_text or "find" in result.scene_shift_text

    def test_item_verb_success_writes_event(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        sword_id = store.add_entity("item", "sword", "A sharp blade.")

        result = verse_act(store, alice_id, "take", target="sword")

        events = store.recent_events(limit=10)
        take_events = [e for e in events if "sword" in e.summary]
        assert len(take_events) == 1
        assert alice_id in take_events[0].entity_ids
        assert sword_id in take_events[0].entity_ids
        # scene_shift mentions verb and item
        assert "take" in result.scene_shift_text or "sword" in result.scene_shift_text
        # No relation added (v1 design)
        assert store.list_relations(from_id=alice_id) == []

    def test_off_list_verb_writes_event_no_side_effect(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        original_location = store.get_attribute(alice_id, "location")

        result = verse_act(store, alice_id, "teleport", target="moon")

        events = store.recent_events(limit=10)
        assert len(events) == 1
        assert "teleport" in result.scene_shift_text
        # Location unchanged
        assert store.get_attribute(alice_id, "location") == original_location

    def test_retired_avatar_raises_before_write(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        store.set_status(alice_id, "retired")

        before_count = len(store.recent_events(limit=100))

        with pytest.raises(ValueError, match="avatar retired"):
            verse_act(store, alice_id, "speak")

        # No event written
        assert len(store.recent_events(limit=100)) == before_count

    def test_unknown_avatar_id_raises(self, store: VerseStore) -> None:
        with pytest.raises(ValueError, match="avatar retired"):
            verse_act(store, 99999, "speak")

    @pytest.mark.parametrize("verb", ["move", "take"])
    def test_move_or_item_verb_without_target_is_event_only(
        self, store: VerseStore, verb: str
    ) -> None:
        """A MOVE/ITEM verb with no target must not write 'tries to take None'
        into canon; it records the bare verb like any event-only action."""
        alice_id = _opt_in(store)
        result = verse_act(store, alice_id, verb)

        summary = store.recent_events(limit=1)[0].summary
        assert "None" not in summary
        assert summary == f"alice {verb}s"
        assert result.scene_shift_text == f"You {verb}."


# ---------------------------------------------------------------------------
# TestVerseMove
# ---------------------------------------------------------------------------


class TestVerseMove:
    def test_success_updates_location_and_returns_place_name(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        store.add_entity("place", "Riverside", "A bend in the river.")
        result = verse_move(store, alice_id, "Riverside")
        assert result == "Riverside"
        assert store.get_attribute(alice_id, "location") == "Riverside"

    def test_no_such_place_raises_and_does_not_change_location(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        original_location = store.get_attribute(alice_id, "location")
        with pytest.raises(ValueError, match="no such place"):
            verse_move(store, alice_id, "Nowhere")
        assert store.get_attribute(alice_id, "location") == original_location

    def test_case_insensitive_place_lookup(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        store.add_entity("place", "Riverside", "A bend in the river.")
        result = verse_move(store, alice_id, "RIVERSIDE")
        assert result == "Riverside"
        assert store.get_attribute(alice_id, "location") == "Riverside"

    def test_retired_place_is_not_resolved(self, store: VerseStore) -> None:
        """A retired place is not a valid move target (parity with verse_act #26)."""
        alice_id = _opt_in(store)
        original_location = store.get_attribute(alice_id, "location")
        gone_id = store.add_entity("place", "Old Mill", "Abandoned mill.")
        store.set_status(gone_id, "retired")
        with pytest.raises(ValueError, match="no such place"):
            verse_move(store, alice_id, "Old Mill")
        assert store.get_attribute(alice_id, "location") == original_location

    def test_retired_avatar_cannot_move(self, store: VerseStore) -> None:
        """A retired avatar must not be relocated (no write before the guard)."""
        alice_id = _opt_in(store)
        store.add_entity("place", "Riverside", "A bend in the river.")
        location_before = store.get_attribute(alice_id, "location")
        store.set_status(alice_id, "retired")
        with pytest.raises(ValueError, match="avatar retired"):
            verse_move(store, alice_id, "Riverside")
        assert store.get_attribute(alice_id, "location") == location_before


# ---------------------------------------------------------------------------
# TestVerseLook
# ---------------------------------------------------------------------------


class TestVerseLook:
    def test_no_target_returns_current_place_summary(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        # opt_in places alice at The Clearing
        result = verse_look(store, alice_id)
        assert result == "A quiet woodland clearing where new stories begin."

    def test_target_existing_entity_returns_summary(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        store.add_entity("item", "Sword", "A bright blade.")
        result = verse_look(store, alice_id, target="sword")
        assert result == "A bright blade."

    def test_target_not_found_returns_none(self, store: VerseStore) -> None:
        alice_id = _opt_in(store)
        result = verse_look(store, alice_id, target="phantom")
        assert result is None

    def test_no_target_no_location_returns_none(self, store: VerseStore) -> None:
        # Create avatar entity directly, without opt_in (no location attribute).
        eid = store.add_entity("avatar", "ghost", "A wandering spirit.")
        result = verse_look(store, eid)
        assert result is None

    def test_retired_target_returns_none(self, store: VerseStore) -> None:
        """A retired entity is not part of the scene — verse_look must not
        describe it (parity with verse_move/verse_act active-only lookups)."""
        alice_id = _opt_in(store)
        gone_id = store.add_entity("item", "Sword", "A bright blade.")
        store.set_status(gone_id, "retired")
        assert verse_look(store, alice_id, target="Sword") is None

    def test_retired_location_returns_none(self, store: VerseStore) -> None:
        """If the avatar's recorded location has since been retired, the
        no-target look must say nothing rather than describe a ghost place
        the system prompt calls 'nowhere'."""
        alice_id = _opt_in(store)  # placed at The Clearing
        place = store.find_entity_by_name("The Clearing", kind="place")
        assert place is not None
        store.set_status(place.id, "retired")
        assert verse_look(store, alice_id) is None


# ---------------------------------------------------------------------------
# TestVerseRecall
# ---------------------------------------------------------------------------


class TestVerseRecall:
    def test_substring_match(self, store: VerseStore) -> None:
        store.add_event(summary="Alice walks to the river", entity_ids=[], source="avatar")
        store.add_event(summary="Bob sits down", entity_ids=[], source="avatar")
        store.add_event(summary="Carol crosses the bridge", entity_ids=[], source="avatar")
        results = verse_recall(store, "river")
        assert len(results) == 1
        assert "river" in results[0].summary

    def test_case_insensitive(self, store: VerseStore) -> None:
        store.add_event(summary="Alice walks to the river", entity_ids=[], source="avatar")
        store.add_event(summary="Bob sits down", entity_ids=[], source="avatar")
        results = verse_recall(store, "RIVER")
        assert len(results) == 1
        assert "river" in results[0].summary

    def test_token_or_match(self, store: VerseStore) -> None:
        store.add_event(summary="Alice walks to the river", entity_ids=[], source="avatar")
        store.add_event(summary="Bob sits down", entity_ids=[], source="avatar")
        store.add_event(summary="Carol crosses the bridge", entity_ids=[], source="avatar")
        results = verse_recall(store, "river bridge")
        assert len(results) == 2

    def test_limit_to_5_newest_first(self, store: VerseStore) -> None:
        for i in range(10):
            store.add_event(summary=f"alpha event {i}", entity_ids=[], source="avatar")
            time.sleep(0.01)
        results = verse_recall(store, "alpha")
        assert len(results) == 5
        # Newest-first: ts of each result must be >= ts of the next
        for i in range(len(results) - 1):
            assert results[i].ts >= results[i + 1].ts

    def test_no_matches_returns_empty(self, store: VerseStore) -> None:
        store.add_event(summary="Something unrelated", entity_ids=[], source="avatar")
        results = verse_recall(store, "nonsense")
        assert results == []

    def test_empty_query_returns_empty(self, store: VerseStore) -> None:
        store.add_event(summary="Something", entity_ids=[], source="avatar")
        assert verse_recall(store, "") == []
        assert verse_recall(store, "   ") == []


# ---------------------------------------------------------------------------
# TestSystemPrompt
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_basic_structure_includes_all_sections(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, nick="alice")
        prompt = build_verse_system_prompt(store, alice_id, "curious traveller")

        assert "You are alice" in prompt
        assert "Persona: curious traveller" in prompt
        assert "Scene:" in prompt
        assert "The Clearing" in prompt
        assert "A quiet woodland clearing where new stories begin." in prompt
        assert "Recent events involving you:" in prompt
        assert "Other avatars present here:" in prompt

    def test_empty_instruct_uses_no_persona_set(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, nick="alice")
        prompt = build_verse_system_prompt(store, alice_id, "")
        assert "Persona: no persona set." in prompt

    def test_recent_events_filtered_by_avatar(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, nick="alice")
        bob_id = _opt_in(store, nick="bob")

        store.add_event(summary="alice does something", entity_ids=[alice_id], source="avatar")
        store.add_event(summary="bob does one thing", entity_ids=[bob_id], source="avatar")
        store.add_event(summary="bob does another thing", entity_ids=[bob_id], source="avatar")

        prompt = build_verse_system_prompt(store, alice_id, "a traveller")

        assert "alice does something" in prompt
        assert "bob does one thing" not in prompt
        assert "bob does another thing" not in prompt

    def test_recent_events_capped_at_5(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, nick="alice")

        summaries = [f"alice event {i}" for i in range(7)]
        for s in summaries:
            store.add_event(summary=s, entity_ids=[alice_id], source="avatar")

        prompt = build_verse_system_prompt(store, alice_id, "a traveller")

        found = [s for s in summaries if s in prompt]
        assert len(found) == 5

    def test_own_events_found_behind_many_unrelated(self, store: VerseStore) -> None:
        """A returning player whose last event is 51+ events back still sees it
        under 'Recent events involving you' (the filter must not run after a
        channel-wide limit)."""
        alice_id = _opt_in(store, nick="alice")
        store.add_event(summary="alice buries the lantern", entity_ids=[alice_id], source="avatar")
        for i in range(60):
            store.add_event(summary=f"unrelated {i}", entity_ids=[], source="avatar")

        prompt = build_verse_system_prompt(store, alice_id, "a traveller")
        assert "- alice buries the lantern" in prompt

    def test_no_events_shows_none_yet_marker(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, nick="alice")
        # opt_in writes no events for alice
        prompt = build_verse_system_prompt(store, alice_id, "a traveller")
        assert "(none yet)" in prompt

    def test_other_avatars_present_filtered_by_location(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, nick="alice")  # placed at The Clearing by opt_in
        _opt_in(store, nick="bob")  # also placed at The Clearing by opt_in

        # Create carol manually at a different location
        carol_id = store.add_entity("avatar", "carol", "A mysterious stranger.")
        link_avatar(store, carol_id, nick="carol")
        store.set_attribute(carol_id, "location", "Riverside")
        # Ensure Riverside exists as a place entity so the location is meaningful
        store.add_entity("place", "Riverside", "A bend in the river.")

        prompt = build_verse_system_prompt(store, alice_id, "a traveller")

        assert "bob" in prompt
        assert "carol" not in prompt

    def test_no_other_avatars_marker(self, store: VerseStore) -> None:
        alice_id = _opt_in(store, nick="alice")
        prompt = build_verse_system_prompt(store, alice_id, "a traveller")
        assert "(no other avatars present)" in prompt

    def test_no_location_uses_nowhere_in_particular(self, store: VerseStore) -> None:
        # Create avatar manually without location attribute
        avatar_id = store.add_entity("avatar", "ghost", "A wandering spirit.")
        link_avatar(store, avatar_id, nick="ghost")
        prompt = build_verse_system_prompt(store, avatar_id, "a spirit")
        assert "nowhere in particular" in prompt

    def test_retired_location_drops_co_located_avatars_and_scene(self, store: VerseStore) -> None:
        """A retired place is a 'ghost location': you cannot be co-located
        at a place that no longer actively exists. While the place is
        active, a second avatar standing there is listed; once the place is
        retired, the co-located block must show the no-other-avatars marker
        and the scene must read 'nowhere in particular'."""
        place_id = store.add_entity("place", "The Glade", "A mossy hollow.")
        alice_id = store.add_entity("avatar", "alice", "A wanderer.")
        store.set_attribute(alice_id, "location", "The Glade")
        bob_id = store.add_entity("avatar", "bob", "Another wanderer.")
        store.set_attribute(bob_id, "location", "The Glade")

        # Positive control: active place -> bob is co-located with alice.
        prompt = build_verse_system_prompt(store, alice_id, "a traveller")
        assert "The Glade" in prompt
        assert "Other avatars present here:" in prompt
        assert "bob" in prompt
        assert "(no other avatars present)" not in prompt

        # Retire the place: it becomes a ghost location.
        store.set_status(place_id, "retired")
        prompt2 = build_verse_system_prompt(store, alice_id, "a traveller")
        assert "(no other avatars present)" in prompt2
        assert "nowhere in particular" in prompt2

    def test_unknown_avatar_id_raises(self, store: VerseStore) -> None:
        with pytest.raises(ValueError, match="avatar not found"):
            build_verse_system_prompt(store, 99999, "something")

    def test_personality_overlay_carries_only_scene_context(self, store: VerseStore) -> None:
        """The verse-mode tool rules and length-cap exception live in
        the verse framework (VERSE_SYSTEM_PROMPT in prompts.py — see
        test_prompts.py for the framework/overlay separation invariant),
        not in the personality overlay. The overlay carries identity,
        persona, scene, recent events, and other avatars — pure context.
        This split keeps the cacheable framework prefix stable across
        turns; verse-only rules ship with the verse framework, and the
        chat framework deliberately omits them."""
        alice_id = _opt_in(store, nick="alice")
        prompt = build_verse_system_prompt(store, alice_id, "a traveller")
        # Scene context is present.
        assert "You are alice" in prompt
        assert "Persona" in prompt
        assert "Scene" in prompt
        assert "Recent events involving you" in prompt
        # Tool rules and length-cap exception live in the framework, not here.
        assert "verse_record" not in prompt
        assert "HARD RULE" not in prompt
        assert "length cap" not in prompt


# ---------------------------------------------------------------------------
# TestOOC
# ---------------------------------------------------------------------------


class TestOOC:
    def test_simple_ooc_wrap(self) -> None:
        assert is_ooc("((hi))") is True

    def test_no_wrap(self) -> None:
        assert is_ooc("hi") is False

    def test_only_prefix(self) -> None:
        assert is_ooc("((hi") is False

    def test_only_suffix(self) -> None:
        assert is_ooc("hi))") is False

    def test_whitespace_around(self) -> None:
        assert is_ooc("  ((hi))  ") is True

    def test_empty_wrap(self) -> None:
        assert is_ooc("(())") is True

    def test_strip_ooc_removes_wrapper(self) -> None:
        assert strip_ooc("((hi))") == "hi"

    def test_strip_ooc_strips_inner_whitespace(self) -> None:
        assert strip_ooc("((  what model are you running?  ))") == ("what model are you running?")

    def test_strip_ooc_passthrough_when_not_wrapped(self) -> None:
        # Not OOC-wrapped — returned stripped but otherwise unchanged.
        assert strip_ooc("  hi  ") == "hi"
        assert strip_ooc("((hi") == "((hi"

    def test_strip_ooc_keeps_inner_parentheses(self) -> None:
        assert strip_ooc("(((nested)))") == "(nested)"

    # Leading // is an ergonomic OOC marker for one-off plain questions,
    # easier to type than wrapping a whole message in ((double parens)).
    def test_slash_prefix_is_ooc(self) -> None:
        assert is_ooc("// what's the weather?") is True

    def test_slash_is_ooc_only_when_leading(self) -> None:
        # // must be the prefix; mid-message it is ordinary text.
        assert is_ooc("go // there") is False
        assert is_ooc("http://example.com") is False

    def test_strip_slash_prefix(self) -> None:
        assert strip_ooc("// what's the weather?") == "what's the weather?"

    def test_strip_slash_not_leading_passthrough(self) -> None:
        # Not a leading // marker — returned stripped but otherwise unchanged.
        assert strip_ooc("go // there") == "go // there"


class TestMakeVerseToolSpecs:
    """Tests for make_verse_tool_specs() (C7c)."""

    def test_verse_act_required_params(self) -> None:
        """verse_act must require only 'verb'; target and details are optional."""
        specs = make_verse_tool_specs()
        act_spec = next(s for s in specs if s["function"]["name"] == "verse_act")
        required = act_spec["function"]["parameters"]["required"]
        assert required == ["verb"]
        props = act_spec["function"]["parameters"]["properties"]
        assert "verb" in props
        assert "target" in props
        assert "details" in props

    def test_verse_move_required_params(self) -> None:
        """verse_move must require 'place_name'."""
        specs = make_verse_tool_specs()
        move_spec = next(s for s in specs if s["function"]["name"] == "verse_move")
        assert move_spec["function"]["parameters"]["required"] == ["place_name"]

    def test_verse_recall_required_params(self) -> None:
        """verse_recall must require 'query'."""
        specs = make_verse_tool_specs()
        recall_spec = next(s for s in specs if s["function"]["name"] == "verse_recall")
        assert recall_spec["function"]["parameters"]["required"] == ["query"]

    def test_verse_look_no_required_params(self) -> None:
        """verse_look has no required params (target is optional)."""
        specs = make_verse_tool_specs()
        look_spec = next(s for s in specs if s["function"]["name"] == "verse_look")
        assert look_spec["function"]["parameters"]["required"] == []


# ---------------------------------------------------------------------------
# TestVerseToolDispatch (C7d)
# ---------------------------------------------------------------------------


class TestVerseToolDispatch:
    """Tests for dispatch_verse_tool_call and make_verse_extra_handlers (C7d)."""

    def test_db_deleted_mid_dispatch_logs_warning_no_raise(
        self, store: VerseStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN OperationalError on write WHEN dispatch called THEN WARNING logged, no raise."""
        import sqlite3
        from contextlib import contextmanager
        from unittest.mock import patch

        alice_id = _opt_in(store)

        # Simulate the DB becoming unavailable mid-session by making
        # write_transaction raise OperationalError (mirrors DB deletion / disk full).
        @contextmanager
        def _boom():  # type: ignore[return]
            raise sqlite3.OperationalError("disk I/O error")
            yield  # pragma: no cover

        with (
            caplog.at_level(logging.WARNING, logger="llm.verse.avatar"),
            patch.object(store, "write_transaction", _boom),
        ):
            # Should not raise
            dispatch_verse_tool_call(store, alice_id, "verse_act", {"verb": "speak"})

        assert any(r.levelno == logging.WARNING for r in caplog.records)

    def test_multiple_tool_calls_one_fails_others_apply(
        self, store: VerseStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """GIVEN 3 dispatches (good, bad-retired, good) THEN 2 events written, 1 WARNING."""
        alice_id = _opt_in(store)
        bob_id = _opt_in(store, nick="bob")

        # Retire alice so her dispatch fails
        store.unlink_avatar(alice_id)
        events_before = len(store.recent_events(limit=100))

        with caplog.at_level(logging.WARNING, logger="llm.verse.avatar"):
            # bob speak — succeeds
            dispatch_verse_tool_call(store, bob_id, "verse_act", {"verb": "speak"})
            # alice speak — fails (retired)
            dispatch_verse_tool_call(store, alice_id, "verse_act", {"verb": "speak"})
            # bob whisper — succeeds
            dispatch_verse_tool_call(store, bob_id, "verse_act", {"verb": "whisper"})

        events_after = store.recent_events(limit=100)
        new_events = len(events_after) - events_before
        assert new_events == 2  # only bob's two succeed
        warning_count = sum(1 for r in caplog.records if r.levelno == logging.WARNING)
        assert warning_count == 1

    def test_make_verse_extra_handlers_returns_four_handlers_with_content(
        self, store: VerseStore
    ) -> None:
        """GIVEN make_verse_extra_handlers THEN returns dict of 4 callables that return .content."""
        alice_id = _opt_in(store)
        handlers = make_verse_extra_handlers(store, alice_id)
        assert set(handlers.keys()) == {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
        }

        # Each handler is callable and returns an object with a .content attribute
        result = handlers["verse_act"]({"verb": "speak"})
        assert hasattr(result, "content")
        import json

        payload = json.loads(result.content)
        assert payload["status"] == "ok"
        assert payload["tool"] == "verse_act"

    def test_make_verse_denial_handlers_rejects_each_advertised_tool(self) -> None:
        """GIVEN verse tool schemas WHEN denial handlers are built THEN every
        tool name maps to a callable that returns ``{"error": ...}`` with
        opt-in onboarding text.

        Used on verse-enabled channels for speakers who haven't joined the
        verse — advertising the schemas keeps the channel's tool surface
        cache-stable, and these handlers turn any actual invocation into a
        rejection the model can self-correct on."""
        import json

        specs = make_verse_tool_specs()
        handlers = make_verse_denial_handlers(specs)

        spec_names = {spec["function"]["name"] for spec in specs}
        assert set(handlers.keys()) == spec_names

        for name, handler in handlers.items():
            result = handler({"unused": "args"})
            assert hasattr(result, "content"), name
            payload = json.loads(result.content)
            assert "error" in payload, name
            assert "opt-in" in payload["error"].lower(), name

    def test_make_verse_denial_handlers_skips_malformed_specs(self) -> None:
        """Malformed schema entries are skipped rather than blowing up — the
        dispatcher passes whatever extra_tools_override carries and we don't
        want a stray entry to crash the request path."""
        bad_specs: list[dict] = [
            {"type": "function"},  # missing "function" body
            {"function": {}},  # missing "name"
            {"function": {"name": "verse_act"}},  # only this one is valid
        ]
        handlers = make_verse_denial_handlers(bad_specs)
        assert set(handlers.keys()) == {"verse_act"}


# ---------------------------------------------------------------------------
# TestDispatchContract (Task 0a.1 — VerseDispatchResult)
# ---------------------------------------------------------------------------


class TestDispatchContract:
    def test_mutation_tools_return_ok_result(self, store: VerseStore) -> None:
        """GIVEN the mutation tools WHEN dispatched THEN returns
        VerseDispatchResult(ok=True, payload={'status':'ok'}). The wrapper's
        observable JSON is unchanged so the model's tool-result payloads
        do not regress."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            dispatch_verse_tool_call,
        )

        alice_id = _opt_in(store)
        store.add_entity("place", "Riverside", "A bend in the river.")
        result = dispatch_verse_tool_call(
            store, alice_id, "verse_move", {"place_name": "riverside"}
        )
        assert isinstance(result, VerseDispatchResult)
        assert result == VerseDispatchResult(ok=True, payload={"status": "ok"})

        result = dispatch_verse_tool_call(store, alice_id, "verse_act", {"verb": "speak"})
        assert result == VerseDispatchResult(
            ok=True, payload={"status": "ok", "result": "You speak."}
        )

    def test_verse_move_to_unknown_place_is_not_ok(self, store: VerseStore) -> None:
        """A move to a place that does not exist must not be reported as a
        success — the model would narrate a move that never happened."""
        from llm.verse.avatar import VerseDispatchResult, dispatch_verse_tool_call

        alice_id = _opt_in(store)
        result = dispatch_verse_tool_call(store, alice_id, "verse_move", {"place_name": "anywhere"})
        assert result == VerseDispatchResult(ok=False, error="no such place")
        assert store.get_attribute(alice_id, "location") == "The Clearing"

    def test_verse_act_failure_text_reaches_model(self, store: VerseStore) -> None:
        """verse_act's in-world failure text ('You can't find that place.')
        is carried in the payload instead of being dropped for a bare ok."""
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        result = dispatch_verse_tool_call(
            store, alice_id, "verse_act", {"verb": "move", "target": "Nowhere"}
        )
        assert result.payload == {"status": "ok", "result": "You can't find that place."}

    def test_verse_act_retired_avatar_is_not_ok(self, store: VerseStore) -> None:
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        store.set_status(alice_id, "retired")
        result = dispatch_verse_tool_call(store, alice_id, "verse_act", {"verb": "speak"})
        assert result.ok is False
        assert result.error == "avatar retired"

    def test_verse_recall_returns_events_payload(self, store: VerseStore) -> None:
        """verse_recall no longer swallows its result: the dispatch payload
        carries the recalled events (≤5, summary + ts), newest first."""
        from llm.verse.avatar import dispatch_verse_tool_call

        alice_id = _opt_in(store)
        for i in range(7):
            store.add_event(summary=f"alpha event {i}", entity_ids=[], source="avatar")
            time.sleep(0.01)
        result = dispatch_verse_tool_call(store, alice_id, "verse_recall", {"query": "alpha"})
        assert result.ok is True
        assert result.payload is not None
        events = result.payload["events"]
        assert len(events) == 5
        assert all(set(e) == {"summary", "ts"} for e in events)
        assert events[0]["summary"] == "alpha event 6"  # newest first

    def test_verse_record_on_retired_avatar_returns_error(self, store: VerseStore) -> None:
        """A retired avatar's verse_record must fail loudly (ok=False with a
        model-facing error) instead of silently swallowing the store's
        not-active ValueError — and must write no event."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            dispatch_verse_tool_call,
        )

        alice_id = _opt_in(store)
        store.unlink_avatar(alice_id)  # retire
        events_before = len(store.recent_events(limit=100))

        result = dispatch_verse_tool_call(
            store, alice_id, "verse_record", {"summary": "did a thing"}
        )

        assert isinstance(result, VerseDispatchResult)
        assert result.ok is False
        assert "retired" in (result.error or "")
        assert len(store.recent_events(limit=100)) == events_before

    def test_unknown_tool_returns_ok_with_warning(
        self, store: VerseStore, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Unknown tool name still doesn't raise; result is ok=True with
        the same payload (preserves today's silent-skip behaviour)."""
        from llm.verse.avatar import (
            VerseDispatchResult,
            dispatch_verse_tool_call,
        )

        alice_id = _opt_in(store)
        with caplog.at_level(logging.WARNING, logger="llm.verse.avatar"):
            result = dispatch_verse_tool_call(store, alice_id, "hallucinated_tool", {"x": 1})
        assert isinstance(result, VerseDispatchResult)
        assert result.ok is True
        assert result.payload == {"status": "ok"}


# ---------------------------------------------------------------------------
# TestHandlerConsumesResult (Task 0a.2 — wrapper surfaces VerseDispatchResult)
# ---------------------------------------------------------------------------


class TestHandlerConsumesResult:
    def test_handler_json_carries_look_description(self, store: VerseStore) -> None:
        """The model-visible verse_look JSON includes the description."""
        from llm.verse.avatar import make_verse_extra_handlers

        alice_id = _opt_in(store)  # opt_in places alice at The Clearing
        handlers = make_verse_extra_handlers(store, alice_id)
        payload = json.loads(handlers["verse_look"]({}).content)
        assert payload["status"] == "ok"
        assert payload["tool"] == "verse_look"
        assert payload["description"] == "A quiet woodland clearing where new stories begin."

    def test_handler_json_carries_recall_events(self, store: VerseStore) -> None:
        """The model-visible verse_recall JSON includes the recalled events."""
        from llm.verse.avatar import make_verse_extra_handlers

        alice_id = _opt_in(store)
        store.add_event(summary="Alice walks to the river", entity_ids=[], source="avatar")
        handlers = make_verse_extra_handlers(store, alice_id)
        payload = json.loads(handlers["verse_recall"]({"query": "river"}).content)
        assert payload["status"] == "ok"
        assert payload["tool"] == "verse_recall"
        assert [e["summary"] for e in payload["events"]] == ["Alice walks to the river"]
        assert all("ts" in e for e in payload["events"])

    def test_handler_emits_error_on_not_ok(
        self, store: VerseStore, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ok=False with error string — handler emits {'status':'error',
        'error': <error>} so the model sees a structured failure."""
        from llm.verse import avatar as avatar_mod

        def fake_dispatch(*a, **k):
            return avatar_mod.VerseDispatchResult(ok=False, error="summary required")

        monkeypatch.setattr(avatar_mod, "dispatch_verse_tool_call", fake_dispatch)
        alice_id = _opt_in(store)
        handlers = avatar_mod.make_verse_extra_handlers(store, alice_id)
        result = handlers["verse_act"]({"verb": "speak"})
        payload = json.loads(result.content)
        assert payload["status"] == "error"
        assert payload["error"] == "summary required"


class TestVerseRecordToolSpec:
    def test_make_verse_tool_specs_returns_six_with_default_max(self) -> None:
        from llm.verse.avatar import make_verse_tool_specs

        specs = make_verse_tool_specs()
        assert len(specs) == 6
        names = {s["function"]["name"] for s in specs}
        assert names == {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
            "verse_edit",
        }
        record = next(s for s in specs if s["function"]["name"] == "verse_record")
        params = record["function"]["parameters"]
        assert params["properties"]["actors"]["maxItems"] == 8
        assert params["required"] == ["summary"]

    def test_make_verse_tool_specs_max_actors_dynamic(self) -> None:
        from llm.verse.avatar import make_verse_tool_specs

        specs = make_verse_tool_specs(max_actors=12)
        record = next(s for s in specs if s["function"]["name"] == "verse_record")
        assert record["function"]["parameters"]["properties"]["actors"]["maxItems"] == 12
