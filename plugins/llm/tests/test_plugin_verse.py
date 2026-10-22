"""Plugin verse: verse commands, avatars, routing, compaction."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import pytest
from llm.identity import Caller, Identity
from llm.plugin import _VERSE_COMPACTION_MODEL, PreflightResult
from llm.service import AssistantResult, Spend

from .conftest import make_completion_response
from .verse.conftest import link_avatar

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


# =============================================================================
# TestVerseoptCommand
# =============================================================================


class TestVerseoptCommand:
    """Tests for the @verseopt in/out command (C3).

    Strategy: use the ``plugin_env`` fixture (mocked LLMService/DB) plus a
    real VerseStore backed by ``tmp_path`` so the avatar table round-trips
    through actual SQLite.  We swap ``plugin._get_or_create_verse_store`` with
    a factory that always returns the same real store, giving us full
    DB coverage without touching production paths.
    """

    SCENE_PREFIX = "You step into The Clearing."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        # Patch store lookup so the command uses our real store.
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Enable verse for the channel.
        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Capability: user has llm.verse.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Instruct text and avatar persona for the user (empty by default).
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        # msg arrives in #afnet
        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Branch 1: verseopt in — happy path (new avatar)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Helper: invoke the wrapped command correctly.
    # wrap(verseopt, [("checkCapability", "llm.verse"), ("literal", ("in", "out"))])
    # produces a callable (irc, msg, args) where args is the token list.
    # ------------------------------------------------------------------

    @staticmethod
    def _call(plugin, irc, msg, mode: str) -> None:
        """Call the wrapped verseopt command with the given mode token."""
        plugin.verseopt(irc, msg, [mode])

    # ------------------------------------------------------------------
    # Branch 1: verseopt in — happy path (new avatar)
    # ------------------------------------------------------------------

    def test_verseopt_in_new_avatar_replies_scene(self, verse_env) -> None:
        """GIVEN verse-enabled channel and llm.verse WHEN @verseopt in THEN scene text replied."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert self.SCENE_PREFIX in reply_text
        assert "You are already opted in." not in reply_text

    def test_verseopt_in_new_avatar_created_in_store(self, verse_env) -> None:
        """GIVEN new opt-in WHEN @verseopt in THEN avatar entity exists in store."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")

        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None

    def test_verseopt_in_uses_avatar_persona(self, verse_env, mocker) -> None:
        """GIVEN user has @avatar set WHEN @verseopt in THEN opt_in_avatar called with persona."""
        plugin, irc, msg, store = verse_env
        irc.state.nickToAccount.return_value = "alice"
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value="Be a wizard.")

        spy = mocker.patch.object(store, "opt_in_avatar", wraps=store.opt_in_avatar)

        self._call(plugin, irc, msg, "in")

        spy.assert_called_once()
        _, _, persona_arg = spy.call_args[0]
        assert persona_arg == "Be a wizard."

    def test_verseopt_in_accountless_gets_no_persona(self, verse_env, mocker) -> None:
        """GIVEN no account WHEN @verseopt in THEN no @avatar is read: keyed on
        the nick, it could be another user's account persona."""
        plugin, irc, msg, store = verse_env
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value="Be a wizard.")
        spy = mocker.patch.object(store, "opt_in_avatar", wraps=store.opt_in_avatar)

        self._call(plugin, irc, msg, "in")

        plugin.db.get_avatar_persona.assert_not_called()
        assert spy.call_args[0][2] == ""

    def test_verseopt_in_ignores_instruction(self, verse_env, mocker) -> None:
        """GIVEN @instruct set but no @avatar WHEN @verseopt in THEN persona is empty.

        The split: @instruct only shapes %ask. The verse must read @avatar.
        """
        plugin, irc, msg, store = verse_env
        plugin.db.get_instruction = mocker.MagicMock(return_value="ask voice")
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        spy = mocker.patch.object(store, "opt_in_avatar", wraps=store.opt_in_avatar)

        self._call(plugin, irc, msg, "in")

        spy.assert_called_once()
        _, _, persona_arg = spy.call_args[0]
        assert persona_arg == ""

    # ------------------------------------------------------------------
    # Branch 2: verseopt in — already opted in
    # ------------------------------------------------------------------

    def test_verseopt_in_already_opted_in_prefixes_message(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt in again THEN reply prefixed with already-in message."""
        plugin, irc, msg, store = verse_env

        # First opt-in creates the avatar.
        self._call(plugin, irc, msg, "in")
        irc.reply.reset_mock()

        # Second opt-in should detect was_already_opted_in=True.
        self._call(plugin, irc, msg, "in")

        reply_text = irc.reply.call_args[0][0]
        assert reply_text.startswith("You are already opted in. ")
        assert self.SCENE_PREFIX in reply_text

    # ------------------------------------------------------------------
    # Branch 3: verseopt in after verseopt out — reactivation
    # ------------------------------------------------------------------

    def test_verseopt_in_after_out_reactivates_without_prefix(self, verse_env) -> None:
        """GIVEN retired avatar WHEN @verseopt in THEN was_already_opted_in=False, no prefix."""
        plugin, irc, msg, store = verse_env

        # Create avatar.
        self._call(plugin, irc, msg, "in")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None

        # Retire it.
        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")
        assert store.find_avatar_by_nick("alice") is None

        # Reactivate.
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@verseopt in")
        self._call(plugin, irc, msg, "in")

        reply_text = irc.reply.call_args[0][0]
        assert not reply_text.startswith("You are already opted in.")
        assert self.SCENE_PREFIX in reply_text

    # ------------------------------------------------------------------
    # Branch 4: verseopt out
    # ------------------------------------------------------------------

    def test_verseopt_out_removes_avatar_link(self, verse_env) -> None:
        """GIVEN active avatar WHEN @verseopt out THEN find_avatar_by_nick returns None."""
        plugin, irc, msg, store = verse_env

        self._call(plugin, irc, msg, "in")
        assert store.find_avatar_by_nick("alice") is not None

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        assert store.find_avatar_by_nick("alice") is None

    def test_verseopt_out_with_no_avatar_replies_not_found(self, verse_env) -> None:
        """GIVEN no avatar WHEN @verseopt out THEN 'no avatar' reply."""
        plugin, irc, msg, store = verse_env

        msg.args = ("#afnet", "@verseopt out")
        self._call(plugin, irc, msg, "out")

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "You don't have an avatar in this channel."

    # ------------------------------------------------------------------
    # Branch 5: verseEnabled=False
    # ------------------------------------------------------------------

    def test_verseopt_in_disabled_channel_replies_not_enabled(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @verseopt in THEN not-enabled message."""
        plugin, irc, msg = plugin_env

        # verse disabled (default in make_registry_side_effect)
        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == (
            "This channel doesn't have a verse. Ask the operator to set verseEnabled."
        )

    def test_verseopt_in_disabled_channel_no_store_created(
        self, plugin_env, mocker, tmp_path
    ) -> None:
        """GIVEN channel without verse WHEN @verseopt in THEN no VerseStore is instantiated."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        store_cls = mocker.patch("llm.plugin.VerseStore")

        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])

        store_cls.assert_not_called()

    # ------------------------------------------------------------------
    # Branch 6: missing llm.verse capability
    # ------------------------------------------------------------------


# =============================================================================
# TestVerseCommand
# =============================================================================


class TestVerseCommand:
    """Tests for the @verse command (C4).

    Uses the same verse_env fixture pattern as TestVerseoptCommand.
    """

    PLACE_NAME = "The Clearing"
    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verse")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(plugin, irc, msg, store) -> int:
        """Opt alice in and return her entity_id."""
        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@verse")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None
        return entity_id

    # ------------------------------------------------------------------
    # Branch 1: avatar present, location set → scene one-liner
    # ------------------------------------------------------------------

    def test_verse_with_location_returns_scene_oneliner(self, verse_env) -> None:
        """GIVEN avatar with location WHEN @verse THEN 'You are at <place>.' reply."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)

        # Add a place and set avatar's location attribute. The store keeps
        # the place NAME in 'location' (opt_in_avatar / verse_move), not an id.
        store.add_entity("place", "The Mill", "A sunlit glade.")
        store.set_attribute(entity_id, "location", "The Mill")

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "You are at The Mill." in reply_text
        assert "A sunlit glade." in reply_text

    def test_a_retired_place_of_the_same_name_is_never_the_scene(self, verse_env) -> None:
        """Dedup leaves retired duplicates behind; the scene is the live one."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)
        retired = store.add_entity("place", "The Docks", "Burned down years ago.")
        store.set_status(retired, "retired")
        store.add_entity("place", "The Docks", "Gulls and rope.")
        store.set_attribute(entity_id, "location", "The Docks")

        plugin.verse(irc, msg, [])

        assert irc.reply.call_args[0][0] == "You are at The Docks. Gulls and rope."

    def test_a_location_naming_only_a_retired_place_is_nowhere(self, verse_env) -> None:
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)
        store.set_status(store.add_entity("place", "The Docks", "Burned down."), "retired")
        store.set_attribute(entity_id, "location", "The Docks")

        plugin.verse(irc, msg, [])

        assert irc.reply.call_args[0][0] == "You are nowhere in particular."

    def test_verse_after_opt_in_names_starter_place(self, verse_env) -> None:
        """GIVEN a fresh opt-in WHEN @verse THEN the reply names the starter place
        opt_in_avatar wrote, not 'nowhere in particular'."""
        plugin, irc, msg, store = verse_env
        self._opt_in(plugin, irc, msg, store)

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text.startswith(f"You are at {self.PLACE_NAME}.")

    # ------------------------------------------------------------------
    # Branch 2: avatar present, no location → "nowhere in particular"
    # ------------------------------------------------------------------

    def test_verse_no_location_returns_nowhere(self, verse_env) -> None:
        """GIVEN avatar whose location names no active place WHEN @verse THEN
        'nowhere in particular' reply."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)
        store.set_attribute(entity_id, "location", "Atlantis")

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "You are nowhere in particular."

    # ------------------------------------------------------------------
    # Branch 3: no avatar → no-avatar message
    # ------------------------------------------------------------------

    def test_verse_no_avatar_replies_no_avatar(self, verse_env) -> None:
        """GIVEN no avatar WHEN @verse THEN no-avatar message."""
        plugin, irc, msg, store = verse_env

        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_AVATAR_REPLY

    # ------------------------------------------------------------------
    # Branch 4: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    def test_verse_disabled_channel_replies_no_verse(self, plugin_env, mocker) -> None:
        """GIVEN channel without verse WHEN @verse THEN no-verse message."""
        plugin, irc, msg = plugin_env

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": False})
        )
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@verse")
        plugin.verse(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_VERSE_REPLY

    # ------------------------------------------------------------------
    # Branch 5: lacking llm.verse capability → Limnoria denial
    # ------------------------------------------------------------------

    def test_verse_no_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse WHEN @verse THEN Limnoria capability denial."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            return_value=False,
        )

        from tests.conftest import make_registry_side_effect

        plugin.registryValue = mocker.MagicMock(
            side_effect=make_registry_side_effect({"verseEnabled": True})
        )

        msg.args = ("#afnet", "@verse")
        plugin.verse(irc, msg, [])

        # Limnoria's ("checkCapability", "llm.verse") wrap gate denies the
        # command before the body runs: it emits errorNoCapability and never
        # replies with a scene. Asserting BOTH sides makes this fail if the
        # gate is removed — the body would then reach NO_AVATAR_REPLY via
        # irc.reply for the unopted user.
        irc.errorNoCapability.assert_called_once()
        irc.reply.assert_not_called()


# =============================================================================
# TestLookCommand
# =============================================================================


class TestLookCommand:
    """Tests for the @look [target] command (C4)."""

    PLACE_NAME = "The Clearing"
    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."
    NO_AVATAR_REPLY = "You don't have an avatar in this channel. Use @verseopt in to join."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@look")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(plugin, irc, msg, store) -> int:
        """Opt alice in and return her entity_id."""
        msg.args = ("#afnet", "@verseopt in")
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()
        msg.args = ("#afnet", "@look")
        entity_id = store.find_avatar_by_nick("alice")
        assert entity_id is not None
        return entity_id

    # ------------------------------------------------------------------
    # Branch 6: no target, avatar present → scene one-liner
    # ------------------------------------------------------------------

    def test_look_no_target_with_avatar_returns_scene(self, verse_env) -> None:
        """GIVEN avatar with location, no target WHEN @look THEN scene one-liner."""
        plugin, irc, msg, store = verse_env
        entity_id = self._opt_in(plugin, irc, msg, store)

        store.add_entity("place", "The Mill", "A sunlit glade.")
        store.set_attribute(entity_id, "location", "The Mill")

        # Wrapped: args=[] means no optional target → target=None inside handler.
        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "You are at The Mill." in reply_text

    # ------------------------------------------------------------------
    # Branch 7: no target, no avatar → no-avatar message
    # ------------------------------------------------------------------

    def test_look_no_target_no_avatar_replies_no_avatar(self, verse_env) -> None:
        """GIVEN no avatar, no target WHEN @look THEN no-avatar message."""
        plugin, irc, msg, store = verse_env

        plugin.look(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == self.NO_AVATAR_REPLY

    # ------------------------------------------------------------------
    # Branch 8: target matches → entity description
    # ------------------------------------------------------------------

    def test_look_target_matches_returns_description(self, verse_env) -> None:
        """GIVEN named entity, target given WHEN @look <name> THEN entity description."""
        plugin, irc, msg, store = verse_env
        store.add_entity("place", self.PLACE_NAME, "A sunlit glade.")

        # Wrapped: target string goes into args list.
        plugin.look(irc, msg, [self.PLACE_NAME])

        reply_text = irc.reply.call_args[0][0]
        assert self.PLACE_NAME in reply_text
        assert "A sunlit glade." in reply_text

    # ------------------------------------------------------------------
    # Branch 9: target doesn't match → "Nothing matches."
    # ------------------------------------------------------------------

    def test_look_target_no_match_replies_nothing_matches(self, verse_env) -> None:
        """GIVEN target with no matching entity WHEN @look <name> THEN 'Nothing matches.'"""
        plugin, irc, msg, store = verse_env

        plugin.look(irc, msg, ["Atlantis"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Nothing matches."

    # ------------------------------------------------------------------
    # Branch 10: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Branch 11: lacking capability → denial
    # ------------------------------------------------------------------


# =============================================================================
# TestWhoCommand
# =============================================================================


class TestWhoCommand:
    """Tests for the @who command (C4)."""

    NO_VERSE_REPLY = "This channel doesn't have a verse. Ask the operator to set verseEnabled."

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@who")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Branch 12: multiple active avatars with locations → comma-joined list
    # ------------------------------------------------------------------

    def test_who_multiple_avatars_returns_list(self, verse_env) -> None:
        """GIVEN two active avatars with locations WHEN @who THEN comma-joined list."""
        plugin, irc, msg, store = verse_env

        # Create two avatars and a place.
        store.add_entity("place", "The Clearing", "A sunlit glade.")
        alice_id = store.add_entity("avatar", "alice")
        bob_id = store.add_entity("avatar", "bob")
        link_avatar(store, alice_id, "alice")
        link_avatar(store, bob_id, "bob")
        store.set_attribute(alice_id, "location", "The Clearing")
        store.set_attribute(bob_id, "location", "The Clearing")

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "alice (at The Clearing)" in reply_text
        assert "bob (at The Clearing)" in reply_text
        # Comma-joined format.
        assert ", " in reply_text

    # ------------------------------------------------------------------
    # Branch 13: no avatars → "Nobody is opted in here yet."
    # ------------------------------------------------------------------

    def test_who_no_avatars_replies_empty(self, verse_env) -> None:
        """GIVEN no active avatars WHEN @who THEN 'Nobody is opted in here yet.'"""
        plugin, irc, msg, store = verse_env

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Nobody is opted in here yet."

    # ------------------------------------------------------------------
    # Branch 14: retired avatars excluded
    # ------------------------------------------------------------------

    def test_who_retired_avatar_excluded(self, verse_env) -> None:
        """GIVEN one active and one retired avatar WHEN @who THEN only active shown."""
        plugin, irc, msg, store = verse_env

        active_id = store.add_entity("avatar", "alice")
        link_avatar(store, active_id, "alice")

        retired_id = store.add_entity("avatar", "bob")
        link_avatar(store, retired_id, "bob")
        store.set_status(retired_id, "retired")

        plugin.who(irc, msg, [])

        reply_text = irc.reply.call_args[0][0]
        assert "alice" in reply_text
        assert "bob" not in reply_text

    # ------------------------------------------------------------------
    # Branch 15: channel without verseEnabled → "no verse" message
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Branch 16: lacking capability → denial
    # ------------------------------------------------------------------


# =============================================================================
# TestVersepurgeCommand
# =============================================================================


class TestVersepurgeCommand:
    """Tests for the @versepurge command (C5).

    Strategy: direct method calls (bypassing wrap), with a real VerseStore
    backed by tmp_path. Token logic is exercised via a ``now_func`` shim passed
    to ``_versepurge_check_token`` so we can fake the clock without monkeypatching.
    """

    @pytest.fixture
    def purge_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore wired for #afnet, with gm capability."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Wire the store into the cache so versepurge can pop it.
        plugin._verse_stores["#afnet"] = store

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Grant llm.verse.gm.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@versepurge #afnet")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Test 1: first call issues token
    # ------------------------------------------------------------------

    def test_first_call_issues_token(self, purge_env) -> None:
        """GIVEN no existing token WHEN @versepurge #afnet THEN 6-char token issued."""
        plugin, irc, msg, store = purge_env

        plugin.versepurge(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "Confirm with @versepurge #afnet" in reply_text
        assert "within 60s" in reply_text

        assert "#afnet" in plugin._versepurge_tokens
        stored_token, expires_at = plugin._versepurge_tokens["#afnet"]
        assert len(stored_token) == 6  # token_hex(3) → 6 hex chars
        import time as _time

        assert expires_at > _time.time()

    def test_no_channel_argument_targets_this_channel(self, purge_env) -> None:
        """In-channel @versepurge with no #chan issues the token for, and
        confirms against, the channel it was issued in."""
        plugin, irc, msg, _store = purge_env

        plugin.versepurge(irc, msg, [])

        assert "Confirm with @versepurge #afnet" in irc.reply.call_args[0][0]
        assert list(plugin._versepurge_tokens) == ["#afnet"]
        token, _ = plugin._versepurge_tokens["#afnet"]
        plugin.versepurge(irc, msg, [token])
        assert irc.reply.call_args[0][0] == "Verse for #afnet purged."

    # ------------------------------------------------------------------
    # Test 2: correct token purges
    # ------------------------------------------------------------------

    def test_second_call_correct_token_purges(self, purge_env, tmp_path) -> None:
        """GIVEN valid token WHEN @versepurge #afnet <token> THEN purged reply, store gone."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        # Step 1: issue token.
        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Step 2: confirm.
        plugin.versepurge(irc, msg, [f"#afnet {stored_token}"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Verse for #afnet purged."

        # Store evicted from cache.
        assert "#afnet" not in plugin._verse_stores

        # DB file removed.
        assert not db_path.exists()

        # Token cleared.
        assert "#afnet" not in plugin._versepurge_tokens

    def test_purge_holds_store_lock_across_unlink(self, purge_env, mocker) -> None:
        """The DB unlink must run WHILE _verse_stores_lock is held.

        The plugin is threaded=True, so a concurrent _get_or_create_verse_store
        (verse @ask on a SupyThread / a loom worker) could otherwise reconstruct
        a store on the files mid-purge and leave a half-written DB behind. Both
        the purge and the constructor take _verse_stores_lock, so the unlink must
        happen inside it.
        """
        from pathlib import Path

        plugin, irc, msg, store = purge_env

        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]

        lock_held_during_unlink: list[bool] = []
        real_unlink = Path.unlink

        def _record(self_path, *a, **kw):
            lock_held_during_unlink.append(plugin._verse_stores_lock.locked())
            return real_unlink(self_path, *a, **kw)

        mocker.patch.object(Path, "unlink", autospec=True, side_effect=_record)

        plugin.versepurge(irc, msg, [f"#afnet {stored_token}"])

        # unlink actually fired and every call saw the lock held.
        assert lock_held_during_unlink
        assert all(lock_held_during_unlink)

    # ------------------------------------------------------------------
    # Test 3: wrong token rejected
    # ------------------------------------------------------------------

    def test_second_call_wrong_token_rejected(self, purge_env) -> None:
        """GIVEN valid token WHEN confirmed with garbage token THEN rejected."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        plugin.versepurge(irc, msg, ["#afnet"])
        irc.reply.reset_mock()

        plugin.versepurge(irc, msg, ["#afnet garbage"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "Token expired or invalid" in reply_text
        assert "Run @versepurge #afnet again" in reply_text

        # DB still exists; token NOT cleared (wrong token doesn't clear unexpired entry).
        assert db_path.exists()
        assert "#afnet" in plugin._versepurge_tokens

    def test_a_token_nobody_issued_purges_nothing(self, purge_env) -> None:
        """Skipping the first step must not skip the confirmation."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        plugin.versepurge(irc, msg, ["#afnet deadbe"])

        assert "Token expired or invalid" in irc.reply.call_args[0][0]
        assert db_path.exists()
        assert "#afnet" in plugin._verse_stores

    # ------------------------------------------------------------------
    # Test 4: token expires after 60s
    # ------------------------------------------------------------------

    def test_token_expires_after_60s(self, purge_env) -> None:
        """GIVEN token issued WHEN 60s pass and correct token presented THEN expired."""
        plugin, irc, msg, store = purge_env
        db_path = store.path

        plugin.versepurge(irc, msg, ["#afnet"])
        stored_token, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Use _versepurge_check_token directly with a fake clock past expiry.
        import time as _time

        future_now = _time.time() + 61.0
        result = plugin._versepurge_check_token("#afnet", stored_token, now_func=lambda: future_now)
        assert result is False

        # Stale entry should be cleared.
        assert "#afnet" not in plugin._versepurge_tokens

        # DB still exists (no purge occurred).
        assert db_path.exists()

    # ------------------------------------------------------------------
    # Test 5: reissue within window invalidates old token
    # ------------------------------------------------------------------

    def test_reissue_within_window_invalidates_old(self, purge_env) -> None:
        """GIVEN unexpired T1 WHEN @versepurge #afnet called again THEN T2 issued, T1 invalid."""
        plugin, irc, msg, store = purge_env

        # Issue T1.
        plugin.versepurge(irc, msg, ["#afnet"])
        token_t1, _ = plugin._versepurge_tokens["#afnet"]
        irc.reply.reset_mock()

        # Issue T2 (while T1 still valid).
        plugin.versepurge(irc, msg, ["#afnet"])
        reply_text = irc.reply.call_args[0][0]
        assert "Previous token invalidated" in reply_text or "invalidated" in reply_text

        token_t2, _ = plugin._versepurge_tokens["#afnet"]
        assert token_t2 != token_t1

        # T1 no longer valid.
        result = plugin._versepurge_check_token("#afnet", token_t1)
        assert result is False

    # ------------------------------------------------------------------
    # Test 6: purge channel with no store yet is a no-op
    # ------------------------------------------------------------------

    def test_purge_for_channel_with_no_store_yet(self, purge_env) -> None:
        """GIVEN channel never had a store WHEN confirmed THEN 'purged' reply, no error."""
        plugin, irc, msg, store = purge_env

        # Issue token for a channel that has no store in cache.
        plugin._verse_stores.pop("#newchan", None)
        plugin.versepurge(irc, msg, ["#newchan"])
        token, _ = plugin._versepurge_tokens["#newchan"]
        irc.reply.reset_mock()

        plugin.versepurge(irc, msg, [f"#newchan {token}"])

        reply_text = irc.reply.call_args[0][0]
        assert reply_text == "Verse for #newchan purged."
        assert "#newchan" not in plugin._versepurge_tokens

    def test_purge_after_restart_unlinks_files_on_disk(self, purge_env, tmp_path, mocker) -> None:
        """GIVEN a verse DB on disk but no store in the cache (stores open
        lazily, so this is every channel after a restart) WHEN purge is
        confirmed THEN the DB and its -wal/-shm files are unlinked."""
        from llm.verse.store import VerseStore

        plugin, irc, msg, _store = purge_env
        data_dir = tmp_path / "verse-data"
        mocker.patch.object(plugin, "_verse_data_dir", return_value=data_dir)
        on_disk = VerseStore(data_dir, "#Restarted")
        on_disk.add_entity("npc", "Bob")
        db = on_disk.path
        db.with_suffix(".db-wal").touch()
        db.with_suffix(".db-shm").touch()
        plugin._verse_stores.pop("#restarted", None)

        plugin.versepurge(irc, msg, ["#Restarted"])
        token, _ = plugin._versepurge_tokens["#Restarted"]
        plugin.versepurge(irc, msg, [f"#Restarted {token}"])

        assert irc.reply.call_args[0][0] == "Verse for #Restarted purged."
        assert not db.exists()
        assert not db.with_suffix(".db-wal").exists()
        assert not db.with_suffix(".db-shm").exists()

    # ------------------------------------------------------------------
    # Test 7: lacking capability is denied
    # ------------------------------------------------------------------

    def test_lacking_capability_denied(self, plugin_env, mocker) -> None:
        """GIVEN user lacks llm.verse.gm WHEN @versepurge THEN error, no token issued."""
        plugin, irc, msg = plugin_env

        mocker.patch(
            "supybot.commands.ircdb.checkCapability",
            return_value=False,
        )

        plugin._versepurge_tokens.clear()
        msg.args = ("#afnet", "@versepurge #afnet")
        plugin.versepurge(irc, msg, ["#afnet"])

        irc.errorNoCapability.assert_called_once()
        assert "#afnet" not in plugin._versepurge_tokens

    # ------------------------------------------------------------------
    # Test 8: llm.verse (not gm) is denied
    # ------------------------------------------------------------------

    def test_purge_unrelated_to_versedump_capability(self, plugin_env, mocker) -> None:
        """GIVEN user has llm.verse but NOT llm.verse.gm WHEN @versepurge THEN denied."""
        plugin, irc, msg = plugin_env

        # Only grant llm.verse (not llm.verse.gm) — at the wrap layer.
        mocker.patch(
            "supybot.commands.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap == "llm.verse",
        )

        plugin._versepurge_tokens.clear()
        msg.args = ("#afnet", "@versepurge #afnet")
        plugin.versepurge(irc, msg, ["#afnet"])

        irc.errorNoCapability.assert_called_once()
        assert "#afnet" not in plugin._versepurge_tokens


# =============================================================================
# TestVersedumpCommand
# =============================================================================


class TestVersedumpCommand:
    """Tests for the @versedump command (C5).

    Strategy: real VerseStore via tmp_path; invoke plugin.versedump directly.
    pyyaml is not installed → yaml branch returns unsupported message.
    """

    @pytest.fixture
    def dump_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore for #afnet, gm capability granted."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Default: pastebin write fails → inline fallback path (preserves
        # the JSON-content assertions in tests written before the
        # publish-and-link change). Tests exercising the URL path
        # override this on the same mock.
        plugin.llm_service.save_markdown_to_http = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@versedump #afnet")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    # ------------------------------------------------------------------
    # Test 9: dump includes entities, attributes, relations, events, avatar_links
    # ------------------------------------------------------------------

    def test_dump_includes_entities_attributes_relations_events_avatar_link(self, dump_env) -> None:
        """GIVEN populated verse WHEN @versedump THEN JSON contains all sections."""
        import json as _json

        plugin, irc, msg, store = dump_env

        # Opt alice in (creates avatar + place + location attribute via opt_in_avatar).
        alice_id = store.opt_in_avatar("alice", None, "").entity_id

        # Add a relation.
        store.add_relation(alice_id, alice_id, "self-aware", "test note")

        # Add an event.
        store.add_event("alice did something", [alice_id], "avatar")

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        data = _json.loads(reply_text)

        assert "entities" in data
        assert len(data["entities"]) >= 1

        # Check attributes present on alice's entity.
        alice_entity = next((e for e in data["entities"] if e["id"] == alice_id), None)
        assert alice_entity is not None
        assert "attributes" in alice_entity

        assert "relations" in data
        assert len(data["relations"]) >= 1

        assert "events" in data
        assert len(data["events"]) >= 1

        assert "avatar_links" in data
        assert any(row["nick"] == "alice" for row in data["avatar_links"])

    # ------------------------------------------------------------------
    # Test 10: default format is JSON
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Test 11: yaml format returns unsupported (pyyaml not installed)
    # ------------------------------------------------------------------

    def test_dump_yaml_format_or_unsupported(self, dump_env) -> None:
        """WHEN @versedump #chan --format=yaml THEN unsupported message (pyyaml absent)."""
        plugin, irc, msg, store = dump_env

        # Pass --format=yaml in the text arg (space-separated from channel).
        plugin.versedump(irc, msg, ["#afnet --format=yaml"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        # pyyaml is not a dependency → unsupported message.
        assert "Unsupported format" in reply_text
        assert "json" in reply_text.lower()

    # ------------------------------------------------------------------
    # Test 12: lacking capability is denied
    # ------------------------------------------------------------------

    def test_no_channel_argument_dumps_this_channel(self, dump_env) -> None:
        plugin, irc, msg, _store = dump_env

        plugin.versedump(irc, msg, [])

        plugin._get_or_create_verse_store.assert_called_once_with("#afnet")
        irc.error.assert_not_called()

    def test_dump_is_a_faithful_backup_field_for_field(self, dump_env) -> None:
        """The dump is the backup taken before an irreversible purge, so every
        exported field must be the stored value, retired entities and aliases
        included."""
        import json as _json

        plugin, irc, msg, store = dump_env
        alice_id = store.opt_in_avatar("alice", None, "").entity_id
        sam_id = store.add_entity("npc", "Old Sam", "a sailor who swallowed a compass")
        store.set_status(sam_id, "retired")
        with store.write_transaction() as conn:
            conn.execute(
                "INSERT INTO entity_alias (entity_id, alias) VALUES (?, ?)", (sam_id, "sam")
            )
        with store.read_connection() as conn:
            created_at, updated_at = conn.execute(
                "SELECT created_at, updated_at FROM entities WHERE id = ?", (sam_id,)
            ).fetchone()

        plugin.versedump(irc, msg, ["#afnet"])

        data = _json.loads(irc.reply.call_args[0][0])
        sam = next(e for e in data["entities"] if e["id"] == sam_id)
        assert sam == {
            "id": sam_id,
            "kind": "npc",
            "name": "Old Sam",
            "summary": "a sailor who swallowed a compass",
            "status": "retired",
            "created_at": created_at,
            "updated_at": updated_at,
            "attributes": {},
        }
        alice = next(e for e in data["entities"] if e["id"] == alice_id)
        assert (alice["kind"], alice["name"], alice["status"]) == ("avatar", "alice", "active")
        assert data["avatar_links"] == [{"entity_id": alice_id, "nick": "alice", "account": None}]
        assert {"entity_id": sam_id, "alias": "sam"} in data["aliases"]

    def test_a_stray_word_does_not_pick_the_channel(self, dump_env) -> None:
        plugin, irc, msg, _store = dump_env

        plugin.versedump(irc, msg, ["foo"])

        plugin._get_or_create_verse_store.assert_called_once_with("#afnet")

    def test_named_channel_wins_over_this_one(self, dump_env) -> None:
        plugin, irc, msg, _store = dump_env

        plugin.versedump(irc, msg, ["#other"])

        plugin._get_or_create_verse_store.assert_called_once_with("#other")

    def test_no_channel_in_pm_is_an_error(self, dump_env) -> None:
        plugin, irc, msg, _store = dump_env
        msg.args = ("testbot", "@versedump")

        plugin.versedump(irc, msg, [])

        irc.error.assert_called_once_with("Specify a channel: @versedump #chan", prefixNick=False)
        plugin._get_or_create_verse_store.assert_not_called()

    def test_dump_publishes_to_pastebin_when_available(self, dump_env, mocker) -> None:
        """WHEN save_markdown_to_http returns URL THEN reply is just the URL.

        Avoids spamming IRC with a fat JSON line when the bot's HTTP
        pastebin is configured.
        """
        plugin, irc, msg, store = dump_env
        published_url = "http://example.com/llm/answer_abc123.html"
        plugin.llm_service.save_markdown_to_http.return_value = published_url

        plugin.versedump(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert published_url in reply_text
        # Body must have been wrapped as a fenced JSON markdown block so
        # the pastebin renders with syntax highlighting (and includes the
        # channel name in a header).
        plugin.llm_service.save_markdown_to_http.assert_called_once()
        markdown_arg = plugin.llm_service.save_markdown_to_http.call_args[0][0]
        assert "```json" in markdown_arg
        assert "#afnet" in markdown_arg


# ===========================================================================
# C6: @avatar double-write — avatar summary syncs with persona text.
# (Was @instruct; split out so @instruct only shapes %ask.)
# ===========================================================================


class TestAvatarVerseSync:
    """@avatar updates avatar.summary when channel is verse-enabled.

    Strategy: real VerseStore backed by tmp_path (no SQLite mocks).
    We patch ``plugin._get_or_create_verse_store`` to return the same real
    store used to assert post-condition, mirroring the TestVerseoptCommand
    pattern.
    """

    @pytest.fixture
    def avatar_verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        irc.state.nickToAccount.return_value = "alice"  # @avatar needs an account

        store = VerseStore(tmp_path / "verse", "#afnet")

        # Patch store lookup so the command uses our real store.
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        # Enable verse for #afnet, disable for everything else.
        def _registry(key, *args):
            if key == "verseEnabled":
                channel = args[0] if args else None
                return channel == "#afnet"
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        # Capability: user has llm.verse.
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # msg arrives in #afnet from alice, authenticated as alice.
        msg.args = ("#afnet", "@avatar hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Wire up db mocks for the avatar path.
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)
        plugin.db.save_avatar_persona = mocker.MagicMock()
        plugin.db.delete_avatar_persona = mocker.MagicMock(return_value=True)

        return plugin, irc, msg, store

    @staticmethod
    def _opt_in(store, nick: str, account: str | None = None) -> int:
        """Opt alice into the verse and return her entity_id."""
        result = store.opt_in_avatar(nick, account, instruct_text="")
        assert result.entity_id is not None
        return result.entity_id

    def test_bare_avatar_shows_the_persona(self, avatar_verse_env) -> None:
        plugin, irc, msg, _store = avatar_verse_env
        plugin.db.get_avatar_persona.return_value = "a sly fox"

        plugin.avatar(irc, msg, [])

        irc.reply.assert_called_once_with("Current persona: a sly fox", prefixNick=False)
        plugin.db.save_avatar_persona.assert_not_called()

    def test_bare_avatar_without_one_says_how(self, avatar_verse_env) -> None:
        plugin, irc, msg, _store = avatar_verse_env

        plugin.avatar(irc, msg, [])

        irc.reply.assert_called_once_with(
            "No persona set. Use @avatar <text> to set one.", prefixNick=False
        )
        plugin.db.save_avatar_persona.assert_not_called()

    def test_in_verse_channel_with_avatar_updates_both(self, avatar_verse_env) -> None:
        """GIVEN verse-enabled channel + active avatar WHEN @avatar THEN both updated."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")

        plugin.avatar(irc, msg, ["curious traveller"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "curious traveller")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == "curious traveller"

    def test_in_verse_channel_without_avatar_only_updates_persona(self, avatar_verse_env) -> None:
        """GIVEN verse-enabled channel + no avatar WHEN @avatar THEN only persona updated."""
        plugin, irc, msg, store = avatar_verse_env

        plugin.avatar(irc, msg, ["hello"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "hello")
        assert store.find_avatar_by_nick("alice") is None

    def test_in_non_verse_channel_only_updates_persona(self, plugin_env, tmp_path, mocker) -> None:
        """GIVEN verseEnabled=False WHEN @avatar THEN only persona updated, no verse store."""
        plugin, irc, msg = plugin_env
        irc.state.nickToAccount.return_value = "alice"  # @avatar needs an account

        get_store_spy = mocker.patch.object(plugin, "_get_or_create_verse_store")

        def _registry(key, *args):
            if key == "verseEnabled":
                return False
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.db.save_avatar_persona = mocker.MagicMock()
        msg.args = ("#other", "@avatar hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.avatar(irc, msg, ["hello"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "hello")
        get_store_spy.assert_not_called()

    def test_clear_in_verse_channel_clears_both(self, avatar_verse_env) -> None:
        """GIVEN active avatar with summary WHEN @avatar clear THEN both cleared."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")
        with store.write_transaction() as conn:
            import time as _time

            conn.execute(
                "UPDATE entities SET summary = ?, updated_at = ? WHERE id = ?",
                ("pirate queen", _time.time(), alice_id),
            )

        msg.args = ("#afnet", "@avatar clear")
        plugin.avatar(irc, msg, ["clear"])

        plugin.db.delete_avatar_persona.assert_called_once_with("alice")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == ""

    def test_verse_write_failure_does_not_update_persona(self, avatar_verse_env, mocker) -> None:
        """GIVEN verse write raises WHEN @avatar THEN persona text unchanged."""
        plugin, irc, msg, store = avatar_verse_env

        alice_id = self._opt_in(store, "alice")

        class _BrokenStore:
            def find_avatar_by_account(self, *a, **kw):
                return None

            def find_avatar_by_nick(self, *a, **kw):
                return alice_id

            def get_entity(self, *a, **kw):
                return store.get_entity(*a, **kw)

            def apply_direct(self, **kw):
                raise RuntimeError("disk full")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=_BrokenStore())

        import pytest as _pytest

        with _pytest.raises(RuntimeError, match="disk full"):
            plugin.avatar(irc, msg, ["foo"])

        plugin.db.save_avatar_persona.assert_not_called()


class TestInstructDoesNotTouchAvatar:
    """@instruct must no longer mirror to avatar.summary. The split's whole point."""

    @pytest.fixture
    def env(self, plugin_env, tmp_path, mocker):
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        irc.state.nickToAccount.return_value = "alice"  # @avatar needs an account
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@instruct hello")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.save_instruction = mocker.MagicMock()
        plugin.db.delete_instruction = mocker.MagicMock(return_value=True)

        return plugin, irc, msg, store

    def test_instruct_in_verse_channel_does_not_touch_avatar(self, env) -> None:
        """GIVEN active avatar WHEN @instruct THEN avatar.summary unchanged."""
        plugin, irc, msg, store = env
        result = store.opt_in_avatar("alice", None, instruct_text="original persona")
        assert result.entity_id is not None
        original_summary = store.get_entity(result.entity_id).summary

        plugin.instruct(irc, msg, ["new ask voice"])

        plugin.db.save_instruction.assert_called_once_with("alice", "new ask voice")
        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.summary == original_summary

    def test_instruct_clear_does_not_touch_avatar(self, env) -> None:
        """GIVEN active avatar WHEN @instruct clear THEN avatar.summary unchanged."""
        plugin, irc, msg, store = env
        result = store.opt_in_avatar("alice", None, instruct_text="keep me")
        assert result.entity_id is not None

        msg.args = ("#afnet", "@instruct clear")
        plugin.instruct(irc, msg, ["clear"])

        plugin.db.delete_instruction.assert_called_once_with("alice")
        entity = store.get_entity(result.entity_id)
        assert entity is not None
        assert entity.summary == "keep me"


class TestAvatarRetiredAvatar:
    """@avatar must not touch a retired avatar's summary."""

    def test_retired_avatar_only_updates_persona(self, plugin_env, tmp_path, mocker) -> None:
        """GIVEN retired avatar WHEN @avatar THEN persona updated, summary unchanged."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        irc.state.nickToAccount.return_value = "alice"  # @avatar needs an account
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        msg.args = ("#afnet", "@avatar foo")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)
        plugin.db.save_avatar_persona = mocker.MagicMock()

        result = store.opt_in_avatar("alice", None, instruct_text="")
        alice_id = result.entity_id
        assert alice_id is not None
        with store.write_transaction() as conn:
            import time as _time

            conn.execute(
                "UPDATE entities SET summary = ?, status = 'retired', updated_at = ? WHERE id = ?",
                ("old summary", _time.time(), alice_id),
            )

        plugin.avatar(irc, msg, ["foo"])

        plugin.db.save_avatar_persona.assert_called_once_with("alice", "foo")
        entity = store.get_entity(alice_id)
        assert entity is not None
        assert entity.summary == "old summary"


class TestVerseRouteForGating:
    """_verse_route_for gating logic (C7b): verseEnabled, llm.verse cap, OOC short-circuit."""

    # ------------------------------------------------------------------
    # Gate 1: verseEnabled=False
    # ------------------------------------------------------------------

    def test_verse_disabled_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=False WHEN _verse_route_for called THEN returns None (gate fires).

        We spy on registryValue to confirm the verseEnabled check is the gate that fires
        rather than the stub unconditionally returning None.
        """
        plugin, _irc, _msg = plugin_env

        # verseEnabled=False (default in plugin_env) — spy to confirm it is consulted.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: False if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "hello world")

        assert result is None
        plugin.registryValue.assert_any_call("verseEnabled", "#afnet")

    # ------------------------------------------------------------------
    # Gate 2: no llm.verse capability
    # ------------------------------------------------------------------

    def test_no_capability_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=True but user lacks llm.verse WHEN called THEN None (quiet fallthrough)."""
        plugin, _irc, _msg = plugin_env

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        cap_check = mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "hello world")

        assert result is None
        cap_check.assert_called_once_with("alice!*@*", "llm.verse")

    # ------------------------------------------------------------------
    # Gate 3: OOC message
    # ------------------------------------------------------------------

    def test_ooc_message_returns_none(self, plugin_env, mocker: MockerFixture) -> None:
        """GIVEN verseEnabled=True and cap granted WHEN message is OOC THEN None."""
        plugin, _irc, _msg = plugin_env

        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", "alice", "((this is OOC))")

        assert result is None

    # ------------------------------------------------------------------
    # Gate 4: all preconditions satisfied but no avatar — None (C7c)
    # ------------------------------------------------------------------

    def test_all_preconditions_satisfied_no_avatar_returns_none(
        self, plugin_env, mocker: MockerFixture, tmp_path
    ) -> None:
        """GIVEN verseEnabled=True, cap granted, plain message, but user has no avatar
        WHEN called THEN None (no opt-in → chat path)."""
        from llm.verse.store import VerseStore

        plugin, _irc, _msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: True if key == "verseEnabled" else ""
        )
        mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)

        result = plugin._verse_route_for("#afnet", "alice", None, "just a plain message")

        assert result is None

    # ------------------------------------------------------------------
    # Regression: @ask dispatch still hits chat path (carried from C7a)
    # ------------------------------------------------------------------


# =============================================================================
# C7c: _verse_route_for system prompt + tool list assembly
# =============================================================================


class TestVerseRouteForC7c:
    """Tests for _verse_route_for returning a populated VerseRoute (C7c).

    Complements TestVerseRouteForGating which covers the None branches.
    """

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """plugin_env + real VerseStore, verse-enabled channel, alice opted in."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            # Permissive trigger so any non-empty message routes: these tests
            # exercise route construction, not the _verse_triggered gate.
            if key == "verseTriggerRegex":
                return "."
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Opt alice in so she has an avatar.
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()

        return plugin, irc, msg, store

    @pytest.mark.parametrize(("account", "loaded"), [(None, False), ("alice", True)])
    def test_persona_needs_an_account(self, verse_env, account, loaded) -> None:
        """@avatar keys on account-or-nick: unauthenticated on nick alice, it
        could be account alice's persona. Roleplay runs without one."""
        plugin, _irc, _msg, _store = verse_env
        plugin.db.get_avatar_persona.return_value = "Be a wizard."

        route = plugin._verse_route_for("#afnet", "alice", account, "hi", force_roleplay=True)

        assert route is not None
        assert ("Be a wizard." in route.system_prompt) is loaded
        assert plugin.db.get_avatar_persona.called is loaded

    def test_avatar_present_returns_route(self, verse_env) -> None:
        """GIVEN opted-in avatar WHEN _verse_route_for called THEN non-None route returned."""
        plugin, _irc, _msg, store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello", force_roleplay=True)

        assert route is not None
        assert route.avatar_id is not None
        assert isinstance(route.system_prompt, str)
        assert "alice" in route.system_prompt
        assert len(route.tools) == 6
        assert route.store is store

    def test_slash_ooc_bypasses_route_even_with_avatar(self, verse_env) -> None:
        """GIVEN an opted-in avatar (so a plain message WOULD route to verse)
        WHEN the message uses the leading // OOC marker THEN None — the
        ergonomic opt-out short-circuits routing.

        Non-vacuous counterpart to test_avatar_present_returns_route: because
        alice has an avatar, the only path to None is the OOC gate, so removing
        // recognition from is_ooc makes this fail (the plain message routes).
        """
        plugin, _irc, _msg, _store = verse_env

        # Sanity: the same message without the // marker DOES route to verse.
        assert (
            plugin._verse_route_for(
                "#afnet", "alice", None, "what model are you?", force_roleplay=True
            )
            is not None
        )

        assert (
            plugin._verse_route_for(
                "#afnet", "alice", None, "// what model are you?", force_roleplay=True
            )
            is None
        )

    def test_no_avatar_falls_through_even_with_rp(self, verse_env, mocker) -> None:
        """GIVEN bob never opted in WHEN @rp forces roleplay THEN chat path:
        no roleplay prompt is built for a caller with no avatar."""
        plugin, _irc, _msg, _store = verse_env
        build = mocker.patch("llm.plugin.build_verse_system_prompt", return_value="SCENE")

        assert plugin._verse_route_for("#afnet", "bob", None, "hello", force_roleplay=True) is None
        build.assert_not_called()

    def test_route_tools_have_expected_names(self, verse_env) -> None:
        """Returned tools must be the six verse tool specs (incl. verse_edit)."""
        plugin, _irc, _msg, _store = verse_env

        route = plugin._verse_route_for("#afnet", "alice", None, "hello", force_roleplay=True)

        assert route is not None
        tool_names = {t["function"]["name"] for t in route.tools}
        assert tool_names == {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
            "verse_edit",
        }


class TestAskWithVerseRoute:
    """Tests that @ask in a verse channel with an opted-in avatar uses the verse path (C7c)."""

    SENTINEL = "VIBEBOT_TEST_SENTINEL_DO_NOT_LEAK"

    @staticmethod
    def _make_result(content: str = "verse reply") -> AssistantResult:
        return AssistantResult(
            content=content,
            grounding_used=False,
            spend=Spend(prompt_tokens=10, completion_tokens=5, cost=0.001),
            model="gpt-4",
        )

    @pytest.fixture
    def verse_ask_env(self, plugin_env, tmp_path, mocker):
        """plugin_env with verse enabled, alice opted in, and assistant_completion stubbed."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            if key == "assistantSystemPrompt":
                return self.SENTINEL
            # Permissive trigger: these tests exercise the verse @ask path, not
            # the _verse_triggered gate.
            if key == "verseTriggerRegex":
                return "."
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@verseopt in")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        # Opt alice in.
        plugin.verseopt(irc, msg, ["in"])
        irc.reply.reset_mock()

        # Stub assistant_completion for the @ask call.
        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = self._make_result()

        msg.args = ("#afnet", "@ask hello")

        return plugin, irc, msg, store

    def test_ask_verse_prompt_includes_channel_overlay_and_scene(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN system_prompt has BOTH the channel
        ``assistantSystemPrompt`` overlay AND the avatar scene context.

        Earlier behavior dropped the channel overlay on verse turns. That cratered
        output length on #afternet — the model under verse produced ~150-token
        list-style replies, while the same model under chat produced 600+ tokens
        with the channel overlay attached. The energy/length pump comes from the
        channel overlay; verse mode must inherit it on top of the scene context.
        """
        plugin, irc, msg, _store = verse_ask_env

        plugin.rp(irc, msg, ["hello"])

        plugin.llm_service.assistant_completion.assert_called_once()
        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        system_prompt = kwargs.get("system_prompt", "") or ""
        # Channel overlay (the sentinel) must appear — its energy is what
        # drove long-form output in chat mode.
        assert self.SENTINEL in system_prompt
        # AND the verse scene context (avatar name) must still be present.
        assert "alice" in system_prompt

    def test_ask_in_verse_appends_verse_tools(self, verse_ask_env) -> None:
        """GIVEN verse route WHEN @ask THEN all 5 verse tool names appear in extra_tools."""
        plugin, irc, msg, _store = verse_ask_env

        plugin.rp(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        extra_tools = kwargs.get("extra_tools") or []
        tool_names = {t["function"]["name"] for t in extra_tools}
        expected = {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
        }
        assert expected.issubset(tool_names)

    def test_every_advertised_tool_on_a_roleplay_turn_can_be_dispatched(
        self, verse_ask_env
    ) -> None:
        """The roleplay turn keeps react, aurora, irc_lookup et al. alongside verse."""
        plugin, irc, msg, _store = verse_ask_env

        plugin.rp(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        handlers = kwargs["extra_handlers"]
        assert {"react", "aurora", "irc_lookup", "verse_act"} <= set(handlers)
        assert {t["function"]["name"] for t in kwargs["extra_tools"]} <= set(handlers)

    def test_chat_record_lets_an_opted_in_avatar_save_canon_from_chat(
        self, verse_ask_env, mocker: MockerFixture
    ) -> None:
        """verseChatRecordEnabled frees verse_record on an ordinary chat turn."""
        import json

        plugin, irc, msg, store = verse_ask_env
        original = plugin.registryValue.side_effect
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: (
                True if key == "verseChatRecordEnabled" else original(key, *a)
            )
        )

        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["route_profile"] == "chat"
        result = kwargs["extra_handlers"]["verse_record"]({"summary": "alice found the brass key"})
        assert json.loads(result.content)["status"] == "ok"
        assert any("brass key" in e.summary for e in store.recent_events(limit=10))

    def test_ask_in_verse_bypasses_token_cap(self, verse_ask_env, mocker: MockerFixture) -> None:
        """GIVEN verse route WHEN @ask THEN route_profile is PROFILE_VERSE.

        PROFILE_VERSE is the only profile not in the profile_max_output dict
        in assistant.py, so it bypasses the token cap applied to PROFILE_CHAT.
        We verify by checking the route_profile passed to assistant_completion.
        """
        from llm.profile import PROFILE_VERSE

        plugin, irc, msg, _store = verse_ask_env

        plugin.rp(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["route_profile"] == PROFILE_VERSE

    def test_ask_in_verse_empty_verse_model_falls_back_to_none(
        self, verse_ask_env, mocker: MockerFixture
    ) -> None:
        """GIVEN an UNSET verseModel WHEN @ask in verse THEN model_override is None.

        Verse reads the per-channel ``verseModel`` and threads it as
        model_override; an empty value becomes ``None`` so the service resolves
        the profile's assistantModel. This is the boundary case — see
        ``test_ask_in_verse_passes_verse_model_override`` for the set case that
        pins the read itself (this empty case alone cannot catch a deleted read).
        """
        plugin, irc, msg, _store = verse_ask_env
        original = plugin.registryValue.side_effect

        def _registry(key, *args):
            if key == "verseModel":
                return ""  # explicit: unset verseModel
            return original(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.rp(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs.get("model_override") is None

    def test_ask_in_verse_passes_verse_model_override(
        self, verse_ask_env, mocker: MockerFixture
    ) -> None:
        """GIVEN a configured verseModel WHEN @ask in verse THEN that exact model
        reaches assistant_completion as model_override.

        This is the load-bearing coupling: bumping the chat assistantModel must
        not silently leave verse on it — verse rides its own (non-reasoning,
        prose) ``verseModel``. Deleting the verseModel read in dispatch, or
        substituting "assistantModel", makes this fail.
        """
        plugin, irc, msg, _store = verse_ask_env
        original = plugin.registryValue.side_effect

        def _registry(key, *args):
            if key == "verseModel":
                return "openrouter/grok-3-non-reasoning"
            return original(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.rp(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs.get("model_override") == "openrouter/grok-3-non-reasoning"

    def test_verse_model_empty_warns_once_per_channel(
        self, verse_ask_env, mocker: MockerFixture
    ) -> None:
        """GIVEN an UNSET verseModel WHEN @ask in verse twice on the same channel
        THEN log.warning fires exactly once (mentioning verseModel + assistantModel)."""
        plugin, irc, msg, _store = verse_ask_env
        original = plugin.registryValue.side_effect

        def _registry(key, *args):
            if key == "verseModel":
                return ""  # explicit: unset verseModel
            return original(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        plugin.rp(irc, msg, ["hello"])
        plugin.rp(irc, msg, ["hello again"])

        verse_warnings = [
            c
            for c in plugin.log.warning.call_args_list
            if c.args and "verseModel" in str(c.args[0])
        ]
        assert len(verse_warnings) == 1  # warn-once-per-channel
        fmt = str(verse_warnings[0].args[0])
        assert "verseModel" in fmt and "assistantModel" in fmt


class TestAskOnVerseChannelWithoutOptIn:
    """Verse-enabled channel + speaker who hasn't opted in: the tool surface
    must still carry verse schemas (cache stability across speakers) but each
    invocation must route to a denying handler so non-opted-in users can't
    drive the canon."""

    @staticmethod
    def _make_result() -> AssistantResult:
        return AssistantResult(
            content="reply",
            grounding_used=False,
            spend=Spend(prompt_tokens=10, completion_tokens=5, cost=0.001),
            model="gpt-4",
        )

    @pytest.fixture
    def env(self, plugin_env, tmp_path, mocker: MockerFixture):
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        # Verse store exists for the channel but the speaker has no avatar —
        # _verse_route_for will return None on that gate.
        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@ask hello")
        msg.prefix = "bob!user@host"
        msg.nick = "bob"
        msg.server_tags = {}

        plugin.llm_service.detect_images.return_value = []
        plugin.llm_service.assistant_completion.side_effect = None
        plugin.llm_service.assistant_completion.return_value = self._make_result()

        return plugin, irc, msg

    def test_extra_tools_include_all_verse_schemas(self, env) -> None:
        """GIVEN verse-enabled channel + non-opted-in speaker WHEN @ask THEN
        the five verse tool schemas are passed to assistant_completion so the
        prefix bytes match the opted-in cohort."""
        plugin, irc, msg = env
        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        extra_tools = kwargs.get("extra_tools") or []
        names = {t["function"]["name"] for t in extra_tools}
        assert {
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
        }.issubset(names)

    def test_extra_handlers_deny_verse_tools(self, env) -> None:
        """GIVEN verse-enabled channel + non-opted-in speaker WHEN @ask THEN
        every verse tool name in extra_handlers returns an ``{"error": ...}``
        payload mentioning opt-in (so the model can self-correct)."""
        import json

        plugin, irc, msg = env
        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        handlers = kwargs.get("extra_handlers") or {}
        for name in (
            "verse_act",
            "verse_move",
            "verse_look",
            "verse_recall",
            "verse_record",
        ):
            assert name in handlers, name
            payload = json.loads(handlers[name]({}).content)
            assert "error" in payload
            assert "opt-in" in payload["error"].lower()

    def test_non_verse_tools_stay_dispatchable_for_a_non_opted_in_speaker(self, env) -> None:
        """Only the verse tools are denied; react, aurora and the rest still run."""
        plugin, irc, msg = env
        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        handlers = kwargs["extra_handlers"]
        assert {"react", "aurora", "irc_lookup"} <= set(handlers)
        assert {t["function"]["name"] for t in kwargs["extra_tools"]} <= set(handlers)

    def test_profile_stays_chat_for_non_opted_in_speaker(self, env) -> None:
        """The advertised tool surface widens, but the system prompt /
        framework must remain the chat profile for non-opted-in speakers —
        forcing PROFILE_VERSE on them would lose the chat framework's
        length cap and tool-behavior rules."""
        from llm.profile import PROFILE_CHAT

        plugin, irc, msg = env
        plugin.ask(irc, msg, ["hello"])

        kwargs = plugin.llm_service.assistant_completion.call_args.kwargs
        assert kwargs["route_profile"] == PROFILE_CHAT


class TestCompactionTimerWiring:
    """E3: plugin wires the daily compaction timer + walks verse-enabled channels."""

    def test_compaction_tick_offloads_pass_to_executor(self, plugin_env, mocker) -> None:
        """``_compaction_tick`` hands ``_run_compaction_pass`` to the executor
        rather than running it inline.

        The daily timer fires on Limnoria's scheduler thread (the IRC
        driver's main loop); ``_run_compaction_pass`` makes a blocking LLM
        call per verse-enabled channel, so running it inline would pin the
        driver — the same bug class as the addressed-message typing lag.
        """
        plugin, _irc, _msg = plugin_env
        plugin._run_compaction_pass = mocker.MagicMock()
        plugin._register_compaction_timer = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock(closing=False)

        plugin._compaction_tick()

        plugin._run_compaction_pass.assert_not_called()  # offloaded, not inline
        plugin._llm_executor.submit.assert_called_once_with(
            "verse_compaction", plugin._run_compaction_pass
        )
        plugin._register_compaction_timer.assert_called_once()  # timer always re-arms

    def test_compaction_tick_skips_submit_when_closing(self, plugin_env, mocker) -> None:
        """During shutdown the pass is not submitted (executor closing), but
        the timer re-arm still runs in ``finally``."""
        plugin, _irc, _msg = plugin_env
        plugin._run_compaction_pass = mocker.MagicMock()
        plugin._register_compaction_timer = mocker.MagicMock()
        plugin._llm_executor = mocker.MagicMock(closing=True)

        plugin._compaction_tick()

        plugin._llm_executor.submit.assert_not_called()
        plugin._run_compaction_pass.assert_not_called()
        plugin._register_compaction_timer.assert_called_once()

    def test_compaction_callback_walks_verse_enabled_channels(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """``_run_compaction_pass`` invokes ``compact_verse`` once per
        verse-enabled channel returned by ``_verse_enabled_channels``."""
        plugin, _irc, _msg = plugin_env

        # Only #afnet is verse-enabled. _verse_enabled_channels already
        # filters by registry; emulate that here.
        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])

        # Defensive re-check inside _run_compaction_pass uses
        # registryValue("verseEnabled", channel) — return True for both
        # so the iteration faithfully reflects the helper's output.
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: True if key == "verseEnabled" else ""
        )

        # Stub stores expose ``_channel`` so the fake compact_verse can
        # introspect which channel it was handed.
        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)

        from llm.verse.compaction import CompactionOutcome

        called_for: list[str] = []

        def fake_compact(store, **kw):
            called_for.append(store._channel)
            return CompactionOutcome("skipped_no_events", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", fake_compact)

        plugin._run_compaction_pass()
        assert called_for == ["#afnet"]

    def test_compaction_failure_does_not_abort_remaining_channels(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """A raise in one channel's compact_verse must not skip later channels."""
        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: True if key == "verseEnabled" else ""
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)

        from llm.verse.compaction import CompactionOutcome

        seen: list[str] = []

        def maybe_bomb(store, **kw):
            seen.append(store._channel)
            if store._channel == "#a":
                raise RuntimeError("fail")
            return CompactionOutcome("skipped_no_events", 0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", maybe_bomb)

        plugin._run_compaction_pass()
        assert "#a" in seen and "#b" in seen


class TestRunCompactionPassCallsAging:
    """Phase 5a: aging runs once per verse-enabled channel inside the
    compaction pass, with its own try/except for failure isolation."""

    def test_aging_called_once_per_enabled_channel(self, plugin_env, mocker, monkeypatch) -> None:
        """_run_compaction_pass calls age_auto_created_entities once per
        channel returned by _verse_enabled_channels."""
        from llm.verse import aging as aging_mod
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: True if key == "verseEnabled" else ""
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse",
            lambda *a, **kw: CompactionOutcome("skipped_disabled", 0, 0),
        )

        called: list[tuple[object, int]] = []

        def _spy_aging(store, *, retire_after_days, now):
            called.append((store, retire_after_days))
            return aging_mod.AgingOutcome(0, 0)

        monkeypatch.setattr("llm.verse.aging.age_auto_created_entities", _spy_aging)

        plugin._run_compaction_pass()

        assert len(called) == 2
        stores = {id(c[0]) for c in called}
        assert len(stores) == 2

    def test_aging_runs_before_compaction_heartbeat(self, plugin_env, mocker, monkeypatch) -> None:
        """Aging must run BEFORE compact_verse for each channel.

        compact_verse's digest heartbeat bumps last_seen_ts=now() on every
        entity in the new digest. If aging runs after it, a long-silent
        auto_created NPC that happens to appear in the compacted events is
        freshly stamped and never older than the retire cutoff — silently
        defeating the retire cutoff. Aging-first reads the true
        last_seen_ts before the heartbeat can resurrect it.
        """
        from llm.verse import aging as aging_mod
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#afnet"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: True if key == "verseEnabled" else ""
        )
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            side_effect=lambda channel: mocker.MagicMock(_channel=channel),
        )

        order: list[str] = []

        def _spy_compact(*_a, **_kw):
            order.append("compact")
            return CompactionOutcome("skipped_disabled", 0, 0)

        def _spy_age(*_a, **_kw):
            order.append("age")
            return aging_mod.AgingOutcome(0, 0)

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _spy_compact)
        monkeypatch.setattr("llm.verse.aging.age_auto_created_entities", _spy_age)

        plugin._run_compaction_pass()

        assert order == ["age", "compact"]

    def test_aging_failure_in_one_channel_does_not_abort_others(
        self, plugin_env, mocker, monkeypatch
    ) -> None:
        """If aging raises for #a, #b still gets aged."""
        from llm.verse import aging as aging_mod
        from llm.verse.compaction import CompactionOutcome

        plugin, _irc, _msg = plugin_env

        mocker.patch.object(plugin, "_verse_enabled_channels", return_value=["#a", "#b"])
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *args: True if key == "verseEnabled" else ""
        )

        def _fake_store_for(channel: str):
            return mocker.MagicMock(_channel=channel)

        mocker.patch.object(plugin, "_get_or_create_verse_store", side_effect=_fake_store_for)
        monkeypatch.setattr(
            "llm.verse.compaction.compact_verse",
            lambda *a, **kw: CompactionOutcome("skipped_disabled", 0, 0),
        )

        seen: list[int] = []

        def _aging(store, *, retire_after_days, now):
            seen.append(id(store))
            if len(seen) == 1:
                raise RuntimeError("simulated aging failure")
            return aging_mod.AgingOutcome(0, 0)

        monkeypatch.setattr("llm.verse.aging.age_auto_created_entities", _aging)

        plugin._run_compaction_pass()

        assert len(seen) == 2


class TestRpCommandGates:
    """@rp shares @ask's rate limit, and on/off toggle without a turn."""

    @pytest.fixture
    def rp_env(self, plugin_env, mocker: MockerFixture):
        plugin, irc, msg = plugin_env
        dispatch = mocker.patch.object(plugin, "_dispatch_with_verse_routing")
        return plugin, irc, msg, dispatch

    def test_rate_limited_caller_gets_no_turn(self, rp_env, mocker: MockerFixture) -> None:
        plugin, irc, msg, dispatch = rp_env
        mocker.patch.object(
            plugin, "_run_preflight", return_value=PreflightResult(True, Caller.from_msg(msg))
        )

        plugin.rp(irc, msg, ["on"])
        plugin.rp(irc, msg, ["Archie kicks the door"])

        dispatch.assert_not_called()
        assert not plugin._roleplay_sticky_active(PreflightResult(False, Caller.from_msg(msg)))

    @pytest.mark.parametrize(
        ("toggle", "active", "reply"),
        [
            ("on", True, "Roleplay mode ON"),
            ("off", False, "Roleplay mode off — back to normal chat."),
        ],
    )
    def test_on_and_off_toggle_sticky_mode(self, rp_env, toggle, active, reply) -> None:
        plugin, irc, msg, dispatch = rp_env
        pf = PreflightResult(False, Caller.from_msg(msg))
        plugin._roleplay_sticky_set(pf, not active)

        plugin.rp(irc, msg, [toggle])

        dispatch.assert_not_called()
        assert plugin._roleplay_sticky_active(pf) is active
        assert irc.reply.call_args.args[0].startswith(reply)


class TestCompactionOutcomeLine:
    """Each documented outcome state reads as itself; only a real compaction
    says "compacted" (docs/guide/operator/verse.md outcome reference)."""

    @pytest.mark.parametrize(
        ("state", "total", "expected"),
        [
            ("compacted", 30, "compacted 30 events"),
            ("skipped_below_floor", 5, "skipped (only 5 events; floor is 20)"),
            ("skipped_no_events", 30, "skipped (no events past retention; total 30)"),
            ("skipped_disabled", 0, "skipped (retention disabled)"),
        ],
    )
    def test_state_heads(self, state, total, expected) -> None:
        from llm.plugin import _format_compaction_outcome
        from llm.verse.compaction import CompactionOutcome

        line = _format_compaction_outcome(
            CompactionOutcome(state, total, 0), None, min_keep_events=20
        )

        assert line == expected


class TestVersecompactCommand:
    """E4: @versecompact owner command — manual retention compaction.

    Strategy mirrors @versepurge tests: direct method
    calls (bypassing wrap), real VerseStore in tmp_path, ircdb.checkCapability
    monkeypatched. The happy-path test inserts >min_keep events older than
    the retention window and monkeypatches compact_verse's compaction client
    constructor so no network call is attempted.
    """

    @pytest.fixture
    def compact_env(self, plugin_env, tmp_path, mocker):
        """plugin_env wired with a real VerseStore for #afnet."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")
        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )
        plugin._verse_stores["#afnet"] = store

        msg.args = ("#afnet", "@versecompact #afnet")
        msg.prefix = "owner!user@host"
        msg.nick = "owner"
        msg.server_tags = {}

        return plugin, irc, msg, store

    def _override_registry(self, plugin, mocker, *, verse_enabled: bool) -> None:
        from tests.conftest import make_registry_side_effect

        defaults = make_registry_side_effect()

        def _registry(key, *args):
            if key == "verseEnabled":
                return verse_enabled
            return defaults(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

    def test_compacts_named_channel(self, compact_env, mocker) -> None:
        """GIVEN >min_keep old events WHEN @versecompact #afnet THEN reply 'compacted'."""
        from tests.verse.conftest import insert_event_at

        plugin, irc, msg, store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=True)

        # Grant llm.verse.gm so the wrap-bypassed direct call still
        # treats the caller as an owner. (Direct .versecompact() call
        # also won't pass through wrap's capability check; the in-method
        # registry guard is what matters.)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        # Substitute a fake compaction client so no network call happens.
        # Capture the model kwarg to assert which model writes the digest.
        captured_models: list[str] = []

        class _FakeClient:
            def call(self, *, op, model, messages):
                from llm.verse.compaction import VerseCallUsage

                captured_models.append(model)
                return "A digest of the past.", VerseCallUsage(
                    prompt_tokens=10, completion_tokens=20, cost=0.0
                )

        mocker.patch(
            "llm.verse.compaction.ProviderVerseClient",
            return_value=_FakeClient(),
        )

        seconds_per_day = 86400
        now_ts = 100_000_000.0
        for i in range(25):
            insert_event_at(
                store,
                summary=f"old{i}",
                entity_ids=[],
                source="avatar",
                ts=now_ts - 60 * seconds_per_day,
            )

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "compaction outcome for #afnet" in reply_text
        assert "compacted" in reply_text
        assert captured_models == [_VERSE_COMPACTION_MODEL]

    def test_manual_compaction_does_not_age_entities(self, compact_env, mocker) -> None:
        """@versecompact compacts without the daily pass's aging sweep, and a
        skipped run does not claim it compacted."""
        plugin, irc, msg, _store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=True)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch("llm.verse.compaction.ProviderVerseClient", return_value=mocker.MagicMock())
        aging = mocker.patch("llm.verse.aging.age_auto_created_entities")

        plugin.versecompact(irc, msg, ["#afnet"])

        aging.assert_not_called()
        reply_text = irc.reply.call_args[0][0]
        assert reply_text.startswith("compaction outcome for #afnet: skipped")

    def test_disabled_verse_says_so(self, compact_env, mocker) -> None:
        """GIVEN verseEnabled=False WHEN @versecompact THEN reply names verseEnabled."""
        plugin, irc, msg, _store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=False)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.reply.assert_called_once()
        reply_text = irc.reply.call_args[0][0]
        assert "verseEnabled" in reply_text

    def test_failure_in_compact_verse_replies_error(self, compact_env, mocker, monkeypatch) -> None:
        """GIVEN compact_verse raises WHEN @versecompact THEN irc.error not crash."""
        plugin, irc, msg, _store = compact_env
        self._override_registry(plugin, mocker, verse_enabled=True)
        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )
        mocker.patch(
            "llm.verse.compaction.ProviderVerseClient",
            return_value=mocker.MagicMock(),
        )

        def _boom(*a, **kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr("llm.verse.compaction.compact_verse", _boom)

        plugin.versecompact(irc, msg, ["#afnet"])

        irc.error.assert_called_once()
        err_text = irc.error.call_args[0][0]
        assert "compaction failed for #afnet" in err_text
        assert "RuntimeError" in err_text


class TestCanonCommand:
    """Tests for the @canon lock|unlock|forget <name> command (author-gated)."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        """Extend plugin_env with a real VerseStore and verse-enabled channel."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env

        store = VerseStore(tmp_path / "verse", "#afnet")

        mocker.patch.object(
            plugin,
            "_get_or_create_verse_store",
            return_value=store,
        )

        def _registry(key, *args):
            if key == "verseEnabled":
                return True
            from tests.conftest import make_registry_side_effect

            return make_registry_side_effect()(key, *args)

        plugin.registryValue = mocker.MagicMock(side_effect=_registry)

        mocker.patch(
            "llm.plugin.ircdb.checkCapability",
            side_effect=lambda prefix, cap: cap.startswith("llm."),
        )

        plugin.db.get_instruction = mocker.MagicMock(return_value=None)
        plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

        msg.args = ("#afnet", "@canon")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        msg.server_tags = {}

        return plugin, irc, msg, store

    def test_canon_lock_sets_author_locked(self, verse_env):
        plugin, irc, msg, store = verse_env
        h = store.add_entity("npc", "Harry", "year 8")
        plugin.canon(irc, msg, ["lock", "Harry"])
        assert store.get_attribute(h, "author_locked") == "1"

    def test_canon_forget_unlocks(self, verse_env):
        plugin, irc, msg, store = verse_env
        h = store.add_entity("npc", "Harry", "year 8")
        plugin.canon(irc, msg, ["lock", "Harry"])
        plugin.canon(irc, msg, ["forget", "Harry"])
        assert store.get_attribute(h, "author_locked") is None

    def test_canon_unknown_character_errors(self, verse_env):
        plugin, irc, msg, store = verse_env
        plugin.canon(irc, msg, ["lock", "Nobody"])
        assert irc.error.called


# =============================================================================
# TestDeloomedPluginBoot
# =============================================================================


# =============================================================================
# Task 7: verseStyleExemplars threaded into the verse route (end-to-end)
# =============================================================================


def test_verse_route_threads_style_exemplars(plugin_env, tmp_path, mocker):
    """GIVEN verseStyleExemplars returns a non-empty list WHEN _verse_route_for
    is called for an opted-in avatar THEN the exemplar text AND the
    'singled these lines out' header both appear in route.system_prompt.

    This is the end-to-end plumbing test: it proves that the caller reads the
    config key and passes it to build_verse_system_prompt.
    """
    from llm.verse.store import VerseStore

    plugin, irc, msg = plugin_env

    store = VerseStore(tmp_path / "verse", "#afnet")
    mocker.patch.object(plugin, "_get_or_create_verse_store", return_value=store)

    def _registry(key, *args):
        if key == "verseEnabled":
            return True
        if key == "verseStyleExemplars":
            return ["the lads marched on the chippy"]
        # Permissive trigger: this test exercises exemplar plumbing, not the gate.
        if key == "verseTriggerRegex":
            return "."
        from tests.conftest import make_registry_side_effect

        return make_registry_side_effect()(key, *args)

    plugin.registryValue = mocker.MagicMock(side_effect=_registry)
    mocker.patch(
        "llm.plugin.ircdb.checkCapability",
        side_effect=lambda prefix, cap: cap.startswith("llm."),
    )
    plugin.db.get_instruction = mocker.MagicMock(return_value=None)
    plugin.db.get_avatar_persona = mocker.MagicMock(return_value=None)

    # Opt alice in via the real command so the store has her avatar.
    msg.args = ("#afnet", "@verseopt in")
    msg.prefix = "alice!user@host"
    msg.nick = "alice"
    msg.server_tags = {}
    plugin.verseopt(irc, msg, ["in"])
    irc.reply.reset_mock()

    route = plugin._verse_route_for("#afnet", "alice", None, "what happened", force_roleplay=True)

    assert route is not None
    assert "the lads marched on the chippy" in route.system_prompt
    assert "singled these lines out" in route.system_prompt


class TestVersePasteStyle:
    """Verse scenes that overflow to a paste use the storybook theme."""

    def test_verse_overflow_paste_uses_story_theme(self, plugin_env):
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        plugin.llm_service.save_markdown_to_http.return_value = "https://e.co/s.html"
        plugin.llm_service.summarize_for_irc.return_value = "A tale."
        scene = "beat one\nbeat two"
        result = AssistantResult(content=scene, spend=Spend.ZERO, was_verse=True)

        plugin._dispatch_assistant_reply(
            irc, msg, result, caller=Caller(Identity("fc42", None), "#test"), response=scene
        )

        assert plugin.llm_service.save_markdown_to_http.call_args.kwargs["style"] == "story"


class TestVerseReactionSendHook:
    def test_verse_reply_records_last_bot_line(self, plugin_env):
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        result = AssistantResult(content="A tale of Methane Max", spend=Spend.ZERO, was_verse=True)
        plugin._dispatch_assistant_reply(
            irc,
            msg,
            result,
            caller=Caller(Identity("fc42", None), "#test"),
            response="A tale of Methane Max",
        )
        last = plugin._last_bot_line.get(("testnet", "#test"))
        assert last is not None
        assert last["text"].endswith("Methane Max")
        assert isinstance(last["ts"], float)

    def test_non_verse_reply_does_not_record(self, plugin_env):
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        result = AssistantResult(content="just chatting", spend=Spend.ZERO, was_verse=False)
        plugin._dispatch_assistant_reply(
            irc,
            msg,
            result,
            caller=Caller(Identity("fc42", None), "#test"),
            response="just chatting",
        )
        assert ("testnet", "#test") not in plugin._last_bot_line

    def test_verse_action_reply_also_records(self, plugin_env):
        # Red-team: a verse line emitted as a /me action returns via the action
        # branch (before the long-reply send) and must still be recorded.
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        result = AssistantResult(
            content="/me summons Methane Max", spend=Spend.ZERO, was_verse=True
        )
        plugin._dispatch_assistant_reply(
            irc,
            msg,
            result,
            caller=Caller(Identity("fc42", None), "#test"),
            response="/me summons Methane Max",
        )
        last = plugin._last_bot_line.get(("testnet", "#test"))
        assert last is not None and "Methane Max" in last["text"]


class TestDoTagmsgReactionCapture:
    @pytest.fixture
    def reaction_env(self, plugin_env, llm_data_dir):
        # _append_reaction_event reads conf.supybot.directories.data() and
        # writes <data>/verse/reactions.jsonl.
        plugin, irc, msg = plugin_env
        irc.network = "testnet"
        return plugin, irc, msg, llm_data_dir

    def _path(self, tmp_path):
        return tmp_path / "verse" / "reactions.jsonl"

    def test_thumbs_up_on_recent_verse_line_logged(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        plugin._last_bot_line[("testnet", "#test")] = {
            "text": "Methane Max hacked the tannoy",
            "ts": time.time(),
        }
        msg.server_tags = {"+draft/react": "\U0001f44d", "+draft/reply": "abc"}
        msg.channel = "#test"
        msg.nick = "fc42"
        plugin.doTagmsg(irc, msg)
        lines = self._path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        ev = json.loads(lines[0])
        assert ev["sentiment"] == "approve"
        assert ev["reactor"] == "fc42"
        assert ev["was_verse"] is True

    def test_every_reaction_is_recorded_when_the_verse_dir_exists(self, reaction_env):
        """In prod data/verse/ already holds the stores; capture must still append."""
        plugin, irc, msg, tmp_path = reaction_env
        (tmp_path / "verse").mkdir()
        plugin._last_bot_line[("testnet", "#test")] = {"text": "x", "ts": time.time()}
        msg.channel = "#test"
        msg.nick = "fc42"

        for emoji in ("\U0001f44d", "\U0001f44e"):
            msg.server_tags = {"+draft/react": emoji}
            plugin.doTagmsg(irc, msg)

        lines = self._path(tmp_path).read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(line)["sentiment"] for line in lines] == ["approve", "disapprove"]

    def test_non_react_tagmsg_ignored(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        msg.server_tags = {"+typing": "active"}
        plugin.doTagmsg(irc, msg)
        assert not self._path(tmp_path).exists()

    def test_no_recent_verse_line_skips(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        msg.server_tags = {"+draft/react": "\U0001f44d"}
        msg.channel = "#test"
        plugin.doTagmsg(irc, msg)
        assert not self._path(tmp_path).exists()

    def test_thumbs_down_recorded_as_disapprove(self, reaction_env):
        plugin, irc, msg, tmp_path = reaction_env
        plugin._last_bot_line[("testnet", "#test")] = {
            "text": "Methane Max hacked the tannoy",
            "ts": time.time(),
        }
        msg.server_tags = {"+draft/react": "\U0001f44e"}  # thumbs down
        msg.channel = "#test"
        msg.nick = "fc42"
        plugin.doTagmsg(irc, msg)
        ev = json.loads(self._path(tmp_path).read_text(encoding="utf-8").strip())
        assert ev["sentiment"] == "disapprove"

    def test_self_reaction_skipped(self, reaction_env):
        # The bot's own reaction (matched case-insensitively) must not be counted.
        plugin, irc, msg, tmp_path = reaction_env
        plugin._last_bot_line[("testnet", "#test")] = {"text": "x", "ts": time.time()}
        msg.server_tags = {"+draft/react": "\U0001f44d"}
        msg.channel = "#test"
        msg.nick = irc.nick.upper()  # different case, still the bot
        plugin.doTagmsg(irc, msg)
        assert not self._path(tmp_path).exists()

    def test_non_channel_tagmsg_ignored(self, reaction_env):
        # A reaction whose target is a nick (private message) is ignored.
        plugin, irc, msg, tmp_path = reaction_env
        plugin._last_bot_line[("testnet", "#test")] = {"text": "x", "ts": time.time()}
        msg.server_tags = {"+draft/react": "\U0001f44d"}
        msg.channel = ""
        msg.args = ("testbot", "")
        msg.nick = "fc42"
        plugin.doTagmsg(irc, msg)
        assert not self._path(tmp_path).exists()

    def test_channel_key_case_insensitive(self, reaction_env):
        # A verse line recorded for "#Test" must be matched by a reaction on "#test".
        plugin, irc, msg, tmp_path = reaction_env
        plugin._record_last_verse_line(
            irc,
            "#Test",
            "Methane Max hacked the tannoy",
            AssistantResult(content="x", spend=Spend.ZERO, was_verse=True),
        )
        msg.server_tags = {"+draft/react": "\U0001f44d"}
        msg.channel = "#test"
        msg.nick = "fc42"
        plugin.doTagmsg(irc, msg)
        ev = json.loads(self._path(tmp_path).read_text(encoding="utf-8").strip())
        assert ev["sentiment"] == "approve"


# =============================================================================
# TestVerseTriggerGate
# =============================================================================


class TestVerseTriggerGate:
    """Tests for the _verse_triggered gate.

    The gate stops every avatar-user message from dropping into verse. A
    message routes to verse only when it names a known in-universe entity
    (other than the speaker's own avatar) OR matches the channel's
    ``verseTriggerRegex``. Everything else falls through to chat.
    """

    @pytest.fixture
    def gate_env(self, plugin_env, tmp_path, mocker):
        """Real VerseStore with an ``alice`` avatar and a ``The Clearing`` place."""
        from llm.verse.store import VerseStore

        plugin, irc, msg = plugin_env
        store = VerseStore(tmp_path / "verse", "#afnet")
        store.opt_in_avatar("alice", None, "")
        avatar_id = store.find_avatar_by_nick("alice")
        store.add_entity("place", "The Clearing", "A quiet glade.")
        return plugin, store, avatar_id

    def _set_regex(self, plugin, mocker, value: str) -> None:
        plugin.registryValue = mocker.MagicMock(
            side_effect=lambda key, *a: value if key == "verseTriggerRegex" else "",
        )

    def test_entity_reference_triggers(self, gate_env, mocker) -> None:
        """A message naming a canon place triggers verse even with no keyword."""
        plugin, store, avatar_id = gate_env
        self._set_regex(plugin, mocker, "")
        assert plugin._verse_triggered("#afnet", store, avatar_id, "I wander into The Clearing")

    def test_keyword_regex_triggers(self, gate_env, mocker) -> None:
        """A configured keyword triggers verse with no entity reference."""
        plugin, store, avatar_id = gate_env
        self._set_regex(plugin, mocker, r"\bverse\b")
        assert plugin._verse_triggered("#afnet", store, avatar_id, "let's do some verse now")

    def test_keyword_regex_is_case_insensitive(self, gate_env, mocker) -> None:
        plugin, store, avatar_id = gate_env
        self._set_regex(plugin, mocker, r"stinky lads")
        assert plugin._verse_triggered("#afnet", store, avatar_id, "the STINKY LADS arrive")

    def test_no_signal_does_not_trigger(self, gate_env, mocker) -> None:
        """No entity, no keyword match → chat path."""
        plugin, store, avatar_id = gate_env
        self._set_regex(plugin, mocker, r"\bverse\b")
        assert not plugin._verse_triggered("#afnet", store, avatar_id, "hi all how are you")

    def test_own_avatar_name_alone_does_not_trigger(self, gate_env, mocker) -> None:
        """Naming only your own avatar is not a signal."""
        plugin, store, avatar_id = gate_env
        self._set_regex(plugin, mocker, "")
        assert not plugin._verse_triggered("#afnet", store, avatar_id, "alice waves hello")

    def test_empty_regex_disables_keyword(self, gate_env, mocker) -> None:
        plugin, store, avatar_id = gate_env
        self._set_regex(plugin, mocker, "")
        assert not plugin._verse_triggered("#afnet", store, avatar_id, "let's do some verse now")

    def test_malformed_regex_is_ignored(self, gate_env, mocker) -> None:
        """A bad regex must not raise into the message path — treated as no match."""
        plugin, store, avatar_id = gate_env
        self._set_regex(plugin, mocker, r"[unclosed(")
        assert not plugin._verse_triggered("#afnet", store, avatar_id, "hello there")


class TestProviderVerseClient:
    """Compaction calls provider.chat with the call's own model.

    The key is resolved from the model inside provider.chat, so a client built
    with one provider's key can never pair it with another provider's model.
    """

    def test_sends_the_call_model_and_returns_usage(self, mocker) -> None:
        """GIVEN a compaction call WHEN it runs THEN provider.chat gets that
        model and the reply's text, tokens and cost come back."""
        from llm.verse import compaction

        chat = mocker.patch(
            "llm.provider.chat",
            return_value=make_completion_response(
                "digest", prompt_tokens=7, completion_tokens=3, cost=0.002
            ),
        )
        messages = [{"role": "user", "content": "compact this"}]
        content, usage = compaction.ProviderVerseClient().call(
            op="compact", model="anthropic/claude-haiku-5-5", messages=messages
        )

        assert chat.call_args.args[0] == "anthropic/claude-haiku-5-5"
        assert chat.call_args.args[1] == messages
        assert content == "digest"
        assert usage == compaction.VerseCallUsage(7, 3, 0.002)


def test_verse_store_cache_folds_channel_case(plugin_env, tmp_path, mocker) -> None:
    """GIVEN '#AfterNet' (e.g. @versedit from a PM) and '#afternet' WHEN the
    store is fetched THEN both resolve to the one cached store and DB file."""
    plugin, _irc, _msg = plugin_env
    mocker.patch.object(plugin, "_verse_data_dir", return_value=tmp_path / "verse")
    plugin._verse_stores.clear()

    a = plugin._get_or_create_verse_store("#AfterNet")
    b = plugin._get_or_create_verse_store("#afternet")

    assert a is b
    assert a.path.name == "_afternet_2de47b99.db"
