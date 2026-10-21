import json

import pytest
from llm.verse.avatar import dispatch_verse_edit
from llm.verse.store import VerseStore


def test_verse_edit_unauthorized_is_noop(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "X"},
        authorized=False,
        account="nobody",
    )
    assert result["status"] == "refused"
    assert store.list_entities_by_kind("npc") == []


def test_verse_edit_authorized_applies(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "Archie"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "ok"
    assert [e.name for e in store.list_entities_by_kind("npc")] == ["Archie"]


def test_verse_edit_apply_error_is_reported(tmp_path):
    """A constructive op that raises in apply_direct (e.g. relating to a
    nonexistent entity) returns status=error with the exception message,
    not an unhandled traceback."""
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_relation",
        payload={"from_id": 999, "to_id": 998, "kind": "knows"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert "does not exist" in result["detail"]


def test_verse_edit_invalid_payload_rejected(tmp_path):
    """validate_payload rejects a malformed payload before apply_direct."""
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc"},  # missing name
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"


def test_verse_edit_rejects_destructive_op(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="delete_event",
        payload={"event_id": 1},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert "op" in result["detail"]


# ---------------------------------------------------------------------------
# Integration: the plugin-layer _verse_edit_handler binds the gate to the
# TRIGGERING user's IRC prefix. These tests assert the capability decides
# whether the store is mutated.
# ---------------------------------------------------------------------------


def _make_plugin(mocker):
    from llm.plugin import LLM

    mocker.patch.object(LLM, "__init__", lambda self, irc: None)
    return LLM.__new__(LLM)


def test_handler_unauthorized_user_mutates_nothing(tmp_path, mocker):
    """An invoking user WITHOUT llm.verse.edit gets a refusal and writes nothing."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=False)
    msg = mocker.MagicMock()
    msg.prefix = "nobody!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="nobody")
    result = handler({"op": "add_entity", "payload": {"kind": "npc", "name": "Ghost"}})

    parsed = json.loads(result.content)
    assert parsed["status"] == "refused"
    assert "error" in parsed  # loop counts this as a failure
    assert store.list_entities_by_kind("npc") == []


def test_handler_authorized_user_applies(tmp_path, mocker):
    """An invoking user WITH llm.verse.edit applies the edit to the store."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="gm")
    result = handler({"op": "add_entity", "payload": {"kind": "npc", "name": "Archie"}})

    parsed = json.loads(result.content)
    assert parsed["status"] == "ok"
    assert [e.name for e in store.list_entities_by_kind("npc")] == ["Archie"]


def test_handler_gate_checks_triggering_prefix(tmp_path, mocker):
    """The capability check is computed against the invoking msg.prefix."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    check = mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    plugin._verse_edit_handler(msg=msg, store=store, account="gm")

    check.assert_called_once_with("gm!user@host", "llm.verse.edit")


def test_handler_authorized_but_destructive_op_refused(tmp_path, mocker):
    """Even an authorized user cannot delete via verse_edit; store is untouched."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "Doomed")
    ev_id = store.add_event("a thing happened", [eid], source="llm")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="gm")
    result = handler({"op": "delete_event", "payload": {"event_id": ev_id}})

    parsed = json.loads(result.content)
    assert parsed["status"] == "error"
    assert store.recent_events()  # event survives


@pytest.mark.parametrize("bad_args", [None, [], "scalar", {"op": "add_entity"}])
def test_handler_tolerates_malformed_args(tmp_path, mocker, bad_args):
    """Non-dict args / missing payload must not raise out of the turn."""
    plugin = _make_plugin(mocker)
    store = VerseStore(tmp_path, "#chan")
    mocker.patch("llm.plugin.ircdb.checkCapability", return_value=True)
    msg = mocker.MagicMock()
    msg.prefix = "gm!user@host"

    handler = plugin._verse_edit_handler(msg=msg, store=store, account="gm")
    result = handler(bad_args)
    parsed = json.loads(result.content)
    assert parsed["status"] in {"ok", "error", "refused"}


def test_verse_edit_add_entity_rejects_duplicate_active_name(tmp_path):
    """add_entity on a name that already exists is refused, so an "edit" the
    model cannot express as update_entity does not become a silent duplicate.
    Mirrors the @versedit add command's guard."""
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "Assgas Archie", "summary": "Y12 windbag"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert "already exists" in result["detail"]
    assert [e.name for e in store.list_entities_by_kind("npc")] == ["Assgas Archie"]


def test_verse_edit_add_entity_duplicate_check_is_case_insensitive(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Assgas Archie")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "assgas archie"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert len(store.list_entities_by_kind("npc")) == 1


def test_verse_edit_add_entity_allows_reuse_of_a_retired_name(tmp_path):
    """Only ACTIVE names collide — a retired character's name is free again."""
    store = VerseStore(tmp_path, "#chan")
    old = store.add_entity("npc", "Rupert")
    store.set_status(old, "retired")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "Rupert"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "ok"


def test_verse_denial_message_names_both_reasons(tmp_path):
    """The denial must not assert "you haven't opted in" — an opted-in user on
    an ordinary chat turn hits this too, and the old text sent them to re-run
    @verse opt-in instead of the route that works."""
    from llm.verse.avatar import make_verse_denial_handlers, make_verse_tool_specs

    handlers = make_verse_denial_handlers(make_verse_tool_specs())
    payload = json.loads(handlers["verse_edit"]({}).content)
    assert "@verse opt-in" in payload["error"]
    assert "@rp" in payload["error"]
    assert "@versedit" in payload["error"]


@pytest.mark.parametrize("kind", ["avatar", "dragon"])
def test_verse_edit_add_entity_rejects_non_canon_kind(tmp_path, kind):
    """kind='avatar' would mint a ghost avatar; an unknown kind would hit the
    schema CHECK as an opaque IntegrityError. Both are rejected up front."""
    store = VerseStore(tmp_path, "#chan")
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": kind, "name": "Smaug"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert "kind" in result["detail"]
    assert store.find_entity_by_name("Smaug") is None


def test_verse_edit_update_rename_onto_existing_name_rejected(tmp_path):
    """Renaming an entity onto another active entity's name would create the
    duplicate the add path already refuses."""
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Archie")
    bob = store.add_entity("npc", "Bob")
    result = dispatch_verse_edit(
        store,
        op="update_entity",
        payload={"entity_id": bob, "name": "archie"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "error"
    assert "already exists" in result["detail"]
    entity = store.get_entity(bob)
    assert entity is not None and entity.name == "Bob"


def test_verse_edit_update_recasing_own_name_allowed(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    bob = store.add_entity("npc", "bob")
    result = dispatch_verse_edit(
        store,
        op="update_entity",
        payload={"entity_id": bob, "name": "Bob"},
        authorized=True,
        account="gm!acct",
    )
    assert result["status"] == "ok"
    entity = store.get_entity(bob)
    assert entity is not None and entity.name == "Bob"


def test_verse_edit_integrity_error_is_reported(tmp_path, monkeypatch):
    """A schema-constraint failure surfaces as status=error, not an opaque
    'Tool execution failed.'"""
    import sqlite3

    store = VerseStore(tmp_path, "#chan")

    def boom(**_kw):
        raise sqlite3.IntegrityError("CHECK constraint failed: kind")

    monkeypatch.setattr(store, "apply_direct", boom)
    result = dispatch_verse_edit(
        store,
        op="add_entity",
        payload={"kind": "npc", "name": "X"},
        authorized=True,
        account="gm!acct",
    )
    assert result == {"status": "error", "detail": "CHECK constraint failed: kind"}
