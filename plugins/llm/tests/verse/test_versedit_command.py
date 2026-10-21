import pytest
from llm.verse.store import VerseStore


def test_versedit_retire_clears_avatar_link(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    res = store.opt_in_avatar(nick="bob", account="bob!acct", instruct_text="")
    store.apply_direct(
        op="set_status",
        payload={"entity_id": res.entity_id, "status": "retired"},
        source="operator",
        provenance="t",
    )
    assert store.find_avatar_by_nick("bob") is None


class TestVerseditCommand:
    """Integration: drive @versedit through the wrapped Limnoria command."""

    @pytest.fixture
    def verse_env(self, plugin_env, tmp_path, mocker):
        plugin, irc, msg = plugin_env
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
        msg.args = ("#afnet", "")
        msg.prefix = "alice!user@host"
        msg.nick = "alice"
        return plugin, irc, msg, store

    def test_add_npc_replies_and_persists(self, verse_env):
        plugin, irc, msg, store = verse_env
        plugin.versedit(irc, msg, ["add", "npc", "Archie", "::", "stinky"])
        reply = irc.reply.call_args[0][0]
        assert "added npc" in reply
        assert "Archie" in reply
        assert store.active_name_exists("Archie")
        ent_id = store.resolve_ref("Archie")
        assert store.get_entity(ent_id).summary == "stinky"

    def test_add_with_leading_channel_works_in_pm(self, verse_env):
        """A leading #channel token targets that channel from a private message.

        In a PM the message carries no channel of its own (msg.channel is None
        and msg.args[0] is the bot's nick), so without the leading-channel
        escape hatch the command can only error. This lets operators batch
        edits in a DM without flooding the channel.
        """
        plugin, irc, msg, store = verse_env
        msg.channel = None
        msg.args = ("testbot", "")
        plugin.versedit(irc, msg, ["#afnet", "add", "npc", "Archie", "::", "stinky"])
        assert not irc.error.called
        reply = irc.reply.call_args[0][0]
        assert "added npc" in reply
        assert store.active_name_exists("Archie")
        assert store.get_entity(store.resolve_ref("Archie")).summary == "stinky"

    def test_unknown_verb_errors(self, verse_env):
        plugin, irc, msg, _store = verse_env
        plugin.versedit(irc, msg, ["frobnicate", "stuff"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("unknown verb" in e for e in errors)

    # --- add error branches --------------------------------------------

    def test_add_bad_usage_errors(self, verse_env):
        plugin, irc, msg, _store = verse_env
        plugin.versedit(irc, msg, ["add", "npc"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("usage: versedit add" in e for e in errors)

    def test_add_bad_kind_errors(self, verse_env):
        plugin, irc, msg, _store = verse_env
        plugin.versedit(irc, msg, ["add", "dragon", "Smaug"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("kind must be" in e for e in errors)

    def test_add_duplicate_name_errors(self, verse_env):
        plugin, irc, msg, store = verse_env
        store.add_entity("npc", "Archie")
        plugin.versedit(irc, msg, ["add", "npc", "Archie"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("already exists" in e for e in errors)

    # --- pin / unpin ---------------------------------------------------

    def test_pin_and_unpin(self, verse_env):
        plugin, irc, msg, store = verse_env
        eid = store.add_entity("npc", "Archie")
        plugin.versedit(irc, msg, ["pin", f"#{eid}"])
        assert [e.name for e in store.list_canon_entities()] == ["Archie"]
        plugin.versedit(irc, msg, ["unpin", f"#{eid}"])
        assert store.list_canon_entities() == []
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any(r.startswith("pinned #") for r in replies)
        assert any(r.startswith("unpinned #") for r in replies)

    # --- set -----------------------------------------------------------

    def test_set_attribute(self, verse_env):
        plugin, irc, msg, store = verse_env
        eid = store.add_entity("npc", "Archie")
        plugin.versedit(irc, msg, ["set", f"#{eid}", "mood", "grumpy"])
        assert store.list_attributes(eid)["mood"] == "grumpy"
        assert "set mood" in irc.reply.call_args[0][0]

    # --- name / desc ---------------------------------------------------

    def test_name_renames(self, verse_env):
        plugin, irc, msg, store = verse_env
        eid = store.add_entity("npc", "Archie")
        plugin.versedit(irc, msg, ["name", f"#{eid}", "Archibald"])
        assert store.get_entity(eid).name == "Archibald"
        assert "renamed" in irc.reply.call_args[0][0]

    def test_name_duplicate_errors(self, verse_env):
        plugin, irc, msg, store = verse_env
        eid = store.add_entity("npc", "Archie")
        store.add_entity("npc", "Betty")
        plugin.versedit(irc, msg, ["name", f"#{eid}", "Betty"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("already exists" in e for e in errors)

    def test_desc_updates_summary(self, verse_env):
        plugin, irc, msg, store = verse_env
        eid = store.add_entity("npc", "Archie")
        plugin.versedit(irc, msg, ["desc", f"#{eid}", "::", "a stinky goblin"])
        assert store.get_entity(eid).summary == "a stinky goblin"
        assert "updated summary" in irc.reply.call_args[0][0]

    # --- retire / restore ----------------------------------------------

    def test_retire_then_restore(self, verse_env):
        plugin, irc, msg, store = verse_env
        eid = store.add_entity("npc", "Archie")
        plugin.versedit(irc, msg, ["retire", f"#{eid}"])
        assert store.get_entity(eid).status == "retired"
        plugin.versedit(irc, msg, ["restore", f"#{eid}"])
        assert store.get_entity(eid).status == "active"
        replies = [c.args[0] for c in irc.reply.call_args_list]
        assert any(r.startswith("retired #") for r in replies)
        assert any(r.startswith("restored #") for r in replies)

    # --- relate / unrelate ---------------------------------------------

    def test_relate_and_unrelate(self, verse_env):
        plugin, irc, msg, store = verse_env
        a = store.add_entity("npc", "Archie")
        b = store.add_entity("npc", "Betty")
        plugin.versedit(irc, msg, ["relate", f"#{a}", "knows", f"#{b}", "::", "old friends"])
        reply = irc.reply.call_args[0][0]
        assert "related" in reply
        rels = store.list_relations(a)
        assert rels and rels[0].kind == "knows"
        rid = rels[0].id
        plugin.versedit(irc, msg, ["unrelate", str(rid)])
        assert store.list_relations(a) == []
        assert "deleted relation" in irc.reply.call_args[0][0]

    # --- event / editevent / delevent ----------------------------------

    def test_event_editevent_delevent(self, verse_env):
        plugin, irc, msg, store = verse_env
        a = store.add_entity("npc", "Archie")
        plugin.versedit(irc, msg, ["event", "Archie", "parps", f"@{a}"])
        ev = store.recent_events(limit=5)[0]
        assert ev.summary == "Archie parps"
        assert "added event" in irc.reply.call_args[0][0]
        plugin.versedit(irc, msg, ["editevent", str(ev.id), "::", "Archie parps loudly"])
        assert store.recent_events(limit=5)[0].summary == "Archie parps loudly"
        assert "edited event" in irc.reply.call_args[0][0]
        plugin.versedit(irc, msg, ["delevent", str(ev.id)])
        assert store.recent_events(limit=5) == []
        assert "deleted event" in irc.reply.call_args[0][0]

    def test_editevent_missing_errors(self, verse_env):
        plugin, irc, msg, _store = verse_env
        plugin.versedit(irc, msg, ["editevent", "999", "::", "nope"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("does not exist" in e for e in errors)

    # --- show ----------------------------------------------------------

    def test_show_renders_entity(self, verse_env):
        plugin, irc, msg, store = verse_env
        eid = store.add_entity("npc", "Archie", summary="stinky")
        plugin.versedit(irc, msg, ["show", f"#{eid}"])
        reply = irc.reply.call_args[0][0]
        assert f"#{eid}" in reply
        assert "Archie" in reply
        assert "stinky" in reply

    def test_unknown_ref_errors(self, verse_env):
        plugin, irc, msg, _store = verse_env
        plugin.versedit(irc, msg, ["show", "Nonesuch"])
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("no active entity named" in e for e in errors)

    # --- channel guard -------------------------------------------------

    def test_non_channel_errors(self, verse_env, mocker):
        plugin, irc, msg, _store = verse_env
        # Drive the unwrapped command body directly so we exercise the
        # "not a channel" guard: pass a non-channel target with no channel arg.
        # Limnoria's wrap() keeps the original callable as the first closure
        # cell; reach through it to bypass the wrap-layer channel injection.
        original = type(plugin).versedit.__closure__[0].cell_contents
        mocker.patch("llm.plugin.ircutils.isChannel", return_value=False)
        msg.args = ("alice", "")  # PM, not a channel
        original(plugin, irc, msg, [], "show Archie")
        errors = [c.args[0] for c in irc.error.call_args_list]
        assert any("Specify a channel" in e for e in errors)
