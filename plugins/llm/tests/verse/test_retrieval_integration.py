"""End-to-end: a locked roster member absent from the message still appears;
a scene-named member + relation + event appear; a retired entity does not."""

from llm.verse.avatar import VERSE_SCENE_MARKER, build_verse_system_prompt

from .conftest import add_alias


def test_full_retrieval(store_with_avatar):
    store, avatar_id = store_with_avatar
    harry = store.add_entity("npc", "Harry", "year 8 ringleader")
    toby = store.add_entity("npc", "Toby", "year 9")
    ghost = store.add_entity("npc", "Ghost")
    store.set_author_locked(harry, True)  # locked roster (not named in message)
    add_alias(store, toby, "Tobes")
    store.add_relation(harry, toby, "rival_of")
    store.add_event("Toby nicked the register", [toby], source="avatar")
    store.add_event("Ghost vanished", [ghost], source="avatar")
    store.set_status(ghost, "retired")

    out = build_verse_system_prompt(
        store,
        avatar_id,
        "be a year 8 boy",
        roster_max_chars=4000,
        message_text="what's Tobes up to?",
    )
    prefix, scene = out.split(VERSE_SCENE_MARKER, 1)
    assert "Harry" in prefix  # locked roster member, even though unmentioned
    assert "Toby" in scene  # resolved via alias 'Tobes'
    assert "rival of" in scene  # 1-hop relation surfaced (kind underscores -> spaces)
    assert "register" in scene  # Toby's event surfaced via event_actor
    assert "Ghost" not in out  # retired -> excluded everywhere
