from llm.verse.avatar import (
    CANON_PRECEDENCE_NOTE,
    VERSE_SCENE_MARKER,
    build_story_world_context,
    build_verse_context_block,
    build_verse_system_prompt,
)
from llm.verse.store import VerseStore


def test_verse_context_block_facts_without_roleplay_framing(tmp_path):
    """The chat-path canon block carries facts (roster + referenced cast) but
    NONE of the roleplay framing — no identity line, persona, or scene marker."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    block = build_verse_context_block(store, "what would Assgas Archie say?", avatar_id=me)
    assert "Assgas Archie: Y11 windbag" in block
    # Facts only — no persona takeover.
    assert "You are" not in block
    assert VERSE_SCENE_MARKER not in block
    assert "Persona" not in block
    # Speaker isn't described back to themselves.
    assert "Hero" not in block


def test_verse_context_block_empty_when_no_canon_and_no_match(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("avatar", "Hero")
    store.add_entity("npc", "Ghost", "unpinned, unmentioned")
    assert build_verse_context_block(store, "hello there") == ""


def test_story_world_context_lists_canon(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    store.add_entity("npc", "Nobody", "not pinned")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    ctx = build_story_world_context(store)
    assert "- Assgas Archie: Y11 windbag" in ctx
    assert "Nobody" not in ctx  # unpinned → not canon


def test_story_world_context_empty_when_no_canon(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    store.add_entity("npc", "Ghost", "unpinned")
    assert build_story_world_context(store) == ""


def test_pinned_entities_appear_in_prompt(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=600)
    assert "Established characters in this world:" in prompt
    assert "Assgas Archie: Y11 windbag" in prompt


def test_roster_omitted_when_none_pinned(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=600)
    assert "Established characters in this world:" not in prompt


def test_roster_respects_char_cap(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    for i in range(30):
        e = store.add_entity("npc", f"Lad{i:02d}", "x" * 40)
        store.apply_direct(
            op="set_pinned",
            payload={"entity_id": e, "pinned": True},
            source="operator",
            provenance="t",
        )
    prompt = build_verse_system_prompt(store, me, "", roster_max_chars=200)
    roster = prompt.split("Established characters in this world:")[1].split(VERSE_SCENE_MARKER)[0]
    assert len(roster) <= 260  # cap + the truncation marker line
    assert "(roster truncated)" in roster


def test_canon_first_scene_after(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8")
    store.set_author_locked(h, True)
    t = store.add_entity("npc", "Toby", "year 9")
    store.add_relation(h, t, "rival_of")
    out = build_verse_system_prompt(
        store,
        avatar_id,
        "be a year 8 boy",
        roster_max_chars=4000,
        message_text="did Harry and Toby fight?",
    )
    assert out.index("Established characters") < out.index(VERSE_SCENE_MARKER)
    assert "Harry" in out and "Toby" in out and "rival of" in out


def test_prefix_byte_identical_when_only_message_changes(store_with_avatar):
    store, avatar_id = store_with_avatar
    h = store.add_entity("npc", "Harry", "year 8")
    store.set_author_locked(h, True)
    a = build_verse_system_prompt(
        store, avatar_id, "p", roster_max_chars=4000, message_text="hi Harry"
    )
    b = build_verse_system_prompt(
        store, avatar_id, "p", roster_max_chars=4000, message_text="yo Toby"
    )
    assert a.split(VERSE_SCENE_MARKER)[0] == b.split(VERSE_SCENE_MARKER)[0]


def _pin(store, eid):
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": eid, "pinned": True},
        source="operator",
        provenance="t",
    )


def test_roster_lines_carry_entity_ids(tmp_path):
    """Every roster/cast line starts with '#<id>'. Without it verse_edit's
    set_attribute/update_entity (which take a numeric entity_id, and have no
    name lookup tool) cannot address anything that already exists — the model
    falls back to add_entity and duplicates the roster."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    _pin(store, archie)

    block = build_verse_context_block(store, "what about Assgas Archie?", avatar_id=me)
    assert f"- #{archie} Assgas Archie: Y11 windbag" in block

    prompt = build_verse_system_prompt(store, me, "persona", message_text="Assgas Archie")
    assert f"- #{archie} Assgas Archie: Y11 windbag" in prompt


def test_roster_line_without_summary_still_carries_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    bare = store.add_entity("npc", "Nameless")
    _pin(store, bare)
    block = build_verse_context_block(store, "tell me about Nameless", avatar_id=me)
    assert f"- #{bare} Nameless" in block


def test_story_world_context_has_no_ids(tmp_path):
    """The storybook generator only needs names to stay true to canon; it never
    edits, so ids would be prose noise."""
    store = VerseStore(tmp_path, "#chan")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    _pin(store, archie)
    out = build_story_world_context(store)
    assert "- Assgas Archie: Y11 windbag" in out
    assert f"#{archie}" not in out


def test_verse_context_block_states_canon_precedence(tmp_path):
    """The chat-path canon block says canon outranks stored memories and older
    turns. Without it a frozen user memory ("the 15 stinky lads are: … year
    11 …") beats the corrected roster, because it arrives later in the
    conversation as a user-role message — prod #afternet, 2026-09-02."""
    store = VerseStore(tmp_path, "#chan")
    archie = store.add_entity("npc", "Assgas Archie", "Y12 windbag")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    block = build_verse_context_block(store, "what year is Assgas Archie in?")
    assert CANON_PRECEDENCE_NOTE in block


def test_verse_system_prompt_states_canon_precedence(tmp_path):
    """Same precedence note on the roleplay path — a verse turn is handed the
    caller's stored memories too."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y12 windbag")
    store.apply_direct(
        op="set_pinned",
        payload={"entity_id": archie, "pinned": True},
        source="operator",
        provenance="t",
    )
    prompt = build_verse_system_prompt(store, me, "")
    assert CANON_PRECEDENCE_NOTE in prompt


def test_verse_context_block_lists_unpinned_referenced_entity(tmp_path):
    """An unpinned entity the message names is not in the roster, so it must
    surface under 'Referenced here:' with its #id line — otherwise the chat
    reply is grounded in nothing and verse_edit cannot address it."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    toby = store.add_entity("npc", "Toby", "year 9 menace")
    block = build_verse_context_block(store, "what about Toby?", avatar_id=me)
    assert "Referenced here:" in block
    assert f"- #{toby} Toby: year 9 menace" in block


def test_verse_context_block_keyword_turn_carries_roster_without_named_entity(tmp_path):
    """A verseTriggerRegex hit is a canon signal on its own: the turn names no
    entity, but the facts block still has to carry the canon roster."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    _pin(store, archie)
    block = build_verse_context_block(store, "tell me a verse story", avatar_id=me)
    assert f"- #{archie} Assgas Archie: Y11 windbag" in block


def test_verse_context_block_scene_is_every_match_but_the_speaker(tmp_path):
    """The speaker's own avatar is dropped from the referenced cast; every
    other entity the message names stays in it."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero", "the speaker")
    toby = store.add_entity("npc", "Toby", "year 9 menace")
    block = build_verse_context_block(store, "Hero meets Toby", avatar_id=me)
    assert f"- #{toby} Toby: year 9 menace" in block
    assert f"#{me} Hero" not in block


def test_verse_context_block_referenced_lists_only_non_canon(tmp_path):
    """'Referenced here:' carries the matched cast NOT already in the roster:
    a pinned entity is listed once (in the roster), the unpinned one under
    'Referenced here:'."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Assgas Archie", "Y11 windbag")
    _pin(store, archie)
    toby = store.add_entity("npc", "Toby", "year 9 menace")
    block = build_verse_context_block(store, "did Assgas Archie and Toby fight?", avatar_id=me)
    referenced = block.split("Referenced here:")[1]
    assert f"- #{toby} Toby: year 9 menace" in referenced
    assert block.count(f"- #{archie} ") == 1


def test_verse_context_block_renders_relations(tmp_path):
    """The facts block carries relations (docs: 'roster, relations, recent
    events'), rendered '- <from> <kind with spaces> <to> (<note>)'."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    harry = store.add_entity("npc", "Harry", "year 8")
    _pin(store, harry)
    toby = store.add_entity("npc", "Toby", "year 9")
    store.add_relation(harry, toby, "rival_of", note="since Y7")
    block = build_verse_context_block(store, "what is Harry up to?", avatar_id=me)
    assert "Known relationships:" in block
    assert "- Harry rival of Toby (since Y7)" in block


def test_verse_context_block_lists_recent_events_of_referenced_cast(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    archie = store.add_entity("npc", "Archie", "Y11 windbag")
    store.add_event(summary="Archie fell in the pond", entity_ids=[archie], source="avatar")
    block = build_verse_context_block(store, "what happened to Archie?", avatar_id=me)
    assert "Recent events involving them:" in block
    assert "- Archie fell in the pond" in block


def test_verse_context_block_roster_respects_char_cap(tmp_path):
    """The chat-path roster is capped like the roleplay one, so a canon-signalled
    chat turn does not carry the whole pinned world."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    for i in range(30):
        _pin(store, store.add_entity("npc", f"Lad{i:02d}", "x" * 40))
    block = build_verse_context_block(
        store, "tell me a verse story", avatar_id=me, roster_max_chars=200
    )
    roster = block.split("Established characters:")[1]
    assert len(roster) <= 260  # cap + the truncation marker line
    assert "(roster truncated)" in roster


def test_verse_system_prompt_lists_unpinned_referenced_entity(tmp_path):
    """A matched entity outside the canon roster gets its own #id line under
    'Characters referenced in this scene:' (verse_edit needs the id); a pinned
    one stays in the roster only."""
    store = VerseStore(tmp_path, "#chan")
    me = store.add_entity("avatar", "Hero")
    harry = store.add_entity("npc", "Harry", "year 8")
    _pin(store, harry)
    toby = store.add_entity("npc", "Toby", "year 9")
    prompt = build_verse_system_prompt(store, me, "p", message_text="did Harry and Toby fight?")
    referenced = prompt.split("Characters referenced in this scene:")[1]
    assert f"- #{toby} Toby: year 9" in referenced
    assert prompt.count(f"- #{harry} ") == 1
