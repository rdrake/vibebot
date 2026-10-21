from llm.verse.store import VerseStore


def test_resolve_ref_numeric_name_is_not_id(tmp_path):
    store = VerseStore(tmp_path, "#chan")
    eid = store.add_entity("npc", "7")  # literally named "7"
    assert store.resolve_ref("7") == eid  # name, not id  # noqa: UP031
    assert store.resolve_ref("#%d" % eid) == eid  # explicit id form  # noqa: UP031
