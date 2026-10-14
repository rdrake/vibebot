"""Shared payload validation for verse proposal/edit ops.

Relocated from the (removed) loom module so the verse_edit tool (avatar.py)
and the @versedit path can validate op payloads without depending on the
loom subsystem. One schema governs all constructive ops; an op with no
schema entry is rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _is_strict_int(v: Any) -> bool:
    """Reject bool, accept int. (bool is a subclass of int in Python.)"""
    return isinstance(v, int) and not isinstance(v, bool)


def _is_int_list(v: Any) -> bool:
    return isinstance(v, list) and all(_is_strict_int(x) for x in v)


_PAYLOAD_SCHEMA: dict[str, tuple[tuple[str, Callable[[Any], bool], str], ...]] = {
    "add_event": (
        ("summary", lambda v: isinstance(v, str), "str"),
        ("entity_ids", _is_int_list, "list[int]"),
    ),
    "set_attribute": (
        ("entity_id", _is_strict_int, "int"),
        ("key", lambda v: isinstance(v, str), "str"),
        ("value", lambda v: isinstance(v, str), "str"),
    ),
    "add_relation": (
        ("from_id", _is_strict_int, "int"),
        ("to_id", _is_strict_int, "int"),
        ("kind", lambda v: isinstance(v, str), "str"),
    ),
    "add_entity": (
        # Not 'avatar': avatars are created by opt-in, never by canon edits.
        ("kind", lambda v: v in ("npc", "place", "faction", "item"), "npc|place|faction|item"),
        ("name", lambda v: isinstance(v, str), "str"),
    ),
    "update_entity": (("entity_id", _is_strict_int, "int"),),
}


def validate_payload(op: str, payload: dict[str, Any]) -> str | None:
    """Return None if *payload* is valid for *op*, else a human reason string.

    Only constructive ops have entries; an op without a schema entry is
    rejected. Used by the verse_edit tool (avatar.py).
    """
    schema = _PAYLOAD_SCHEMA.get(op)
    if schema is None:
        return f"unknown or non-constructive op: {op!r}"
    for key, predicate, label in schema:
        if key not in payload:
            return f"missing {key}"
        if not predicate(payload[key]):
            return f"{key} not {label}"
    return None
