"""Pytest fixtures for verse tests — real SQLite, no mocks."""

from __future__ import annotations

import json as _json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from llm.verse.store import VerseStore


@pytest.fixture
def verse_db_dir(tmp_path: Path) -> Path:
    """Per-test directory for verse SQLite files."""
    d = tmp_path / "verse"
    d.mkdir()
    return d


@pytest.fixture
def store(verse_db_dir: Path) -> VerseStore:
    """A migrated VerseStore on a real per-test SQLite file."""
    return VerseStore(verse_db_dir, "#test")


@pytest.fixture
def store_with_avatar(store: VerseStore) -> tuple[VerseStore, int]:
    """(store, avatar_id) — an opted-in avatar named 'me' for prompt/retrieval tests."""
    avatar_id = store.opt_in_avatar(nick="me", account="me-acct", instruct_text="").entity_id
    return store, avatar_id


def link_avatar(store: VerseStore, entity_id: int, nick: str, account: str | None = None) -> None:
    """Seed an avatar_link row directly (prod links only via opt_in_avatar)."""
    with store.write_transaction() as conn:
        conn.execute(
            "INSERT INTO avatar_link (entity_id, nick, account) VALUES (?, ?, ?)",
            (entity_id, nick, account),
        )


def add_alias(store: VerseStore, entity_id: int, alias: str) -> None:
    """Seed an entity_alias row directly (no prod writer exists)."""
    with store.write_transaction() as conn:
        conn.execute(
            "INSERT INTO entity_alias (entity_id, alias) VALUES (?, ?)", (entity_id, alias)
        )


def fixture_text(name: str) -> str:
    """Return the text contents of a file under
    ``plugins/llm/tests/verse/fixtures/<name>``."""
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


def insert_event_at(
    store: Any,
    *,
    summary: str,
    entity_ids: Iterable[int],
    source: str,
    ts: float,
) -> int:
    """Test helper: insert an events row with a caller-specified ``ts``.

    Production code always stamps ``ts`` to ``time.time()``; this helper
    bypasses that for retention/compaction tests. Lives in conftest so
    no production class carries test-only methods.
    """
    ids = list(entity_ids)
    with store.write_transaction() as conn:
        cur = conn.execute(
            "INSERT INTO events (ts, summary, entity_ids, source) VALUES (?, ?, ?, ?)",
            (ts, summary, _json.dumps(ids), source),
        )
        event_id = int(cur.lastrowid)
        for eid in dict.fromkeys(ids):
            if conn.execute("SELECT 1 FROM entities WHERE id=?", (eid,)).fetchone():
                conn.execute(
                    "INSERT OR IGNORE INTO event_actor (event_id, entity_id) VALUES (?, ?)",
                    (event_id, eid),
                )
        return event_id
