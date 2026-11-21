
from __future__ import annotations

import os
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.assistant_store import database_connection

MEMORY_CATEGORIES = frozenset(
    {"profile", "preference", "instruction", "project", "other"}
)
MAX_KEY_LENGTH = 120
MAX_VALUE_LENGTH = 5000
MAX_SEARCH_LENGTH = 200
DEFAULT_LIMIT = 50
MAX_LIMIT = 100


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _owner_id(owner_id: str | None = None) -> str:
    value = (owner_id or os.getenv("ASSISTANT_OWNER_ID", "local-user")).strip()
    if not value or len(value) > 120:
        raise ValueError("owner_id must contain 1 to 120 characters.")
    return value


def _normalise_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().casefold()


def _validate_memory(
    category: str,
    memory_key: str,
    memory_value: str,
) -> tuple[str, str, str]:
    category = category.strip().lower()
    memory_key = memory_key.strip()
    memory_value = memory_value.strip()

    if category not in MEMORY_CATEGORIES:
        raise ValueError(
            f"category must be one of: {', '.join(sorted(MEMORY_CATEGORIES))}."
        )
    if not memory_key or len(memory_key) > MAX_KEY_LENGTH:
        raise ValueError(
            f"memory_key must contain 1 to {MAX_KEY_LENGTH} characters."
        )
    if not memory_value or len(memory_value) > MAX_VALUE_LENGTH:
        raise ValueError(
            f"memory_value must contain 1 to {MAX_VALUE_LENGTH} characters."
        )
    return category, memory_key, memory_value


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def init_memory_db(db_path: str | Path | None = None) -> None:
    """Create memory storage and indexes without touching existing tables."""
    with database_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS assistant_memories (
                memory_id TEXT PRIMARY KEY,
                owner_id TEXT NOT NULL,
                category TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                normalized_key TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'user_explicit',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(owner_id, category, normalized_key)
            )
            """
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_assistant_memories_owner_category
               ON assistant_memories(owner_id, category, updated_at DESC)"""
        )
        connection.execute(
            """CREATE INDEX IF NOT EXISTS idx_assistant_memories_owner_updated
               ON assistant_memories(owner_id, updated_at DESC)"""
        )


def save_memory(
    *,
    memory_key: str,
    memory_value: str,
    category: str = "profile",
    source: str = "user_explicit",
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Insert a memory or update the same owner/category/key in place."""
    category, memory_key, memory_value = _validate_memory(
        category, memory_key, memory_value
    )
    if source != "user_explicit":
        raise ValueError("Only explicitly user-provided memories may be saved in Batch 2.")

    resolved_owner = _owner_id(owner_id)
    normalized_key = _normalise_key(memory_key)
    now = _now()
    proposed_id = str(uuid.uuid4())
    init_memory_db(db_path)

    with database_connection(db_path, immediate=True) as connection:
        connection.execute(
            """
            INSERT INTO assistant_memories (
                memory_id, owner_id, category, memory_key, normalized_key,
                memory_value, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(owner_id, category, normalized_key)
            DO UPDATE SET
                memory_key = excluded.memory_key,
                memory_value = excluded.memory_value,
                source = excluded.source,
                updated_at = excluded.updated_at
            """,
            (
                proposed_id,
                resolved_owner,
                category,
                memory_key,
                normalized_key,
                memory_value,
                source,
                now,
                now,
            ),
        )
        row = connection.execute(
            """SELECT memory_id, owner_id, category, memory_key,
                      memory_value, source, created_at, updated_at
               FROM assistant_memories
               WHERE owner_id = ? AND category = ? AND normalized_key = ?""",
            (resolved_owner, category, normalized_key),
        ).fetchone()

    result = _row_to_dict(row)
    if result is None:  # Defensive: the upsert must always produce a row.
        raise RuntimeError("Memory save completed without a retrievable record.")
    return result


def get_memory(
    memory_id: str,
    *,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    init_memory_db(db_path)
    with database_connection(db_path) as connection:
        row = connection.execute(
            """SELECT memory_id, owner_id, category, memory_key,
                      memory_value, source, created_at, updated_at
               FROM assistant_memories WHERE memory_id = ? AND owner_id = ?""",
            (memory_id, _owner_id(owner_id)),
        ).fetchone()
    return _row_to_dict(row)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def list_memories(
    *,
    query: str | None = None,
    category: str | None = None,
    limit: int = DEFAULT_LIMIT,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """List memories, optionally filtering by category and literal text."""
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"limit must be between 1 and {MAX_LIMIT}.")
    if category is not None:
        category = category.strip().lower()
        if category not in MEMORY_CATEGORIES:
            raise ValueError(
                f"category must be one of: {', '.join(sorted(MEMORY_CATEGORIES))}."
            )
    if query is not None:
        query = query.strip()
        if len(query) > MAX_SEARCH_LENGTH:
            raise ValueError(f"query cannot exceed {MAX_SEARCH_LENGTH} characters.")

    clauses = ["owner_id = ?"]
    parameters: list[Any] = [_owner_id(owner_id)]
    if category:
        clauses.append("category = ?")
        parameters.append(category)
    if query:
        pattern = f"%{_escape_like(query)}%"
        clauses.append("(memory_key LIKE ? ESCAPE '\\' OR memory_value LIKE ? ESCAPE '\\')")
        parameters.extend([pattern, pattern])
    parameters.append(limit)

    init_memory_db(db_path)
    with database_connection(db_path) as connection:
        rows = connection.execute(
            f"""SELECT memory_id, owner_id, category, memory_key,
                       memory_value, source, created_at, updated_at
                FROM assistant_memories
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC
                LIMIT ?""",
            parameters,
        ).fetchall()
    return [dict(row) for row in rows]


def update_memory(
    memory_id: str,
    *,
    memory_key: str,
    memory_value: str,
    category: str,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Update an existing memory; return None when it is not owned/found."""
    category, memory_key, memory_value = _validate_memory(
        category, memory_key, memory_value
    )
    resolved_owner = _owner_id(owner_id)
    now = _now()
    init_memory_db(db_path)

    try:
        with database_connection(db_path, immediate=True) as connection:
            cursor = connection.execute(
                """UPDATE assistant_memories
                   SET category = ?, memory_key = ?, normalized_key = ?,
                       memory_value = ?, updated_at = ?
                   WHERE memory_id = ? AND owner_id = ?""",
                (
                    category,
                    memory_key,
                    _normalise_key(memory_key),
                    memory_value,
                    now,
                    memory_id,
                    resolved_owner,
                ),
            )
            if cursor.rowcount == 0:
                return None
            row = connection.execute(
                """SELECT memory_id, owner_id, category, memory_key,
                          memory_value, source, created_at, updated_at
                   FROM assistant_memories WHERE memory_id = ? AND owner_id = ?""",
                (memory_id, resolved_owner),
            ).fetchone()
    except sqlite3.IntegrityError as exc:
        raise ValueError("A memory with this category and key already exists.") from exc

    return _row_to_dict(row)


def delete_memory(
    memory_id: str,
    *,
    owner_id: str | None = None,
    db_path: str | Path | None = None,
) -> bool:
    init_memory_db(db_path)
    with database_connection(db_path, immediate=True) as connection:
        cursor = connection.execute(
            "DELETE FROM assistant_memories WHERE memory_id = ? AND owner_id = ?",
            (memory_id, _owner_id(owner_id)),
        )
    return cursor.rowcount > 0

