
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def database_path(db_path: str | Path | None = None) -> Path:
    """Resolve the shared assistant DB path and ensure its parent exists."""
    configured = db_path if db_path is not None else os.getenv(
        "UNIFIED_AI_DB_PATH", "cache.db"
    )
    path = Path(configured).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def database_connection(
    db_path: str | Path | None = None,
    *,
    immediate: bool = False,
) -> Iterator[sqlite3.Connection]:
    """Yield a transactional SQLite connection and always close it."""
    connection = sqlite3.connect(
        str(database_path(db_path)),
        timeout=30,
        isolation_level="DEFERRED",
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        if immediate:
            connection.execute("BEGIN IMMEDIATE")
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

