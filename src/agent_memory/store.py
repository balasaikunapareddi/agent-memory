"""Long-term memory store (SQLite).

Everything the agent must not forget lives here: facts about a user,
learned preferences, and the conversation turns of every session.
The store is a plain SQLite file, so memory survives process restarts
with no network and no API key.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """SQLite-backed long-term memory, keyed by user id."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        # Agent frameworks (CrewAI included) execute tools on worker
        # threads, not the thread that built the store. check_same_thread
        # must be off, and every access goes through the lock.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS facts (
                user_id    TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );
            CREATE TABLE IF NOT EXISTS preferences (
                user_id    TEXT NOT NULL,
                key        TEXT NOT NULL,
                value      TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (user_id, key)
            );
            CREATE TABLE IF NOT EXISTS turns (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

    # -- facts ---------------------------------------------------------

    def remember_fact(self, user_id: str, key: str, value: str) -> None:
        """Store (or overwrite) one fact about a user."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO facts (user_id, key, value, created_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
                "created_at = excluded.created_at",
                (user_id, key, value, _now()),
            )
            self._conn.commit()

    def recall_facts(self, user_id: str, key: str | None = None) -> dict[str, str]:
        """Return all facts for a user, or just one when key is given."""
        with self._lock:
            if key is None:
                rows = self._conn.execute(
                    "SELECT key, value FROM facts WHERE user_id = ? ORDER BY key",
                    (user_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT key, value FROM facts WHERE user_id = ? AND key = ?",
                    (user_id, key),
                ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def forget_fact(self, user_id: str, key: str) -> bool:
        """Delete one fact. Returns True if something was deleted."""
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM facts WHERE user_id = ? AND key = ?", (user_id, key)
            )
            self._conn.commit()
            return cur.rowcount > 0

    # -- preferences ---------------------------------------------------

    def set_preference(self, user_id: str, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO preferences (user_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value, "
                "updated_at = excluded.updated_at",
                (user_id, key, value, _now()),
            )
            self._conn.commit()

    def get_preferences(self, user_id: str) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM preferences WHERE user_id = ? ORDER BY key",
                (user_id,),
            ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    # -- conversation turns --------------------------------------------

    def add_turn(self, session_id: str, user_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, user_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, user_id, role, content, _now()),
            )
            self._conn.commit()

    def get_turns(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM turns WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
