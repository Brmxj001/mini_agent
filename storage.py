"""SQLite 持久化：Session、消息历史与待办。"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    name TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    summary_until_message_id INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(user_id, name)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    data TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_todos_scope ON todos(user_id, session_id, status);
"""


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@dataclass
class Session:
    id: int
    user_id: str
    name: str
    summary: str
    summary_until_message_id: int


@dataclass
class Message:
    id: int
    role: str
    data: dict


@dataclass
class Todo:
    id: int
    user_id: str
    session_id: int
    content: str
    status: str
    created_at: str
    completed_at: str | None


class Storage:
    def __init__(self, db_path: str = "data/agent.db"):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------- sessions
    def get_or_create_session(self, user_id: str, name: str) -> Session:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND name = ?", (user_id, name)
        ).fetchone()
        if row is None:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO sessions"
                    "(user_id, name, summary, summary_until_message_id, created_at) "
                    "VALUES (?, ?, '', 0, ?)",
                    (user_id, name, _now()),
                )
            row = self.conn.execute(
                "SELECT * FROM sessions WHERE user_id = ? AND name = ?",
                (user_id, name),
            ).fetchone()
        return self._session(row)

    def get_session(self, session_id: int) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._session(row) if row else None

    def set_summary(self, session_id: int, summary: str, until_message_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE sessions SET summary = ?, summary_until_message_id = ? WHERE id = ?",
                (summary, until_message_id, session_id),
            )

    @staticmethod
    def _session(row: sqlite3.Row) -> Session:
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            summary=row["summary"],
            summary_until_message_id=row["summary_until_message_id"],
        )

    # ------------------------------------------------------------- messages
    def append_message(self, session_id: int, role: str, message: dict) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO messages(session_id, role, data, created_at) VALUES (?, ?, ?, ?)",
                (session_id, role, json.dumps(message, ensure_ascii=False), _now()),
            )
        return cursor.lastrowid

    def append_exchange(
        self, session_id: int, assistant_message: dict, tool_messages: list[dict]
    ) -> None:
        """assistant(tool_calls) 与其全部 tool 结果必须原子写入，避免结构断裂。"""
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages(session_id, role, data, created_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    "assistant",
                    json.dumps(assistant_message, ensure_ascii=False),
                    _now(),
                ),
            )
            for message in tool_messages:
                self.conn.execute(
                    "INSERT INTO messages(session_id, role, data, created_at) VALUES (?, ?, ?, ?)",
                    (
                        session_id,
                        "tool",
                        json.dumps(message, ensure_ascii=False),
                        _now(),
                    ),
                )

    def active_messages(self, session: Session) -> list[Message]:
        rows = self.conn.execute(
            "SELECT id, role, data FROM messages "
            "WHERE session_id = ? AND id > ? ORDER BY id",
            (session.id, session.summary_until_message_id),
        ).fetchall()
        return [
            Message(id=row["id"], role=row["role"], data=json.loads(row["data"]))
            for row in rows
        ]

    # ---------------------------------------------------------------- todos
    def add_todo(self, user_id: str, session_id: int, content: str) -> int:
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO todos(user_id, session_id, content, status, created_at) "
                "VALUES (?, ?, ?, 'pending', ?)",
                (user_id, session_id, content, _now()),
            )
        return cursor.lastrowid

    def list_todos(self, user_id: str, session_id: int) -> list[Todo]:
        rows = self.conn.execute(
            "SELECT * FROM todos WHERE user_id = ? AND session_id = ? ORDER BY id",
            (user_id, session_id),
        ).fetchall()
        return [self._todo(row) for row in rows]

    def complete_todo(
        self, user_id: str, session_id: int, todo_id: int
    ) -> Todo | None:
        row = self.conn.execute(
            "SELECT * FROM todos WHERE id = ? AND user_id = ? AND session_id = ?",
            (todo_id, user_id, session_id),
        ).fetchone()
        if row is None:
            return None
        if row["status"] != "completed":
            with self.conn:
                self.conn.execute(
                    "UPDATE todos SET status = 'completed', completed_at = ? WHERE id = ?",
                    (_now(), todo_id),
                )
        return self._todo(
            self.conn.execute("SELECT * FROM todos WHERE id = ?", (todo_id,)).fetchone()
        )

    @staticmethod
    def _todo(row: sqlite3.Row) -> Todo:
        return Todo(
            id=row["id"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            content=row["content"],
            status=row["status"],
            created_at=row["created_at"],
            completed_at=row["completed_at"],
        )
