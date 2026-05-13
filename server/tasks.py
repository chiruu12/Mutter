import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from server.llm import LLMClient


class Task(BaseModel):
    id: int | None = None
    description: str
    due: str | None = None
    priority: str = "medium"
    done: bool = False
    created_at: str | None = None


TASK_EXTRACT_PROMPT = """Extract task details from this spoken text.
Return JSON: {"description": "<what to do>", "due": "<when if mentioned else null>", "priority": "high|medium|low"}
Infer priority from urgency cues."""


class TaskStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT NOT NULL,
                    due TEXT,
                    priority TEXT DEFAULT 'medium',
                    done BOOLEAN DEFAULT 0,
                    created_at TEXT NOT NULL
                )"""
            )

    def extract_and_store(self, llm: LLMClient, content: str) -> Task:
        result = llm.complete_json(
            system=TASK_EXTRACT_PROMPT,
            user=content,
        )
        task = Task(
            description=result["description"],
            due=result.get("due"),
            priority=result.get("priority", "medium"),
            created_at=datetime.now().isoformat(),
        )
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO tasks (description, due, priority, done, created_at) VALUES (?, ?, ?, ?, ?)",
                (task.description, task.due, task.priority, task.done, task.created_at),
            )
            task.id = cursor.lastrowid
        return task

    def list_tasks(self, include_done: bool = False) -> list[Task]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            query = "SELECT * FROM tasks"
            if not include_done:
                query += " WHERE done = 0"
            query += " ORDER BY created_at DESC"
            rows = conn.execute(query).fetchall()
        return [Task(**dict(row)) for row in rows]

    def complete_task(self, task_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))

    def delete_task(self, task_id: int) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
