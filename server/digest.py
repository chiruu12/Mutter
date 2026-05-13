import logging
import time
from datetime import datetime, timedelta

from pydantic import BaseModel

from server.llm import LLMClient
from server.notes import NoteStore
from server.tasks import TaskStore

log = logging.getLogger("mutter.digest")


class Digest(BaseModel):
    date: str
    summary: str
    pending_tasks: list[dict]
    notes_count: int


DIGEST_PROMPT = """Summarize this person's day based on their tasks and notes.
Be concise. Use bullet points. Highlight what's due today and what was captured.

Tasks:
{tasks}

Recent notes:
{notes}

Write a brief daily digest (3-5 bullet points max)."""


def generate_digest(llm: LLMClient, tasks: TaskStore, notes: NoteStore) -> Digest:
    t0 = time.perf_counter()
    pending = tasks.list_tasks(include_done=False)
    task_lines = "\n".join(
        f"- {t.description} (due: {t.due or 'none'}, priority: {t.priority})"
        for t in pending
    ) or "No pending tasks."

    since = datetime.now() - timedelta(hours=24)
    recent_notes = notes.list_since(since)
    note_lines = "\n".join(
        f"- {n.content[:120]}" for n in recent_notes
    ) or "No notes in the last 24 hours."

    summary = llm.complete(
        system=DIGEST_PROMPT.format(tasks=task_lines, notes=note_lines),
        user=f"Generate daily digest for {datetime.now().strftime('%B %d, %Y')}.",
    )

    elapsed = time.perf_counter() - t0
    log.info("[digest] generated in %.1fs (%d tasks, %d notes)", elapsed, len(pending), len(recent_notes))

    return Digest(
        date=datetime.now().strftime("%Y-%m-%d"),
        summary=summary,
        pending_tasks=[t.model_dump() for t in pending],
        notes_count=len(recent_notes),
    )
