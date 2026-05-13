import logging
import time
from enum import Enum

from pydantic import BaseModel

from server.llm import LLMClient

log = logging.getLogger("mutter.router")


class IntentType(str, Enum):
    TASK = "task"
    NOTE = "note"
    QUERY = "query"


class RouterResult(BaseModel):
    intent: IntentType
    content: str


ROUTER_SYSTEM_PROMPT = """Classify the text into exactly one category. Pick ONE of: task, note, query.

task = the speaker wants to DO something or be REMINDED. Words like "need to", "remind me", "don't forget", "have to", "should", "by Friday".
note = the speaker is stating a fact, idea, or observation. No action needed.
query = the speaker is ASKING a question. Contains "what", "when", "how", "did I say", "?".

Return JSON with two fields:
- "intent": must be exactly one of "task", "note", or "query"
- "content": the cleaned up text

Example input: "um I need to email John by Friday"
Example output: {"intent": "task", "content": "Email John by Friday"}

Example input: "the architecture uses a three stage pipeline"
Example output: {"intent": "note", "content": "The architecture uses a three-stage pipeline"}

Example input: "what did I say about the budget"
Example output: {"intent": "query", "content": "What did I say about the budget?"}"""


def classify(llm: LLMClient, transcription: str) -> RouterResult:
    t0 = time.perf_counter()
    result = llm.complete_json(
        system=ROUTER_SYSTEM_PROMPT,
        user=transcription,
    )
    try:
        parsed = RouterResult(**result)
    except Exception:
        log.warning("[router] failed to parse LLM output, falling back to NOTE: %s", result)
        parsed = RouterResult(intent=IntentType.NOTE, content=transcription)
    elapsed = time.perf_counter() - t0
    log.info("[router] classified as %s in %.1fs", parsed.intent.value.upper(), elapsed)
    return parsed
