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


ROUTER_SYSTEM_PROMPT = """You classify spoken text into exactly one category.

TASK: The speaker wants to do something later, or is setting a reminder.
Examples: "I need to email John by Friday", "remind me to buy groceries", "don't forget the meeting at 3pm"

NOTE: The speaker is capturing a thought, idea, observation, or information.
Examples: "The main architecture uses a three-stage pipeline", "met with Sarah today, she mentioned the Q3 budget is tight"

QUERY: The speaker is asking a question about something they said before.
Examples: "what did I say about the pitch deck?", "when was that meeting with Sarah?"

Respond with JSON: {"intent": "task|note|query", "content": "<cleaned version>"}
Strip filler words. Fix grammar. Keep meaning intact."""


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
