import asyncio
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.agent import run_agent
from server.config import ModelsConfig, get_settings
from server.digest import generate_digest
from server.llm import LLMClient, LLMError
from server.notes import NoteStore
from server.query import answer_query
from server.router import IntentType, classify
from server.tasks import TaskStore
from server.tools import ToolExecutor
from server.whisper_client import WhisperClient

log = logging.getLogger("mutter.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    models = ModelsConfig()
    app.state.llm = LLMClient(settings, models)
    app.state.whisper = WhisperClient(settings.whisper_model)
    app.state.tasks = TaskStore(Path("data/mutter.db"))
    app.state.notes = NoteStore(settings.chroma_url)
    app.state.tools = ToolExecutor(app.state.tasks, app.state.notes)
    log.info("[server] started on %s:%d", settings.server_host, settings.server_port)
    yield


app = FastAPI(title="Mutter", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextInput(BaseModel):
    text: str


class QueryInput(BaseModel):
    question: str


class AgentInput(BaseModel):
    message: str


def _handle_intent(app_state, intent_type: IntentType, content: str) -> dict:
    if intent_type == IntentType.TASK:
        task = app_state.tasks.extract_and_store(app_state.llm, content)
        return {"intent": "task", **task.model_dump()}
    elif intent_type == IntentType.NOTE:
        note = app_state.notes.clean_and_store(app_state.llm, content)
        return {"intent": "note", **note.model_dump()}
    else:
        result = answer_query(app_state.llm, app_state.notes, content)
        return {"intent": "query", **result.model_dump()}


@app.post("/process")
async def process_audio(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        try:
            t_whisper = time.perf_counter()
            transcription = await asyncio.to_thread(
                app.state.whisper.transcribe_file, tmp_path
            )
            whisper_ms = int((time.perf_counter() - t_whisper) * 1000)
        except Exception as e:
            log.error("[server] whisper failed: %s", e)
            raise HTTPException(status_code=422, detail=f"Transcription failed: {e}")
        t_router = time.perf_counter()
        result = await asyncio.to_thread(classify, app.state.llm, transcription)
        router_ms = int((time.perf_counter() - t_router) * 1000)
        t_handler = time.perf_counter()
        response = await asyncio.to_thread(
            _handle_intent, app.state, result.intent, result.content
        )
        handler_ms = int((time.perf_counter() - t_handler) * 1000)
        total_ms = int((time.perf_counter() - t0) * 1000)
        response["transcription"] = transcription
        response["pipeline"] = {
            "whisper_ms": whisper_ms,
            "router_ms": router_ms,
            "handler_ms": handler_ms,
            "total_ms": total_ms,
        }
        log.info("[server] /process completed in %dms", total_ms)
        return response
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/process/text")
async def process_text(body: TextInput):
    t0 = time.perf_counter()
    try:
        t_router = time.perf_counter()
        result = await asyncio.to_thread(classify, app.state.llm, body.text)
        router_ms = int((time.perf_counter() - t_router) * 1000)
        t_handler = time.perf_counter()
        response = await asyncio.to_thread(
            _handle_intent, app.state, result.intent, result.content
        )
        handler_ms = int((time.perf_counter() - t_handler) * 1000)
        total_ms = int((time.perf_counter() - t0) * 1000)
        response["transcription"] = body.text
        response["pipeline"] = {
            "whisper_ms": 0,
            "router_ms": router_ms,
            "handler_ms": handler_ms,
            "total_ms": total_ms,
        }
        log.info("[server] /process/text completed in %dms", total_ms)
        return response
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/tasks")
async def list_tasks():
    return await asyncio.to_thread(app.state.tasks.list_tasks)


@app.get("/notes")
async def list_notes():
    return await asyncio.to_thread(app.state.notes.list_recent)


@app.post("/query")
async def query_kb(body: QueryInput):
    try:
        return await asyncio.to_thread(
            answer_query, app.state.llm, app.state.notes, body.question
        )
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/agent")
async def agent_endpoint(body: AgentInput):
    try:
        response = await asyncio.to_thread(
            run_agent, app.state.llm, app.state.tools, body.message
        )
        return {"response": response}
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/digest")
async def digest():
    try:
        return await asyncio.to_thread(
            generate_digest, app.state.llm, app.state.tasks, app.state.notes
        )
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))


DICTATION_CLEANUP_PROMPT = """Clean up this speech-to-text transcription for typing.
Fix misheard words, spelling, and punctuation. Remove filler words (um, uh, like, you know).
If the speaker repeated themselves, keep only the final version.
Preserve the speaker's exact intended meaning and phrasing — do not summarize or rephrase.
Return just the cleaned text, nothing else."""


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        raw = await asyncio.to_thread(
            app.state.whisper.transcribe_file, tmp_path
        )
        if not raw.strip():
            return {"text": "", "raw": ""}
        cleaned = await asyncio.to_thread(
            app.state.llm.complete,
            DICTATION_CLEANUP_PROMPT,
            raw,
            None,
            "note_cleanup",
        )
        elapsed = time.perf_counter() - t0
        log.info("[server] /transcribe completed in %.1fs (raw=%d chars, cleaned=%d chars)", elapsed, len(raw), len(cleaned))
        return {"text": cleaned.strip(), "raw": raw}
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.error("[server] transcribe failed: %s", e)
        raise HTTPException(status_code=422, detail=f"Transcription failed: {e}")
    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
async def health():
    status = {"status": "ok", "chroma": "disconnected", "llm": "disconnected"}
    try:
        coll = app.state.notes.collection
        if coll is not None:
            await asyncio.to_thread(coll.count)
            status["chroma"] = "connected"
        else:
            status["status"] = "degraded"
    except Exception:
        status["status"] = "degraded"
    try:
        await asyncio.to_thread(
            app.state.llm.complete, "Say ok.", "test", 0.0, "router"
        )
        status["llm"] = "connected"
    except Exception:
        status["status"] = "degraded"
    return status
