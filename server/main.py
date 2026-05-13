import asyncio
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.agent import run_agent
from server.config import ModelsConfig, get_settings
from server.llm import LLMClient
from server.notes import NoteStore
from server.query import answer_query
from server.router import IntentType, classify
from server.tasks import TaskStore
from server.tools import ToolExecutor
from server.whisper_client import WhisperClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    models = ModelsConfig()
    app.state.llm = LLMClient(settings, models)
    app.state.whisper = WhisperClient(settings.whisper_model)
    app.state.tasks = TaskStore(Path("data/mutter.db"))
    app.state.notes = NoteStore(settings.chroma_url)
    app.state.tools = ToolExecutor(app.state.tasks, app.state.notes)
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
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        transcription = await asyncio.to_thread(
            app.state.whisper.transcribe_file, tmp_path
        )
        result = await asyncio.to_thread(classify, app.state.llm, transcription)
        return await asyncio.to_thread(
            _handle_intent, app.state, result.intent, result.content
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/process/text")
async def process_text(body: TextInput):
    result = await asyncio.to_thread(classify, app.state.llm, body.text)
    return await asyncio.to_thread(
        _handle_intent, app.state, result.intent, result.content
    )


@app.get("/tasks")
async def list_tasks():
    return await asyncio.to_thread(app.state.tasks.list_tasks)


@app.get("/notes")
async def list_notes():
    return await asyncio.to_thread(app.state.notes.list_recent)


@app.post("/query")
async def query_kb(body: QueryInput):
    return await asyncio.to_thread(
        answer_query, app.state.llm, app.state.notes, body.question
    )


@app.post("/agent")
async def agent_endpoint(body: AgentInput):
    response = await asyncio.to_thread(
        run_agent, app.state.llm, app.state.tools, body.message
    )
    return {"response": response}


@app.get("/health")
async def health():
    return {"status": "ok"}
