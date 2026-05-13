import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from server.config import get_settings
from server.llm import LLMClient
from server.notes import NoteStore
from server.query import answer_query
from server.router import IntentType, classify
from server.tasks import TaskStore
from server.whisper_client import WhisperClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.llm = LLMClient(settings)
    app.state.whisper = WhisperClient(settings.whisper_model)
    app.state.tasks = TaskStore(Path("data/mutter.db"))
    app.state.notes = NoteStore(settings.chroma_url)
    yield


app = FastAPI(title="Mutter", lifespan=lifespan)


class TextInput(BaseModel):
    text: str


class QueryInput(BaseModel):
    question: str


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
async def process_audio(file: UploadFile):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    transcription = app.state.whisper.transcribe_file(tmp_path)
    result = classify(app.state.llm, transcription)
    return _handle_intent(app.state, result.intent, result.content)


@app.post("/process/text")
async def process_text(body: TextInput):
    result = classify(app.state.llm, body.text)
    return _handle_intent(app.state, result.intent, result.content)


@app.get("/tasks")
async def list_tasks():
    return app.state.tasks.list_tasks()


@app.get("/notes")
async def list_notes():
    return app.state.notes.list_recent()


@app.post("/query")
async def query_kb(body: QueryInput):
    return answer_query(app.state.llm, app.state.notes, body.question)


@app.get("/health")
async def health():
    return {"status": "ok"}
