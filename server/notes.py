import uuid
from datetime import datetime
from urllib.parse import urlparse

import chromadb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from server.llm import LLMClient


class Note(BaseModel):
    id: str
    content: str
    raw: str
    created_at: str


NOTE_CLEANUP_PROMPT = """Clean up this spoken transcription into a well-written note.
Remove filler words. Fix grammar and punctuation. Keep meaning and tone.
Do not add information. Return just the cleaned text."""


class NoteStore:
    def __init__(self, chroma_url: str) -> None:
        parsed = urlparse(chroma_url)
        self.client = chromadb.HttpClient(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8000,
        )
        self.collection = self.client.get_or_create_collection("mutter_notes")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def clean_and_store(self, llm: LLMClient, content: str) -> Note:
        cleaned = llm.complete(
            system=NOTE_CLEANUP_PROMPT,
            user=content,
        )
        note_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        embedding = self.embedder.encode(cleaned).tolist()
        self.collection.add(
            ids=[note_id],
            documents=[cleaned],
            embeddings=[embedding],
            metadatas=[{"created_at": created_at, "raw": content}],
        )
        return Note(id=note_id, content=cleaned, raw=content, created_at=created_at)

    def search(self, query: str, n_results: int = 5) -> list[Note]:
        embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
        )
        notes = []
        for i, doc_id in enumerate(results["ids"][0]):
            notes.append(
                Note(
                    id=doc_id,
                    content=results["documents"][0][i],
                    raw=results["metadatas"][0][i].get("raw", ""),
                    created_at=results["metadatas"][0][i].get("created_at", ""),
                )
            )
        return notes

    def list_recent(self, limit: int = 20) -> list[Note]:
        results = self.collection.get(
            include=["documents", "metadatas"],
            limit=limit,
        )
        notes = []
        for i, doc_id in enumerate(results["ids"]):
            notes.append(
                Note(
                    id=doc_id,
                    content=results["documents"][i],
                    raw=results["metadatas"][i].get("raw", ""),
                    created_at=results["metadatas"][i].get("created_at", ""),
                )
            )
        notes.sort(key=lambda n: n.created_at, reverse=True)
        return notes
