import logging
import time
import uuid
from datetime import datetime
from urllib.parse import urlparse

import chromadb
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from server.llm import LLMClient

log = logging.getLogger("mutter.notes")


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
        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 8000
        self._collection = None
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self._connect()

    def _connect(self) -> None:
        try:
            client = chromadb.HttpClient(host=self._host, port=self._port)
            self._collection = client.get_or_create_collection("mutter_notes")
            log.info("[notes] connected to ChromaDB at %s:%d", self._host, self._port)
        except Exception as e:
            log.warning("[notes] ChromaDB unavailable at startup: %s", e)
            self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._connect()
        return self._collection

    def clean_and_store(self, llm: LLMClient, content: str) -> Note:
        t0 = time.perf_counter()
        cleaned = llm.complete(
            system=NOTE_CLEANUP_PROMPT,
            user=content,
            agent="note_cleanup",
        )
        t_clean = time.perf_counter() - t0
        note_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        try:
            t1 = time.perf_counter()
            embedding = self.embedder.encode(cleaned).tolist()
            t_embed = time.perf_counter() - t1
            self.collection.add(
                ids=[note_id],
                documents=[cleaned],
                embeddings=[embedding],
                metadatas=[{"created_at": created_at, "raw": content}],
            )
            total = time.perf_counter() - t0
            log.info("[notes] stored in %.1fs (cleanup=%.1fs, embed=%.1fs)", total, t_clean, t_embed)
        except Exception as e:
            log.warning("[notes] ChromaDB unreachable, note cleaned but not stored: %s", e)
        return Note(id=note_id, content=cleaned, raw=content, created_at=created_at)

    def store_raw(self, content: str) -> Note:
        note_id = str(uuid.uuid4())
        created_at = datetime.now().isoformat()
        try:
            embedding = self.embedder.encode(content).tolist()
            self.collection.add(
                ids=[note_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[{"created_at": created_at, "raw": content}],
            )
            log.info("[notes] stored raw note %s", note_id[:8])
        except Exception as e:
            log.warning("[notes] ChromaDB unreachable, raw note not stored: %s", e)
        return Note(id=note_id, content=content, raw=content, created_at=created_at)

    def search(self, query: str, n_results: int = 5) -> list[Note]:
        t0 = time.perf_counter()
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
        elapsed = time.perf_counter() - t0
        log.info("[notes] search returned %d results in %.1fs", len(notes), elapsed)
        return notes

    def list_recent(self, limit: int = 20) -> list[Note]:
        if self._collection is None:
            self._connect()
        if self._collection is None:
            return []
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

    def list_since(self, since: datetime, limit: int = 50) -> list[Note]:
        cutoff = since.isoformat()
        all_notes = self.list_recent(limit=limit)
        return [n for n in all_notes if n.created_at >= cutoff]
