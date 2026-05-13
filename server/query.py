from pydantic import BaseModel

from server.llm import LLMClient
from server.notes import NoteStore


class QueryResult(BaseModel):
    answer: str
    sources: list[str]


QUERY_ANSWER_PROMPT = """Answer the user's question using ONLY the context provided.
If the context doesn't contain enough information, say so.
Do not make up information.

Context:
{context}

Question: {question}"""


def answer_query(llm: LLMClient, notes: NoteStore, question: str) -> QueryResult:
    results = notes.search(question, n_results=5)
    if not results:
        return QueryResult(answer="No notes found to answer this question.", sources=[])
    context = "\n\n".join(
        f"[{note.created_at}] {note.content}" for note in results
    )
    answer = llm.complete(
        system=QUERY_ANSWER_PROMPT.format(context=context, question=question),
        user=question,
    )
    source_ids = [note.id for note in results]
    return QueryResult(answer=answer, sources=source_ids)
