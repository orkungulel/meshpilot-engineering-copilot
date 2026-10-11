"""HTTP API.

    POST /ask     {"question": "...", "mode": "agent" | "rag"}  ->  answer, citations, tool calls
    GET  /health

The index, the embedding model and the Anthropic client are loaded once at
startup and shared by every request. Run with: uvicorn rag.api:app
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Literal

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import config
from rag.agent import ToolCall, run_agent
from rag.corpus import read_manifest
from rag.embeddings import Embedder
from rag.generate import Answer, generate_answer
from rag.index import Hit, VectorIndex
from rag.llm import LLMError, make_client

log = logging.getLogger("rag.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # "agent": the model searches (several times if needed) and calculates with tools.
    # "rag": one search, one answer.
    mode: Literal["agent", "rag"] = "agent"


class CitationOut(BaseModel):
    doc_id: str
    title: str
    page: int  # 1-based PDF page


class ToolCallOut(BaseModel):
    step: int
    tool: str
    input: dict
    output_summary: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool  # false when the documents do not contain an answer the system can cite
    citations: list[CitationOut]
    tool_calls: list[ToolCallOut]  # the agent's tool calls, in order; empty in "rag" mode
    model: str  # the model that wrote the answer


class QAService:
    """Everything a request needs, built once at startup."""

    def __init__(self, index: VectorIndex, client, model: str, titles: dict[str, str], index_name: str, k: int = 5):
        self.index = index
        self.client = client
        self.model = model
        self.titles = titles
        self.index_name = index_name
        self.k = k
        # FastAPI runs these handlers in a thread pool, and the embedding model's
        # tokenizer is not safe to share between threads. Retrieval takes
        # milliseconds, so it runs one request at a time; the slow model call does not.
        self._search_lock = threading.Lock()

    def search(self, query: str) -> list[Hit]:
        with self._search_lock:
            return self.index.search(query, self.k)

    def ask(self, question: str, mode: str = "agent") -> tuple[Answer, list[ToolCall]]:
        if mode == "rag":
            return generate_answer(self.client, self.model, question, self.search(question), self.titles), []
        result = run_agent(self.client, self.model, question, self.search, self.titles)
        return result.answer, result.tool_calls


def load_service() -> QAService:
    """Fail at startup, with a clear message, rather than on the first request."""
    if not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError("ANTHROPIC_API_KEY is not set (for Docker, pass it with --env-file .env)")
    index_config = config.INDEX_CONFIGS[config.SERVED_INDEX]
    index_dir = config.INDEX_DIR / index_config.name
    if not (index_dir / "index.faiss").exists():
        raise RuntimeError(f"no index at {index_dir}; run `python -m rag.cli build-index --config {index_config.name}`")
    index = VectorIndex.load(index_dir, Embedder(index_config.embedding_model))
    titles = {doc.doc_id: doc.title for doc in read_manifest(config.MANIFEST_PATH)}
    return QAService(index, make_client(), config.ANSWER_MODEL, titles, index_config.name)


def create_app(service: QAService | None = None) -> FastAPI:
    """The tests pass in a service built from fakes; in production it is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        app.state.service = service or load_service()
        yield

    app = FastAPI(title="Engineering document QA", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        s = app.state.service
        return {"status": "ok", "index": s.index_name, "chunks": len(s.index.chunks), "model": s.model}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        started = time.perf_counter()
        try:
            answer, tool_calls = app.state.service.ask(request.question, request.mode)
        except LLMError as exc:
            log.warning("ask failed: %s (%s)", exc.failure, exc)
            raise HTTPException(status_code=502, detail=f"the model returned no usable answer ({exc.failure})")
        except anthropic.APIError as exc:
            log.warning("ask failed: Anthropic API error: %s", exc)
            raise HTTPException(status_code=502, detail="the call to the Anthropic API failed")
        log.info(
            "ask mode=%s answerable=%s citations=%d tool_calls=%d model=%s seconds=%.1f",
            request.mode,
            answer.answerable,
            len(answer.citations),
            len(tool_calls),
            answer.model,
            time.perf_counter() - started,
        )
        return AskResponse(
            answer=answer.text,
            answerable=answer.answerable,
            citations=[CitationOut(doc_id=c.doc_id, title=c.title, page=c.page) for c in answer.citations],
            tool_calls=[
                ToolCallOut(step=c.step, tool=c.tool, input=c.input, output_summary=c.output_summary) for c in tool_calls
            ],
            model=answer.model,
        )

    return app


app = create_app()
