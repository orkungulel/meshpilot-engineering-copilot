import pytest
from fakes import FakeClaude, FakeEmbedder, FakeResponse, tool_use_reply
from fastapi.testclient import TestClient

from rag.api import QAService, create_app, load_service
from rag.chunking import Chunk
from rag.generate import NOT_FOUND
from rag.index import VectorIndex

CHUNKS = [
    Chunk("gec07:57:0", "gec07", 57, "soil nails are commonly spaced 1.5 m apart"),
    Chunk("gec10:40:0", "gec10", 40, "drilled shaft side resistance in rock sockets"),
]
TITLES = {"gec07": "Soil Nail Walls (GEC 7)", "gec10": "Drilled Shafts (GEC 10)"}


def client_for(*replies) -> TestClient:
    service = QAService(VectorIndex.build(CHUNKS, FakeEmbedder()), FakeClaude(replies), "claude-sonnet-5-5", TITLES, "R1")
    return TestClient(create_app(service))


def test_ask_returns_the_answer_with_document_and_page_citations():
    with client_for({"answerable": True, "answer": "About 1.5 m.", "citations": ["S1"]}) as client:
        response = client.post("/ask", json={"question": "How far apart are soil nails spaced?", "mode": "rag"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "About 1.5 m.",
        "answerable": True,
        "citations": [{"doc_id": "gec07", "title": "Soil Nail Walls (GEC 7)", "page": 57}],
        "tool_calls": [],
        "model": "claude-sonnet-5-5",
    }


def test_ask_says_so_when_the_documents_do_not_contain_the_answer():
    with client_for({"answerable": False, "answer": "", "citations": []}) as client:
        body = client.post("/ask", json={"question": "What does Eurocode 7 require?", "mode": "rag"}).json()

    assert body["answer"] == NOT_FOUND
    assert body["answerable"] is False
    assert body["citations"] == []


@pytest.mark.parametrize("question", ["", "x" * 2001])
def test_empty_or_oversized_questions_are_rejected(question):
    with client_for() as client:
        assert client.post("/ask", json={"question": question}).status_code == 422


def test_a_refusal_becomes_a_502_with_its_reason():
    with client_for(FakeResponse(None, stop_reason="refusal", content=[])) as client:
        response = client.post("/ask", json={"question": "How far apart are soil nails spaced?", "mode": "rag"})

    assert response.status_code == 502
    assert "refusal" in response.json()["detail"]


def test_agent_is_the_default_and_its_tool_calls_are_returned():
    replies = [
        tool_use_reply("search_documents", {"query": "soil nail spacing"}),
        tool_use_reply("calculator", {"expression": "1.5 * 4"}, call_id="call-2"),
        {"answerable": True, "answer": "Four nails span 6 m.", "citations": ["S1"]},
    ]
    with client_for(*replies) as client:
        body = client.post("/ask", json={"question": "How long a wall do four nail spacings cover?"}).json()

    assert body["answer"] == "Four nails span 6 m."
    assert body["citations"] == [{"doc_id": "gec07", "title": "Soil Nail Walls (GEC 7)", "page": 57}]
    assert [(c["step"], c["tool"], c["input"]) for c in body["tool_calls"]] == [
        (1, "search_documents", {"query": "soil nail spacing"}),
        (2, "calculator", {"expression": "1.5 * 4"}),
    ]
    assert body["tool_calls"][1]["output_summary"] == "= 6"


def test_an_unknown_mode_is_rejected():
    with client_for() as client:
        assert client.post("/ask", json={"question": "Spacing?", "mode": "magic"}).status_code == 422


def test_health_reports_the_served_index():
    with client_for() as client:
        assert client.get("/health").json() == {"status": "ok", "index": "R1", "chunks": 2, "model": "claude-sonnet-5-5"}


def test_startup_fails_clearly_without_an_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        load_service()


def test_startup_fails_clearly_without_an_index(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr("rag.config.INDEX_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="build-index"):
        load_service()
