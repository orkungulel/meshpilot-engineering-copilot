import logging

import pytest
from fakes import FakeClaude, FakeEmbedder, FakeResponse
from fakes import tool_use_reply as tool_use

from rag.agent import MAX_STEPS, ToolCall, run_agent
from rag.chunking import Chunk
from rag.generate import NOT_FOUND, Citation
from rag.index import VectorIndex
from rag.llm import LLMError

CHUNKS = [
    Chunk("gec07:57:0", "gec07", 57, "soil nails are commonly spaced 1.5 m apart"),
    Chunk("gec07:58:0", "gec07", 58, "nail inclination is typically 15 degrees"),
    Chunk("gec10:40:0", "gec10", 40, "drilled shaft side resistance in rock sockets"),
]
TITLES = {"gec07": "Soil Nail Walls (GEC 7)", "gec10": "Drilled Shafts (GEC 10)"}
INDEX = VectorIndex.build(CHUNKS, FakeEmbedder())


def search(query: str):
    return INDEX.search(query, 2)


def final(citations=("S1",), answerable=True, answer="About 1.5 m.") -> dict:
    return {"answerable": answerable, "answer": answer, "citations": list(citations)}


def run(*replies):
    claude = FakeClaude(replies)
    return run_agent(claude, "claude-sonnet-5-5", "How far apart are soil nails spaced?", search, TITLES), claude


def last_tool_result(request: dict) -> dict:
    return request["messages"][-1]["content"][0]


def test_search_then_answer_cites_the_page_and_logs_the_call():
    result, claude = run(tool_use("search_documents", {"query": "soil nail spacing"}), final(["S1"]))

    assert result.answer.text == "About 1.5 m."
    assert result.answer.citations == [Citation("gec07", "Soil Nail Walls (GEC 7)", 57)]
    assert result.steps == 2
    [call] = result.tool_calls
    assert (call.step, call.tool, call.input) == (1, "search_documents", {"query": "soil nail spacing"})
    assert call.output_summary.startswith("S1 gec07 p.57")
    tool_result = last_tool_result(claude.requests[1])
    assert tool_result["tool_use_id"] == "call-1" and not tool_result["is_error"]
    assert '<source id="S1" document="Soil Nail Walls (GEC 7)" page="57">' in tool_result["content"]


def test_calculator_result_goes_back_to_the_model():
    result, claude = run(
        tool_use("search_documents", {"query": "nail spacing"}),
        tool_use("calculator", {"expression": "1.5 * 4"}, call_id="call-2"),
        final(["S1"], answer="Four nails span 6 m."),
    )

    assert last_tool_result(claude.requests[2])["content"] == "6"
    assert [c.tool for c in result.tool_calls] == ["search_documents", "calculator"]
    assert result.tool_calls[1].output_summary == "= 6"


def test_at_most_one_tool_per_step_and_no_tools_on_the_last_step():
    searches = [tool_use("search_documents", {"query": f"query {i}"}, call_id=f"call-{i}") for i in range(MAX_STEPS - 1)]

    result, claude = run(*searches, final(["S1"]))

    assert result.steps == MAX_STEPS
    assert len(claude.requests) == MAX_STEPS
    for request in claude.requests[:-1]:
        assert request["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert claude.requests[-1]["tool_choice"] == {"type": "none"}


def test_tool_errors_are_reported_to_the_model_and_the_run_continues():
    result, claude = run(
        tool_use("calculator", {"expression": "import os"}),
        tool_use("delete_files", {}, call_id="call-2"),
        tool_use("search_documents", {"query": "nail spacing"}, call_id="call-3"),
        final(["S1"]),
    )

    assert last_tool_result(claude.requests[1])["is_error"] is True
    assert last_tool_result(claude.requests[2])["is_error"] is True
    assert "no tool called delete_files" in last_tool_result(claude.requests[2])["content"]
    assert result.answer.answerable


def test_citing_a_source_never_shown_gives_no_answer():
    result, _ = run(tool_use("search_documents", {"query": "nail spacing"}), final(["S7"]))

    assert result.answer.text == NOT_FOUND
    assert result.answer.unsupported_citations == ["S7"]


def test_an_excerpt_found_again_keeps_its_source_id():
    result, claude = run(
        tool_use("search_documents", {"query": "soil nail spacing"}),
        tool_use("search_documents", {"query": "soil nails spaced apart"}, call_id="call-2"),
        final(["S1"]),
    )

    first, second = (c.output_summary for c in result.tool_calls)
    assert first.split(", ")[0] == second.split(", ")[0] == "S1 gec07 p.57"
    assert len(result.answer.sources) == len(set(result.answer.sources))


def test_thinking_blocks_are_sent_back_unchanged():
    _, claude = run(tool_use("search_documents", {"query": "nail spacing"}, thinking=True), final(["S1"]))

    assistant_turn = claude.requests[1]["messages"][1]
    assert assistant_turn["role"] == "assistant"
    assert [block.type for block in assistant_turn["content"]] == ["thinking", "tool_use"]


def test_a_refusal_stops_the_run():
    with pytest.raises(LLMError) as excinfo:
        run(tool_use("search_documents", {"query": "nail spacing"}), FakeResponse(None, stop_reason="refusal", content=[]))

    assert excinfo.value.failure == "refusal"


def test_every_tool_call_is_logged(caplog):
    with caplog.at_level(logging.INFO, logger="rag.agent"):
        run(tool_use("calculator", {"expression": "2 + 2"}), final([]))

    assert 'tool call: step=1 tool=calculator input={"expression": "2 + 2"} -> = 4' in caplog.text


def test_tool_calls_are_plain_data():
    result, _ = run(tool_use("calculator", {"expression": "2 + 2"}), final([]))

    assert result.tool_calls == [ToolCall(1, "calculator", {"expression": "2 + 2"}, "= 4", result.tool_calls[0].seconds)]
