from fakes import FakeClaude

from rag.chunking import Chunk
from rag.generate import NOT_FOUND, Citation, closed_book_answer, generate_answer
from rag.index import Hit

HITS = [
    Hit(Chunk("gec07:57:0", "gec07", 57, "Nails are commonly spaced 1.5 m apart."), 0.81),
    Hit(Chunk("gec07:57:1", "gec07", 57, "Spacing may be reduced near the wall top."), 0.77),
    Hit(Chunk("gec10:40:0", "gec10", 40, "Side resistance in rock sockets."), 0.42),
]
TITLES = {"gec07": "Soil Nail Walls (GEC 7)", "gec10": "Drilled Shafts (GEC 10)"}
QUESTION = "What nail spacing is common?"


def answer_with(reply: dict, hits=HITS):
    client = FakeClaude([reply])
    return generate_answer(client, "claude-sonnet-5-5", QUESTION, hits, TITLES), client


def test_cited_source_ids_become_document_and_page_citations():
    answer, _ = answer_with({"answerable": True, "answer": " About 1.5 m. ", "citations": ["S1", "S3"]})

    assert answer.answerable
    assert answer.text == "About 1.5 m."
    assert answer.citations == [Citation("gec07", "Soil Nail Walls (GEC 7)", 57), Citation("gec10", "Drilled Shafts (GEC 10)", 40)]
    assert answer.sources == ["gec07:57:0", "gec07:57:1", "gec10:40:0"]


def test_two_chunks_from_the_same_page_give_one_citation():
    answer, _ = answer_with({"answerable": True, "answer": "1.5 m", "citations": ["S2", "S1"]})

    assert answer.citations == [Citation("gec07", "Soil Nail Walls (GEC 7)", 57)]


def test_prompt_puts_titled_sources_before_the_question_and_limits_citations_to_them():
    _, client = answer_with({"answerable": True, "answer": "1.5 m", "citations": ["S1"]})

    [request] = client.requests
    prompt = request["messages"][0]["content"]
    assert '<source id="S1" document="Soil Nail Walls (GEC 7)" page="57">' in prompt
    assert prompt.index("<sources>") < prompt.index(f"Question: {QUESTION}")
    schema = request["output_config"]["format"]["schema"]
    assert schema["properties"]["citations"]["items"]["enum"] == ["S1", "S2", "S3"]


def test_unanswerable_reply_gives_the_standard_message_and_no_citations():
    answer, _ = answer_with({"answerable": False, "answer": "The sources discuss spacing only.", "citations": ["S1"]})

    assert not answer.answerable
    assert answer.text == NOT_FOUND
    assert answer.citations == []


def test_an_answer_without_citations_is_not_given():
    answer, _ = answer_with({"answerable": True, "answer": "1.5 m", "citations": []})

    assert not answer.answerable
    assert answer.text == NOT_FOUND


def test_citations_of_sources_that_were_not_shown_are_dropped_and_recorded():
    answer, _ = answer_with({"answerable": True, "answer": "1.5 m", "citations": ["S9"]})

    assert answer.unsupported_citations == ["S9"]
    assert not answer.answerable  # its only citation was invalid


def test_no_retrieved_chunks_means_no_api_call():
    answer, client = answer_with({}, hits=[])

    assert answer.text == NOT_FOUND
    assert client.requests == []


def test_closed_book_answer_sends_no_sources():
    client = FakeClaude([{"answerable": True, "answer": "Around 1.5 m."}])

    answer = closed_book_answer(client, "claude-sonnet-5-5", QUESTION)

    assert answer.text == "Around 1.5 m."
    assert answer.citations == []
    assert "<sources>" not in client.requests[0]["messages"][0]["content"]
