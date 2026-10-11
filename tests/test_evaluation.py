import pytest
from fakes import FakeEmbedder

from rag.chunking import Chunk
from rag.evaluation import (
    answerable_record,
    answers_table,
    compare_retrieval,
    evaluate_retrieval,
    judge_check_cases,
    majority,
    read_run,
    retrieval_table,
    review_markdown,
    summarize_answers,
    summarize_retrieval,
    unanswerable_record,
    write_run,
)
from rag.generate import Answer, Citation
from rag.index import VectorIndex
from rag.judge import Verdict
from rag.questions import Location, Question, UnanswerableQuestion

CHUNKS = [
    Chunk("gec07:12:0", "gec07", 12, "soil nail wall facing design and nail spacing"),
    Chunk("gec07:30:0", "gec07", 30, "corrosion protection of soil nail bars"),
    Chunk("gec10:40:0", "gec10", 40, "drilled shaft side resistance in rock sockets"),
    Chunk("gec12:88:0", "gec12", 88, "driven pile dynamic testing and wave equation analysis"),
]


def question(text: str, doc_id: str, page: int, also_acceptable=(), question_id: str = "q1") -> Question:
    return Question(question_id, text, doc_id, page, "1.5 m", "fact", "human", tuple(also_acceptable))


@pytest.fixture
def index():
    return VectorIndex.build(CHUNKS, FakeEmbedder())


# --- retrieval -----------------------------------------------------------------


def test_records_the_rank_of_the_expected_page_and_document(index):
    # The facing chunk (p.12) matches best, so the document is found at rank 1
    # but the expected page (p.30) only at rank 2.
    [record] = evaluate_retrieval(index, [question("soil nail wall facing design", "gec07", 30)], k_max=4)

    assert record["doc_rank"] == 1
    assert record["page_rank"] == 2
    assert [h["chunk_id"] for h in record["hits"]][:2] == ["gec07:12:0", "gec07:30:0"]


def test_page_outside_the_top_k_has_no_rank(index):
    [record] = evaluate_retrieval(index, [question("soil nail wall facing design", "gec07", 30)], k_max=1)

    assert record["page_rank"] is None


def test_an_also_acceptable_page_counts_as_found(index):
    q = question("drilled shaft side resistance", "gec99", 1, also_acceptable=[Location("gec10", 40)])

    [record] = evaluate_retrieval(index, [q], k_max=4)

    assert record["page_rank"] == 1


def test_summary_counts_questions_found_within_each_k():
    records = [
        {"page_rank": 1, "doc_rank": 1},
        {"page_rank": 3, "doc_rank": 1},
        {"page_rank": None, "doc_rank": 7},
        {"page_rank": 12, "doc_rank": 2},
    ]

    metrics = summarize_retrieval(records)

    assert [metrics[k]["hits"] for k in ("recall@1", "recall@5", "recall@20", "doc_recall@5")] == [1, 2, 3, 3]
    assert metrics["recall@5"]["rate"] == 0.5
    low, high = metrics["recall@5"]["ci95"]
    assert low < 0.5 < high


def test_compare_counts_agreements_and_disagreements_at_k():
    a = [{"id": "q1", "page_rank": 1}, {"id": "q2", "page_rank": None}, {"id": "q3", "page_rank": 6}, {"id": "q4", "page_rank": 2}]
    b = [{"id": "q1", "page_rank": 2}, {"id": "q2", "page_rank": 3}, {"id": "q3", "page_rank": None}, {"id": "q4", "page_rank": None}]

    result = compare_retrieval(a, b, k=5)

    assert (result["both"], result["neither"]) == (1, 1)
    assert (result["only_a"], result["only_b"]) == (["q4"], ["q2"])
    assert result["p_value"] == 1.0


def test_compare_refuses_runs_on_different_questions():
    with pytest.raises(ValueError):
        compare_retrieval([{"id": "q1", "page_rank": 1}], [{"id": "q2", "page_rank": 1}], k=5)


def test_a_run_folder_round_trips_and_is_never_overwritten(tmp_path):
    write_run(tmp_path / "run", {"kind": "retrieval"}, [{"id": "q1"}])

    assert read_run(tmp_path / "run") == ({"kind": "retrieval"}, [{"id": "q1"}])
    with pytest.raises(FileExistsError):
        write_run(tmp_path / "run", {"kind": "retrieval"}, [])


def test_table_marks_runs_made_with_uncommitted_changes():
    metric = {"hits": 3, "n": 4, "rate": 0.75, "ci95": [0.3, 0.95]}
    summary = {
        "index": {"config": {"name": "R1", "embedding_model": "m", "chunk_words": 300, "overlap_words": 60}},
        "metrics": {k: metric for k in ("recall@1", "recall@5", "recall@20", "doc_recall@5")},
    }

    table = retrieval_table([("clean-run", {**summary, "git_dirty": False}), ("dirty-run", {**summary, "git_dirty": True})])

    assert "| clean-run | R1 | m | 300 / 60 | 3/4 (75%; 95% CI 30%–95%) |" in table
    assert "| dirty-run \\* |" in table
    assert "uncommitted" in table


# --- answers -------------------------------------------------------------------


def make_answer(answerable: bool = True, citations=(("gec07", 57),)) -> Answer:
    cited = [Citation(doc_id, doc_id.upper(), page) for doc_id, page in citations] if answerable else []
    usage = {"input_tokens": 900, "output_tokens": 40}
    return Answer("1.5 m" if answerable else "not found", answerable, cited, ["gec07:57:0"], "claude-sonnet-5-5", usage, 2.0, [])


def judged(q: Question, rep: int, correct: bool, answer: Answer | None = None) -> dict:
    verdict = Verdict(correct, "reason", "claude-opus-5-5", {"input_tokens": 300, "output_tokens": 20})
    return answerable_record(q, rep, answer or make_answer(), verdict)


def test_citing_any_expected_page_counts_as_citing_it():
    q = question("spacing?", "gec99", 1, also_acceptable=[Location("gec07", 57)])

    assert judged(q, 0, True)["cited_expected_page"]
    assert not judged(q, 0, True, make_answer(citations=(("gec07", 58),)))["cited_expected_page"]


def test_majority_needs_more_than_half_of_the_runs():
    records = [{"id": "a", "kind": "answerable", "ok": v} for v in (True, False, True)]
    records += [{"id": "b", "kind": "answerable", "ok": v} for v in (True, False)]

    assert majority(records, "answerable", lambda r: r["ok"]) == {"a": True, "b": False}


def test_answer_summary_counts_per_question_majorities_and_each_run():
    q1 = question("spacing?", "gec07", 57, question_id="q1")
    q2 = question("facing?", "gec07", 57, question_id="q2")
    unanswerable = UnanswerableQuestion("u1", "What does Eurocode 7 say?", "human")
    records = [judged(q1, rep, True) for rep in range(3)]
    records += [judged(q2, rep, correct) for rep, correct in enumerate((True, False, False))]
    records += [unanswerable_record(unanswerable, rep, make_answer(answerable=False)) for rep in range(3)]

    metrics = summarize_answers(records, mode="rag")

    assert (metrics["correct"]["hits"], metrics["correct"]["n"]) == (1, 2)  # q2 right in only 1 of 3 runs
    assert metrics["correct"]["by_rep"] == [{"hits": 2, "n": 2}, {"hits": 1, "n": 2}, {"hits": 1, "n": 2}]
    assert metrics["said_not_found"]["hits"] == 0
    assert (metrics["abstained_when_unanswerable"]["hits"], metrics["abstained_when_unanswerable"]["n"]) == (1, 1)
    assert metrics["usage"]["answer_input_tokens_mean"] == 900
    assert metrics["usage"]["judge_input_tokens_mean"] == 300


def test_closed_book_summary_has_no_citation_metric():
    metrics = summarize_answers([judged(question("spacing?", "gec07", 57), 0, True)], mode="closed-book")

    assert "cited_expected_page" not in metrics


def test_judge_checks_cover_known_good_and_known_bad_answers():
    q1 = Question("q1", "Spacing?", "gec07", 57, "1.5 m", "number", "human")
    q2 = Question("q2", "Facing?", "gec07", 60, "shotcrete", "fact", "human")

    cases = judge_check_cases([q1, q2])

    assert [(check, answer, should_pass) for q, check, answer, should_pass in cases if q is q1] == [
        ("reference_answer", "1.5 m", True),
        ("empty_answer", "", False),
        ("i_dont_know", "I don't know.", False),
        ("other_questions_answer", "shotcrete", False),
    ]


def test_review_and_table_show_each_answer_and_its_verdict():
    records = [judged(question("Spacing?", "gec07", 57), 0, True)]
    summary = {
        "mode": "rag",
        "index_config": "R1",
        "answer_model": "claude-sonnet-5-5",
        "judge_model": "claude-opus-5-5",
        "git_dirty": False,
        "metrics": summarize_answers(records, mode="rag"),
    }

    review = review_markdown("run-1", summary, records)
    table = answers_table([("run-1", summary)])

    assert "**Reference answer:** 1.5 m" in review
    assert "**Judge:** correct. reason" in review
    assert "| run-1 | rag | R1 | claude-sonnet-5-5 | 1/1 |" in table
