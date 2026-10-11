import json

import pytest

from rag import config
from rag.corpus import Document, read_manifest
from rag.questions import (
    Location,
    Question,
    UnanswerableQuestion,
    content_warnings,
    load_questions,
    load_unanswerable,
    shares_consecutive_words,
    validate_questions,
    validate_unanswerable,
)

DOCS = [
    Document("gec07", "Soil Nail Walls", "FHWA", "2015", "X", "https://x", "US", sha256="ab" * 32, pages="40"),
    Document("gec10", "Drilled Shafts", "FHWA", "2018", "Y", "https://y", "US"),  # not pinned yet
]


def make_question(**overrides) -> Question:
    fields = dict(
        id="q01",
        question="How far apart are soil nails usually spaced?",
        doc_id="gec07",
        page=12,
        reference_answer="1.5 m",
        type="number",
        written_by="human",
    )
    return Question(**{**fields, **overrides})


def test_load_reads_one_question_per_line_with_optional_extra_pages(tmp_path):
    path = tmp_path / "questions.jsonl"
    rows = [
        {"id": "q01", "question": "Q?", "doc_id": "gec07", "page": 12, "reference_answer": "A",
         "type": "fact", "written_by": "human"},
        {"id": "q02", "question": "Q2?", "doc_id": "gec07", "page": 3, "reference_answer": "B",
         "type": "fact", "written_by": "human", "also_acceptable": [{"doc_id": "gec10", "page": 7}]},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")

    q1, q2 = load_questions(path)

    assert q1.gold == {Location("gec07", 12)}
    assert q2.gold == {Location("gec07", 3), Location("gec10", 7)}


def test_load_names_the_line_of_a_malformed_question(tmp_path):
    path = tmp_path / "questions.jsonl"
    path.write_text('{"id": "q01", "question": "Q?"}\n')

    with pytest.raises(ValueError, match="line 1"):
        load_questions(path)


def test_a_well_formed_question_has_no_errors():
    assert validate_questions([make_question()], DOCS) == []


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"doc_id": "gec99"}, "unknown document gec99"),
        ({"page": 41}, "page 41 is outside gec07 (1-40)"),
        ({"page": 0}, "page 0 is outside"),
        ({"type": "opinion"}, "type 'opinion'"),
        ({"written_by": "someone"}, "written_by 'someone'"),
        ({"reference_answer": " "}, "empty question or reference answer"),
        ({"also_acceptable": (Location("gec07", 99),)}, "page 99 is outside"),
    ],
)
def test_structural_errors_are_reported(overrides, message):
    [error] = validate_questions([make_question(**overrides)], DOCS)

    assert message in error


def test_an_empty_question_set_is_an_error():
    assert validate_questions([], DOCS) == ["the question file is empty"]


def test_page_range_is_not_checked_for_an_unpinned_document():
    assert validate_questions([make_question(doc_id="gec10", page=500)], DOCS) == []


def test_duplicate_ids_and_too_many_questions_per_document_are_errors():
    questions = [make_question(id=f"q{i}") for i in range(4)] + [make_question(id="q0")]

    errors = validate_questions(questions, DOCS)

    assert any("q0: id used 2 times" in e for e in errors)
    assert any("gec07: 5 questions, more than 3" in e for e in errors)


def test_content_warnings_catch_a_wrong_page_and_copied_wording(tmp_path, make_pdf):
    page_text = "Soil nails are typically installed at a horizontal spacing of 1.5 m in most walls."
    make_pdf(["Front matter", page_text], name="gec07.pdf")
    questions = [
        make_question(id="ok", page=2, question="What nail spacing is common?", reference_answer="1.5 m"),
        make_question(id="wrong-page", page=1, reference_answer="1.5 m"),
        make_question(id="copied", page=2, question="Soil nails are typically installed at a horizontal spacing of what?"),
    ]

    warnings = content_warnings(questions, raw_dir=tmp_path)

    assert not any(w.startswith("ok:") for w in warnings)
    assert any(w.startswith("wrong-page: 1.5 from the reference answer is not on gec07 p.1") for w in warnings)
    assert any(w.startswith("copied: copies 8 or more consecutive words") for w in warnings)


def test_shares_consecutive_words_ignores_case_and_punctuation():
    page = "The factor of safety, against sliding, shall be at least 1.5."
    assert shares_consecutive_words("the FACTOR of safety against sliding shall be?", page, 8)
    assert not shares_consecutive_words("Which sliding safety factor is required?", page, 8)


def test_unanswerable_questions_load_and_validate(tmp_path):
    path = tmp_path / "unanswerable.jsonl"
    path.write_text('{"id": "u01", "question": "What does Eurocode 7 require?", "written_by": "human"}\n')

    [question] = load_unanswerable(path)

    assert question == UnanswerableQuestion("u01", "What does Eurocode 7 require?", "human")
    assert validate_unanswerable([question], answerable_ids={"q01"}) == []
    assert validate_unanswerable([question], answerable_ids={"u01"}) == ["u01: id is not unique across both question files"]


def test_committed_question_set_is_valid():
    if not config.QUESTIONS_PATH.exists():
        pytest.skip("eval/questions.jsonl has not been written yet")
    questions = load_questions(config.QUESTIONS_PATH)

    assert validate_questions(questions, read_manifest(config.MANIFEST_PATH)) == []


def test_committed_unanswerable_set_is_valid():
    if not config.UNANSWERABLE_PATH.exists():
        pytest.skip("eval/unanswerable.jsonl has not been written yet")
    answerable_ids = {q.id for q in load_questions(config.QUESTIONS_PATH)} if config.QUESTIONS_PATH.exists() else set()

    assert validate_unanswerable(load_unanswerable(config.UNANSWERABLE_PATH), answerable_ids) == []
