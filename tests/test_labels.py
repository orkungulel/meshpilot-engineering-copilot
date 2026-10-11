from pathlib import Path

import pytest

from rag.generate import NOT_FOUND
from rag.labels import LabelError, agreement, agreement_text, items_to_label, label_sheet, read_labels


def answered(question_id: str, rep: int, correct: bool = True, answerable: bool = True) -> dict:
    return {
        "id": question_id,
        "rep": rep,
        "kind": "answerable",
        "question": f"Question {question_id}?",
        "reference_answer": f"Reference {question_id}",
        "answer": f"Answer {question_id}" if answerable else NOT_FOUND,
        "answerable": answerable,
        "correct": correct,
        "judge_reason": f"Judge's reason for {question_id}",
    }


def unanswerable(question_id: str, rep: int) -> dict:
    return {
        "id": question_id,
        "rep": rep,
        "kind": "unanswerable",
        "question": "?",
        "answer": NOT_FOUND,
        "answerable": False,
    }


# Two runs in the order the runner writes them: q1 abstained in its first run.
RECORDS = [
    answered("q1", 0, correct=False, answerable=False),
    answered("q2", 0),
    unanswerable("u1", 0),
    answered("q1", 1),
    answered("q2", 1),
    unanswerable("u1", 1),
]


def fill(sheet: str, *labels: str) -> str:
    """Write the labels into a sheet's empty label lines, in order."""
    parts = sheet.split("label (y/n): ")
    assert len(parts) == len(labels) + 1
    return parts[0] + "".join(f"label (y/n): {label}{rest}" for label, rest in zip(labels, parts[1:]))


def test_one_answer_per_question_from_its_first_run_that_answered():
    assert [(r["id"], r["rep"]) for r in items_to_label(RECORDS)] == [("q1", 1), ("q2", 0)]
    assert [(r["id"], r["rep"]) for r in items_to_label(RECORDS, all_reps=True)] == [("q1", 1), ("q2", 0), ("q2", 1)]


def test_the_sheet_shows_the_rubric_and_every_answer_but_no_verdict():
    sheet = label_sheet(Path("run-1"), items_to_label(RECORDS))

    assert "> 2. Nothing in it contradicts the reference answer." in sheet
    assert "## 1. q1, run 2" in sheet and "## 2. q2, run 1" in sheet
    assert "**Reference answer:** Reference q1" in sheet
    assert "**Answer:** Answer q1" in sheet
    assert "Judge's reason" not in sheet
    assert "python eval/run_eval.py agreement run-1" in sheet
    assert all(line == "label (y/n): " for line in sheet.splitlines() if line.startswith("label"))


def test_a_filled_in_sheet_gives_one_grade_per_answer():
    sheet = fill(label_sheet(Path("run-1"), items_to_label(RECORDS)), "y", " No ")

    expected = {("q1", 1): True, ("q2", 0): False}
    assert read_labels(sheet) == expected
    assert read_labels(sheet.replace("\n", "\r\n")) == expected  # saved by a Windows editor


def test_every_missing_or_unreadable_label_is_reported():
    sheet = fill(label_sheet(Path("run-1"), items_to_label(RECORDS, all_reps=True)), "", "maybe", "y")

    with pytest.raises(LabelError) as excinfo:
        read_labels(sheet)

    assert str(excinfo.value).splitlines() == [
        "item 1 (q1, run 2): not labelled yet",
        "item 2 (q2, run 1): 'maybe' is not y or n",
    ]


def test_a_file_without_answers_is_rejected():
    with pytest.raises(LabelError, match="no answers found"):
        read_labels("# Blind labels\n\nlabel (y/n): y\n")


def test_agreement_separates_a_lenient_judge_from_a_strict_one():
    records = [answered("q1", 0, True), answered("q2", 0, True), answered("q3", 0, False), answered("q4", 0, False)]
    labels = {("q1", 0): True, ("q2", 0): False, ("q3", 0): True, ("q4", 0): False}

    result = agreement(records, labels)

    assert (result["agreement"]["hits"], result["agreement"]["n"]) == (2, 4)
    assert (result["both_correct"], result["both_incorrect"]) == (1, 1)
    assert result["only_judge_correct"] == 1  # q2: the judge passed what the person failed
    assert result["only_human_correct"] == 1  # q3: the judge failed what the person passed
    assert result["cohens_kappa"] == 0.0  # same pass rates, agreeing half the time: chance level
    assert [d["id"] for d in result["disagreements"]] == ["q2", "q3"]
    text = agreement_text(result)
    assert "q2, run 1: you incorrect, judge correct. Judge's reason for q2" in text
    assert ["judge:", "correct", "1", "1"] in [line.split() for line in text.splitlines()]


def test_labels_for_answers_that_are_not_in_the_run_are_rejected():
    with pytest.raises(LabelError, match="q9, run 1"):
        agreement(RECORDS, {("q9", 0): True})
