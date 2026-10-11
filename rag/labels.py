"""Blind labels: an engineer grades one run's answers by hand, to check the judge.

The judge's numbers are trusted only once a person has graded the answers of
one full run without seeing the judge's verdicts. label_sheet writes those
answers to a Markdown file, with the judge's rubric and an empty label for each
answer. agreement then compares the person's labels with the judge's verdicts.

Only real answers are labelled. When the system said the documents do not
contain the answer, the rubric makes that incorrect whoever grades it, so those
cases would only inflate the agreement.
"""

import re
from pathlib import Path

from rag.evaluation import format_rate, rate
from rag.judge import RUBRIC
from rag.metrics import cohens_kappa

LABELS_FILE = "labels.md"
AGREEMENT_FILE = "agreement.json"
LABEL_PROMPT = "label (y/n):"
ITEM_HEADER = re.compile(r"^## (\d+)\. (\S+), run (\d+)[ \t\r]*$", re.MULTILINE)
LABEL_LINE = re.compile(r"^label \(y/n\):(.*)$", re.MULTILINE | re.IGNORECASE)
YES, NO = {"y", "yes"}, {"n", "no"}


class LabelError(ValueError):
    """The labels are incomplete, or do not match the run."""


def items_to_label(records: list[dict], all_reps: bool = False) -> list[dict]:
    """The judged answers to label, in question order.

    By default one per question, from its first run that gave an answer: the
    runs of one question are not independent, so labelling all of them triples
    the work for little extra information. all_reps labels every run.
    """
    order = {question_id: i for i, question_id in enumerate(dict.fromkeys(r["id"] for r in records))}
    answered = sorted(
        (r for r in records if r["kind"] == "answerable" and r["answerable"]),
        key=lambda r: (order[r["id"]], r["rep"]),
    )
    if all_reps:
        return answered
    first: dict[str, dict] = {}
    for r in answered:
        first.setdefault(r["id"], r)
    return list(first.values())


def label_sheet(run_dir: Path, items: list[dict]) -> str:
    """The Markdown file a person fills in: every answer with its reference, never the judge's verdict."""
    lines = [
        f"# Blind labels: {run_dir.name}",
        "",
        "Grade each answer against its reference answer, using the judge's rubric:",
        "",
        *(f"> {line}".rstrip() for line in RUBRIC.splitlines()),
        "",
        "Fill in every label before you look at the judge's verdicts for this run",
        "(`review.md` and `per_question.jsonl` in this folder).",
        f'Write y (correct) or n (incorrect) after each "{LABEL_PROMPT}", then run:',
        "",
        f"    python eval/run_eval.py agreement {run_dir}",
    ]
    for number, r in enumerate(items, start=1):
        lines += [
            "",
            f"## {number}. {r['id']}, run {r['rep'] + 1}",
            "",
            f"**Question:** {r['question']}",
            "",
            f"**Reference answer:** {r['reference_answer']}",
            "",
            f"**Answer:** {r['answer']}",
            "",
            f"{LABEL_PROMPT} ",
        ]
    return "\n".join(lines) + "\n"


def read_labels(text: str) -> dict[tuple[str, int], bool]:
    """(question id, rep) -> the person's verdict, from a filled-in sheet.

    Raises LabelError listing every item that does not have exactly one y or n.
    """
    parts = ITEM_HEADER.split(text)[1:]  # the text before the first item is the instructions
    if not parts:
        raise LabelError("no answers found: was this file written by label-sheet?")
    labels: dict[tuple[str, int], bool] = {}
    problems = []
    for i in range(0, len(parts), 4):
        number, question_id, run, body = parts[i : i + 4]
        item = f"item {number} ({question_id}, run {run})"
        key = (question_id, int(run) - 1)
        values = [value.strip().lower() for value in LABEL_LINE.findall(body)]
        if len(values) != 1:
            problems.append(f'{item}: {len(values)} "{LABEL_PROMPT}" lines instead of one')
        elif not values[0]:
            problems.append(f"{item}: not labelled yet")
        elif values[0] not in YES | NO:
            problems.append(f"{item}: {values[0]!r} is not y or n")
        elif key in labels:
            problems.append(f"{item}: this answer is in the file twice")
        else:
            labels[key] = values[0] in YES
    if problems:
        raise LabelError("\n".join(problems))
    return labels


def agreement(records: list[dict], labels: dict[tuple[str, int], bool]) -> dict:
    """The person's labels against the judge's verdicts on the same answers."""
    if not labels:
        raise LabelError("no labels to compare")
    judged = {(r["id"], r["rep"]): r for r in records if r["kind"] == "answerable"}
    unknown = [f"{question_id}, run {rep + 1}" for question_id, rep in labels if (question_id, rep) not in judged]
    if unknown:
        raise LabelError("labelled answers that are not in this run: " + "; ".join(unknown))
    pairs = [(labels[key], judged[key]) for key in labels]  # (the person's verdict, the record)
    both_correct = sum(human and r["correct"] for human, r in pairs)
    both_incorrect = sum(not human and not r["correct"] for human, r in pairs)
    only_judge_correct = sum(r["correct"] and not human for human, r in pairs)  # the judge was more lenient
    only_human_correct = sum(human and not r["correct"] for human, r in pairs)  # the judge was stricter
    n = len(pairs)
    kappa = cohens_kappa(both_correct, only_human_correct, only_judge_correct, both_incorrect)
    return {
        "n": n,
        "agreement": rate(both_correct + both_incorrect, n),
        "cohens_kappa": None if kappa is None else round(kappa, 3),
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
        "only_judge_correct": only_judge_correct,
        "only_human_correct": only_human_correct,
        "human_correct": rate(both_correct + only_human_correct, n),
        "judge_correct": rate(both_correct + only_judge_correct, n),
        "disagreements": [
            {
                "id": r["id"],
                "run": r["rep"] + 1,
                "human_correct": human,
                "judge_correct": r["correct"],
                "judge_reason": r["judge_reason"],
            }
            for human, r in pairs
            if human != r["correct"]
        ],
    }


def agreement_text(result: dict) -> str:
    """The agreement for the terminal: rates, the two-by-two table and every disagreement."""

    def grade(correct: bool) -> str:
        return "correct" if correct else "incorrect"

    kappa = result["cohens_kappa"]
    lines = [
        f"{result['n']} answers graded by you and by the judge",
        f"agreement        {format_rate(result['agreement'])}",
        f"Cohen's kappa    {'undefined (both gave every answer the same grade)' if kappa is None else f'{kappa:.2f}'}",
        f"correct (you)    {format_rate(result['human_correct'])}",
        f"correct (judge)  {format_rate(result['judge_correct'])}",
        "",
        f"{'':18}{'you: correct':>14}{'you: incorrect':>16}",
        f"{'judge: correct':18}{result['both_correct']:>14}{result['only_judge_correct']:>16}",
        f"{'judge: incorrect':18}{result['only_human_correct']:>14}{result['both_incorrect']:>16}",
    ]
    if result["disagreements"]:
        lines += ["", "Disagreements, with the judge's reason:"]
    for d in result["disagreements"]:
        lines.append(
            f"  {d['id']}, run {d['run']}: you {grade(d['human_correct'])}, judge {grade(d['judge_correct'])}. "
            f"{d['judge_reason']}"
        )
    return "\n".join(lines)


def agreement_line(run_name: str, result: dict) -> str:
    """One line for the report."""
    kappa = "undefined" if result["cohens_kappa"] is None else f"{result['cohens_kappa']:.2f}"
    return (
        f"Judge vs blind labels {run_name}: agreement {format_rate(result['agreement'])}, Cohen's kappa {kappa}; "
        f"judge more lenient on {result['only_judge_correct']}, stricter on {result['only_human_correct']}"
    )
