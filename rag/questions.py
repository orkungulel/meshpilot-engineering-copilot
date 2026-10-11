"""The evaluation question set: format, loading and validation.

Each line of eval/questions.jsonl is one question with the page that answers it.
Pages are 1-based PDF page indices, the numbering the loader and the citations
use, not the number printed on the page.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rag.corpus import Document
from rag.pdf_loader import load_pages

QUESTION_TYPES = {"fact", "number", "definition", "procedure", "calculation"}
AUTHORS = {"human", "claude-drafted-human-verified"}
MAX_PER_DOCUMENT = 3  # spreads the questions across the corpus
COPIED_WORDS = 8  # a question sharing this many consecutive words with its page was probably copied


@dataclass(frozen=True)
class Location:
    doc_id: str
    page: int


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    doc_id: str
    page: int
    reference_answer: str
    type: str
    written_by: str
    # Other pages that state the same answer. Only for facts that genuinely appear twice.
    also_acceptable: tuple[Location, ...] = ()

    @property
    def gold(self) -> set[Location]:
        """Every location that counts as retrieving the answer."""
        return {Location(self.doc_id, self.page), *self.also_acceptable}


@dataclass(frozen=True)
class UnanswerableQuestion:
    """An in-domain question the corpus does not answer; the right reply is to say so."""

    id: str
    question: str
    written_by: str


def load_questions(path: Path) -> list[Question]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row["also_acceptable"] = tuple(Location(**loc) for loc in row.get("also_acceptable", []))
                questions.append(Question(**row))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name} line {line_number}: {exc}") from exc
    return questions


def load_unanswerable(path: Path) -> list[UnanswerableQuestion]:
    questions = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                try:
                    questions.append(UnanswerableQuestion(**json.loads(line)))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{path.name} line {line_number}: {exc}") from exc
    return questions


def validate_unanswerable(questions: list[UnanswerableQuestion], answerable_ids: set[str]) -> list[str]:
    errors = []
    for question_id, count in Counter(q.id for q in questions).items():
        if count > 1 or question_id in answerable_ids:
            errors.append(f"{question_id}: id is not unique across both question files")
    for q in questions:
        if not q.question.strip():
            errors.append(f"{q.id}: empty question")
        if q.written_by not in AUTHORS:
            errors.append(f"{q.id}: written_by {q.written_by!r} is not one of {sorted(AUTHORS)}")
    return errors


def validate_questions(questions: list[Question], docs: list[Document]) -> list[str]:
    """Structural errors. Needs only the manifest, so it also runs in CI."""
    if not questions:
        return ["the question file is empty"]
    errors = []
    page_counts = {d.doc_id: int(d.pages) if d.pages else None for d in docs}
    for question_id, count in Counter(q.id for q in questions).items():
        if count > 1:
            errors.append(f"{question_id}: id used {count} times")
    for doc_id, count in Counter(q.doc_id for q in questions).items():
        if count > MAX_PER_DOCUMENT:
            errors.append(f"{doc_id}: {count} questions, more than {MAX_PER_DOCUMENT}")
    for q in questions:
        if not q.question.strip() or not q.reference_answer.strip():
            errors.append(f"{q.id}: empty question or reference answer")
        if q.type not in QUESTION_TYPES:
            errors.append(f"{q.id}: type {q.type!r} is not one of {sorted(QUESTION_TYPES)}")
        if q.written_by not in AUTHORS:
            errors.append(f"{q.id}: written_by {q.written_by!r} is not one of {sorted(AUTHORS)}")
        for loc in sorted(q.gold, key=lambda loc: (loc.doc_id, loc.page)):
            if loc.doc_id not in page_counts:
                errors.append(f"{q.id}: unknown document {loc.doc_id}")
            elif loc.page < 1 or (page_counts[loc.doc_id] and loc.page > page_counts[loc.doc_id]):
                last_page = page_counts[loc.doc_id] or "?"  # "?" until the manifest is pinned
                errors.append(f"{q.id}: page {loc.page} is outside {loc.doc_id} (1-{last_page})")
    return errors


def content_warnings(questions: list[Question], raw_dir: Path) -> list[str]:
    """Checks against the text of each question's page. Needs the PDFs, so it runs locally."""
    warnings = []
    pages_by_doc: dict[str, list[str] | None] = {}
    for q in questions:
        if q.doc_id not in pages_by_doc:
            path = raw_dir / f"{q.doc_id}.pdf"
            pages_by_doc[q.doc_id] = [p.text for p in load_pages(path, q.doc_id)] if path.exists() else None
        pages = pages_by_doc[q.doc_id]
        if not pages or not 1 <= q.page <= len(pages):
            continue
        page_text = pages[q.page - 1]
        missing = sorted(numbers_in(q.reference_answer) - numbers_in(page_text))
        if missing:
            warnings.append(
                f"{q.id}: {', '.join(missing)} from the reference answer is not on {q.doc_id} p.{q.page}"
                " (wrong page, or a printed page number used instead of the PDF page?)"
            )
        if shares_consecutive_words(q.question, page_text, COPIED_WORDS):
            warnings.append(f"{q.id}: copies {COPIED_WORDS} or more consecutive words from its page; paraphrase it")
    return warnings


def numbers_in(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")))


def shares_consecutive_words(question: str, page_text: str, n: int) -> bool:
    question_words = re.findall(r"[a-z0-9]+", question.lower())
    page_words = re.findall(r"[a-z0-9]+", page_text.lower())
    page_ngrams = {tuple(page_words[i : i + n]) for i in range(len(page_words) - n + 1)}
    return any(tuple(question_words[i : i + n]) in page_ngrams for i in range(len(question_words) - n + 1))
