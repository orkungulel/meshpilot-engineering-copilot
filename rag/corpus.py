"""Download the documents listed in corpus/manifest.csv and verify every file.

The PDFs are not committed (data/ is gitignored). The manifest records where
each document comes from, its licence and, once pinned, its SHA-256 and page
count. Anyone can then rebuild exactly the same corpus, and a publisher
silently replacing a file is caught instead of shifting the page numbers the
evaluation relies on.
"""

import csv
import hashlib
import urllib.request
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import pymupdf

USER_AGENT = "rag-prototype/0.1 (research use)"


class CorpusError(Exception):
    pass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    publisher: str
    year: str
    publication_number: str
    url: str
    licence: str
    sha256: str = ""  # empty until pinned with `download --pin`
    pages: str = ""  # page count, pinned together with the hash


COLUMNS = [f.name for f in fields(Document)]


def read_manifest(path: Path) -> list[Document]:
    with open(path, newline="", encoding="utf-8") as f:
        return [Document(**row) for row in csv.DictReader(f)]


def write_manifest(path: Path, docs: list[Document]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(asdict(d) for d in docs)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pdf_page_count(path: Path) -> int:
    """Page count of a real, readable PDF.

    The header check matters: PyMuPDF "repairs" an HTML error page saved as
    .pdf into a one-page document instead of raising an error.
    """
    with open(path, "rb") as f:
        if b"%PDF-" not in f.read(1024):
            raise CorpusError(f"{path.name} is not a PDF (no %PDF- header); the URL may point to a web page")
    try:
        with pymupdf.open(path, filetype="pdf") as pdf:
            return pdf.page_count
    except Exception as exc:
        raise CorpusError(f"{path.name} is not a readable PDF: {exc}") from exc


def verify(doc: Document, path: Path) -> None:
    """Fail if a pinned document's file differs from the pinned hash or page count."""
    if doc.sha256 and sha256_of(path) != doc.sha256:
        raise CorpusError(
            f"{doc.doc_id}: SHA-256 differs from the manifest. The publisher may have "
            "changed the file, which can move pages that eval/questions.jsonl refers to. "
            "Check the questions before re-pinning."
        )
    if doc.pages and pdf_page_count(path) != int(doc.pages):
        raise CorpusError(f"{doc.doc_id}: page count differs from the manifest")


def download(doc: Document, raw_dir: Path) -> Path:
    """Fetch the PDF into raw_dir/<doc_id>.pdf unless a verified copy is already there."""
    target = raw_dir / f"{doc.doc_id}.pdf"
    if target.exists():
        verify(doc, target)
        return target
    raw_dir.mkdir(parents=True, exist_ok=True)
    partial = raw_dir / f"{doc.doc_id}.pdf.part"
    request = urllib.request.Request(doc.url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as f:
            while block := response.read(1 << 20):
                f.write(block)
        pdf_page_count(partial)
        verify(doc, partial)
    except Exception:
        partial.unlink(missing_ok=True)  # never leave a half-written or rejected file behind
        raise
    partial.rename(target)
    return target


def pin(doc: Document, path: Path) -> Document:
    """Record the file's hash and page count for a document that has none yet."""
    if doc.sha256:
        return doc
    return replace(doc, sha256=sha256_of(path), pages=str(pdf_page_count(path)))
