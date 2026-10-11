"""Read a PDF into pages of plain text with PyMuPDF.

Page numbers are 1-based PDF page indices: the number a PDF viewer shows in its
page box. The printed page label ("iv", "3-12") often differs because of front
matter, so it is kept separately and used for display only.
"""

from dataclasses import dataclass
from pathlib import Path

import pymupdf

# PyMuPDF's default keeps typographic ligatures as single characters, so "ﬁeld"
# would not match "field". Clearing this flag expands them into plain letters.
TEXT_FLAGS = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES


@dataclass(frozen=True)
class Page:
    doc_id: str
    number: int  # 1-based PDF page index, used for citations and the evaluation
    label: str  # printed page label if the PDF defines one, otherwise ""
    text: str


def load_pages(pdf_path: Path, doc_id: str) -> list[Page]:
    """Return every page of the PDF, including pages without extractable text."""
    with pymupdf.open(pdf_path) as pdf:
        return [
            Page(
                doc_id=doc_id,
                number=index + 1,
                label=page.get_label(),
                text=page.get_text("text", flags=TEXT_FLAGS),
            )
            for index, page in enumerate(pdf)
        ]
