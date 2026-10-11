from pathlib import Path

import pymupdf
import pytest


def write_pdf(path: Path, page_texts: list[str], labels: list[dict] | None = None) -> Path:
    """Write a PDF with one page per text ("" gives a blank page)."""
    pdf = pymupdf.open()
    for text in page_texts:
        page = pdf.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(72, 72, 523, 770), text, fontsize=10)
    if labels:
        pdf.set_page_labels(labels)
    pdf.save(path)
    return path


@pytest.fixture
def make_pdf(tmp_path):
    def make(page_texts: list[str], name: str = "doc.pdf", labels: list[dict] | None = None) -> Path:
        return write_pdf(tmp_path / name, page_texts, labels)

    return make
