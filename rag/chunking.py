"""Split page text into overlapping word windows that never cross a page boundary.

Keeping every chunk inside one page means each retrieved chunk maps to exactly
one (document, page) pair, so citations and the page-level evaluation are
unambiguous. The price is that a sentence running across a page break is split.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from rag.pdf_loader import Page


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>:<page>:<position on the page>", stable across rebuilds
    doc_id: str
    page: int  # 1-based PDF page index
    text: str


def chunk_page(page: Page, size: int, overlap: int) -> list[Chunk]:
    """Windows of `size` words, a new one starting every `size - overlap` words.

    The last window on a page may be shorter than `size`. A page without words
    gives no chunks.
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"need size > 0 and 0 <= overlap < size, got {size=}, {overlap=}")
    words = page.text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    for start in range(0, len(words), step):
        chunks.append(
            Chunk(
                chunk_id=f"{page.doc_id}:{page.number}:{len(chunks)}",
                doc_id=page.doc_id,
                page=page.number,
                text=" ".join(words[start : start + size]),
            )
        )
        if start + size >= len(words):  # this window reached the end of the page
            break
    return chunks


def chunk_pages(pages: Iterable[Page], size: int, overlap: int) -> list[Chunk]:
    return [chunk for page in pages for chunk in chunk_page(page, size, overlap)]
