import pytest

from rag.chunking import chunk_page, chunk_pages
from rag.pdf_loader import Page


def page_of(n_words: int, number: int = 1, prefix: str = "w") -> Page:
    return Page(doc_id="doc", number=number, label="", text=" ".join(f"{prefix}{i}" for i in range(n_words)))


def test_short_page_gives_one_chunk_with_all_words():
    chunks = chunk_page(page_of(3), size=10, overlap=2)

    assert [c.text for c in chunks] == ["w0 w1 w2"]


def test_windows_overlap_by_the_requested_number_of_words():
    chunks = chunk_page(page_of(10), size=4, overlap=1)

    assert [c.text for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_last_window_may_be_shorter():
    chunks = chunk_page(page_of(11), size=4, overlap=1)

    assert chunks[-1].text == "w9 w10"


@pytest.mark.parametrize("n_words, size, overlap", [(1, 5, 0), (37, 8, 3), (300, 150, 30), (1000, 300, 60)])
def test_every_word_is_covered_in_order_and_no_chunk_is_too_long(n_words, size, overlap):
    chunks = chunk_page(page_of(n_words), size=size, overlap=overlap)

    assert all(len(c.text.split()) <= size for c in chunks)
    rebuilt = chunks[0].text.split()
    for chunk in chunks[1:]:
        words = chunk.text.split()
        if overlap:
            assert words[:overlap] == rebuilt[-overlap:]
        rebuilt += words[overlap:]
    assert rebuilt == page_of(n_words).text.split()


def test_page_without_words_gives_no_chunks():
    assert chunk_page(Page("doc", 1, "", " \n "), size=4, overlap=1) == []


def test_chunks_never_cross_a_page_and_keep_their_page_number():
    pages = [page_of(6, number=1, prefix="a"), page_of(6, number=2, prefix="b")]

    chunks = chunk_pages(pages, size=4, overlap=1)

    for chunk in chunks:
        prefix = "a" if chunk.page == 1 else "b"
        assert all(word.startswith(prefix) for word in chunk.text.split())
    assert [c.chunk_id for c in chunks] == ["doc:1:0", "doc:1:1", "doc:2:0", "doc:2:1"]


@pytest.mark.parametrize("size, overlap", [(0, 0), (4, 4), (4, 5), (4, -1)])
def test_invalid_window_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_page(page_of(10), size=size, overlap=overlap)
