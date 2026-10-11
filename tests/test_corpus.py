from dataclasses import replace

import pytest

from rag.corpus import (
    CorpusError,
    Document,
    download,
    pin,
    read_manifest,
    sha256_of,
    write_manifest,
)


@pytest.fixture
def source_pdf(make_pdf):
    return make_pdf(["Soil nail walls.", "Drilled shafts.", "Driven piles."], name="source.pdf")


def doc_for(path, **overrides) -> Document:
    fields = dict(
        doc_id="gec07",
        title="Soil Nail Walls",
        publisher="FHWA",
        year="2015",
        publication_number="FHWA-NHI-14-007",
        url=path.as_uri(),  # file:// URL, so the tests need no network
        licence="US Government work",
    )
    return Document(**{**fields, **overrides})


def test_manifest_round_trips_including_unpinned_fields(tmp_path, source_pdf):
    path = tmp_path / "manifest.csv"
    docs = [doc_for(source_pdf), doc_for(source_pdf, doc_id="gec10", title='Drilled Shafts, "2018"')]

    write_manifest(path, docs)

    assert read_manifest(path) == docs


def test_download_saves_the_pdf_under_its_doc_id(tmp_path, source_pdf):
    path = download(doc_for(source_pdf), tmp_path / "raw")

    assert path == tmp_path / "raw" / "gec07.pdf"
    assert sha256_of(path) == sha256_of(source_pdf)


def test_pin_records_hash_and_page_count(tmp_path, source_pdf):
    doc = doc_for(source_pdf)
    path = download(doc, tmp_path / "raw")

    pinned = pin(doc, path)

    assert pinned.sha256 == sha256_of(source_pdf)
    assert pinned.pages == "3"
    assert pin(pinned, path) is pinned  # an existing pin is never overwritten


def test_hash_mismatch_is_rejected_and_nothing_is_left_behind(tmp_path, source_pdf):
    doc = doc_for(source_pdf, sha256="0" * 64)

    with pytest.raises(CorpusError, match="SHA-256"):
        download(doc, tmp_path / "raw")
    assert list((tmp_path / "raw").iterdir()) == []


def test_existing_verified_copy_is_not_downloaded_again(tmp_path, source_pdf):
    doc = pin(doc_for(source_pdf), source_pdf)
    download(doc, tmp_path / "raw")
    source_pdf.unlink()  # a second download attempt would now fail

    assert download(doc, tmp_path / "raw").exists()


def test_existing_copy_that_no_longer_matches_its_pin_is_rejected(tmp_path, source_pdf, make_pdf):
    doc = pin(doc_for(source_pdf), source_pdf)
    path = download(doc, tmp_path / "raw")
    path.write_bytes(make_pdf(["A different edition."], name="other.pdf").read_bytes())

    with pytest.raises(CorpusError, match="SHA-256"):
        download(doc, tmp_path / "raw")


def test_html_error_page_is_rejected(tmp_path):
    page = tmp_path / "not_found.pdf"
    page.write_text("<html><body>404 Not Found</body></html>")

    with pytest.raises(CorpusError, match="not a PDF"):
        download(doc_for(page), tmp_path / "raw")


def test_page_count_mismatch_is_rejected(tmp_path, source_pdf):
    doc = replace(pin(doc_for(source_pdf), source_pdf), pages="4")

    with pytest.raises(CorpusError, match="page count"):
        download(doc, tmp_path / "raw")
