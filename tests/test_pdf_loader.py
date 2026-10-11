import glob

import pymupdf
import pytest

from rag.pdf_loader import load_pages


def test_pages_are_numbered_from_one_and_blank_pages_are_kept(make_pdf):
    path = make_pdf(["Bearing capacity of shallow footings.", "", "Settlement of embankments."])

    pages = load_pages(path, "gec06")

    assert [p.number for p in pages] == [1, 2, 3]
    assert {p.doc_id for p in pages} == {"gec06"}
    assert "Bearing capacity" in pages[0].text
    assert pages[1].text.strip() == ""
    assert "Settlement" in pages[2].text


def test_printed_labels_are_kept_separately_from_page_numbers(make_pdf):
    # Two roman-numbered front-matter pages, then the body starts again at 1.
    labels = [
        {"startpage": 0, "style": "r", "firstpagenum": 1},
        {"startpage": 2, "style": "D", "firstpagenum": 1},
    ]
    path = make_pdf(["Preface", "Contents", "Chapter 1"], labels=labels)

    pages = load_pages(path, "doc")

    assert [p.number for p in pages] == [1, 2, 3]
    assert [p.label for p in pages] == ["i", "ii", "1"]


def test_ligatures_are_expanded(tmp_path):
    font = next(
        (f for f in glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)),
        None,
    )
    if font is None:
        pytest.skip("needs a font with ligature glyphs (DejaVu Sans)")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_font(fontname="F0", fontfile=font)
    page.insert_text((72, 72), "ﬁeld ﬂow", fontname="F0")  # "ﬁeld ﬂow"
    path = tmp_path / "ligatures.pdf"
    pdf.save(path)

    assert load_pages(path, "doc")[0].text.split() == ["field", "flow"]
