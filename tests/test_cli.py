import json

from fakes import FakeClaude, FakeEmbedder, tool_use_reply

from rag.cli import main
from rag.corpus import Document, read_manifest, write_manifest


def manifest_with(tmp_path, urls: dict[str, str]):
    path = tmp_path / "manifest.csv"
    docs = [
        Document(doc_id, f"Title {doc_id}", "FHWA", "2016", "X-1", url, "US Government work")
        for doc_id, url in urls.items()
    ]
    write_manifest(path, docs)
    return path


def test_download_with_pin_writes_hashes_into_the_manifest(tmp_path, make_pdf):
    manifest = manifest_with(tmp_path, {"a": make_pdf(["one", "two"], name="a.pdf").as_uri()})

    exit_code = main(["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "download", "--pin"])

    assert exit_code == 0
    [doc] = read_manifest(manifest)
    assert len(doc.sha256) == 64
    assert doc.pages == "2"


def test_one_failing_document_does_not_stop_the_others(tmp_path, make_pdf, capsys):
    manifest = manifest_with(
        tmp_path,
        {"missing": (tmp_path / "nope.pdf").as_uri(), "b": make_pdf(["text"], name="b.pdf").as_uri()},
    )

    exit_code = main(["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "download"])

    assert exit_code == 1
    assert (tmp_path / "raw" / "b.pdf").exists()
    assert "FAILED    missing" in capsys.readouterr().err


def test_check_flags_documents_without_a_text_layer(tmp_path, make_pdf, capsys):
    words = " ".join(["soil"] * 30)
    manifest = manifest_with(
        tmp_path,
        {
            "text": make_pdf([words, words], name="text.pdf").as_uri(),
            "scan": make_pdf(["", "", words], name="scan.pdf").as_uri(),
        },
    )
    main(["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "download"])

    main(["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "check"])

    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()[1:]}
    assert "scanned" not in lines["text"]
    assert "scanned" in lines["scan"]


def test_build_index_then_search_finds_the_matching_page(tmp_path, make_pdf, monkeypatch, capsys):
    monkeypatch.setattr("rag.cli.Embedder", FakeEmbedder)
    manifest = manifest_with(
        tmp_path,
        {
            "nails": make_pdf(["soil nail wall facing design"], name="nails.pdf").as_uri(),
            "shafts": make_pdf(["intro", "drilled shaft side resistance in rock"], name="shafts.pdf").as_uri(),
        },
    )
    paths = ["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "--index-dir", str(tmp_path / "index")]
    assert main([*paths, "download"]) == 0

    assert main([*paths, "build-index", "--config", "R2"]) == 0
    capsys.readouterr()
    assert main([*paths, "search", "--config", "R2", "-k", "1", "side resistance of drilled shafts"]) == 0

    assert "shafts p.2" in capsys.readouterr().out
    info = json.loads((tmp_path / "index" / "R2" / "build_info.json").read_text())
    assert info["config"]["name"] == "R2"
    assert sorted(info["documents"]) == ["nails", "shafts"]


def test_build_index_refuses_an_incomplete_corpus(tmp_path, make_pdf, monkeypatch):
    monkeypatch.setattr("rag.cli.Embedder", FakeEmbedder)
    manifest = manifest_with(tmp_path, {"a": make_pdf(["text"], name="a.pdf").as_uri()})

    exit_code = main(["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"),
                      "--index-dir", str(tmp_path / "index"), "build-index"])

    assert exit_code == 1
    assert not (tmp_path / "index").exists()


def test_ask_prints_the_answer_and_its_citations(tmp_path, make_pdf, monkeypatch, capsys):
    monkeypatch.setattr("rag.cli.Embedder", FakeEmbedder)
    claude = FakeClaude([{"answerable": True, "answer": "Nails are spaced about 1.5 m apart.", "citations": ["S1"]}])
    monkeypatch.setattr("rag.cli.make_client", lambda: claude)
    manifest = manifest_with(tmp_path, {"nails": make_pdf(["soil nail spacing is about 1.5 m"], name="nails.pdf").as_uri()})
    paths = ["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "--index-dir", str(tmp_path / "index")]
    main([*paths, "download"])
    main([*paths, "build-index", "--config", "R2"])
    capsys.readouterr()

    assert main([*paths, "ask", "--config", "R2", "How far apart are soil nails?"]) == 0

    assert capsys.readouterr().out.splitlines() == ["Nails are spaced about 1.5 m apart.", "  - Title nails, page 1"]


def test_sources_lists_where_to_save_each_document_and_its_url(tmp_path, capsys):
    manifest = manifest_with(tmp_path, {"gec07": "https://example.org/gec07.pdf"})

    main(["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "sources"])

    assert capsys.readouterr().out.split() == [str(tmp_path / "raw" / "gec07.pdf"), "https://example.org/gec07.pdf"]


def test_ask_with_the_agent_prints_its_tool_calls_first(tmp_path, make_pdf, monkeypatch, capsys):
    monkeypatch.setattr("rag.cli.Embedder", FakeEmbedder)
    claude = FakeClaude(
        [
            tool_use_reply("search_documents", {"query": "soil nail spacing"}),
            {"answerable": True, "answer": "About 1.5 m.", "citations": ["S1"]},
        ]
    )
    monkeypatch.setattr("rag.cli.make_client", lambda: claude)
    manifest = manifest_with(tmp_path, {"nails": make_pdf(["soil nail spacing is about 1.5 m"], name="nails.pdf").as_uri()})
    paths = ["--manifest", str(manifest), "--raw-dir", str(tmp_path / "raw"), "--index-dir", str(tmp_path / "index")]
    main([*paths, "download"])
    main([*paths, "build-index", "--config", "R2"])
    capsys.readouterr()

    assert main([*paths, "ask", "--config", "R2", "--agent", "How far apart are soil nails?"]) == 0

    assert capsys.readouterr().out.splitlines() == [
        '[step 1] search_documents {"query": "soil nail spacing"} -> S1 nails p.1',
        "About 1.5 m.",
        "  - Title nails, page 1",
    ]
