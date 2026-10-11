import pytest
from fakes import FakeEmbedder

from rag.chunking import Chunk
from rag.config import IndexConfig
from rag.corpus import sha256_of
from rag.index import VectorIndex, build_corpus_index

CHUNKS = [
    Chunk("gec07:12:0", "gec07", 12, "soil nail wall facing design and nail spacing"),
    Chunk("gec10:40:0", "gec10", 40, "drilled shaft side resistance in rock sockets"),
    Chunk("gec12:88:0", "gec12", 88, "driven pile dynamic testing and wave equation analysis"),
]


def test_search_ranks_the_most_similar_chunk_first():
    index = VectorIndex.build(CHUNKS, FakeEmbedder())

    hits = index.search("side resistance of a drilled shaft", k=3)

    assert hits[0].chunk.doc_id == "gec10"
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_k_larger_than_the_index_returns_every_chunk_once():
    hits = VectorIndex.build(CHUNKS, FakeEmbedder()).search("pile", k=10)

    assert sorted(h.chunk.chunk_id for h in hits) == sorted(c.chunk_id for c in CHUNKS)


def test_saved_index_gives_the_same_results_after_loading(tmp_path):
    index = VectorIndex.build(CHUNKS, FakeEmbedder())
    index.save(tmp_path / "R1", {"embedding_model": "fake-model"})

    loaded = VectorIndex.load(tmp_path / "R1", FakeEmbedder())

    assert loaded.search("nail spacing", k=3) == index.search("nail spacing", k=3)


def test_loading_with_a_different_embedding_model_is_refused(tmp_path):
    VectorIndex.build(CHUNKS, FakeEmbedder()).save(tmp_path / "R1", {"embedding_model": "fake-model"})

    with pytest.raises(ValueError, match="built with fake-model"):
        VectorIndex.load(tmp_path / "R1", FakeEmbedder(model_name="other-model"))


def test_empty_corpus_and_bad_k_are_rejected():
    with pytest.raises(ValueError):
        VectorIndex.build([], FakeEmbedder())
    with pytest.raises(ValueError):
        VectorIndex.build(CHUNKS, FakeEmbedder()).search("pile", k=0)


def test_build_info_records_config_and_chunks_over_the_token_limit(make_pdf):
    pdfs = {
        "short": make_pdf(["one two three", "four five"], name="short.pdf"),
        "long": make_pdf([" ".join(f"w{i}" for i in range(40))], name="long.pdf"),
    }
    config = IndexConfig("T", "fake-model", chunk_words=20, overlap_words=5)
    # Fake token count is words + 2, so 20-word chunks (22 tokens) exceed a 21-token limit.
    index, info = build_corpus_index(pdfs, config, FakeEmbedder(max_tokens=21))

    assert info["documents"] == {"long": sha256_of(pdfs["long"]), "short": sha256_of(pdfs["short"])}
    assert info["pages"] == 3
    assert info["chunks"] == len(index.chunks) == 5  # long page: 3 windows, short pages: 1 each
    assert info["chunks_over_token_limit"] == 2  # the two full 20-word windows
    assert info["config"]["chunk_words"] == 20
    assert info["embedding_model"] == "fake-model"
