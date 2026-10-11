"""Checks the Embedder wrapper's contract with a stand-in sentence_transformers module (no torch)."""

import sys
import types

import numpy as np
import pytest

from rag.embeddings import QUERY_PROMPTS, Embedder


class RecordingModel:
    max_seq_length = 512

    def __init__(self, name, device, **options):
        self.calls = []
        self.options = options

    def encode(self, texts, **kwargs):
        self.calls.append(kwargs)
        return np.ones((len(texts), 4), dtype=np.float64) / 2.0

    def tokenizer(self, texts, add_special_tokens, verbose):
        return {"input_ids": [[101, *range(len(t.split())), 102] for t in texts]}


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = RecordingModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_query_prompt_is_used_for_queries_only(fake_sentence_transformers):
    embedder = Embedder("BAAI/bge-small-en-v1.5")

    embedder.embed_query("allowable bearing pressure")
    embedder.embed_passages(["a chunk of text"])

    query_call, passage_call = embedder.model.calls
    assert query_call["prompt"] == QUERY_PROMPTS["BAAI/bge-small-en-v1.5"]
    assert passage_call["prompt"] == ""


def test_vectors_are_normalised_float32(fake_sentence_transformers):
    embedder = Embedder("sentence-transformers/all-MiniLM-L6-v2")

    vectors = embedder.embed_passages(["x", "y"])

    assert embedder.model.calls[0]["normalize_embeddings"] is True
    assert vectors.dtype == np.float32
    assert embedder.query_prompt == ""  # no prompt for models outside QUERY_PROMPTS


def test_token_count_includes_special_tokens(fake_sentence_transformers):
    embedder = Embedder("BAAI/bge-small-en-v1.5")

    assert embedder.count_tokens(["three word chunk"]) == [5]
    assert embedder.max_tokens == 512


def test_a_pinned_revision_reaches_the_model_and_no_revision_changes_nothing(fake_sentence_transformers):
    assert Embedder("BAAI/bge-small-en-v1.5", revision="5c38ec7").model.options == {"revision": "5c38ec7"}
    assert Embedder("BAAI/bge-small-en-v1.5").model.options == {}  # the call R1-R3 make, unchanged
