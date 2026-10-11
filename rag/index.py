"""A FAISS index over chunk embeddings, saved to disk together with the chunks.

IndexFlatIP does exact, brute-force search; with normalised vectors the inner
product is the cosine similarity. At this corpus size that takes milliseconds
per query, and exact search keeps approximation error out of the retrieval
evaluation.
"""

import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import faiss
import numpy as np

from rag.chunking import Chunk, chunk_pages
from rag.config import IndexConfig
from rag.corpus import sha256_of
from rag.pdf_loader import load_pages


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine similarity between the query and the chunk


class VectorIndex:
    def __init__(self, faiss_index: faiss.Index, chunks: list[Chunk], embedder):
        if faiss_index.ntotal != len(chunks):
            raise ValueError("the FAISS index and the chunk list differ in length")
        self.faiss_index = faiss_index
        self.chunks = chunks  # row i of the FAISS index holds the vector of chunks[i]
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder) -> "VectorIndex":
        if not chunks:
            raise ValueError("nothing to index")
        vectors = np.ascontiguousarray(embedder.embed_passages([c.text for c in chunks]), dtype=np.float32)
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)
        return cls(faiss_index, chunks, embedder)

    def search(self, query: str, k: int) -> list[Hit]:
        """The k chunks most similar to the query, best first."""
        if k < 1:
            raise ValueError("k must be at least 1")
        query_vector = np.ascontiguousarray(self.embedder.embed_query(query), dtype=np.float32).reshape(1, -1)
        scores, rows = self.faiss_index.search(query_vector, min(k, len(self.chunks)))
        return [Hit(self.chunks[row], float(score)) for row, score in zip(rows[0], scores[0])]

    def save(self, directory: Path, info: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.faiss_index, str(directory / "index.faiss"))
        with open(directory / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        (directory / "build_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, embedder) -> "VectorIndex":
        info = json.loads((directory / "build_info.json").read_text(encoding="utf-8"))
        if info["embedding_model"] != embedder.model_name:
            # Query vectors from another model are not comparable with the stored ones.
            raise ValueError(f"index was built with {info['embedding_model']}, not {embedder.model_name}")
        with open(directory / "chunks.jsonl", encoding="utf-8") as f:
            chunks = [Chunk(**json.loads(line)) for line in f]
        return cls(faiss.read_index(str(directory / "index.faiss")), chunks, embedder)


def build_corpus_index(pdfs: dict[str, Path], config: IndexConfig, embedder) -> tuple[VectorIndex, dict]:
    """Load, chunk and embed every PDF. Also returns a record of how the index was built."""
    pages = [page for doc_id, path in sorted(pdfs.items()) for page in load_pages(path, doc_id)]
    chunks = chunk_pages(pages, config.chunk_words, config.overlap_words)
    started = time.perf_counter()
    index = VectorIndex.build(chunks, embedder)
    embedding_seconds = time.perf_counter() - started
    tokens = embedder.count_tokens([c.text for c in chunks])
    info = {
        "config": asdict(config),
        "embedding_model": embedder.model_name,
        "query_prompt": embedder.query_prompt,
        "documents": {doc_id: sha256_of(path) for doc_id, path in sorted(pdfs.items())},
        "pages": len(pages),
        "chunks": len(chunks),
        "model_max_tokens": embedder.max_tokens,
        "chunk_tokens_median": statistics.median(tokens),
        "chunk_tokens_max": max(tokens),
        # The model silently drops the text beyond its limit from these chunks' embeddings.
        "chunks_over_token_limit": sum(1 for n in tokens if n > embedder.max_tokens),
        "embedding_seconds": round(embedding_seconds, 1),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": package_versions(),
    }
    return index, info


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("pymupdf", "faiss-cpu", "numpy", "sentence-transformers", "transformers", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions
