"""Sentence embeddings from a local sentence-transformers model (no paid API).

Vectors are L2-normalised, so the inner product the index computes is the
cosine similarity.
"""

import numpy as np

# The bge models were trained with this instruction in front of queries (not
# passages). Models not listed here get no prefix.
QUERY_PROMPTS = {
    "BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: ",
}


class Embedder:
    """A sentence-transformers model on CPU. The tests use a fake with the same attributes."""

    def __init__(self, model_name: str, batch_size: int = 32, revision: str | None = None):
        # Imported here because torch is heavy and the tests never need it.
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        # A pinned revision (a Hugging Face commit) fixes the exact weights. Without one,
        # the call is unchanged from before, so the geotechnical indexes load as they always have.
        self.revision = revision
        pinned = {"revision": revision} if revision else {}
        self.model = SentenceTransformer(model_name, device="cpu", **pinned)
        self.max_tokens: int = self.model.max_seq_length  # longer inputs are cut off silently
        self.query_prompt = QUERY_PROMPTS.get(model_name, "")
        self.batch_size = batch_size

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prompt="")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], prompt=self.query_prompt)[0]

    def count_tokens(self, texts: list[str]) -> list[int]:
        """Tokens the model would see, special tokens included, before truncation."""
        encoded = self.model.tokenizer(texts, add_special_tokens=True, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def _encode(self, texts: list[str], prompt: str) -> np.ndarray:
        # An explicit prompt (even "") also stops sentence-transformers from
        # applying a default prompt from the model's own configuration.
        vectors = self.model.encode(
            texts,
            prompt=prompt,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vectors, dtype=np.float32)
