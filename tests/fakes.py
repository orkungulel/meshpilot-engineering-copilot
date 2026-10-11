"""Test doubles for the embedding model and the Claude API."""

import json
import zlib
from types import SimpleNamespace

import numpy as np


class FakeEmbedder:
    """Deterministic bag-of-words vectors: texts that share words get similar vectors.

    Same attributes and methods as rag.embeddings.Embedder, without torch or a download.
    """

    def __init__(self, model_name: str = "fake-model", dim: int = 256, max_tokens: int = 512):
        self.model_name = model_name
        self.dim = dim
        self.max_tokens = max_tokens
        self.query_prompt = ""

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def count_tokens(self, texts: list[str]) -> list[int]:
        return [len(text.split()) + 2 for text in texts]  # words plus two special tokens

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            vector[zlib.crc32(word.encode()) % self.dim] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class FakeResponse:
    """The parts of an Anthropic message response the code reads."""

    def __init__(self, text=None, model="claude-sonnet-5-5", stop_reason="end_turn", content=None, usage=(120, 30)):
        self.content = content if content is not None else [SimpleNamespace(type="text", text=text)]
        self.model = model
        self.stop_reason = stop_reason
        self.usage = SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1])


class FakeClaude:
    """Stands in for anthropic.Anthropic. Replies are scripted; every request is recorded.

    Each scripted reply is a dict (sent back as the JSON text), a FakeResponse,
    or a function that receives the request and returns either. With `respond`,
    that one function answers every request, which suits concurrent callers.
    """

    def __init__(self, replies=(), respond=None):
        self.replies = list(replies)
        self.respond = respond  # optional function used for every request instead of the list
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request):
        # Snapshot the message list: callers such as the agent keep appending to it,
        # while the real SDK serialises it at call time.
        self.requests.append({**request, "messages": list(request["messages"])})
        reply = self.respond if self.respond else self.replies.pop(0)
        if callable(reply):
            reply = reply(request)
        if isinstance(reply, dict):
            reply = FakeResponse(json.dumps(reply), model=request["model"])
        return reply


def tool_use_reply(name: str, tool_input: dict, call_id: str = "call-1", thinking: bool = False) -> FakeResponse:
    """A reply in which the model calls one tool (optionally after a thinking block)."""
    content = [SimpleNamespace(type="thinking", thinking="")] if thinking else []
    content.append(SimpleNamespace(type="tool_use", id=call_id, name=name, input=tool_input))
    return FakeResponse(content=content, stop_reason="tool_use")
