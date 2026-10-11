"""Calls to the Claude API that must return a JSON object matching a schema.

Answer generation and the judge both go through complete_json(), so refusal
handling, truncation checks and usage accounting are written once.
"""

import json
import time
from dataclasses import dataclass

import anthropic

# If a safety classifier declines a request, the API re-runs it on the fallback
# model Anthropic recommends for that refusal category, inside the same call.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(Exception):
    """A call that returned no usable output. `failure` names the class for the eval's error log."""

    def __init__(self, failure: str, message: str, model: str | None = None, usage: dict | None = None):
        super().__init__(message)
        self.failure = failure  # "refusal", "truncated" or "invalid_output"
        self.model = model
        self.usage = usage or {}


@dataclass(frozen=True)
class Completion:
    data: dict  # the JSON object the model returned
    model: str  # the model that actually served the request (differs if a fallback ran)
    usage: dict
    latency_s: float


def make_client() -> anthropic.Anthropic:
    """The SDK reads ANTHROPIC_API_KEY from the environment; rag.config loads .env first."""
    return anthropic.Anthropic()


def complete_json(
    client, model: str, system: str, prompt: str, schema: dict, max_tokens: int = 16000
) -> Completion:
    started = time.perf_counter()
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        # Structured output: the reply is guaranteed to be JSON matching the schema.
        output_config={"format": {"type": "json_schema", "schema": schema}},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    latency_s = time.perf_counter() - started
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    check_stop_reason(response, usage)
    return Completion(data=reply_json(response, usage), model=response.model, usage=usage, latency_s=round(latency_s, 2))


def check_stop_reason(response, usage: dict) -> None:
    """Check why the model stopped before reading what it wrote."""
    if response.stop_reason == "refusal":
        raise LLMError("refusal", "the model declined the request", response.model, usage)
    if response.stop_reason == "max_tokens":
        raise LLMError("truncated", "the reply hit max_tokens", response.model, usage)


def reply_json(response, usage: dict) -> dict:
    """The JSON object in the reply's text block (thinking blocks before it are skipped)."""
    text = next((block.text for block in response.content if block.type == "text"), None)
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise LLMError("invalid_output", f"no JSON object in the reply: {exc}", response.model, usage) from exc
