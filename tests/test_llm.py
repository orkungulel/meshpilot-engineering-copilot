from types import SimpleNamespace

import pytest
from fakes import FakeClaude, FakeResponse

from rag.llm import FALLBACK_BETA, LLMError, complete_json

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


def call(client):
    return complete_json(client, "claude-sonnet-5-5", "system text", "user text", SCHEMA)


def test_returns_the_json_object_with_served_model_and_usage():
    client = FakeClaude([{"ok": True}])

    completion = call(client)

    assert completion.data == {"ok": True}
    assert completion.model == "claude-sonnet-5-5"
    assert completion.usage == {"input_tokens": 120, "output_tokens": 30}


def test_request_asks_for_schema_output_and_server_side_fallback():
    client = FakeClaude([{"ok": True}])
    call(client)

    [request] = client.requests
    assert request["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}}
    assert request["betas"] == [FALLBACK_BETA]
    assert request["fallbacks"] == "default"
    assert request["system"] == "system text"
    assert request["messages"] == [{"role": "user", "content": "user text"}]


def test_thinking_blocks_before_the_text_are_skipped():
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text='{"ok": false}')]

    assert call(FakeClaude([FakeResponse(content=content)])).data == {"ok": False}


def test_the_model_that_actually_served_is_reported():
    # After a server-side fallback the response names the fallback model.
    client = FakeClaude([FakeResponse('{"ok": true}', model="claude-sonnet-5")])

    assert call(client).model == "claude-sonnet-5"


@pytest.mark.parametrize(
    "response, failure",
    [
        (FakeResponse(None, stop_reason="refusal", content=[]), "refusal"),
        (FakeResponse('{"ok": tr', stop_reason="max_tokens"), "truncated"),
        (FakeResponse("not json"), "invalid_output"),
        (FakeResponse(content=[]), "invalid_output"),
    ],
)
def test_unusable_replies_raise_with_a_failure_class(response, failure):
    with pytest.raises(LLMError) as excinfo:
        call(FakeClaude([response]))

    assert excinfo.value.failure == failure
