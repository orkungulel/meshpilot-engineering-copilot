"""The Perplexity adapter with a scripted transport: nothing here reaches the Perplexity API."""

import io
import json
import urllib.error

import pytest
from test_meshpilot_agent import PREDICTION

import meshpilot.perplexity as pplx
from meshpilot.agent import run_chat
from meshpilot.api import chat_provider
from rag.llm import LLMError

KEY = "test-key-value-that-must-not-appear"


def function_call(name, arguments, call_id="call-1"):
    return {"type": "function_call", "id": "fc-1", "call_id": call_id, "name": name,
            "arguments": json.dumps(arguments), "status": "completed", "thought_signature": "opaque"}


def message(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text, "annotations": []}]}


def reply(*output, status="completed", cost=0.0002):
    return {"id": "resp-1", "status": status, "model": "openai/gpt-6-luna", "output": list(output),
            "usage": {"input_tokens": 900, "output_tokens": 40, "cost": {"total_cost": cost, "tool_calls_cost": 0}}}


def scripted(monkeypatch, *replies):
    """A PerplexityClient whose requests are recorded and answered from a script."""
    client = pplx.PerplexityClient(KEY)
    sent, queue = [], list(replies)
    monkeypatch.setattr(client, "send", lambda body: sent.append(body) or queue.pop(0))
    return client, sent


def test_requests_carry_only_our_function_tools_and_bounded_output(monkeypatch):
    client, sent = scripted(monkeypatch, reply(message(json.dumps({"answer": "No fillets.", "citations": []}))))
    run_chat(client, "openai/gpt-6-luna", "Where are the fillets?", PREDICTION)
    [body] = sent
    assert {tool["type"] for tool in body["tools"]} == {"function"}  # no web_search, fetch_url or sandbox
    assert "preset" not in body and body["max_steps"] == 1 and body["store"] is False
    assert body["max_output_tokens"] == pplx.MAX_OUTPUT_TOKENS
    assert body["response_format"]["type"] == "json_schema"
    assert body["instructions"].startswith("You help an engineer")
    assert body["tool_choice"] == "auto"


def test_a_tool_call_round_trip_highlights_and_replays_the_call_unchanged(monkeypatch):
    call = function_call("show_predicted_class", {"class_name": "Fillet"})
    client, sent = scripted(monkeypatch, reply(call),
                            reply(message(json.dumps({"answer": "3 faces are predicted as Fillet.", "citations": []}))))
    result = run_chat(client, "openai/gpt-6-luna", "Show the fillets", PREDICTION)
    assert result.geometry[0]["face_ids"] == [0, 1, 2]
    assert result.answer == "3 faces are predicted as Fillet."
    follow_up = sent[1]["input"]
    assert call in follow_up  # thought_signature and all, as the docs require
    [output] = [item for item in follow_up if item["type"] == "function_call_output"]
    assert output["call_id"] == "call-1" and json.loads(output["output"])["predicted_faces"] == 3
    assert result.usage == {"input_tokens": 1800, "output_tokens": 80}


def test_a_reply_with_any_other_output_is_rejected(monkeypatch):
    client, _ = scripted(monkeypatch, reply({"type": "search_results", "results": []}))
    with pytest.raises(pplx.PerplexityError, match="never requests"):
        run_chat(client, "openai/gpt-6-luna", "Where are the fillets?", PREDICTION)


def test_a_cut_off_reply_is_reported_as_truncated(monkeypatch):
    client, _ = scripted(monkeypatch, reply(message('{"answer": "The mod'), status="incomplete"))
    with pytest.raises(LLMError) as error:
        run_chat(client, "openai/gpt-6-luna", "Where are the fillets?", PREDICTION)
    assert error.value.failure == "truncated"


def test_http_errors_carry_the_message_but_never_the_key(monkeypatch):
    def refuse(request, timeout):
        body = io.BytesIO(json.dumps({"error": {"message": "invalid model"}}).encode())
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, body)

    monkeypatch.setattr(pplx.urllib.request, "urlopen", refuse)
    client = pplx.PerplexityClient(KEY)
    with pytest.raises(pplx.PerplexityError) as error:
        client.send({"model": "openai/gpt-6-luna"})
    assert "HTTP 400: invalid model" in str(error.value)
    assert KEY not in str(error.value) and KEY not in repr(client)


@pytest.mark.parametrize("env, expected", [
    ({"PERPLEXITY_API_KEY": "x"}, "perplexity"),
    ({"ANTHROPIC_API_KEY": "x"}, "claude"),
    ({"ANTHROPIC_API_KEY": "x", "PERPLEXITY_API_KEY": "x"}, "claude"),
    ({"ANTHROPIC_API_KEY": "x", "PERPLEXITY_API_KEY": "x", "MESHPILOT_CHAT_PROVIDER": "perplexity"}, "perplexity"),
    ({"MESHPILOT_CHAT_PROVIDER": "perplexity"}, None),  # chosen, but its key is missing: the chat is off
    ({}, None),
])
def test_the_provider_comes_from_the_setting_or_the_key_present(monkeypatch, env, expected):
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "PERPLEXITY_API_KEY", "MESHPILOT_CHAT_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    assert chat_provider() == expected
