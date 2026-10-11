"""Perplexity's Agent API as a chat provider, behind the part of the Anthropic client the chat uses.

meshpilot.agent.run_chat calls client.beta.messages.create(...) and reads an
Anthropic-shaped reply. PerplexityClient translates both ways for
POST https://api.perplexity.ai/v1/agent, so the chat loop, the tools and the
highlighting stay as they are:
- system -> instructions; messages -> input items; tools -> function tools;
  the final-answer JSON schema -> response_format.
- Only our own function tools are sent: never web_search, fetch_url, sandbox
  or a preset, which bring per-call charges and search settings. A reply
  containing any other kind of output is rejected.
- max_steps is 1 (no server-side tool loop), store is false, the output is
  capped at max_output_tokens, and nothing is retried automatically.
- The key comes from PERPLEXITY_API_KEY and is only ever put in the
  Authorization header; it is not shown by repr() or in errors.
Anthropic-only options (beta flags, server-side fallbacks) have no Perplexity
equivalent and are dropped.
"""

import json
import urllib.error
import urllib.request
from types import SimpleNamespace

from rag.llm import LLMError

ENDPOINT = "https://api.perplexity.ai/v1/agent"
# The cheapest OpenAI model in Perplexity's Agent API list (October 2026); its
# function calling is checked by the smoke test before anything relies on it.
DEFAULT_MODEL = "openai/gpt-6-luna"
MAX_OUTPUT_TOKENS = 1200
TIMEOUT_S = 60
# Output items our requests can produce; anything else would mean a tool we never asked for ran.
EXPECTED_OUTPUT_ITEMS = {"message", "function_call"}


class PerplexityError(LLMError):
    """A Perplexity request that failed or returned something the chat cannot use."""


def to_input_items(messages: list[dict]) -> list[dict]:
    """Anthropic-style messages as Agent API input items. The model's own turns are replayed unchanged."""
    items = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            items.append({"type": "message", "role": message["role"], "content": content})
            continue
        for block in content:
            kind = block["type"] if isinstance(block, dict) else block.type
            if kind == "perplexity_output":  # every item of an earlier reply, as Perplexity returned it
                items.extend(block.items)
            elif kind == "tool_result":
                items.append({"type": "function_call_output", "call_id": block["tool_use_id"], "output": block["content"]})
    return items


def to_function_tool(tool: dict) -> dict:
    return {"type": "function", "name": tool["name"], "description": tool["description"],
            "parameters": tool["input_schema"], "strict": True}


def to_anthropic_reply(reply: dict):
    """An Agent API response in the shape run_chat reads: content blocks, stop_reason, usage, model."""
    if reply.get("status") in ("failed", "cancelled"):
        raise PerplexityError("api_error", f"the response {reply.get('status')}: {reply.get('error')}", reply.get("model"))
    output = reply.get("output") or []
    unexpected = sorted({item.get("type") for item in output} - EXPECTED_OUTPUT_ITEMS)
    if unexpected:
        raise PerplexityError("unexpected_output", f"the reply contains {unexpected}, which this app never requests", reply.get("model"))
    content = [SimpleNamespace(type="perplexity_output", items=output)]
    texts, refused = [], False
    for item in output:
        if item["type"] == "function_call":
            try:
                arguments = json.loads(item.get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                raise PerplexityError("invalid_output", f"tool arguments are not JSON: {exc}", reply.get("model")) from None
            content.append(SimpleNamespace(type="tool_use", id=item["call_id"], name=item["name"], input=arguments))
        else:
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    texts.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    refused = True
    if texts:
        content.append(SimpleNamespace(type="text", text="".join(texts)))
    if refused:
        stop_reason = "refusal"
    elif any(block.type == "tool_use" for block in content):
        stop_reason = "tool_use"
    elif reply.get("status") == "incomplete":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    usage = reply.get("usage") or {}
    cost = (usage.get("cost") or {}).get("total_cost")
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=reply.get("model"),
        usage=SimpleNamespace(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0)),
        cost_usd=cost,  # Perplexity's own figure for the request, when it reports one
    )


class PerplexityClient:
    """Stands in for anthropic.Anthropic in meshpilot.agent.run_chat."""

    def __init__(self, api_key: str, max_output_tokens: int = MAX_OUTPUT_TOKENS):
        if not api_key:
            raise ValueError("PERPLEXITY_API_KEY is empty")
        self._api_key = api_key
        self.max_output_tokens = max_output_tokens
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def __repr__(self) -> str:
        return f"PerplexityClient(endpoint={ENDPOINT!r}, api_key=<hidden>)"

    def request_body(self, *, model, system, messages, tools, tool_choice, output_config, max_tokens) -> dict:
        """The Agent API request for one chat step; only documented fields, only function tools."""
        schema = output_config["format"]["schema"]
        return {
            "model": model,
            "instructions": system,
            "input": to_input_items(messages),
            "tools": [to_function_tool(tool) for tool in tools],
            "tool_choice": tool_choice["type"],  # "auto" or "none"
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
            "max_output_tokens": min(max_tokens, self.max_output_tokens),
            "max_steps": 1,
            "store": False,
            "stream": False,
        }

    def create(self, *, model, system, messages, tools, tool_choice, output_config, max_tokens, **anthropic_only):
        body = self.request_body(model=model, system=system, messages=messages, tools=tools,
                                 tool_choice=tool_choice, output_config=output_config, max_tokens=max_tokens)
        return to_anthropic_reply(self.send(body))

    def send(self, body: dict) -> dict:
        """One POST, with no automatic retry."""
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                     "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            raise PerplexityError("api_error", f"Perplexity returned HTTP {exc.code}: {error_message(exc)}", body["model"]) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise PerplexityError("api_error", f"could not reach Perplexity: {getattr(exc, 'reason', exc)}", body["model"]) from None


def error_message(exc: urllib.error.HTTPError) -> str:
    """The message from an error body {"error": {"message": ...}}; the request (and its key) is never included."""
    try:
        error = json.loads(exc.read().decode("utf-8", "replace")).get("error") or {}
        return str(error.get("message") or error)[:300]
    except (ValueError, OSError):
        return exc.reason or "no details"
