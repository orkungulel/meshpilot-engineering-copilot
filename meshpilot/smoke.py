"""A bounded smoke test of the chat with a real provider: python -m meshpilot.smoke [--run]

Three scenarios, each one question about a saved prediction, through the same
code as the viewer's POST /chat (the provider's model chooses the tools; code
computes the faces to highlight):
  1. a supported class: the geometry tool must highlight exactly its predicted faces;
  2. a document question: the answer must cite a local document (needs the RAG index);
  3. an unsupported class: nothing may be highlighted, so no capability is invented.

Without --run it shows the plan and the text that would be sent, and sends
nothing. With --run each request is counted and priced before it is sent,
and the run stops before a request that could pass --max-requests or
--max-usd. The worst case for a request is its size at 2 characters per
token plus its full output allowance. Nothing is retried automatically.
Token usage and cost go to the screen and to OUT/smoke/. The API key is never
printed or written.
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import anthropic

from meshpilot.agent import run_chat, system_prompt, tool_definitions
from meshpilot.api import OUTPUTS_DIR, find_predictions, load_chat_backend, load_prediction
from rag.llm import LLMError

# US dollars per million input and output tokens, as listed by Anthropic and in
# Perplexity's Agent API model list in October 2026 (lower tier, which our
# bounded requests stay in). Check the lists before relying on the figures.
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00),
    "openai/gpt-6-luna": (0.10, 0.50),
    "anthropic/claude-haiku-5-5": (0.10, 0.50),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00),
    "perplexity/sonar": (0.25, 2.50),
}
DOCUMENT_QUESTION = "According to the MeshPilot notes, what does the CutEnd class mean? Cite the page."


def price(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


class BudgetExceeded(Exception):
    """The next request could pass the approved number of requests, input size or spend."""


@dataclass
class Budget:
    max_requests: int
    max_usd: float
    max_input_tokens: int = 6000
    requests: int = 0
    spent_usd: float = 0.0
    log: list[dict] = field(default_factory=list)

    def check(self, model: str, input_tokens: int, max_output_tokens: int) -> float:
        """The worst-case cost of the next request; raises BudgetExceeded if it may not be sent."""
        worst = price(model, input_tokens, max_output_tokens)
        if worst is None:
            raise BudgetExceeded(f"no listed price for {model}, so its cost cannot be bounded")
        if self.requests >= self.max_requests:
            raise BudgetExceeded(f"all {self.max_requests} approved requests are used")
        if input_tokens > self.max_input_tokens:
            raise BudgetExceeded(f"the request is about {input_tokens} input tokens, over the {self.max_input_tokens} limit")
        if self.spent_usd + worst > self.max_usd:
            raise BudgetExceeded(f"${self.spent_usd:.4f} spent; the next request could cost up to ${worst:.4f}, "
                                 f"past the ${self.max_usd:.2f} limit")
        return worst


class BudgetedClient:
    """Wraps a chat client so each request is checked against the budget before it is sent, then priced."""

    def __init__(self, client, budget: Budget, scenario: str = ""):
        self.client = client
        self.budget = budget
        self.scenario = scenario
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **request):
        max_output = min(request["max_tokens"], getattr(self.client, "max_output_tokens", request["max_tokens"]))
        size = len(json.dumps({key: request[key] for key in ("system", "messages", "tools")}, default=str))
        worst = self.budget.check(request["model"], size // 2 + 1, max_output)
        self.budget.requests += 1  # counted before sending: a request that fails may still be billed
        entry = {"scenario": self.scenario, "worst_case_usd": worst}
        try:
            response = self.client.beta.messages.create(**request)
        except Exception as exc:
            self.budget.spent_usd += worst  # the outcome is unknown, so assume the worst
            self.budget.log.append({**entry, "error": type(exc).__name__, "cost_usd": worst, "cost_source": "worst case"})
            raise
        usage = response.usage
        reported = getattr(response, "cost_usd", None)
        cost = reported if reported is not None else price(response.model, usage.input_tokens, usage.output_tokens)
        cost = worst if cost is None else cost
        self.budget.spent_usd += cost
        self.budget.log.append({**entry, "model": response.model, "input_tokens": usage.input_tokens,
                                "output_tokens": usage.output_tokens, "cost_usd": cost,
                                "cost_source": "reported by the provider" if reported is not None else "tokens at list price"})
        return response


def scenarios(prediction: dict, supported: str, unsupported: str, document_question: str) -> list[dict]:
    names = prediction["class_names"]
    if supported not in names or unsupported in names:
        raise ValueError(f"{supported!r} must be one of the model's classes and {unsupported!r} must not be")
    labels = prediction["predicted"]["labels"]
    supported_faces = [f for f, label in enumerate(labels) if label == names.index(supported)]
    return [
        {"name": "supported class", "question": f"Show the faces predicted as {supported}.",
         "expects": f"show_predicted_class highlights exactly the {len(supported_faces)} faces predicted as {supported}",
         "check": lambda r: any(g["class_name"] == supported and g["face_ids"] == supported_faces for g in r.geometry)},
        {"name": "document question", "question": document_question, "needs_documents": True,
         "expects": "the answer cites at least one local document",
         "check": lambda r: bool(r.citations)},
        {"name": "unsupported class", "question": f"Show the faces predicted as {unsupported}.",
         "expects": f"nothing is highlighted, since the model has no {unsupported} class (read the answer as well)",
         "check": lambda r: not r.geometry},
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded smoke test of the MeshPilot chat (sends only with --run).")
    parser.add_argument("--run", action="store_true", help="send the requests: real, billed API calls")
    parser.add_argument("--scenarios", default="1,2,3", help="which scenarios to run, e.g. 1,3")
    parser.add_argument("--max-requests", type=int, default=6, help="API requests allowed in total, follow-ups included")
    parser.add_argument("--max-usd", type=float, default=0.25, help="worst-case spend allowed in total")
    parser.add_argument("--max-steps", type=int, default=3, help="API requests allowed per scenario")
    parser.add_argument("--supported-class", default="Fillet")
    parser.add_argument("--unsupported-class", default="Hole")
    parser.add_argument("--document-question", default=DOCUMENT_QUESTION)
    parser.add_argument("--prediction", help="prediction id, as GET /predictions lists it (default: the first)")
    parser.add_argument("--out", type=Path, default=OUTPUTS_DIR)
    args = parser.parse_args(argv)

    backend = load_chat_backend()
    if backend is None:
        print("No API key for the chat provider (PERPLEXITY_API_KEY or ANTHROPIC_API_KEY in .env). Nothing was sent.")
        return 2
    found = find_predictions(args.out)
    if not found or (args.prediction and args.prediction not in found):
        print(f"No prediction {args.prediction or ''} under {args.out}: run meshpilot.predict first. Nothing was sent.")
        return 2
    prediction_id = args.prediction or next(iter(found))
    prediction = load_prediction(found[prediction_id])
    chosen = [int(n) for n in args.scenarios.split(",")]
    plan = [s for n, s in enumerate(scenarios(prediction, args.supported_class, args.unsupported_class,
                                              args.document_question), 1) if n in chosen]
    documents = backend.search is not None
    real = "MOCK (scripted, no API call)" if backend.mock else f"{backend.provider} (real calls, billed)"

    print(f"provider: {real}; model: {backend.model}; documents: {'local index' if documents else 'not available'}")
    print(f"limits: {args.max_requests} requests, ${args.max_usd:.2f} worst-case spend, {args.max_steps} requests per scenario, "
          "no automatic retries")
    print("\nSent with every request: these instructions, the tool definitions "
          f"({', '.join(t['name'] for t in tool_definitions(prediction['class_names'], documents))}) and the question;")
    print("after a tool call, the model's previous output and the tool's result (class, counts and patch sizes;")
    print("document excerpts when the local index is used). Never sent: meshes, coordinates, face ids, labels or weights.")
    print("---- instructions ----\n" + system_prompt(prediction, documents, args.max_steps) +"\n----------------------")
    for n, s in enumerate(plan, 1):
        blocked = s.get("needs_documents") and not documents
        print(f"{n}. {s['name']}: {s['question']!r}\n   expects: {s['expects']}" + ("\n   BLOCKED: no local document index" if blocked else ""))
    if not args.run:
        print("\nDry run: nothing was sent. Add --run to send these requests (real, billed).")
        return 0

    budget = Budget(args.max_requests, args.max_usd)
    client = backend.client.with_options(max_retries=0) if hasattr(backend.client, "with_options") else backend.client
    results, stopped = [], None
    for s in plan:
        outcome = {"scenario": s["name"], "question": s["question"], "expects": s["expects"]}
        if s.get("needs_documents") and not documents:
            results.append({**outcome, "outcome": "BLOCKED", "reason": "no local document index; nothing was sent"})
            continue
        budgeted = BudgetedClient(client, budget, s["name"])
        try:
            result = run_chat(budgeted, backend.model, s["question"], prediction, backend.search, backend.titles,
                              max_steps=args.max_steps)
        except BudgetExceeded as exc:
            stopped = str(exc)
            results.append({**outcome, "outcome": "STOPPED", "reason": f"budget: {exc}"})
            break
        except (LLMError, anthropic.APIError) as exc:
            results.append({**outcome, "outcome": "FAIL", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        results.append({
            **outcome,
            "outcome": "PASS" if s["check"](result) else "FAIL",
            "answer": result.answer,
            "document_citations": [{"title": c.title, "page": c.page} for c in result.citations],
            "geometry": [{"class_name": g["class_name"], "face_count": g["face_count"], "patch_sizes": g["patch_sizes"]}
                         for g in result.geometry],
            "tool_calls": [f"{c.tool}({json.dumps(c.input)})" for c in result.tool_calls],
            "model_served": result.model,
            "usage": result.usage,
        })

    print()
    for r in results:
        print(f"{r['outcome']:7} {r['scenario']}: {r.get('reason') or r.get('answer')}")
        for key in ("tool_calls", "geometry", "document_citations"):
            if r.get(key):
                print(f"        {key}: {r[key]}")
    tokens_in = sum(e.get("input_tokens", 0) for e in budget.log)
    tokens_out = sum(e.get("output_tokens", 0) for e in budget.log)
    print(f"\n{budget.requests} requests, {tokens_in} input and {tokens_out} output tokens, about ${budget.spent_usd:.4f}")
    record = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": backend.provider, "model": backend.model, "prediction": prediction_id,
        "documents_available": documents,
        "limits": {"max_requests": args.max_requests, "max_usd": args.max_usd, "max_steps": args.max_steps},
        "results": results, "requests": budget.log,
        "totals": {"requests": budget.requests, "input_tokens": tokens_in, "output_tokens": tokens_out,
                   "cost_usd": round(budget.spent_usd, 6)},
        "stopped": stopped,
    }
    out_dir = args.out / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"smoke_{record['time_utc'].replace(':', '').replace('+0000', 'Z')}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    ran = [r for r in results if r["outcome"] != "BLOCKED"]
    return 0 if ran and all(r["outcome"] == "PASS" for r in ran) else 1


if __name__ == "__main__":
    sys.exit(main())
