"""Evaluation logic: retrieval recall, answer correctness, and the run folders that hold results.

Retrieval: for every question the index returns its top chunks, and the record
keeps the rank of the first chunk from an expected page and from an expected
document. recall@k is the share of questions whose page rank is at most k; with
one expected page per question this is the same as the hit rate at k.

Answers: every question is answered in several runs because the model's output
varies. A question counts as correct when it was answered correctly in more
than half of its runs, so each question is again a single yes/no and the same
Wilson intervals and McNemar test apply as for retrieval.
"""

import json
import statistics
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from rag.generate import Answer
from rag.index import Hit, VectorIndex
from rag.judge import Verdict
from rag.metrics import mcnemar_exact_p, wilson_interval
from rag.questions import Question, UnanswerableQuestion

# recall@5 is the headline. recall@1 shows ranking quality; recall@20 is the
# ceiling a reranker over the top 20 could reach (DECISIONS.md, entry 11).
RECALL_KS = (1, 5, 20)


# --- shared -------------------------------------------------------------------


def rate(successes: int, n: int) -> dict:
    low, high = wilson_interval(successes, n)
    return {"hits": successes, "n": n, "rate": round(successes / n, 4), "ci95": [round(low, 4), round(high, 4)]}


def format_rate(metric: dict) -> str:
    low, high = metric["ci95"]
    return f"{metric['hits']}/{metric['n']} ({metric['rate']:.0%}; 95% CI {low:.0%}–{high:.0%})"


def paired_comparison(outcomes_a: dict[str, bool], outcomes_b: dict[str, bool]) -> dict:
    """Compare two configurations question by question (outcome True = success)."""
    if outcomes_a.keys() != outcomes_b.keys():
        raise ValueError("the two runs used different questions")
    only_a = sorted(q for q in outcomes_a if outcomes_a[q] and not outcomes_b[q])
    only_b = sorted(q for q in outcomes_a if outcomes_b[q] and not outcomes_a[q])
    both = sum(outcomes_a[q] and outcomes_b[q] for q in outcomes_a)
    return {
        "n": len(outcomes_a),
        "both": both,
        "neither": len(outcomes_a) - both - len(only_a) - len(only_b),
        "only_a": only_a,
        "only_b": only_b,
        "p_value": mcnemar_exact_p(len(only_a), len(only_b)),
    }


def write_run(run_dir: Path, summary: dict, records: list[dict], errors: list[dict] | None = None) -> None:
    run_dir.mkdir(parents=True)  # fails rather than overwrite an earlier run
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_jsonl(run_dir / "per_question.jsonl", records)
    if errors:
        # Calls that produced nothing to score. They are listed, never counted as wrong answers.
        write_jsonl(run_dir / "errors.jsonl", errors)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_run(run_dir: Path) -> tuple[dict, list[dict]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    with open(run_dir / "per_question.jsonl", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return summary, records


# --- retrieval ----------------------------------------------------------------


def first_rank(hits: list[Hit], matches: Callable[[Hit], bool]) -> int | None:
    """1-based rank of the first matching hit, or None if no hit matches."""
    return next((rank for rank, hit in enumerate(hits, start=1) if matches(hit)), None)


def found_within(rank: int | None, k: int) -> bool:
    return rank is not None and rank <= k


def evaluate_retrieval(index: VectorIndex, questions: list[Question], k_max: int = max(RECALL_KS)) -> list[dict]:
    records = []
    for q in questions:
        gold_pages = {(loc.doc_id, loc.page) for loc in q.gold}
        gold_docs = {doc_id for doc_id, _ in gold_pages}
        hits = index.search(q.question, k_max)
        records.append(
            {
                "id": q.id,
                "question": q.question,
                "gold": [{"doc_id": doc_id, "page": page} for doc_id, page in sorted(gold_pages)],
                "page_rank": first_rank(hits, lambda h: (h.chunk.doc_id, h.chunk.page) in gold_pages),
                "doc_rank": first_rank(hits, lambda h: h.chunk.doc_id in gold_docs),
                "hits": [
                    {"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "page": h.chunk.page, "score": round(h.score, 4)}
                    for h in hits
                ],
            }
        )
    return records


def summarize_retrieval(records: list[dict]) -> dict:
    n = len(records)
    metrics = {f"recall@{k}": rate(sum(found_within(r["page_rank"], k) for r in records), n) for k in RECALL_KS}
    metrics["doc_recall@5"] = rate(sum(found_within(r["doc_rank"], 5) for r in records), n)
    return metrics


def compare_retrieval(records_a: list[dict], records_b: list[dict], k: int) -> dict:
    """Paired comparison of two retrieval runs on the same questions at recall@k."""
    outcomes_a = {r["id"]: found_within(r["page_rank"], k) for r in records_a}
    outcomes_b = {r["id"]: found_within(r["page_rank"], k) for r in records_b}
    return {"k": k, **paired_comparison(outcomes_a, outcomes_b)}


def retrieval_table(runs: list[tuple[str, dict]]) -> str:
    """Markdown table with one row per retrieval run, for the README."""
    columns = ("recall@1", "recall@5", "recall@20", "doc_recall@5")
    lines = [
        "| Run | Index | Embedding model | Chunk words / overlap | " + " | ".join(columns) + " |",
        "|---" * (4 + len(columns)) + "|",
    ]
    for name, summary in runs:
        index_config = summary["index"]["config"]
        cells = [
            run_label(name, summary),
            index_config["name"],
            index_config["embedding_model"],
            f"{index_config['chunk_words']} / {index_config['overlap_words']}",
            *(format_rate(summary["metrics"][column]) for column in columns),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + dirty_note(runs)


def run_label(name: str, summary: dict) -> str:
    return name if summary["git_dirty"] is False else f"{name} \\*"


def dirty_note(runs: list[tuple[str, dict]]) -> str:
    if all(summary["git_dirty"] is False for _, summary in runs):
        return ""
    return "\n\n\\* Run with uncommitted code or questions: rerun from a clean commit before reporting it."


# --- answers ------------------------------------------------------------------


def answer_fields(answer: Answer) -> dict:
    return {
        "answer": answer.text,
        "answerable": answer.answerable,
        "citations": [asdict(c) for c in answer.citations],
        "sources": answer.sources,
        "unsupported_citations": answer.unsupported_citations,
        "model": answer.model,
        "usage": answer.usage,
        "latency_s": answer.latency_s,
    }


def answerable_record(q: Question, rep: int, answer: Answer, verdict: Verdict) -> dict:
    gold = {(loc.doc_id, loc.page) for loc in q.gold}
    return {
        "id": q.id,
        "rep": rep,
        "kind": "answerable",
        "question": q.question,
        "reference_answer": q.reference_answer,
        **answer_fields(answer),
        "correct": verdict.correct,
        "judge_reason": verdict.reason,
        "judge_model": verdict.model,
        "judge_usage": verdict.usage,
        "cited_expected_page": any((c.doc_id, c.page) in gold for c in answer.citations),
    }


def unanswerable_record(q: UnanswerableQuestion, rep: int, answer: Answer) -> dict:
    return {"id": q.id, "rep": rep, "kind": "unanswerable", "question": q.question, **answer_fields(answer)}


# metric name: (which questions it covers, the outcome of one record)
ANSWER_OUTCOMES: dict[str, tuple[str, Callable[[dict], bool]]] = {
    "correct": ("answerable", lambda r: r["correct"]),
    "said_not_found": ("answerable", lambda r: not r["answerable"]),  # wrongly, since these are answerable
    "cited_expected_page": ("answerable", lambda r: r["cited_expected_page"]),
    "abstained_when_unanswerable": ("unanswerable", lambda r: not r["answerable"]),
}


def majority(records: list[dict], kind: str, outcome: Callable[[dict], bool]) -> dict[str, bool]:
    """Per question: did the outcome hold in more than half of its scored runs?"""
    runs: dict[str, list[bool]] = {}
    for record in records:
        if record["kind"] == kind:
            runs.setdefault(record["id"], []).append(outcome(record))
    return {question_id: sum(results) > len(results) / 2 for question_id, results in runs.items()}


def summarize_answers(records: list[dict], mode: str) -> dict:
    metrics = {}
    reps = sorted({r["rep"] for r in records})
    for name, (kind, outcome) in ANSWER_OUTCOMES.items():
        if name == "cited_expected_page" and mode == "closed-book":
            continue  # closed-book answers have no citations
        per_question = majority(records, kind, outcome)
        if not per_question:
            continue
        by_rep = []
        for rep in reps:
            results = [outcome(r) for r in records if r["kind"] == kind and r["rep"] == rep]
            by_rep.append({"hits": sum(results), "n": len(results)})
        metrics[name] = {**rate(sum(per_question.values()), len(per_question)), "by_rep": by_rep}
    metrics["usage"] = usage_summary(records)
    return metrics


def usage_summary(records: list[dict]) -> dict:
    def mean_of(key: str, field: str) -> float | None:
        values = [r[key][field] for r in records if r.get(key)]
        return round(statistics.mean(values)) if values else None

    latencies = [r["latency_s"] for r in records if r["latency_s"]]
    steps = [r["steps"] for r in records if "steps" in r]  # agent runs only
    return {
        "agent_steps_mean": round(statistics.mean(steps), 2) if steps else None,
        "answer_input_tokens_mean": mean_of("usage", "input_tokens"),
        "answer_output_tokens_mean": mean_of("usage", "output_tokens"),
        "judge_input_tokens_mean": mean_of("judge_usage", "input_tokens"),
        "judge_output_tokens_mean": mean_of("judge_usage", "output_tokens"),
        "answer_latency_s_median": round(statistics.median(latencies), 2) if latencies else None,
    }


def answer_outcomes(records: list[dict]) -> dict[str, bool]:
    kind, outcome = ANSWER_OUTCOMES["correct"]
    return majority(records, kind, outcome)


def answers_table(runs: list[tuple[str, dict]]) -> str:
    """Markdown table with one row per answer run, for the README."""
    columns = {
        "correct": "Correct (majority of runs)",
        "said_not_found": 'Wrongly said "not in the documents"',
        "abstained_when_unanswerable": "Abstained on unanswerable",
        "cited_expected_page": "Cited the expected page",
    }
    lines = [
        "| Run | Mode | Index | Answer model | Correct per run | " + " | ".join(columns.values()) + " |",
        "|---" * (5 + len(columns)) + "|",
    ]
    for name, summary in runs:
        metrics = summary["metrics"]
        per_run = ", ".join(f"{r['hits']}/{r['n']}" for r in metrics["correct"]["by_rep"])
        cells = [
            run_label(name, summary),
            summary["mode"],
            summary.get("index_config") or "-",
            summary["answer_model"],
            per_run,
            *(format_rate(metrics[key]) if key in metrics else "-" for key in columns),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + dirty_note(runs)


def review_markdown(run_name: str, summary: dict, records: list[dict]) -> str:
    """Every answer next to its reference and the judge's verdict, for a person to check."""
    lines = [
        f"# Answer review: {run_name}",
        "",
        f"Mode `{summary['mode']}`, index `{summary.get('index_config') or '-'}`, "
        f"answer model `{summary['answer_model']}`, judge model `{summary['judge_model']}`.",
        "For each answer: would you grade it the same way as the judge?",
    ]
    for r in sorted(records, key=lambda r: (r["kind"], r["id"], r["rep"])):
        citations = "; ".join(f"{c['title']} p.{c['page']}" for c in r["citations"]) or "none"
        lines += ["", f"## {r['id']} (run {r['rep'] + 1}, {r['kind']})", "", f"**Question:** {r['question']}", ""]
        if r["kind"] == "answerable":
            lines += [f"**Reference answer:** {r['reference_answer']}", ""]
        for call in r.get("tool_calls", []):
            lines += [f"- step {call['step']}: `{call['tool']}` {json.dumps(call['input'])} -> {call['output_summary']}"]
        if r.get("tool_calls"):
            lines += [""]
        lines += [f"**System answer:** {r['answer']}", "", f"**Citations:** {citations}", ""]
        if r["kind"] == "answerable":
            verdict = "correct" if r["correct"] else "incorrect"
            lines += [f"**Judge:** {verdict}. {r['judge_reason']}"]
        else:
            lines += [f"**Abstained:** {'yes' if not r['answerable'] else 'no'}"]
    return "\n".join(lines) + "\n"


# --- judge checks ---------------------------------------------------------------


def judge_check_cases(questions: list[Question]) -> list[tuple[Question, str, str, bool]]:
    """Answers whose correct grade is known in advance: (question, check, answer, should_pass).

    The reference answer itself must pass. An empty answer, "I don't know" and
    another question's reference answer must fail.
    """
    cases = []
    for i, q in enumerate(questions):
        cases += [
            (q, "reference_answer", q.reference_answer, True),
            (q, "empty_answer", "", False),
            (q, "i_dont_know", "I don't know.", False),
        ]
        if len(questions) > 1:
            other = questions[(i + 1) % len(questions)]
            cases.append((q, "other_questions_answer", other.reference_answer, False))
    return cases
