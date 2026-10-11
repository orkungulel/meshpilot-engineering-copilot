"""Evaluation runner.

    python eval/run_eval.py validate                         # check the question files
    python eval/run_eval.py retrieval --config R1            # recall@k of one index configuration
    python eval/run_eval.py judge-check                      # does the judge grade known answers correctly?
    python eval/run_eval.py answers --mode rag --config R1   # judged answer correctness (API calls)
    python eval/run_eval.py answers --mode closed-book       # the same questions without retrieval
    python eval/run_eval.py answers --mode agent --config R1 # the tool-using agent (A3)
    python eval/run_eval.py report                           # markdown tables of all runs
    python eval/run_eval.py compare RUN_A RUN_B              # paired comparison of two runs
    python eval/run_eval.py label-sheet RUN                  # answers to grade by hand, without verdicts
    python eval/run_eval.py agreement RUN                    # those grades against the judge's

Each run is written to eval/results/<timestamp>_<kind>_.../: summary.json holds
the metrics and everything needed to reproduce them, per_question.jsonl the
outcome for every question, errors.jsonl any call that produced nothing to
score, and for answer runs review.md puts every answer next to its reference.
A labelled answer run also holds labels.md and agreement.json.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rag import config
from rag.agent import run_agent
from rag.corpus import read_manifest, sha256_of
from rag.embeddings import Embedder
from rag.evaluation import (
    answer_outcomes,
    answerable_record,
    answers_table,
    compare_retrieval,
    evaluate_retrieval,
    format_rate,
    judge_check_cases,
    paired_comparison,
    read_run,
    retrieval_table,
    review_markdown,
    summarize_answers,
    summarize_retrieval,
    unanswerable_record,
    write_run,
)
from rag.generate import closed_book_answer, generate_answer
from rag.index import VectorIndex
from rag.judge import judge_answer
from rag.labels import (
    AGREEMENT_FILE,
    LABELS_FILE,
    LabelError,
    agreement,
    agreement_line,
    agreement_text,
    items_to_label,
    label_sheet,
    read_labels,
)
from rag.llm import LLMError, make_client
from rag.questions import (
    Question,
    UnanswerableQuestion,
    content_warnings,
    load_questions,
    load_unanswerable,
    validate_questions,
    validate_unanswerable,
)


def git_state() -> dict:
    """The commit a run used, and whether code or questions had uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True, check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        # Earlier results that are not committed yet do not make a new run dirty.
        changes = git("status", "--porcelain", "--", ".", ":(exclude)eval/results").strip()
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": None}
    return {"git_commit": commit, "git_dirty": bool(changes)}


def load_valid_questions(args: argparse.Namespace) -> list[Question] | None:
    if not args.questions.exists():
        print(f"ERROR   {args.questions} does not exist yet", file=sys.stderr)
        return None
    questions = load_questions(args.questions)
    errors = validate_questions(questions, read_manifest(args.manifest))
    for error in errors:
        print(f"ERROR   {error}", file=sys.stderr)
    return None if errors else questions


def load_valid_unanswerable(args: argparse.Namespace, answerable_ids: set[str]) -> list[UnanswerableQuestion] | None:
    if not args.unanswerable.exists():
        return []
    questions = load_unanswerable(args.unanswerable)
    errors = validate_unanswerable(questions, answerable_ids)
    for error in errors:
        print(f"ERROR   {error}", file=sys.stderr)
    return None if errors else questions


def have_credentials() -> bool:
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    print("ERROR   no Anthropic API key: set ANTHROPIC_API_KEY (see .env.example)", file=sys.stderr)
    return False


def check_served(served: str, requested: str) -> None:
    """A score from a model other than the one being evaluated measures nothing, so it is not scored.

    The served id may only differ by a dated snapshot suffix, such as "-20260901".
    """
    if not re.fullmatch(re.escape(requested) + r"(-\d{8})?", served):
        raise LLMError("served_model_mismatch", f"requested {requested} but {served} answered (fallback?)", served)


def error_record(question_id: str, rep: int, exc: Exception) -> dict:
    return {
        "id": question_id,
        "rep": rep,
        "failure": getattr(exc, "failure", type(exc).__name__),
        "message": str(exc),
        "model": getattr(exc, "model", None),
        "usage": getattr(exc, "usage", {}),
    }


def run_folder(args: argparse.Namespace, label: str) -> tuple[Path, str]:
    finished = datetime.now(timezone.utc)
    return args.results_dir / f"{finished:%Y%m%d-%H%M%S}_{label}", finished.isoformat(timespec="seconds")


def cmd_validate(args: argparse.Namespace) -> int:
    questions = load_valid_questions(args)
    if questions is None:
        return 1
    unanswerable = load_valid_unanswerable(args, {q.id for q in questions})
    if unanswerable is None:
        return 1
    for warning in content_warnings(questions, args.raw_dir):
        print(f"WARNING {warning}")
    print(f"{len(questions)} answerable and {len(unanswerable)} unanswerable questions, no structural errors")
    return 0


def cmd_retrieval(args: argparse.Namespace) -> int:
    questions = load_valid_questions(args)
    if questions is None:
        return 1
    index_config = config.INDEX_CONFIGS[args.config]
    index_dir = args.index_dir / index_config.name
    provenance = git_state()
    index = VectorIndex.load(index_dir, Embedder(index_config.embedding_model))
    records = evaluate_retrieval(index, questions)
    run_dir, created_at = run_folder(args, f"retrieval_{index_config.name}")
    summary = {
        "kind": "retrieval",
        "created_at": created_at,
        **provenance,
        "questions_sha256": sha256_of(args.questions),
        "n_questions": len(questions),
        "questions_with_extra_pages": sum(1 for q in questions if q.also_acceptable),
        "index": json.loads((index_dir / "build_info.json").read_text(encoding="utf-8")),
        "metrics": summarize_retrieval(records),
    }
    write_run(run_dir, summary, records)
    for name, metric in summary["metrics"].items():
        print(f"{name:13} {format_rate(metric)}")
    print(f"written to {run_dir}")
    return 0


def cmd_answers(args: argparse.Namespace) -> int:
    questions = load_valid_questions(args)
    if questions is None:
        return 1
    unanswerable = load_valid_unanswerable(args, {q.id for q in questions})
    if unanswerable is None:
        return 1
    if args.limit:  # a pilot on the first few questions
        questions, unanswerable = questions[: args.limit], unanswerable[: args.limit]
    if not have_credentials():
        return 1
    client = make_client()
    titles = {doc.doc_id: doc.title for doc in read_manifest(args.manifest)}
    index_config, build_info, retrieved = None, None, {}
    if args.mode in ("rag", "agent"):
        index_config = config.INDEX_CONFIGS[args.config]
        index_dir = args.index_dir / index_config.name
        index = VectorIndex.load(index_dir, Embedder(index_config.embedding_model))
        build_info = json.loads((index_dir / "build_info.json").read_text(encoding="utf-8"))
    if args.mode == "rag":
        # Retrieval is deterministic, so it runs once per question, before the parallel
        # API calls (the embedding model's tokenizer is not safe to share between threads).
        retrieved = {q.id: index.search(q.question, args.k) for q in [*questions, *unanswerable]}
    search_lock = threading.Lock()

    def agent_search(query: str):
        # The agent chooses its queries while running, so its searches share a lock instead.
        with search_lock:
            return index.search(query, args.k)

    def run_case(case: tuple[Question | UnanswerableQuestion, int]) -> tuple[dict | None, dict | None]:
        q, rep = case
        try:
            agent_fields = {}
            if args.mode == "rag":
                answer = generate_answer(client, args.model, q.question, retrieved[q.id], titles)
            elif args.mode == "agent":
                result = run_agent(client, args.model, q.question, agent_search, titles)
                answer = result.answer
                agent_fields = {"tool_calls": [asdict(c) for c in result.tool_calls], "steps": result.steps}
            else:
                answer = closed_book_answer(client, args.model, q.question)
            check_served(answer.model, args.model)
            if isinstance(q, UnanswerableQuestion):
                return {**unanswerable_record(q, rep, answer), **agent_fields}, None
            verdict = judge_answer(client, args.judge_model, q.question, q.reference_answer, answer.text)
            check_served(verdict.model, args.judge_model)
            return {**answerable_record(q, rep, answer, verdict), **agent_fields}, None
        except Exception as exc:  # one failed case must not lose the finished ones; it is logged, not scored
            return None, error_record(q.id, rep, exc)

    provenance = git_state()
    cases = [(q, rep) for rep in range(args.reps) for q in [*questions, *unanswerable]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(run_case, cases))
    records = [record for record, _ in outcomes if record]
    errors = [error for _, error in outcomes if error]

    label = f"answers_{args.mode}" + (f"_{index_config.name}" if index_config else "") + ("_pilot" if args.limit else "")
    run_dir, created_at = run_folder(args, label)
    summary = {
        "kind": "answers",
        "mode": args.mode,
        "created_at": created_at,
        **provenance,
        "answer_model": args.model,
        "judge_model": args.judge_model,
        "reps": args.reps,
        "k": args.k if index_config else None,
        "index_config": index_config.name if index_config else None,
        "index": build_info,
        "questions_sha256": sha256_of(args.questions),
        "unanswerable_sha256": sha256_of(args.unanswerable) if args.unanswerable.exists() else None,
        "limit": args.limit,
        "n_answerable": len(questions),
        "n_unanswerable": len(unanswerable),
        "n_errors": len(errors),
        "metrics": summarize_answers(records, args.mode),
    }
    write_run(run_dir, summary, records, errors)
    (run_dir / "review.md").write_text(review_markdown(run_dir.name, summary, records), encoding="utf-8")
    for name, metric in summary["metrics"].items():
        if "ci95" in metric:
            per_run = ", ".join(f"{r['hits']}/{r['n']}" for r in metric["by_rep"])
            print(f"{name:28} {format_rate(metric)}   per run: {per_run}")
    print(f"{len(errors)} calls produced nothing to score" + (" (see errors.jsonl)" if errors else ""))
    print(f"written to {run_dir}")
    return 0


def cmd_judge_check(args: argparse.Namespace) -> int:
    questions = load_valid_questions(args)
    if questions is None:
        return 1
    if args.limit:
        questions = questions[: args.limit]
    if not have_credentials():
        return 1
    client = make_client()

    def run_case(case: tuple[Question, str, str, bool]) -> tuple[dict | None, dict | None]:
        q, check, answer, should_pass = case
        try:
            verdict = judge_answer(client, args.judge_model, q.question, q.reference_answer, answer)
            check_served(verdict.model, args.judge_model)
        except Exception as exc:  # logged, not scored
            return None, {**error_record(q.id, 0, exc), "check": check}
        record = {
            "id": q.id,
            "check": check,
            "answer": answer,
            "should_pass": should_pass,
            "judged_correct": verdict.correct,
            "as_expected": verdict.correct == should_pass,
            "judge_reason": verdict.reason,
            "judge_model": verdict.model,
            "judge_usage": verdict.usage,
        }
        return record, None

    provenance = git_state()
    cases = judge_check_cases(questions)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(run_case, cases))
    records = [record for record, _ in outcomes if record]
    errors = [error for _, error in outcomes if error]
    checks = {}
    for check in dict.fromkeys(c for _, c, _, _ in cases):
        rows = [r for r in records if r["check"] == check]
        checks[check] = {"as_expected": sum(r["as_expected"] for r in rows), "n": len(rows)}
    run_dir, created_at = run_folder(args, "judge-check")
    summary = {
        "kind": "judge-check",
        "created_at": created_at,
        **provenance,
        "judge_model": args.judge_model,
        "questions_sha256": sha256_of(args.questions),
        "n_errors": len(errors),
        "checks": checks,
    }
    write_run(run_dir, summary, records, errors)
    for check, counts in checks.items():
        print(f"{check:24} {counts['as_expected']}/{counts['n']} graded as expected")
    for r in records:
        if not r["as_expected"]:
            print(f"UNEXPECTED {r['id']} {r['check']}: {r['judge_reason']}")
    print(f"written to {run_dir}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    runs: dict[str, list[tuple[str, dict]]] = {"retrieval": [], "answers": [], "judge-check": []}
    for summary_path in sorted(args.results_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs[summary["kind"]].append((summary_path.parent.name, summary))
    if not any(runs.values()):
        print("no runs yet")
    if runs["retrieval"]:
        print("### Retrieval\n\n" + retrieval_table(runs["retrieval"]) + "\n")
    answer_runs = [(name, s) for name, s in runs["answers"] if "correct" in s["metrics"]]
    if answer_runs:
        print("### Answers\n\n" + answers_table(answer_runs) + "\n")
    for name, summary in runs["judge-check"]:
        checks = ", ".join(f"{check} {c['as_expected']}/{c['n']}" for check, c in summary["checks"].items())
        print(f"Judge check {name}: {checks}")
    for name, _ in runs["answers"]:
        agreement_path = args.results_dir / name / AGREEMENT_FILE
        if agreement_path.exists():
            print(agreement_line(name, json.loads(agreement_path.read_text(encoding="utf-8"))))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    summary_a, records_a = read_run(args.run_a)
    summary_b, records_b = read_run(args.run_b)
    if summary_a["kind"] != summary_b["kind"] or summary_a["kind"] not in ("retrieval", "answers"):
        print("compare needs two retrieval runs or two answer runs", file=sys.stderr)
        return 1
    same_files = all(summary_a.get(key) == summary_b.get(key) for key in ("questions_sha256", "unanswerable_sha256", "limit"))
    if not same_files:
        print("the runs used different question files, so they cannot be compared pairwise", file=sys.stderr)
        return 1
    try:
        if summary_a["kind"] == "retrieval":
            result, measure = compare_retrieval(records_a, records_b, args.k), f"recall@{args.k}"
        else:
            result, measure = paired_comparison(answer_outcomes(records_a), answer_outcomes(records_b)), "correct answers"
    except ValueError:
        print("the runs scored different questions (see errors.jsonl), so they cannot be compared pairwise", file=sys.stderr)
        return 1
    print(
        f"{measure} on {result['n']} questions: both {result['both']}, neither {result['neither']}, "
        f"only A {len(result['only_a'])}, only B {len(result['only_b'])}"
    )
    print(f"exact McNemar p = {result['p_value']:.3f}")
    print(f"only A ({args.run_a.name}): {', '.join(result['only_a']) or '-'}")
    print(f"only B ({args.run_b.name}): {', '.join(result['only_b']) or '-'}")
    return 0


def read_answer_run(run_dir: Path) -> tuple[dict, list[dict]] | None:
    if not (run_dir / "summary.json").exists():
        print(f"{run_dir} is not a run folder", file=sys.stderr)
        return None
    summary, records = read_run(run_dir)
    if summary["kind"] != "answers":
        print(f"{run_dir} is a {summary['kind']} run; labels are for answer runs", file=sys.stderr)
        return None
    return summary, records


def cmd_label_sheet(args: argparse.Namespace) -> int:
    run = read_answer_run(args.run)
    if run is None:
        return 1
    _, records = run
    path = args.run / LABELS_FILE
    if path.exists():
        print(f"{path} already exists and may hold your labels, so it is not overwritten", file=sys.stderr)
        return 1
    items = items_to_label(records, args.all_reps)
    if not items:
        print("this run has no judged answers to label", file=sys.stderr)
        return 1
    path.write_text(label_sheet(args.run, items), encoding="utf-8")
    print(f"{len(items)} answers to label in {path}")
    print("Label them before opening review.md or per_question.jsonl: those show the judge's verdicts.")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    run = read_answer_run(args.run)
    if run is None:
        return 1
    summary, records = run
    labels_path = args.run / LABELS_FILE
    if not labels_path.exists():
        print(f"no {LABELS_FILE} in {args.run}: write one with label-sheet first", file=sys.stderr)
        return 1
    try:
        result = agreement(records, read_labels(labels_path.read_text(encoding="utf-8")))
    except LabelError as exc:
        for problem in str(exc).splitlines():
            print(f"ERROR   {problem}", file=sys.stderr)
        return 1
    result = {"judge_model": summary["judge_model"], "labels_sha256": sha256_of(labels_path), **result}
    print(agreement_text(result))
    path = args.run / AGREEMENT_FILE
    if not path.exists():
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"written to {path}")
    elif json.loads(path.read_text(encoding="utf-8"))["labels_sha256"] != result["labels_sha256"]:
        # A label changed after the judge's verdicts were seen is no longer blind.
        print(f"{LABELS_FILE} has changed since {path} was written; that file keeps the first, blind result")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python eval/run_eval.py")
    parser.add_argument("--questions", type=Path, default=config.QUESTIONS_PATH)
    parser.add_argument("--unanswerable", type=Path, default=config.UNANSWERABLE_PATH)
    parser.add_argument("--manifest", type=Path, default=config.MANIFEST_PATH)
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DIR)
    parser.add_argument("--index-dir", type=Path, default=config.INDEX_DIR)
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("validate", help="check the question files").set_defaults(func=cmd_validate)

    retrieval = commands.add_parser("retrieval", help="recall@k of one index configuration")
    retrieval.add_argument("--config", choices=config.INDEX_CONFIGS, default=config.DEFAULT_INDEX)
    retrieval.set_defaults(func=cmd_retrieval)

    answers = commands.add_parser("answers", help="judged answer correctness (makes API calls)")
    answers.add_argument("--mode", choices=("rag", "closed-book", "agent"), default="rag")
    answers.add_argument("--config", choices=config.INDEX_CONFIGS, default=config.DEFAULT_INDEX)
    answers.add_argument("--k", type=int, default=5, help="chunks given to the model")
    answers.add_argument("--reps", type=int, default=3, help="runs per question; answers vary between runs")
    answers.add_argument("--limit", type=int, default=0, help="only the first N questions of each file (pilot)")
    answers.add_argument("--workers", type=int, default=4, help="questions answered in parallel")
    answers.add_argument("--model", default=config.ANSWER_MODEL)
    answers.add_argument("--judge-model", default=config.JUDGE_MODEL)
    answers.set_defaults(func=cmd_answers)

    judge_check = commands.add_parser("judge-check", help="grade answers whose verdict is known in advance")
    judge_check.add_argument("--limit", type=int, default=0)
    judge_check.add_argument("--workers", type=int, default=4)
    judge_check.add_argument("--judge-model", default=config.JUDGE_MODEL)
    judge_check.set_defaults(func=cmd_judge_check)

    commands.add_parser("report", help="markdown tables of all runs").set_defaults(func=cmd_report)

    compare = commands.add_parser("compare", help="paired comparison of two runs")
    compare.add_argument("run_a", type=Path)
    compare.add_argument("run_b", type=Path)
    compare.add_argument("--k", type=int, default=5)
    compare.set_defaults(func=cmd_compare)

    sheet = commands.add_parser("label-sheet", help="write labels.md: the run's answers to grade by hand")
    sheet.add_argument("run", type=Path)
    sheet.add_argument("--all-reps", action="store_true", help="every run of each question, not just the first answer")
    sheet.set_defaults(func=cmd_label_sheet)

    agree = commands.add_parser("agreement", help="the grades in labels.md against the judge's verdicts")
    agree.add_argument("run", type=Path)
    agree.set_defaults(func=cmd_agreement)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
