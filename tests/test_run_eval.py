"""End-to-end tests of eval/run_eval.py with a small saved index and the fake embedder."""

import importlib.util
import json
from dataclasses import asdict

import pytest
from fakes import FakeClaude, FakeEmbedder, FakeResponse, tool_use_reply

from rag import config
from rag.chunking import Chunk
from rag.corpus import Document, sha256_of, write_manifest
from rag.index import VectorIndex
from rag.judge import JUDGE_PROMPT

CHUNKS = [
    Chunk("gec07:12:0", "gec07", 12, "soil nail wall facing design and nail spacing"),
    Chunk("gec10:40:0", "gec10", 40, "drilled shaft side resistance in rock sockets"),
]
QUESTIONS = [
    {"id": "q1", "question": "nail spacing in a soil nail wall", "doc_id": "gec07", "page": 12},
    {"id": "q2", "question": "side resistance of drilled shafts in rock", "doc_id": "gec10", "page": 40},
]


@pytest.fixture
def run_eval(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_eval", config.ROOT / "eval" / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "Embedder", FakeEmbedder)
    return module


def write_questions(path, rows):
    defaults = {"reference_answer": "answer", "type": "fact", "written_by": "human"}
    path.write_text("".join(json.dumps({**defaults, **row}) + "\n" for row in rows))


@pytest.fixture
def workspace(tmp_path):
    """Manifest, saved R1 and R2 indexes and a question file; returns the common CLI arguments."""
    write_manifest(tmp_path / "manifest.csv", [Document(d, d, "FHWA", "2016", "X", "https://x", "US") for d in ("gec07", "gec10")])
    for name in ("R1", "R2"):
        index_config = config.INDEX_CONFIGS[name]
        info = {"embedding_model": index_config.embedding_model, "config": asdict(index_config)}
        VectorIndex.build(CHUNKS, FakeEmbedder(index_config.embedding_model)).save(tmp_path / "index" / name, info)
    write_questions(tmp_path / "questions.jsonl", QUESTIONS)
    return [
        "--manifest", str(tmp_path / "manifest.csv"),
        "--questions", str(tmp_path / "questions.jsonl"),
        "--unanswerable", str(tmp_path / "unanswerable.jsonl"),
        "--index-dir", str(tmp_path / "index"),
        "--results-dir", str(tmp_path / "results"),
        "--raw-dir", str(tmp_path / "raw"),
    ]


def run_dirs(tmp_path):
    return sorted((tmp_path / "results").iterdir())


def test_retrieval_run_writes_metrics_with_provenance(run_eval, workspace, tmp_path):
    assert run_eval.main([*workspace, "retrieval", "--config", "R1"]) == 0

    [run_dir] = run_dirs(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert run_dir.name.endswith("_retrieval_R1")
    assert summary["metrics"]["recall@5"]["hits"] == 2
    assert summary["questions_sha256"] == sha256_of(tmp_path / "questions.jsonl")
    assert summary["index"]["config"]["name"] == "R1"
    assert {"git_commit", "git_dirty"} <= summary.keys()
    assert len((run_dir / "per_question.jsonl").read_text().splitlines()) == 2


def test_invalid_questions_stop_the_run(run_eval, workspace, tmp_path, capsys):
    write_questions(tmp_path / "questions.jsonl", [{**QUESTIONS[0], "doc_id": "gec99"}])

    assert run_eval.main([*workspace, "retrieval"]) == 1
    assert "unknown document gec99" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_report_lists_every_run_and_compare_pairs_them(run_eval, workspace, tmp_path, capsys):
    run_eval.main([*workspace, "retrieval", "--config", "R1"])
    run_eval.main([*workspace, "retrieval", "--config", "R2"])
    capsys.readouterr()

    run_eval.main([*workspace, "report"])
    table = capsys.readouterr().out
    run_r1, run_r2 = run_dirs(tmp_path)
    assert f"| {run_r1.name}" in table and f"| {run_r2.name}" in table

    assert run_eval.main([*workspace, "compare", str(run_r1), str(run_r2)]) == 0
    assert "recall@5 on 2 questions: both 2, neither 0, only A 0, only B 0" in capsys.readouterr().out


def test_compare_refuses_runs_on_different_question_files(run_eval, workspace, tmp_path, capsys):
    run_eval.main([*workspace, "retrieval", "--config", "R1"])
    write_questions(tmp_path / "questions.jsonl", [{**QUESTIONS[0], "question": "reworded"}, QUESTIONS[1]])
    run_eval.main([*workspace, "retrieval", "--config", "R2"])
    run_r1, run_r2 = run_dirs(tmp_path)

    assert run_eval.main([*workspace, "compare", str(run_r1), str(run_r2)]) == 1
    assert "different question files" in capsys.readouterr().err


def test_missing_question_file_is_reported_without_a_traceback(run_eval, workspace, tmp_path, capsys):
    (tmp_path / "questions.jsonl").unlink()

    assert run_eval.main([*workspace, "validate"]) == 1
    assert "does not exist yet" in capsys.readouterr().err


# --- answer runs (fake Claude) -------------------------------------------------


def section(prompt: str, tag: str) -> str:
    return prompt[prompt.index(f"<{tag}>\n") + len(tag) + 3 : prompt.index(f"\n</{tag}>")]


def fake_claude(judge_correct, served_model=None) -> FakeClaude:
    """Answers cite their first source and abstain on Eurocode questions; judge_correct decides the verdict."""

    def respond(request):
        prompt = request["messages"][0]["content"]
        if request["system"] == JUDGE_PROMPT:
            correct = judge_correct(section(prompt, "question"), section(prompt, "reference_answer"), section(prompt, "system_answer"))
            return {"reason": "compared with the reference", "correct": correct}
        reply = {"answerable": "Eurocode" not in prompt, "answer": "answer", "citations": ["S1"]}
        return FakeResponse(json.dumps(reply), model=served_model or request["model"])

    return FakeClaude(respond=respond)


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def test_answers_run_scores_judged_answers_and_writes_a_review(run_eval, workspace, tmp_path, monkeypatch, api_key):
    unanswerable = {"id": "u1", "question": "What does Eurocode 7 require?", "written_by": "human"}
    (tmp_path / "unanswerable.jsonl").write_text(json.dumps(unanswerable) + "\n")
    client = fake_claude(judge_correct=lambda question, reference, answer: "nail" in question)
    monkeypatch.setattr(run_eval, "make_client", lambda: client)

    assert run_eval.main([*workspace, "answers", "--config", "R1", "--reps", "2", "--workers", "2"]) == 0

    [run_dir] = run_dirs(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    metrics = summary["metrics"]
    assert run_dir.name.endswith("_answers_rag_R1")
    assert (metrics["correct"]["hits"], metrics["correct"]["n"]) == (1, 2)
    assert metrics["correct"]["by_rep"] == [{"hits": 1, "n": 2}, {"hits": 1, "n": 2}]
    assert metrics["cited_expected_page"]["hits"] == 2
    assert (metrics["abstained_when_unanswerable"]["hits"], metrics["abstained_when_unanswerable"]["n"]) == (1, 1)
    assert summary["n_errors"] == 0
    assert len((run_dir / "per_question.jsonl").read_text().splitlines()) == 6  # 3 questions x 2 runs
    assert "**Judge:** correct." in (run_dir / "review.md").read_text()
    judge_requests = [r for r in client.requests if r["system"] == JUDGE_PROMPT]
    assert len(judge_requests) == 4  # unanswerable questions are not judged
    assert {r["model"] for r in judge_requests} == {config.JUDGE_MODEL}


def test_answers_from_another_model_are_logged_not_scored(run_eval, workspace, tmp_path, monkeypatch, api_key):
    monkeypatch.setattr(run_eval, "make_client", lambda: fake_claude(lambda *_: True, served_model="claude-sonnet-5"))

    assert run_eval.main([*workspace, "answers", "--reps", "1", "--workers", "1", "--model", "claude-sonnet-5-5"]) == 0

    [run_dir] = run_dirs(tmp_path)
    errors = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    assert [e["failure"] for e in errors] == ["served_model_mismatch", "served_model_mismatch"]
    assert "correct" not in json.loads((run_dir / "summary.json").read_text())["metrics"]


def test_answers_need_an_api_key(run_eval, workspace, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

    assert run_eval.main([*workspace, "answers"]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_judge_check_exposes_a_lenient_judge(run_eval, workspace, tmp_path, monkeypatch, capsys, api_key):
    write_questions(tmp_path / "questions.jsonl", [{**QUESTIONS[0], "reference_answer": "1.5 m"}, {**QUESTIONS[1], "reference_answer": "shotcrete"}])
    # This judge wrongly accepts "I don't know"; the check has to show that.
    lenient = fake_claude(judge_correct=lambda question, reference, answer: answer in (reference, "I don't know."))
    monkeypatch.setattr(run_eval, "make_client", lambda: lenient)

    assert run_eval.main([*workspace, "judge-check", "--workers", "1"]) == 0

    lines = {line.split()[0]: line for line in capsys.readouterr().out.splitlines()}
    assert "2/2 graded as expected" in lines["reference_answer"]
    assert "0/2 graded as expected" in lines["i_dont_know"]
    assert "2/2 graded as expected" in lines["other_questions_answer"]


def test_report_and_compare_cover_answer_runs(run_eval, workspace, tmp_path, monkeypatch, capsys, api_key):
    monkeypatch.setattr(run_eval, "make_client", lambda: fake_claude(lambda question, *_: "nail" in question))
    run_eval.main([*workspace, "answers", "--mode", "closed-book", "--reps", "1", "--workers", "1"])
    run_eval.main([*workspace, "answers", "--mode", "rag", "--reps", "1", "--workers", "1"])
    closed_book, rag = run_dirs(tmp_path)
    capsys.readouterr()

    run_eval.main([*workspace, "report"])
    report = capsys.readouterr().out
    assert "### Answers" in report and closed_book.name in report and rag.name in report

    assert run_eval.main([*workspace, "compare", str(closed_book), str(rag)]) == 0
    assert "correct answers on 2 questions: both 1, neither 1" in capsys.readouterr().out


@pytest.mark.parametrize(
    "served, requested, accepted",
    [
        ("claude-sonnet-5-5", "claude-sonnet-5-5", True),
        ("claude-sonnet-5-5-20260901", "claude-sonnet-5-5", True),  # a dated snapshot of the same model
        ("claude-sonnet-5", "claude-sonnet-5-5", False),  # a fallback model
        ("claude-sonnet-5-5", "claude-sonnet-5", False),  # a prefix is not the same model
    ],
)
def test_served_model_must_be_the_requested_model(run_eval, served, requested, accepted):
    if accepted:
        run_eval.check_served(served, requested)
    else:
        with pytest.raises(run_eval.LLMError, match="served_model_mismatch|answered"):
            run_eval.check_served(served, requested)


def test_retrieval_runs_once_per_question_however_many_runs(run_eval, workspace, monkeypatch, api_key):
    searches = []
    original_search = VectorIndex.search
    monkeypatch.setattr(VectorIndex, "search", lambda self, query, k: searches.append(query) or original_search(self, query, k))
    monkeypatch.setattr(run_eval, "make_client", lambda: fake_claude(lambda *_: True))

    run_eval.main([*workspace, "answers", "--reps", "3", "--workers", "2"])

    assert sorted(searches) == sorted(q["question"] for q in QUESTIONS)


def test_an_unexpected_error_in_one_case_is_logged_and_the_rest_are_scored(run_eval, workspace, tmp_path, monkeypatch, api_key):
    def judge(question, reference, answer):
        if "drilled" in question:
            raise KeyError("boom")
        return True

    monkeypatch.setattr(run_eval, "make_client", lambda: fake_claude(judge))

    assert run_eval.main([*workspace, "answers", "--reps", "1", "--workers", "1"]) == 0

    [run_dir] = run_dirs(tmp_path)
    [error] = [json.loads(line) for line in (run_dir / "errors.jsonl").read_text().splitlines()]
    assert (error["id"], error["failure"]) == ("q2", "KeyError")
    assert json.loads((run_dir / "summary.json").read_text())["metrics"]["correct"]["n"] == 1


def test_pilot_runs_are_labelled_and_runs_with_different_scored_questions_are_not_compared(
    run_eval, workspace, tmp_path, monkeypatch, capsys, api_key
):
    monkeypatch.setattr(run_eval, "make_client", lambda: fake_claude(lambda *_: True))
    run_eval.main([*workspace, "answers", "--reps", "1", "--workers", "1", "--limit", "1"])
    [pilot] = run_dirs(tmp_path)
    assert pilot.name.endswith("_answers_rag_R1_pilot")

    # Same files and limit, but the second run lost q1 to an error.
    def fail_on_nails(question, reference, answer):
        if "nail" in question:
            raise KeyError("boom")
        return True

    monkeypatch.setattr(run_eval, "make_client", lambda: fake_claude(fail_on_nails))
    run_eval.main([*workspace, "answers", "--reps", "1", "--workers", "1", "--limit", "1", "--mode", "closed-book"])
    pilot_closed_book = next(d for d in run_dirs(tmp_path) if d != pilot)

    assert run_eval.main([*workspace, "compare", str(pilot), str(pilot_closed_book)]) == 1
    assert "scored different questions" in capsys.readouterr().err


def agent_claude(judge_correct) -> FakeClaude:
    """The agent searches once with the question, then answers citing S1."""

    def respond(request):
        if request["system"] == JUDGE_PROMPT:
            prompt = request["messages"][0]["content"]
            correct = judge_correct(section(prompt, "question"), section(prompt, "reference_answer"), section(prompt, "system_answer"))
            return {"reason": "compared with the reference", "correct": correct}
        if len(request["messages"]) == 1:  # nothing searched yet
            return tool_use_reply("search_documents", {"query": request["messages"][0]["content"]})
        return {"answerable": True, "answer": "answer", "citations": ["S1"]}

    return FakeClaude(respond=respond)


def test_agent_runs_record_tool_calls_and_steps(run_eval, workspace, tmp_path, monkeypatch, api_key):
    monkeypatch.setattr(run_eval, "make_client", lambda: agent_claude(lambda *_: True))

    args = ["answers", "--mode", "agent", "--reps", "1", "--workers", "2", "--model", "claude-sonnet-5-5"]
    assert run_eval.main([*workspace, *args]) == 0

    [run_dir] = run_dirs(tmp_path)
    assert run_dir.name.endswith("_answers_agent_R1")
    records = [json.loads(line) for line in (run_dir / "per_question.jsonl").read_text().splitlines()]
    assert [r["steps"] for r in records] == [2, 2]
    assert all(r["tool_calls"][0]["tool"] == "search_documents" for r in records)
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["metrics"]["correct"]["hits"] == 2
    assert summary["metrics"]["cited_expected_page"]["hits"] == 2
    assert summary["metrics"]["usage"]["agent_steps_mean"] == 2
    assert "`search_documents`" in (run_dir / "review.md").read_text()


# --- blind labels ----------------------------------------------------------------


def answer_run(run_eval, workspace, tmp_path, monkeypatch, judge_correct, reps: int = 1):
    monkeypatch.setattr(run_eval, "make_client", lambda: fake_claude(judge_correct))
    run_eval.main([*workspace, "answers", "--reps", str(reps), "--workers", "1"])
    [run_dir] = run_dirs(tmp_path)
    return run_dir


def test_blind_labels_are_compared_with_the_judge(run_eval, workspace, tmp_path, monkeypatch, capsys, api_key):
    # The judge passes the soil nail answer and fails the drilled shaft answer.
    run_dir = answer_run(run_eval, workspace, tmp_path, monkeypatch, lambda question, *_: "nail" in question)

    assert run_eval.main([*workspace, "label-sheet", str(run_dir)]) == 0
    sheet = (run_dir / "labels.md").read_text()
    assert "compared with the reference" not in sheet  # the judge's reasons stay hidden
    (run_dir / "labels.md").write_text(sheet.replace("label (y/n): ", "label (y/n): y"))  # both correct
    capsys.readouterr()

    assert run_eval.main([*workspace, "agreement", str(run_dir)]) == 0

    out = capsys.readouterr().out
    assert "agreement        1/2" in out
    assert "q2, run 1: you correct, judge incorrect. compared with the reference" in out
    saved = json.loads((run_dir / "agreement.json").read_text())
    assert (saved["only_human_correct"], saved["only_judge_correct"]) == (1, 0)
    assert saved["judge_model"] == config.JUDGE_MODEL
    run_eval.main([*workspace, "report"])
    assert f"Judge vs blind labels {run_dir.name}: agreement 1/2" in capsys.readouterr().out


def test_an_unfinished_sheet_is_refused(run_eval, workspace, tmp_path, monkeypatch, capsys, api_key):
    run_dir = answer_run(run_eval, workspace, tmp_path, monkeypatch, lambda *_: True)
    run_eval.main([*workspace, "label-sheet", str(run_dir)])

    assert run_eval.main([*workspace, "agreement", str(run_dir)]) == 1

    err = capsys.readouterr().err
    assert "item 1 (q1, run 1): not labelled yet" in err and "item 2 (q2, run 1): not labelled yet" in err
    assert not (run_dir / "agreement.json").exists()


def test_labels_are_never_overwritten_and_the_first_agreement_is_kept(
    run_eval, workspace, tmp_path, monkeypatch, capsys, api_key
):
    run_dir = answer_run(run_eval, workspace, tmp_path, monkeypatch, lambda *_: True, reps=2)
    assert run_eval.main([*workspace, "label-sheet", str(run_dir), "--all-reps"]) == 0
    labels = run_dir / "labels.md"
    assert labels.read_text().count("label (y/n): ") == 4  # 2 questions x 2 runs
    labels.write_text(labels.read_text().replace("label (y/n): ", "label (y/n): n"))

    assert run_eval.main([*workspace, "label-sheet", str(run_dir)]) == 1
    assert labels.read_text().count("label (y/n): n") == 4

    run_eval.main([*workspace, "agreement", str(run_dir)])
    blind = (run_dir / "agreement.json").read_text()
    labels.write_text(labels.read_text().replace("label (y/n): n", "label (y/n): y"))  # changed after seeing the judge
    capsys.readouterr()

    assert run_eval.main([*workspace, "agreement", str(run_dir)]) == 0
    assert "keeps the first, blind result" in capsys.readouterr().out
    assert (run_dir / "agreement.json").read_text() == blind


def test_labels_are_only_for_answer_runs(run_eval, workspace, tmp_path, capsys):
    run_eval.main([*workspace, "retrieval"])
    [run_dir] = run_dirs(tmp_path)

    assert run_eval.main([*workspace, "label-sheet", str(run_dir)]) == 1
    assert "labels are for answer runs" in capsys.readouterr().err
