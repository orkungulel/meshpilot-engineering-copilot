"""The bounded smoke test with a scripted client: nothing here reaches a real API."""

import json

from fakes import FakeClaude, tool_use_reply
from test_meshpilot_api import folder  # noqa: F401  (the fixture: a 2-face mesh with classes A and B)

import meshpilot.smoke as smoke
from meshpilot.api import ChatBackend

KEY = "test-key-value-that-must-not-appear"
SCENARIOS = ["--supported-class", "B", "--unsupported-class", "Hole"]


def answer(text):
    return {"answer": text, "citations": []}


def run(monkeypatch, folder, backend, *args):
    monkeypatch.setattr(smoke, "load_chat_backend", lambda: backend)
    return smoke.main(["--out", str(folder / "outputs"), *SCENARIOS, *args])


def backend(*replies, model="claude-sonnet-5-5"):
    claude = FakeClaude(list(replies))
    return claude, ChatBackend(claude, model, None, {}, "claude")


def test_without_a_key_nothing_is_sent(monkeypatch, capsys, folder):
    assert run(monkeypatch, folder, None, "--run") == 2
    assert "Nothing was sent" in capsys.readouterr().out


def test_a_dry_run_shows_what_would_be_sent_and_sends_nothing(monkeypatch, capsys, folder):
    claude, chat = backend()
    assert run(monkeypatch, folder, chat) == 0
    out = capsys.readouterr().out
    assert "Never sent: meshes, coordinates, face ids, labels or weights" in out
    assert "You help an engineer" in out  # the instructions, shown in full
    assert "BLOCKED: no local document index" in out
    assert "Dry run: nothing was sent" in out and claude.requests == []


def test_a_run_checks_each_scenario_records_cost_and_never_shows_the_key(monkeypatch, capsys, folder):
    monkeypatch.setenv("PERPLEXITY_API_KEY", KEY)
    claude, chat = backend(tool_use_reply("show_predicted_class", {"class_name": "B"}), answer("2 faces are B."),
                           answer("The model has no Hole class, so nothing can be shown."))
    assert run(monkeypatch, folder, chat, "--run") == 0
    out = capsys.readouterr().out
    assert "PASS    supported class" in out and "PASS    unsupported class" in out
    assert "BLOCKED document question" in out  # no index here: skipped, nothing sent
    assert len(claude.requests) == 3
    [path] = (folder / "outputs" / "smoke").glob("smoke_*.json")
    text = path.read_text(encoding="utf-8")
    record = json.loads(text)
    assert record["totals"]["requests"] == 3 and record["totals"]["cost_usd"] > 0
    assert all(entry["cost_source"] == "tokens at list price" for entry in record["requests"])
    assert KEY not in out and KEY not in text


def test_the_run_stops_before_a_request_past_the_approved_count(monkeypatch, capsys, folder):
    claude, chat = backend(tool_use_reply("show_predicted_class", {"class_name": "B"}), answer("unused"))
    assert run(monkeypatch, folder, chat, "--run", "--max-requests", "1") == 1
    assert "STOPPED supported class: budget: all 1 approved requests are used" in capsys.readouterr().out
    assert len(claude.requests) == 1


def test_a_model_without_a_listed_price_is_never_called(monkeypatch, capsys, folder):
    claude, chat = backend(answer("unused"), model="some/unpriced-model")
    assert run(monkeypatch, folder, chat, "--run", "--scenarios", "1") == 1
    assert "cannot be bounded" in capsys.readouterr().out
    assert claude.requests == []


def test_highlighting_some_class_for_an_unsupported_one_fails(monkeypatch, capsys, folder):
    claude, chat = backend(tool_use_reply("show_predicted_class", {"class_name": "A"}), answer("These look like holes."))
    assert run(monkeypatch, folder, chat, "--run", "--scenarios", "3") == 1
    assert "FAIL    unsupported class" in capsys.readouterr().out
