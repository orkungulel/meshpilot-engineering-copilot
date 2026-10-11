"""The MeshPilot viewer API, serving a tiny prediction written to a temporary folder."""

import json

import pytest
from fakes import FakeClaude, tool_use_reply
from fastapi.testclient import TestClient

from meshpilot.api import LIMITATIONS, ChatBackend, create_app
from rag.corpus import sha256_of

PREDICTION_ID = "runs_x/checkpoint_last/part"


@pytest.fixture
def folder(tmp_path):
    """A two-triangle mesh with labels, and a prediction for it as meshpilot.predict writes it."""
    obj = tmp_path / "part.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n", encoding="utf-8")
    labels = tmp_path / "part.labels"
    labels.write_text("0\n1\n", encoding="utf-8")
    record = {
        "kind": "model_prediction",
        "mock": False,
        "mesh": {"stem": "part", "obj": str(obj), "obj_sha256": sha256_of(obj),
                 "vertices": 4, "faces": 2, "split": "test"},
        "model": {"checkpoint": "runs_x/checkpoint_last.pt", "checkpoint_sha256": "abc",
                  "n_class": 2, "label_kind": "fusion360_seg"},
        "preprocessing": {},
        "software": {},
        "class_names": ["A", "B"],
        "predicted_face_counts": {"A": 0, "B": 2},
        "face_labels": [1, 1],
        "ground_truth": {"labels_file": str(labels), "labels_sha256": sha256_of(labels),
                         "face_counts": {"A": 1, "B": 1}, "single_mesh_scores": {}},
    }
    prediction_dir = tmp_path / "outputs" / PREDICTION_ID
    prediction_dir.mkdir(parents=True)
    (prediction_dir / "prediction.json").write_text(json.dumps(record), encoding="utf-8")
    return tmp_path


def client(folder) -> TestClient:
    return TestClient(create_app(folder / "outputs"))


def write_verify(folder, checkpoint_sha256):
    verify = {"checkpoint_sha256": checkpoint_sha256, "test_split": {"meshes": 3},
              "measured": {"mean_iou": 0.5}, "recorded_in_history": {"mean_iou": 0.5},
              "mesh_check": {"stem": "part", "faces_predicted_differently": 0}, "agrees": True}
    path = folder / "outputs" / "runs_x" / "checkpoint_last" / "verify.json"
    path.write_text(json.dumps(verify), encoding="utf-8")


def test_a_prediction_is_listed_and_served_with_its_mesh_and_labels(folder):
    api = client(folder)
    assert api.get("/predictions").json() == [
        {"id": PREDICTION_ID, "stem": "part", "checkpoint": "runs_x/checkpoint_last.pt", "mock": False}
    ]
    served = api.get(f"/predictions/{PREDICTION_ID}").json()
    assert served["mock"] is False
    assert served["mesh"]["triangles"] == [0, 1, 2, 0, 1, 3]
    assert len(served["mesh"]["positions"]) == 12
    assert served["predicted"]["labels"] == [1, 1]
    assert served["ground_truth"]["labels"] == [0, 1]  # served apart from the predictions
    assert len(served["class_colors"]) == 2
    assert "not machining features" in served["dataset_note"]
    assert served["limitations"] == LIMITATIONS
    assert served["benchmark"] is None


def test_test_split_context_comes_only_from_the_same_checkpoint(folder):
    write_verify(folder, "abc")
    benchmark = client(folder).get(f"/predictions/{PREDICTION_ID}").json()["benchmark"]
    assert benchmark["agrees"] is True
    assert benchmark["mesh_check"]["faces_predicted_differently"] == 0
    write_verify(folder, "another checkpoint")
    assert client(folder).get(f"/predictions/{PREDICTION_ID}").json()["benchmark"] is None


def test_a_mesh_changed_after_the_prediction_is_refused(folder):
    with open(folder / "part.obj", "a", encoding="utf-8") as f:
        f.write("v 2 2 2\n")
    response = client(folder).get(f"/predictions/{PREDICTION_ID}")
    assert response.status_code == 409
    assert "changed" in response.json()["detail"]


def test_only_listed_predictions_are_served(folder):
    assert client(folder).get("/predictions/runs_x/checkpoint_last/other").status_code == 404


def test_the_viewer_page_loads_nothing_from_the_network(folder):
    response = client(folder).get("/")
    assert response.status_code == 200
    assert "MeshPilot" in response.text
    assert "http" not in response.text
    assert response.headers["cache-control"] == "no-cache"  # never an older copy of the page


def test_health_counts_the_predictions_and_says_whether_the_chat_is_on(folder):
    assert client(folder).get("/health").json() == {"status": "ok", "predictions": 1, "chat": False,
                                                    "chat_provider": None, "chat_model": None,
                                                    "chat_max_requests": 3, "documents": False}


def test_without_an_api_key_the_chat_is_off_but_the_viewer_works(folder):
    api = client(folder)
    response = api.post("/chat", json={"question": "Show B", "prediction_id": PREDICTION_ID})
    assert response.status_code == 503
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    assert api.get(f"/predictions/{PREDICTION_ID}").status_code == 200


def test_chat_returns_model_predictions_and_document_citations_apart(folder):
    claude = FakeClaude([tool_use_reply("show_predicted_class", {"class_name": "B"}),
                         {"answer": "The model predicted 2 faces as B.", "citations": []}])
    backend = ChatBackend(claude, "claude-sonnet-5-5", None, {}, "mock")
    api = TestClient(create_app(folder / "outputs", chat=backend))
    reply = api.post("/chat", json={"question": "Show B", "prediction_id": PREDICTION_ID}).json()
    assert reply["answer"] == "The model predicted 2 faces as B."
    assert reply["document_citations"] == []
    [geometry] = reply["geometry"]
    assert (geometry["kind"], geometry["class_name"], geometry["face_ids"]) == ("model_prediction", "B", [0, 1])
    assert geometry["patch_sizes"] == [2]  # the two faces share an edge
    assert reply["documents_available"] is False
    assert (reply["mock"], reply["provider"]) == (True, "mock")
    health = api.get("/health").json()
    assert (health["chat"], health["chat_provider"]) == (True, "mock")  # the viewer labels the mock as such


def test_chat_about_an_unknown_prediction_is_refused(folder):
    backend = ChatBackend(FakeClaude([]), "claude-sonnet-5-5", None, {}, "mock")
    api = TestClient(create_app(folder / "outputs", chat=backend))
    assert api.post("/chat", json={"question": "Show B", "prediction_id": "nothing"}).status_code == 404


def test_preset_questions_name_only_classes_the_model_has(folder):
    path = folder / "outputs" / PREDICTION_ID / "prediction.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["class_names"] = ["Fillet", "Chamfer"]  # a Fusion 360 model without the RevolveSide class
    path.write_text(json.dumps(record), encoding="utf-8")
    presets = client(folder).get(f"/predictions/{PREDICTION_ID}").json()["preset_questions"]
    assert presets == ["Show the faces predicted as Fillet.", "Where did the model predict Chamfer faces?"]


def test_one_chat_question_makes_at_most_the_capped_number_of_requests(folder):
    # A model that keeps asking for tools is stopped at the cap (3 by default), with tools off on the last request.
    calls = [tool_use_reply("show_predicted_class", {"class_name": "B"}, call_id=f"call-{n}") for n in range(5)]
    claude = FakeClaude(calls)
    api = TestClient(create_app(folder / "outputs", chat=ChatBackend(claude, "claude-sonnet-5-5", None, {}, "mock")))
    response = api.post("/chat", json={"question": "Show B", "prediction_id": PREDICTION_ID})
    assert response.status_code == 502 and "step_limit" in response.json()["detail"]
    assert len(claude.requests) == 3
    assert claude.requests[-1]["tool_choice"] == {"type": "none"}
