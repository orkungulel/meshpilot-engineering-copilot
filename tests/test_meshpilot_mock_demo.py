"""The key-free MOCK demo: synthetic mesh, rule-based MOCK labels, scripted chat, no API."""

import numpy as np
from fastapi.testclient import TestClient

from meshpilot import mock_demo
from meshpilot.agent import run_chat
from meshpilot.api import ChatBackend, create_app, load_prediction
from meshpilot.regions import edge_connected_patches


def test_the_synthetic_mesh_is_watertight_and_labelled_by_the_rule():
    verts, faces = mock_demo.stepped_shaft(segments=12)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, uses = np.unique(edges, axis=0, return_counts=True)
    assert (uses == 2).all()  # every edge shared by exactly two triangles
    labels = mock_demo.rule_labels(verts, faces)
    assert set(labels.tolist()) == {0, 1}
    assert len(edge_connected_patches(faces, np.flatnonzero(labels == 1))) == 3  # two caps and the step


def test_the_viewer_serves_it_as_mock_data(tmp_path):
    path = mock_demo.write_demo(tmp_path)
    served = load_prediction(path)
    assert served["mock"] is True and "No model ran" in served["mock_rule"]
    assert served["limitations"][0].startswith("MOCK data: no model ran")
    api = TestClient(create_app(tmp_path, chat=ChatBackend(mock_demo.ScriptedMockChat(), "mock-script", None, {}, "mock")))
    assert api.get("/health").json()["chat_provider"] == "mock"
    reply = api.post("/chat", json={"question": "Show the faces predicted as ExtrudeEnd", "prediction_id": mock_demo.PREDICTION_ID}).json()
    assert reply["mock"] is True and reply["answer"].startswith("MOCK reply")
    assert reply["geometry"][0]["face_count"] == sum(1 for label in served["predicted"]["labels"] if label == 1)


def test_the_scripted_chat_asks_for_a_class_instead_of_inventing_one(tmp_path):
    prediction = load_prediction(mock_demo.write_demo(tmp_path))
    result = run_chat(mock_demo.ScriptedMockChat(), "mock-script", "Where are the holes?", prediction)
    assert result.geometry == [] and result.answer.startswith("MOCK reply")
