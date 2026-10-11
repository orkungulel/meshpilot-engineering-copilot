"""MeshPilot's chat and its geometry tool, run against a scripted Claude: no API key, no paid calls."""

import json

import numpy as np
import pytest
from fakes import FakeClaude, FakeEmbedder
from fakes import tool_use_reply as tool_use

from meshpilot.agent import run_chat
from meshpilot.regions import PATCH_NOTE, edge_connected_patches, predicted_class_regions
from rag.chunking import Chunk
from rag.generate import Citation
from rag.index import VectorIndex

# Faces 0 and 1 share an edge; face 2 stands apart.
PREDICTION = {
    "class_names": ["Fillet", "Chamfer"],
    "dataset_note": "Each class is a CAD modelling operation.",
    "mesh": {"stem": "part", "faces": 3, "triangles": [0, 1, 2, 1, 2, 3, 4, 5, 6]},
    "predicted": {"labels": [0, 0, 0]},
}
INDEX = VectorIndex.build([Chunk("guide:12:0", "guide", 12, "fillet radius guidance for machined parts")], FakeEmbedder())
TITLES = {"guide": "Design guide"}


def search(query: str):
    return INDEX.search(query, 1)


def tool_result(request: dict) -> dict:
    return request["messages"][-1]["content"][0]


def test_patches_join_faces_that_share_an_edge_not_just_a_vertex():
    faces = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]])
    assert edge_connected_patches(faces, [0, 1, 2]) == [[0, 1], [2]]
    touching_at_a_corner = np.array([[0, 1, 2], [2, 3, 4]])
    assert edge_connected_patches(touching_at_a_corner, [0, 1]) == [[0], [1]]


def test_regions_are_reported_as_predictions_with_their_caveat():
    regions = predicted_class_regions(PREDICTION, "Fillet")
    assert regions["kind"] == "model_prediction"
    assert (regions["face_ids"], regions["patch_sizes"]) == ([0, 1, 2], [2, 1])
    assert regions["note"] == PATCH_NOTE
    assert predicted_class_regions(PREDICTION, "Chamfer")["face_count"] == 0
    with pytest.raises(ValueError):
        predicted_class_regions(PREDICTION, "Hole")


def test_the_tool_decides_the_highlight_and_the_model_gets_only_counts():
    claude = FakeClaude([tool_use("show_predicted_class", {"class_name": "Fillet"}),
                         {"answer": "The model predicted 3 faces as Fillet.", "citations": []}])
    result = run_chat(claude, "claude-sonnet-5-5", "Where are the fillets?", PREDICTION)
    assert result.geometry[0]["face_ids"] == [0, 1, 2]
    assert result.citations == []
    sent = json.loads(tool_result(claude.requests[1])["content"])
    assert (sent["predicted_faces"], sent["patches"], sent["largest_patch_sizes"]) == (3, 2, [2, 1])
    assert "face_ids" not in sent  # face ids go to the viewer, not into the model's context
    assert result.tool_calls[0].output_summary == "3 faces in 2 patches"


def test_without_documents_search_is_not_offered_and_the_prompt_says_so():
    claude = FakeClaude([{"answer": "No documents are available.", "citations": []}])
    run_chat(claude, "claude-sonnet-5-5", "What fillet radius should I use?", PREDICTION)
    request = claude.requests[0]
    assert [tool["name"] for tool in request["tools"]] == ["calculator", "show_predicted_class"]
    assert request["tools"][1]["input_schema"]["properties"]["class_name"]["enum"] == ["Fillet", "Chamfer"]
    assert "document index is not available" in request["system"]


def test_with_documents_citations_are_mapped_and_unshown_ids_dropped():
    claude = FakeClaude([tool_use("search_documents", {"query": "fillet radius"}),
                         {"answer": "Use the radius the guide gives.", "citations": ["S1", "S7"]}])
    result = run_chat(claude, "claude-sonnet-5-5", "What fillet radius?", PREDICTION, search, TITLES)
    assert claude.requests[0]["tools"][0]["name"] == "search_documents"
    assert result.citations == [Citation("guide", "Design guide", 12)]
    assert result.unsupported_citations == ["S7"]
    assert result.geometry == []  # no prediction was used, so none is shown as evidence


def test_an_unknown_class_is_reported_back_to_the_model():
    claude = FakeClaude([tool_use("show_predicted_class", {"class_name": "Hole"}),
                         {"answer": "The model has no Hole class.", "citations": []}])
    result = run_chat(claude, "claude-sonnet-5-5", "Show the holes", PREDICTION)
    assert tool_result(claude.requests[1])["is_error"] is True
    assert result.geometry == []


def test_one_tool_per_step_and_thinking_blocks_go_back_unchanged():
    reply = tool_use("show_predicted_class", {"class_name": "Fillet"}, thinking=True)
    claude = FakeClaude([reply, {"answer": "Done.", "citations": []}])
    run_chat(claude, "claude-sonnet-5-5", "Show the fillets", PREDICTION)
    assert claude.requests[0]["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    assert claude.requests[1]["messages"][1] == {"role": "assistant", "content": reply.content}
