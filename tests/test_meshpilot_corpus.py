"""The MeshPilot demo corpus build, on made-up records: provenance, filling, the PDF and its index."""

import copy
import json

import numpy as np
import pymupdf
import pytest
from fakes import FakeEmbedder

from meshpilot import corpus
from meshpilot.outputs import single_mesh_scores
from rag.corpus import sha256_of

NAMES = ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd"]


@pytest.fixture
def records(tmp_path):
    """A consistent pair of predict and verify records, and every file they describe."""
    checkpoint = tmp_path / "checkpoint_last.pt"
    checkpoint.write_bytes(b"not real weights")
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["seen"], "test": ["part", "other"]}), encoding="utf-8")
    obj = tmp_path / "part.obj"
    obj.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 1 2 4\n", encoding="utf-8")
    labels = tmp_path / "part.labels"
    labels.write_text("0\n1\n", encoding="utf-8")
    face_labels = [0, 2]
    scores = single_mesh_scores(np.array([0, 1]), np.array(face_labels), 8)
    prediction = {
        "mesh": {"stem": "part", "obj": str(obj), "obj_sha256": sha256_of(obj), "faces": 2, "vertices": 4,
                 "split": "test", "split_file": str(split), "split_sha256": sha256_of(split)},
        "model": {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_of(checkpoint), "epochs_completed": 20,
                  "epochs_planned": 20, "chosen_by_test_miou": False, "input_features": "xyz", "k_eig": 64,
                  "C_width": 128, "N_block": 4, "n_class": 8},
        "class_names": NAMES,
        "face_labels": face_labels,
        "predicted_face_counts": {n: face_labels.count(i) for i, n in enumerate(NAMES)},
        "ground_truth": {"labels_file": str(labels), "labels_sha256": sha256_of(labels),
                         "face_counts": {n: [0, 1].count(i) for i, n in enumerate(NAMES)},
                         "single_mesh_scores": {**scores, "per_class_iou": {NAMES[c]: v for c, v in scores["per_class_iou"].items()}}},
    }
    test_iou = {n: 0.1 * (i + 1) for i, n in enumerate(NAMES)}
    verify = {"checkpoint_sha256": sha256_of(checkpoint), "epochs_completed": 20,
              "test_split": {"meshes": 2, "split_file": str(split)},
              "measured": {"mean_iou": float(np.mean(list(test_iou.values()))), "accuracy": 0.55, "per_class_iou": test_iou},
              "recorded_in_history": {"mean_iou": float(np.mean(list(test_iou.values()))), "accuracy": 0.55},
              "tolerances": {"scores": 1e-6}, "agrees": True}
    return prediction, verify, tmp_path


def test_a_placeholder_without_a_value_stops_the_build():
    with pytest.raises(corpus.BuildError, match="unresolved placeholders: y"):
        corpus.fill("a {{x}} and {{ y }}", {"x": "1"})
    assert corpus.fill("a {{x}}", {"x": "1"}) == "a 1"


@pytest.mark.parametrize("tamper, problem", [
    (lambda p, v, d: v.update(agrees=False), "did not agree"),
    (lambda p, v, d: v.update(checkpoint_sha256="0" * 64), "different checkpoints"),
    (lambda p, v, d: (d / "checkpoint_last.pt").write_bytes(b"other weights"), "checkpoint file"),
    (lambda p, v, d: (d / "split.json").write_text(json.dumps({"train": ["part"], "test": ["other"]})), "split file"),
    (lambda p, v, d: p["mesh"].update(stem="seen"), "not a test-only mesh"),
    (lambda p, v, d: (d / "part.labels").write_text("1\n1\n"), "labels"),
    (lambda p, v, d: p["ground_truth"]["single_mesh_scores"].update(accuracy=0.9), "recomputation"),
])
def test_provenance_checks_catch_each_mismatch(records, tamper, problem):
    prediction, verify, folder = copy.deepcopy(records[0]), copy.deepcopy(records[1]), records[2]
    corpus.check_provenance(prediction, verify)  # consistent records pass
    tamper(prediction, verify, folder)
    with pytest.raises(corpus.BuildError, match=problem):
        corpus.check_provenance(prediction, verify)


def test_the_real_template_becomes_a_checked_pdf_with_its_own_index(records):
    prediction, verify, folder = records
    split = corpus.check_provenance(prediction, verify)
    values = corpus.fields(prediction, verify, split, {"verify": "a" * 64, "prediction": "b" * 64, "template": "c" * 64})
    markdown = corpus.fill(corpus.TEMPLATE.read_text(encoding="utf-8"), values)
    pdf = folder / "notes.pdf"
    layout = corpus.build_pdf(markdown, pdf)
    report = corpus.check_pdf(pdf, layout, prediction, verify)
    assert [heading.split(".")[0] for _, heading in report["pages"][1:8]] == ["1", "2", "3", "4", "5", "6", "7"]
    assert report["pages"][-1] == (9, "Sources") and report["links"] == 3
    with pymupdf.open(pdf) as document:
        uris = {link["uri"] for page in document for link in page.get_links()}
    assert "https://arxiv.org/abs/2012.00888v3" in uris

    info = corpus.build_index(pdf, {"test": True}, embedder=FakeEmbedder(), index_dir=folder / "index")
    assert info["pages"] == 9 and info["chunks"] == 9
    search, titles = corpus.load_search(embedder=FakeEmbedder(), index_dir=folder / "index")
    top = search("Limitations patches are not feature instances research prototype")[0]
    assert top.chunk.page == 6 and corpus.DOC_ID in titles


def test_a_pdf_whose_table_differs_from_the_records_fails_its_checks(records):
    prediction, verify, folder = records
    split = corpus.check_provenance(prediction, verify)
    values = corpus.fields(prediction, verify, split, {"verify": "a" * 64, "prediction": "b" * 64, "template": "c" * 64})
    values["test_split_iou_table"] = values["test_split_iou_table"].replace("0.100", "0.999")
    pdf = folder / "notes.pdf"
    layout = corpus.build_pdf(corpus.fill(corpus.TEMPLATE.read_text(encoding="utf-8"), values), pdf)
    with pytest.raises(corpus.BuildError, match="test-split table row for ExtrudeSide"):
        corpus.check_pdf(pdf, layout, prediction, verify)
