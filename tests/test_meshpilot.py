"""MeshPilot without the thesis assets: file formats, scores and safety checks.

The checkpoints, meshes and diffusion_net library stay on the owner's PC, so
the end-to-end prediction is checked there. These tests cover what decides
whether a prediction can be trusted and shown: face order, restricted
checkpoint loading, matching input features, the library's origin, and
labelling single-mesh scores as such. Tests that need torch skip without it.
"""

import json
import pickle
import sys
import types

import numpy as np
import pytest

from meshpilot.model import import_diffusion_net, input_channels, load_checkpoint
from meshpilot.outputs import (
    PALETTE,
    UNLABELED_COLOR,
    face_colors,
    read_labels,
    read_obj,
    single_mesh_scores,
    write_ply,
)
from meshpilot.predict import SINGLE_MESH_NOTE, main, report


class Payload:
    """Stands in for any object a pickle could smuggle into a checkpoint."""


def write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def isolated_imports(monkeypatch):
    """Undo the sys.path entry and the diffusion_net modules a test adds."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name.split(".")[0] == "diffusion_net"}
    yield
    for name in [name for name in sys.modules if name.split(".")[0] == "diffusion_net"]:
        del sys.modules[name]
    sys.modules.update(saved)


def test_obj_keeps_the_file_order(tmp_path):
    obj = write(tmp_path / "m.obj", "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 2\nf 1 2 3\nf 4/1 3/2 2/3\n")
    verts, faces = read_obj(obj)
    assert verts[3].tolist() == [0, 0, 2]
    assert faces.tolist() == [[0, 1, 2], [3, 2, 1]]


def test_obj_refuses_polygons(tmp_path):
    obj = write(tmp_path / "m.obj", "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n")
    with pytest.raises(ValueError, match="triangle"):
        read_obj(obj)


def test_labels_are_read_as_integers(tmp_path):
    assert read_labels(write(tmp_path / "m.labels", "3\n-1\n0\n")).tolist() == [3, -1, 0]


def test_face_colors_grey_out_unlabelled_faces():
    colors = face_colors(np.array([1, -1]), n_class=8)
    assert colors[0].tolist() == PALETTE[1].tolist()
    assert colors[1].tolist() == UNLABELED_COLOR.tolist()


@pytest.mark.parametrize("labels, n_class", [([8], 8), ([-2], 8), ([0], len(PALETTE) + 1)])
def test_face_colors_refuse_labels_they_cannot_show(labels, n_class):
    with pytest.raises(ValueError):
        face_colors(np.array(labels), n_class)


def test_ply_gives_each_face_its_own_coloured_corners(tmp_path):
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 1, 2], [3, 2, 1]])
    colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    write_ply(tmp_path / "m.ply", verts, faces, colors)
    lines = (tmp_path / "m.ply").read_text(encoding="ascii").splitlines()
    body = lines[lines.index("end_header") + 1:]
    assert "element vertex 6" in lines and "element face 2" in lines
    assert body[:4] == ["0 0 0 255 0 0", "1 0 0 255 0 0", "0 1 0 255 0 0", "0 0 1 0 0 255"]
    assert body[6:] == ["3 0 1 2", "3 3 4 5"]


def test_single_mesh_scores_leave_out_unlabelled_faces_and_absent_classes():
    truth = np.array([0, 0, 1, -1])
    pred = np.array([0, 1, 1, 2])  # class 2 is predicted only on the unlabelled face
    scores = single_mesh_scores(truth, pred, n_class=4)
    assert scores["scored_faces"] == 3
    assert scores["accuracy"] == pytest.approx(2 / 3)
    assert scores["per_class_iou"] == {0: 0.5, 1: 0.5}
    assert scores["mean_iou"] == 0.5


def test_input_features_must_match_the_stored_weights():
    checkpoint = {"args": {"input_features": "hks"}, "model_state": {"first_lin.weight": np.zeros((128, 3))}}
    with pytest.raises(ValueError, match="'hks'"):
        input_channels(checkpoint)
    checkpoint["args"]["input_features"] = "xyz"
    assert input_channels(checkpoint) == 3


def test_diffusion_net_is_imported_from_the_given_folder_without_bytecode(tmp_path, isolated_imports):
    with pytest.raises(FileNotFoundError):
        import_diffusion_net(tmp_path)
    (tmp_path / "diffusion_net").mkdir()
    write(tmp_path / "diffusion_net" / "__init__.py", "from .layers import *\n")
    write(tmp_path / "diffusion_net" / "layers.py", "VALUE = 1\n")
    assert import_diffusion_net(tmp_path).VALUE == 1
    assert not list(tmp_path.rglob("__pycache__"))


def test_another_copy_of_diffusion_net_is_refused(tmp_path, isolated_imports):
    (tmp_path / "given" / "diffusion_net").mkdir(parents=True)
    write(tmp_path / "given" / "diffusion_net" / "__init__.py", "")
    other = types.ModuleType("diffusion_net")
    other.__file__ = str(tmp_path / "other" / "diffusion_net" / "__init__.py")
    sys.modules["diffusion_net"] = other  # as if imported earlier from elsewhere
    with pytest.raises(ImportError, match="not from"):
        import_diffusion_net(tmp_path / "given")


def test_checkpoints_are_loaded_without_running_pickled_code(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "checkpoint_last.pt"
    torch.save({"epoch": 0, "model_state": {}, "args": {}, "payload": Payload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(path)


def test_checkpoints_must_hold_what_the_training_script_saves(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "weights.pt"
    torch.save({"model_state": {}}, path)
    with pytest.raises(ValueError, match="not a DiffusionNet training checkpoint"):
        load_checkpoint(path)


def test_operator_check_ignores_eigenvector_signs_but_not_real_differences():
    torch = pytest.importorskip("torch")
    from meshpilot.verify import projector, relative_difference

    evecs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    mass = torch.ones(3)
    flipped = evecs * torch.tensor([-1.0, 1.0])  # the same eigenvectors, one with the other sign
    assert relative_difference(flipped, evecs) > 0
    assert relative_difference(projector(flipped, mass), projector(evecs, mass)) == 0
    laplacian = torch.eye(3).to_sparse()
    assert relative_difference(laplacian * 1.5, laplacian) == pytest.approx(0.5)


def test_predict_refuses_a_mesh_from_the_training_split(tmp_path, capsys):
    split = write(tmp_path / "split.json", json.dumps({"train": ["seen"], "test": ["held_out"]}))
    code = main([
        "--checkpoint", str(tmp_path / "missing.pt"),
        "--meshes-dir", str(tmp_path),
        "--stem", "seen",
        "--split", str(split),
        "--class-names", str(tmp_path / "names.json"),
    ])
    assert code == 2
    assert "only held-out meshes" in capsys.readouterr().err


def test_report_labels_predictions_and_single_mesh_scores():
    record = {
        "model": {"checkpoint": "c.pt", "epochs_completed": 14, "epochs_planned": 20,
                  "input_features": "xyz", "chosen_by_test_miou": True},
        "mesh": {"stem": "m", "obj": "m.obj", "faces": 2, "split": "test"},
        "predicted_face_counts": {"A": 2, "B": 0},
        "ground_truth": {
            "face_counts": {"A": 1, "B": 1},
            "single_mesh_scores": {"accuracy": 0.5, "scored_faces": 2,
                                   "per_class_iou": {"A": 0.5, "B": 0.0}, "mean_iou": 0.25},
        },
    }
    text = report(record)
    assert "not ground truth" in text
    assert "chosen by test mIoU" in text
    assert SINGLE_MESH_NOTE in text
