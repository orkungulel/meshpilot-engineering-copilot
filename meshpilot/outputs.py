"""Everything after the model: mesh files, class colours, PLY output and single-mesh scores.

Only numpy is needed here, so these parts are tested in CI, which has no torch.
"""

import os
from pathlib import Path

import numpy as np

# Where predict, verify, the smoke test and the corpus build write and read their records (gitignored).
OUTPUTS_DIR = Path(os.getenv("MESHPILOT_OUTPUTS", "meshpilot_assets/outputs"))

# The label value the datasets use for a face without a class.
UNLABELED = -1

# The first eight Tableau 10 colours: distinct hues that stay apart on a shaded mesh.
PALETTE = np.array(
    [
        [78, 121, 167],
        [242, 142, 43],
        [225, 87, 89],
        [118, 183, 178],
        [89, 161, 79],
        [237, 201, 72],
        [176, 122, 161],
        [255, 157, 167],
    ],
    dtype=np.uint8,
)
UNLABELED_COLOR = np.array([200, 200, 200], dtype=np.uint8)


def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertex positions and 0-based face corners, in the order the OBJ file lists them.

    The `.labels` file, the PLY output and the viewer all count faces in this
    order, so the mesh reader used for inference is checked against it.
    """
    verts, faces = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                corners = [int(token.split("/")[0]) for token in line.split()[1:]]
                if len(corners) != 3:
                    raise ValueError(f"{path}: only triangle meshes are supported, found a face with {len(corners)} corners")
                faces.append(corners)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if (faces < 1).any():
        raise ValueError(f"{path}: relative (negative) OBJ indices are not supported")
    return np.asarray(verts, dtype=np.float64).reshape(-1, 3), faces - 1


def read_labels(path: Path) -> np.ndarray:
    """One label per face, read the way the training script reads them."""
    return np.loadtxt(path).astype(np.int64).reshape(-1)


def face_colors(labels: np.ndarray, n_class: int) -> np.ndarray:
    """An RGB colour per face from its class; unlabelled faces are light grey."""
    if n_class > len(PALETTE):
        raise ValueError(f"the palette has {len(PALETTE)} colours but the model has {n_class} classes")
    labels = np.asarray(labels)
    if ((labels < UNLABELED) | (labels >= n_class)).any():
        raise ValueError(f"labels must be {UNLABELED} or a class in 0..{n_class - 1}")
    colors = np.tile(UNLABELED_COLOR, (len(labels), 1))
    known = labels != UNLABELED
    colors[known] = PALETTE[labels[known]]
    return colors


def write_ply(path: Path, verts: np.ndarray, faces: np.ndarray, colors: np.ndarray) -> None:
    """An ASCII PLY with one flat colour per face.

    Each face gets three vertices of its own, coloured with the face's colour,
    because some viewers (Blender, for one) read vertex colours but ignore face
    colours. Face i of the file is face i of the input mesh.
    """
    corners = np.asarray(verts, dtype=np.float64)[np.asarray(faces)].reshape(-1, 3)
    corner_colors = np.repeat(np.asarray(colors, dtype=np.uint8), 3, axis=0)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(corners)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    lines += [f"{x:.9g} {y:.9g} {z:.9g} {r} {g} {b}" for (x, y, z), (r, g, b) in zip(corners, corner_colors)]
    lines += [f"3 {3 * i} {3 * i + 1} {3 * i + 2}" for i in range(len(faces))]
    Path(path).write_text("\n".join(lines) + "\n", encoding="ascii")


def confusion_matrix(truth: np.ndarray, pred: np.ndarray, n_class: int) -> np.ndarray:
    """Counts of (true class, predicted class) over the faces that have a label."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    if truth.shape != pred.shape:
        raise ValueError(f"{truth.shape=} but {pred.shape=}")
    scored = truth != UNLABELED
    counts = np.zeros((n_class, n_class), dtype=np.int64)
    np.add.at(counts, (truth[scored], pred[scored]), 1)
    return counts


def scores(counts: np.ndarray) -> dict:
    """Face accuracy and per-class IoU, computed as the thesis training script computes them.

    IoU is TP / (TP + FP + FN) per class. A class that appears in neither the
    truth nor the prediction has no IoU and is left out of the mean.
    """
    tp = np.diag(counts)
    union = counts.sum(axis=0) + counts.sum(axis=1) - tp
    iou = {c: float(tp[c] / union[c]) for c in range(len(counts)) if union[c]}
    total = int(counts.sum())
    return {
        "scored_faces": total,
        "accuracy": float(tp.sum() / total) if total else None,
        "per_class_iou": iou,
        "mean_iou": float(np.mean(list(iou.values()))) if iou else None,
    }


def single_mesh_scores(truth: np.ndarray, pred: np.ndarray, n_class: int) -> dict:
    """Scores on one mesh. They illustrate one prediction; they are not the model's benchmark score."""
    return scores(confusion_matrix(truth, pred, n_class))
