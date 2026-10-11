"""Faces predicted as one class, grouped into patches of edge-connected faces.

The chat's geometry tool reports these. A patch is a set of faces with the
same predicted class that share edges. It is not a validated feature: one
hole can be split into several patches, and two touching features can merge
into one.
"""

import numpy as np

PATCH_NOTE = (
    "Model predictions, not ground truth. A patch is a group of edge-connected faces predicted as the same "
    "class; the number of patches is not a validated count of features."
)


def edge_connected_patches(faces: np.ndarray, selected) -> list[list[int]]:
    """The selected faces grouped by shared edges, largest group first (then by lowest face id)."""
    parent = {int(f): int(f) for f in selected}

    def root(f: int) -> int:
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    edge_owner = {}  # an edge, as a sorted vertex pair -> a selected face that uses it
    for f in parent:
        a, b, c = (int(v) for v in faces[f])
        for edge in ((a, b), (b, c), (c, a)):
            key = (min(edge), max(edge))
            if key in edge_owner:
                parent[root(f)] = root(edge_owner[key])
            else:
                edge_owner[key] = f
    groups: dict[int, list[int]] = {}
    for f in sorted(parent):
        groups.setdefault(root(f), []).append(f)
    return sorted(groups.values(), key=lambda group: (-len(group), group[0]))


def predicted_class_regions(prediction: dict, class_name: str) -> dict:
    """What show_predicted_class returns, for a prediction as the viewer API serves it."""
    names = prediction["class_names"]
    if class_name not in names:
        raise ValueError(f"unknown class {class_name!r}; the model's classes are {', '.join(names)}")
    class_id = names.index(class_name)
    labels = np.asarray(prediction["predicted"]["labels"])
    faces = np.asarray(prediction["mesh"]["triangles"]).reshape(-1, 3)
    face_ids = np.flatnonzero(labels == class_id).tolist()
    return {
        "kind": "model_prediction",
        "class_name": class_name,
        "class_id": class_id,
        "face_ids": face_ids,
        "face_count": len(face_ids),
        "mesh_face_count": int(len(labels)),
        "patch_sizes": [len(patch) for patch in edge_connected_patches(faces, face_ids)],
        "note": PATCH_NOTE,
    }
