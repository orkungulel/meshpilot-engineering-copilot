"""A key-free MOCK demo of the viewer: python -m meshpilot.mock_demo

It runs from a fresh clone, without the thesis checkpoint, the dataset or an
API key. Nothing in it comes from the model or from a real provider, and the
viewer says so:
- **The mesh:** a synthetic stepped shaft generated here, not a dataset part.
- **The labels:** a geometric rule, recorded with "mock": true, so the viewer
  shows "MOCK data". Triangles facing along the shaft axis are labelled
  ExtrudeEnd and the rest ExtrudeSide.
- **The chat:** a scripted client that calls the real geometry tool and marks
  every reply MOCK, so the viewer shows "MOCK chat".

The viewer, the highlighting code and the chat loop are the real ones, unchanged.
"""

import argparse
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from meshpilot.api import ChatBackend, create_app
from rag.corpus import sha256_of

NAMES = ["ExtrudeSide", "ExtrudeEnd", "CutSide", "CutEnd", "Fillet", "Chamfer", "RevolveSide", "RevolveEnd"]
RULE = ("MOCK labels from a geometric rule, not a model: triangles whose normal is within about 25 degrees of the "
        "shaft axis are labelled ExtrudeEnd, all others ExtrudeSide. No model ran.")
PREDICTION_ID = "mock/rule-based/stepped-shaft"


def stepped_shaft(segments: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """A watertight mesh of two coaxial cylinders (radius 1 and 0.6) joined by a flat step."""
    angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    rings = [(1.0, 0.0), (1.0, 0.6), (0.6, 0.6), (0.6, 1.4)]  # (radius, height), bottom to top
    verts = [[r * np.cos(a), r * np.sin(a), z] for r, z in rings for a in angles]
    bottom, top = len(verts), len(verts) + 1
    verts += [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]
    ring = lambda k, i: k * segments + i % segments
    faces = []
    for i in range(segments):
        faces.append([bottom, ring(0, i + 1), ring(0, i)])  # bottom cap
        for k in range(3):  # lower side, the step, upper side
            faces += [[ring(k, i), ring(k, i + 1), ring(k + 1, i + 1)], [ring(k, i), ring(k + 1, i + 1), ring(k + 1, i)]]
        faces.append([top, ring(3, i), ring(3, i + 1)])  # top cap
    return np.asarray(verts), np.asarray(faces)


def rule_labels(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    corners = verts[faces]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    axial = np.abs(normals[:, 2]) / np.linalg.norm(normals, axis=1)
    return np.where(axial > 0.9, NAMES.index("ExtrudeEnd"), NAMES.index("ExtrudeSide"))


def write_demo(out: Path) -> Path:
    """Write the synthetic mesh and its MOCK prediction where the viewer API finds predictions."""
    folder = out / PREDICTION_ID
    folder.mkdir(parents=True, exist_ok=True)
    verts, faces = stepped_shaft()
    obj = folder / "stepped-shaft.obj"
    obj.write_text("".join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in verts)
                   + "".join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces), encoding="utf-8")
    labels = rule_labels(verts, faces)
    none = "none: MOCK demo, no model ran"
    record = {
        "kind": "mock_prediction",
        "mock": True,
        "mock_rule": RULE,
        "mesh": {"stem": "stepped-shaft-synthetic", "obj": str(obj.resolve()), "obj_sha256": sha256_of(obj),
                 "vertices": int(len(verts)), "faces": int(len(faces)), "split": "synthetic (not from any dataset)"},
        "model": {"checkpoint": none, "checkpoint_sha256": "", "run": none, "epochs_completed": 0, "epochs_planned": 0,
                  "chosen_by_test_miou": False, "input_features": "none", "k_eig": 0, "C_width": 0, "N_block": 0,
                  "n_class": len(NAMES), "label_kind": "fusion360_seg"},
        "preprocessing": {"mesh_reader": none, "normalisation": none, "operators": none, "input": none},
        "software": {"diffusion_net_sha256": "", "torch": none, "numpy": np.__version__, "scipy": none, "potpourri3d": none},
        "class_names": NAMES,
        "predicted_face_counts": {name: int(np.sum(labels == i)) for i, name in enumerate(NAMES)},
        "face_labels": labels.tolist(),
        "ground_truth": None,
    }
    (folder / "prediction.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return folder / "prediction.json"


def reply(content: list, stop_reason: str):
    return SimpleNamespace(content=content, stop_reason=stop_reason, model="mock-script",
                           usage=SimpleNamespace(input_tokens=0, output_tokens=0))


class ScriptedMockChat:
    """Stands in for a provider: asks the real geometry tool for a class the question names, then answers MOCK."""

    def __init__(self):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, *, tools, messages, tool_choice, **request):
        last = messages[-1]["content"]
        if isinstance(last, str):  # the question: no tool has run yet
            names = next(t["input_schema"]["properties"]["class_name"]["enum"]
                         for t in tools if t["name"] == "show_predicted_class")
            asked = next((n for n in sorted(names, key=len, reverse=True)
                          if re.search(rf"\b{re.escape(n)}\b", last, re.IGNORECASE)), None)
            if asked and tool_choice["type"] != "none":
                call = SimpleNamespace(type="tool_use", id="mock-call-1", name="show_predicted_class",
                                       input={"class_name": asked})
                return reply([call], "tool_use")
            text = ("MOCK reply (scripted, no model): name one of the classes, for example ExtrudeEnd, and the "
                    "geometry tool will highlight its triangles.")
        else:
            result = json.loads(last[0]["content"]) if not last[0].get("is_error") else None
            text = (f"MOCK reply (scripted, no model): the geometry tool highlighted the {result['predicted_faces']} "
                    f"triangles labelled {result['class_name']} by the demo's geometric rule." if result
                    else "MOCK reply (scripted, no model): the geometry tool returned an error.")
        return reply([SimpleNamespace(type="text", text=json.dumps({"answer": text, "citations": []}))], "end_turn")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the key-free MOCK viewer demo (no model, no API key).")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--out", type=Path, default=Path("meshpilot_assets/mock_demo"))
    args = parser.parse_args(argv)
    write_demo(args.out)
    print("MOCK DEMO: a synthetic mesh, labels from a geometric rule and a scripted chat.")
    print("No model runs and no API is called. Open http://127.0.0.1:%d" % args.port)
    import uvicorn

    uvicorn.run(create_app(args.out, chat=ChatBackend(ScriptedMockChat(), "mock-script", None, {}, "mock")),
                host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
