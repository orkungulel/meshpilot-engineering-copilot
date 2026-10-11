"""Predict a class for every face of one test mesh: python -m meshpilot.predict ...

Writes into OUT/<run folder>/<checkpoint>/<mesh>/, where OUT defaults to the
gitignored meshpilot_assets/outputs:
  prediction.json   the predicted labels and their provenance
  prediction.ply    the mesh coloured by predicted class
  ground_truth.ply  the mesh coloured by the dataset's labels, when it has them
  report.md         a short summary; its scores are for this one mesh only
"""

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

from meshpilot.model import build_model, import_diffusion_net, load_checkpoint, predict_faces, read_mesh, source_sha256
from meshpilot.outputs import face_colors, read_labels, single_mesh_scores, write_ply
from rag.corpus import sha256_of

LIBRARIES = ("torch", "numpy", "scipy", "potpourri3d", "robust_laplacian")
SINGLE_MESH_NOTE = "Scores on one mesh: an illustration of this prediction, not the model's benchmark performance."


def split_side(split_path: Path, stem: str) -> str:
    """Which side of the recorded split a mesh is on: 'train', 'test' or 'none'."""
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    if stem in split["test"]:
        return "test"
    if stem in split["train"]:
        return "train"
    return "none"


def counts(labels: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {name: int(np.sum(labels == i)) for i, name in enumerate(class_names)}


def report(record: dict) -> str:
    model, mesh, truth = record["model"], record["mesh"], record["ground_truth"]
    lines = [
        f"# Predicted classes for {mesh['stem']}",
        "",
        "Real inference with the checkpoint below; these are model predictions, not ground truth.",
        "",
        f"- Checkpoint: `{model['checkpoint']}`, epoch {model['epochs_completed']} of {model['epochs_planned']}, "
        f"{model['input_features']} input"
        + (" (this checkpoint was chosen by test mIoU)" if model["chosen_by_test_miou"] else ""),
        f"- Mesh: `{mesh['obj']}`, {mesh['faces']} faces, in the recorded {mesh['split']} split",
        "",
        "| Class | Predicted faces | Labelled faces | IoU on this mesh |",
        "|---|---|---|---|",
    ]
    for name, n_pred in record["predicted_face_counts"].items():
        if truth is None:
            lines.append(f"| {name} | {n_pred} | – | – |")
            continue
        n_true = truth["face_counts"][name]
        iou = truth["single_mesh_scores"]["per_class_iou"].get(name)
        lines.append(f"| {name} | {n_pred} | {n_true} | {'–' if iou is None else f'{iou:.3f}'} |")
    if truth is not None:
        scores = truth["single_mesh_scores"]
        lines += [
            "",
            f"Face accuracy on this mesh: {scores['accuracy']:.3f} over {scores['scored_faces']} labelled faces. "
            f"Mean IoU over the {len(scores['per_class_iou'])} classes present: {scores['mean_iou']:.3f}.",
            "",
            SINGLE_MESH_NOTE,
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict a class for every face of one test mesh.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="a checkpoint_*.pt from a thesis run folder")
    parser.add_argument("--meshes-dir", type=Path, required=True, help="folder holding <stem>.obj and, if labelled, <stem>.labels")
    parser.add_argument("--stem", required=True, help="the mesh's file name without its extension")
    parser.add_argument("--split", type=Path, required=True, help="the run's split.json; the mesh must be in its test list")
    parser.add_argument("--class-names", type=Path, required=True, help="JSON list of class names, indexed by label")
    parser.add_argument("--diffusion-net-src", type=Path, default=Path("meshpilot_assets/src"),
                        help="folder holding the diffusion_net package used in training")
    parser.add_argument("--out", type=Path, default=Path("meshpilot_assets/outputs"))
    args = parser.parse_args(argv)

    side = split_side(args.split, args.stem)
    if side != "test":
        print(f"{args.stem} is not in the test list of {args.split} (found: {side}); only held-out meshes are used",
              file=sys.stderr)
        return 2

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["args"]
    n_class = int(config["effective_n_class"])
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    if len(class_names) != n_class:
        print(f"{args.class_names} names {len(class_names)} classes but the model has {n_class}", file=sys.stderr)
        return 2

    diffusion_net = import_diffusion_net(args.diffusion_net_src)
    model = build_model(checkpoint, diffusion_net)
    obj_path = args.meshes_dir / f"{args.stem}.obj"
    verts, faces = read_mesh(obj_path)
    pred = predict_faces(model, checkpoint, verts, faces, diffusion_net)

    labels_path = args.meshes_dir / f"{args.stem}.labels"
    truth = read_labels(labels_path) if labels_path.is_file() else None
    if truth is not None and len(truth) != len(faces):
        print(f"{labels_path} has {len(truth)} labels for {len(faces)} faces", file=sys.stderr)
        return 2

    out_dir = args.out / args.checkpoint.parent.name / args.checkpoint.stem / args.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ply(out_dir / "prediction.ply", verts, faces, face_colors(pred, n_class))

    ground_truth = None
    if truth is not None:
        write_ply(out_dir / "ground_truth.ply", verts, faces, face_colors(truth, n_class))
        scores = single_mesh_scores(truth, pred, n_class)
        scores["per_class_iou"] = {class_names[c]: iou for c, iou in scores["per_class_iou"].items()}
        ground_truth = {
            "labels_file": str(labels_path),
            "labels_sha256": sha256_of(labels_path),
            "face_counts": counts(truth, class_names),
            "single_mesh_scores": {"note": SINGLE_MESH_NOTE, **scores},
        }

    record = {
        "kind": "model_prediction",
        "mock": False,
        "mesh": {
            "stem": args.stem,
            "obj": str(obj_path),
            "obj_sha256": sha256_of(obj_path),
            "vertices": int(len(verts)),
            "faces": int(len(faces)),
            "split": side,
            "split_file": str(args.split),
            "split_sha256": sha256_of(args.split),
        },
        "model": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_of(args.checkpoint),
            "run": config["out_dir"],
            "epochs_completed": int(checkpoint["epoch"]) + 1,  # the checkpoint counts epochs from 0
            "epochs_planned": int(config["epochs"]),
            "chosen_by_test_miou": "best_test_iou" in checkpoint,
            "input_features": config["input_features"],
            "k_eig": int(config["k_eig"]),
            "C_width": int(config["C_width"]),
            "N_block": int(config["N_block"]),
            "n_class": n_class,
            "label_kind": config.get("resolved_label_schema", {}).get("label_kind"),
        },
        "preprocessing": {
            "mesh_reader": "potpourri3d.read_mesh; face order checked against the OBJ file",
            "normalisation": "diffusion_net.geometry.normalize_positions: subtract the vertex mean, divide by the largest radius",
            "operators": f"diffusion_net.geometry.get_operators with k_eig={int(config['k_eig'])}, computed without a cache",
            "input": "normalised XYZ positions" if config["input_features"] == "xyz" else "HKS (autoscale, 16 scales)",
        },
        "software": {
            "diffusion_net_source": str(args.diffusion_net_src),
            "diffusion_net_sha256": source_sha256(args.diffusion_net_src),
            **{library: version(library) for library in LIBRARIES},
        },
        "class_names": class_names,
        "predicted_face_counts": counts(pred, class_names),
        "face_labels": pred.tolist(),
        "ground_truth": ground_truth,
    }
    (out_dir / "prediction.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report(record), encoding="utf-8")

    print(f"wrote {out_dir}")
    if ground_truth is not None:
        s = ground_truth["single_mesh_scores"]
        print(f"this mesh only: face accuracy {s['accuracy']:.3f}, mean IoU {s['mean_iou']:.3f} "
              f"over {len(s['per_class_iou'])} classes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
