"""Check the inference path against the training run: python -m meshpilot.verify ...

For every test mesh, the training script cached the normalised positions, the
operators and the labels it used (diffusion_net_train.py writes
cache/test_<split>_keig<k>_zb<z>_nc<n>.pt). This command compares:

1. for one mesh, the operators computed here with the cached ones, and the
   predictions made from each set;
2. over the whole cached test split, the mIoU, accuracy and per-class IoU the
   loaded checkpoint scores with the values the run's history.json recorded
   for that epoch.

If both agree, the checkpoint, the library, the model construction and the
preprocessing are the ones the run used. It writes verify.json into
OUT/<run folder>/<checkpoint>/, where the viewer shows it as benchmark
context, and exits with 0 when everything agrees and 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from meshpilot.model import (
    build_model,
    compute_operators,
    import_diffusion_net,
    load_checkpoint,
    predict_from_operators,
    read_mesh,
)
from meshpilot.outputs import confusion_matrix, read_labels, scores
from rag.corpus import sha256_of

# The order of the lists in the training script's dataset cache.
CACHE_FIELDS = ("verts", "faces", "frames", "mass", "L", "evals", "evecs", "gradX", "gradY", "labels")
# Largest operator difference accepted, relative to the largest cached value.
OPERATOR_TOLERANCE = 1e-4
# Largest difference accepted between a measured and a recorded score.
SCORE_TOLERANCE = 1e-6


def relative_difference(a, b) -> float:
    """Largest absolute difference between two tensors, relative to the largest value of b."""
    if a.is_sparse:
        difference, reference = (a - b).coalesce().values(), b.coalesce().values()
    else:
        difference, reference = a - b, b
    if difference.numel() == 0:
        return 0.0
    return float(difference.abs().max()) / (float(reference.abs().max()) or 1.0)


def projector(evecs, mass):
    """The projection onto the eigenvectors' span.

    Eigenvectors are defined only up to sign, and up to rotation within a
    repeated eigenvalue; the projection they define is not, so it is what is
    compared.
    """
    return evecs @ (evecs * mass[:, None]).T


def check_mesh(model, config: dict, entry: dict, obj_path: Path, labels_path: Path, diffusion_net) -> dict:
    """Operators and predictions for one mesh: computed here vs cached in training."""
    import torch

    verts, faces = read_mesh(obj_path)
    ops = compute_operators(verts, faces, int(config["k_eig"]), diffusion_net)
    result = {
        "faces_identical": bool(ops["faces"].shape == entry["faces"].shape and torch.equal(ops["faces"], entry["faces"])),
        "labels_identical": bool(np.array_equal(read_labels(labels_path), entry["labels"].numpy())),
    }
    if not result["faces_identical"]:
        return result
    differences = {name: relative_difference(ops[name], entry[name])
                   for name in ("verts", "frames", "mass", "L", "evals", "gradX", "gradY")}
    differences["evecs (projector)"] = relative_difference(
        projector(ops["evecs"], ops["mass"]), projector(entry["evecs"], entry["mass"]))
    fresh = predict_from_operators(model, config, ops, diffusion_net)
    cached = predict_from_operators(model, config, entry, diffusion_net)
    result["max_relative_difference"] = differences
    result["faces"] = int(len(fresh))
    result["faces_predicted_differently"] = int((fresh != cached).sum())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check inference against the training run's test cache and history.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="a checkpoint_*.pt from a thesis run folder")
    parser.add_argument("--cache", type=Path, required=True, help="the training script's test cache (test_*.pt)")
    parser.add_argument("--history", type=Path, required=True, help="the run's history.json")
    parser.add_argument("--split", type=Path, required=True, help="the run's split.json; the cache follows its test list")
    parser.add_argument("--meshes-dir", type=Path, required=True, help="folder holding <stem>.obj and <stem>.labels")
    parser.add_argument("--stem", required=True, help="the test mesh whose operators are compared")
    parser.add_argument("--class-names", type=Path, required=True, help="JSON list of class names, indexed by label")
    parser.add_argument("--diffusion-net-src", type=Path, default=Path("meshpilot_assets/src"),
                        help="folder holding the diffusion_net package used in training")
    parser.add_argument("--out", type=Path, default=Path("meshpilot_assets/outputs"))
    args = parser.parse_args(argv)

    import torch

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["args"]
    n_class = int(config["effective_n_class"])
    class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    diffusion_net = import_diffusion_net(args.diffusion_net_src)
    model = build_model(checkpoint, diffusion_net)

    print("loading the test cache with weights_only=True (takes a while)...")
    cache = torch.load(args.cache, map_location="cpu", weights_only=True, mmap=True)
    entries = [dict(zip(CACHE_FIELDS, values)) for values in zip(*cache)]
    test = json.loads(args.split.read_text(encoding="utf-8"))["test"]
    if len(entries) != len(test) or args.stem not in test:
        print(f"the cache has {len(entries)} meshes, the split's test list {len(test)}, "
              f"and {args.stem} must be in that list", file=sys.stderr)
        return 1

    index = test.index(args.stem)
    mesh_check = {"stem": args.stem, "cache_index": index,
                  **check_mesh(model, config, entries[index], args.meshes_dir / f"{args.stem}.obj",
                               args.meshes_dir / f"{args.stem}.labels", diffusion_net)}

    print(f"predicting the {len(entries)} cached test meshes...")
    counts = np.zeros((n_class, n_class), dtype=np.int64)
    for entry in entries:
        pred = predict_from_operators(model, config, entry, diffusion_net)
        counts += confusion_matrix(entry["labels"].numpy(), pred, n_class)
    measured = scores(counts)

    history = json.loads(args.history.read_text(encoding="utf-8"))
    recorded = next(h for h in history if h["epoch"] == checkpoint["epoch"])
    recorded_iou = {int(c): v for c, v in recorded["test_per_class_iou"].items() if v is not None}
    scores_agree = (
        abs(measured["mean_iou"] - recorded["test_iou"]) <= SCORE_TOLERANCE
        and abs(measured["accuracy"] - recorded["test_acc"]) <= SCORE_TOLERANCE
        and measured["per_class_iou"].keys() == recorded_iou.keys()
        and all(abs(measured["per_class_iou"][c] - v) <= SCORE_TOLERANCE for c, v in recorded_iou.items())
    )
    mesh_agrees = (
        mesh_check["faces_identical"]
        and mesh_check["labels_identical"]
        and max(mesh_check["max_relative_difference"].values()) <= OPERATOR_TOLERANCE
        and mesh_check["faces_predicted_differently"] == 0
    )

    record = {
        "kind": "benchmark_context",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_of(args.checkpoint),
        "epochs_completed": int(checkpoint["epoch"]) + 1,
        "test_split": {
            "meshes": len(entries),
            "cache_file": str(args.cache),
            "cache_sha256": sha256_of(args.cache),
            "split_file": str(args.split),
            "history_file": str(args.history),
        },
        "measured": {
            "scored_faces": measured["scored_faces"],
            "mean_iou": measured["mean_iou"],
            "accuracy": measured["accuracy"],
            "per_class_iou": {class_names[c]: v for c, v in measured["per_class_iou"].items()},
        },
        "recorded_in_history": {
            "mean_iou": recorded["test_iou"],
            "accuracy": recorded["test_acc"],
            "per_class_iou": {class_names[c]: v for c, v in recorded_iou.items()},
        },
        "tolerances": {"operators_relative": OPERATOR_TOLERANCE, "scores": SCORE_TOLERANCE},
        "mesh_check": mesh_check,
        "agrees": bool(scores_agree and mesh_agrees),
    }
    out_dir = args.out / args.checkpoint.parent.name / args.checkpoint.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "verify.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    differences = mesh_check.get("max_relative_difference", {})
    print(f"mesh {args.stem}: faces identical {mesh_check['faces_identical']}, "
          f"labels identical {mesh_check['labels_identical']}, "
          f"largest relative operator difference {max(differences.values(), default=float('nan')):.1e}, "
          f"faces predicted differently {mesh_check.get('faces_predicted_differently')}")
    print(f"test split ({len(entries)} meshes): mIoU {measured['mean_iou']:.6f} "
          f"(history.json {recorded['test_iou']:.6f}), accuracy {measured['accuracy']:.6f} "
          f"(history.json {recorded['test_acc']:.6f})")
    print(f"{'agrees' if record['agrees'] else 'DOES NOT AGREE'}; wrote {out_dir / 'verify.json'}")
    return 0 if record["agrees"] else 1


if __name__ == "__main__":
    sys.exit(main())
