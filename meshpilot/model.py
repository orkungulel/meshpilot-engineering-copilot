"""Load a thesis DiffusionNet checkpoint and predict a class for every face.

Each step repeats what the thesis training script (diffusion_net_train.py)
does on its test split: the same mesh reader, the same normalisation, the same
operators and input features, and the same model constructor. The
diffusion_net library is imported from the copy taken from the training
cluster and never from another version. torch and the library are imported
only when a prediction is made, so the rest of the app runs without them.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np

from meshpilot.outputs import read_obj

# Input channels for each input feature, as in diffusion_net_train.py.
INPUT_CHANNELS = {"xyz": 3, "hks": 16}
# What diffusion_net_train.py saves in every checkpoint.
CHECKPOINT_KEYS = {"epoch", "model_state", "args"}


def import_diffusion_net(src_dir: Path):
    """Import diffusion_net from src_dir/diffusion_net, and refuse any other copy."""
    src_dir = Path(src_dir).resolve()
    package = src_dir / "diffusion_net"
    if not (package / "__init__.py").is_file():
        raise FileNotFoundError(f"no diffusion_net package in {src_dir}")
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # leave the copied source folder untouched
    try:
        import diffusion_net
    finally:
        sys.dont_write_bytecode = write_bytecode
    if Path(diffusion_net.__file__).resolve().parent != package:
        raise ImportError(f"diffusion_net was imported from {diffusion_net.__file__}, not from {package}")
    return diffusion_net


def source_sha256(src_dir: Path) -> str:
    """One hash over the library's .py files, recording which source made a prediction."""
    digest = hashlib.sha256()
    for path in sorted((Path(src_dir) / "diffusion_net").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    """Load a checkpoint with weights_only=True: tensors and plain containers, never arbitrary objects."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    missing = CHECKPOINT_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"{path} is not a DiffusionNet training checkpoint: it has no {sorted(missing)}")
    input_channels(checkpoint)  # a config that does not fit the weights fails here, not later
    return checkpoint


def input_channels(checkpoint: dict) -> int:
    """The input channels named by the run config, checked against the stored weights.

    This is what stops XYZ input being fed to HKS weights, or the reverse.
    """
    features = checkpoint["args"]["input_features"]
    if features not in INPUT_CHANNELS:
        raise ValueError(f"unknown input features {features!r}")
    stored = checkpoint["model_state"]["first_lin.weight"].shape[1]
    if stored != INPUT_CHANNELS[features]:
        raise ValueError(
            f"the run config says {features!r} ({INPUT_CHANNELS[features]} channels) "
            f"but the stored weights take {stored}"
        )
    return stored


def build_model(checkpoint: dict, diffusion_net):
    """The network as diffusion_net_train.py builds it, with the checkpoint's weights."""
    import torch

    config = checkpoint["args"]
    model = diffusion_net.layers.DiffusionNet(
        C_in=input_channels(checkpoint),
        C_out=int(config["effective_n_class"]),
        C_width=int(config["C_width"]),
        N_block=int(config["N_block"]),
        last_activation=lambda x: torch.nn.functional.log_softmax(x, dim=-1),
        outputs_at="faces",
        dropout=bool(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state"])  # strict: every weight must fit
    model.eval()
    return model


def read_mesh(obj_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces read with potpourri3d, as in training, with the file's face order checked."""
    import potpourri3d as pp3d

    verts, faces = pp3d.read_mesh(str(obj_path))
    if not np.array_equal(faces, read_obj(obj_path)[1]):
        raise ValueError(f"{obj_path}: the mesh reader changed the order or number of faces")
    return verts, faces


def compute_operators(verts: np.ndarray, faces: np.ndarray, k_eig: int, diffusion_net) -> dict:
    """Normalised positions and operators, built as the training script's dataset builds them."""
    import torch

    v = torch.tensor(np.ascontiguousarray(verts)).float()
    f = torch.tensor(np.ascontiguousarray(faces)).long()
    v = diffusion_net.geometry.normalize_positions(v)
    # No op_cache_dir: the library's operator cache loads .npz files with
    # allow_pickle=True, so the operators are always computed afresh.
    frames, mass, L, evals, evecs, gradX, gradY = diffusion_net.geometry.get_operators(
        v, f, k_eig=k_eig, op_cache_dir=None
    )
    return {"verts": v, "faces": f, "frames": frames, "mass": mass, "L": L,
            "evals": evals, "evecs": evecs, "gradX": gradX, "gradY": gradY}


def predict_from_operators(model, config: dict, ops: dict, diffusion_net) -> np.ndarray:
    """One class per face from prepared operators, as the training script's test pass computes it."""
    import torch

    if config["input_features"] == "xyz":
        features = ops["verts"]
    else:
        features = diffusion_net.geometry.compute_hks_autoscale(ops["evals"], ops["evecs"], INPUT_CHANNELS["hks"])
    with torch.no_grad():
        log_probs = model(features, ops["mass"], L=ops["L"], evals=ops["evals"], evecs=ops["evecs"],
                          gradX=ops["gradX"], gradY=ops["gradY"], faces=ops["faces"])
    return torch.max(log_probs, dim=1).indices.numpy()


def predict_faces(model, checkpoint: dict, verts: np.ndarray, faces: np.ndarray, diffusion_net) -> np.ndarray:
    """One class per face of a mesh, computed as the training script computes it on its test split."""
    config = checkpoint["args"]
    ops = compute_operators(verts, faces, int(config["k_eig"]), diffusion_net)
    return predict_from_operators(model, config, ops, diffusion_net)
