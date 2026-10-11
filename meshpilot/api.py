"""MeshPilot viewer: python -m uvicorn meshpilot.api:app --port 8001

    GET /                    the viewer page
    GET /predictions         the predictions found in the outputs folder
    GET /predictions/{id}    one prediction with its mesh, labels and provenance
    POST /chat               {"question": "...", "prediction_id": "..."}  ->  answer, document
                             citations and model-prediction results, kept apart
    GET /health

It serves what `python -m meshpilot.predict` and `python -m meshpilot.verify`
wrote, and runs no model itself, so it needs neither torch nor the
checkpoints. It is an app of its own, apart from rag.api: the document RAG
still starts without any MeshPilot file, and the viewer without the RAG index
or an API key. The chat needs an API key; it searches documents only when the
RAG index is built.
"""

import json
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

import anthropic
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from meshpilot.agent import run_chat
from meshpilot.corpus import INDEX_DIR, load_search
from meshpilot.outputs import OUTPUTS_DIR, PALETTE, UNLABELED_COLOR, read_labels, read_obj
from meshpilot.perplexity import DEFAULT_MODEL as PERPLEXITY_DEFAULT_MODEL
from meshpilot.perplexity import PerplexityClient
from rag import config
from rag.corpus import sha256_of
from rag.index import Hit
from rag.llm import LLMError, make_client

log = logging.getLogger("meshpilot.api")

# API requests one chat question may make, each billed; the last has tools off, so the model must answer.
CHAT_MAX_STEPS = int(os.getenv("MESHPILOT_CHAT_MAX_STEPS", "3"))
VIEWER_PAGE = Path(__file__).parent / "static" / "viewer.html"

# What a class means depends on the dataset the run was trained on.
DATASET_NOTES = {
    "fusion360_seg": (
        "Fusion 360 Gallery segmentation: each class is the CAD modelling operation that created "
        "the face (extrude, cut, fillet, chamfer or revolve). They are not machining features."
    ),
}

# Demo questions for the chat, per dataset. Each names the class it asks about,
# and only questions about the model's own classes are offered.
PRESET_QUESTIONS = {
    "fusion360_seg": [
        ("Show the faces predicted as Fillet.", "Fillet"),
        ("Where did the model predict Chamfer faces?", "Chamfer"),
        ("Did the model predict any RevolveSide faces on this part?", "RevolveSide"),
    ],
}

LIMITATIONS = [
    "The class colours are model predictions, not ground truth. The dataset's own labels appear only in "
    "the separate ground-truth layer.",
    "Test-split IoU describes the whole test split. It is not a confidence for any face of this mesh.",
    "Scores on this one mesh illustrate this prediction. They are not the model's benchmark performance.",
    "The model was trained without rotation or other augmentation, on meshes converted by the thesis "
    "pipeline. Other poses, tessellations and sizes are untested.",
    "This is a local demo, not a production system.",
]
MOCK_LIMITATIONS = [
    "MOCK data: no model ran. These labels are not model predictions, and nothing here measures the model.",
    "This is a local demo, not a production system.",
]


class PredictionError(Exception):
    """A prediction whose files no longer match what was recorded when it was made."""


@dataclass(frozen=True)
class ChatBackend:
    """What the chat needs: a provider's client, and document search when the RAG index is built."""

    client: object  # anthropic.Anthropic, meshpilot.perplexity.PerplexityClient, or a scripted fake
    model: str
    search: Callable[[str], list[Hit]] | None  # None without a document index
    titles: dict[str, str]
    # "claude" or "perplexity": real calls, billed per question. "mock": a scripted fake
    # client, no API call; the viewer labels its replies MOCK.
    provider: str = "claude"

    @property
    def mock(self) -> bool:
        return self.provider == "mock"


def chat_provider() -> str | None:
    """MESHPILOT_CHAT_PROVIDER ("claude" or "perplexity") if set, else the first provider with a key.

    None means the chosen provider has no key: the chat is off and the viewer still works.
    """
    has_key = {
        "claude": bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")),
        "perplexity": bool(os.getenv("PERPLEXITY_API_KEY")),
    }
    chosen = os.getenv("MESHPILOT_CHAT_PROVIDER", "").strip().lower()
    if chosen and chosen not in has_key:
        raise RuntimeError(f"MESHPILOT_CHAT_PROVIDER must be claude or perplexity, not {chosen!r}")
    if chosen:
        return chosen if has_key[chosen] else None
    return next((name for name, present in has_key.items() if present), None)


def load_document_search() -> tuple[Callable[[str], list[Hit]] | None, dict[str, str]]:
    """Search over the MeshPilot demo notes (meshpilot.corpus), on this machine, once they are built.

    The chat searches only these notes; the document RAG's geotechnical index stays with rag.api.
    """
    search, titles = load_search()
    if search is None:
        log.warning("document search is off: no demo-corpus index at %s (python -m meshpilot.corpus build)", INDEX_DIR)
    return search, titles


def load_chat_backend() -> ChatBackend | None:
    """The chosen provider's client, plus local document search if the index exists. None without a key."""
    provider = chat_provider()
    if provider is None:
        log.warning("no API key for the chat provider: the chat is off; the viewer still works")
        return None
    search, titles = load_document_search()
    if provider == "perplexity":
        client = PerplexityClient(os.environ["PERPLEXITY_API_KEY"])
        return ChatBackend(client, os.getenv("PERPLEXITY_MODEL", PERPLEXITY_DEFAULT_MODEL), search, titles, "perplexity")
    return ChatBackend(make_client(), config.ANSWER_MODEL, search, titles, "claude")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    prediction_id: str


def find_predictions(outputs_dir: Path) -> dict[str, Path]:
    """prediction.json files under outputs_dir, keyed by their folder relative to it."""
    return {
        path.parent.relative_to(outputs_dir).as_posix(): path
        for path in sorted(Path(outputs_dir).glob("**/prediction.json"))
    }


def benchmark_context(prediction_path: Path, checkpoint_sha256: str) -> dict | None:
    """verify.json for the same checkpoint, if `meshpilot.verify` has been run on it."""
    path = prediction_path.parent.parent / "verify.json"
    if not path.is_file():
        return None
    verify = json.loads(path.read_text(encoding="utf-8"))
    if verify.get("checkpoint_sha256") != checkpoint_sha256:
        return None
    return {key: verify.get(key) for key in ("test_split", "measured", "recorded_in_history", "mesh_check", "tolerances", "agrees")}


def load_prediction(path: Path) -> dict:
    """A prediction with its mesh and labels, checked against the hashes recorded when it was made."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    mesh = record["mesh"]
    obj = Path(mesh["obj"])
    if sha256_of(obj) != mesh["obj_sha256"]:
        raise PredictionError(f"{obj} has changed since the prediction was made")
    verts, faces = read_obj(obj)
    if len(record["face_labels"]) != len(faces):
        raise PredictionError(f"{len(record['face_labels'])} predicted labels for {len(faces)} faces")

    truth = record.get("ground_truth")
    if truth is not None:
        labels_file = Path(truth["labels_file"])
        if sha256_of(labels_file) != truth["labels_sha256"]:
            raise PredictionError(f"{labels_file} has changed since the prediction was made")
        truth = {
            "labels": read_labels(labels_file).tolist(),
            "face_counts": truth["face_counts"],
            "single_mesh_scores": truth["single_mesh_scores"],
        }

    model = record["model"]
    return {
        "kind": record["kind"],
        "mock": record["mock"],
        "mock_rule": record.get("mock_rule"),  # how a MOCK prediction's labels were made
        "class_names": record["class_names"],
        "class_colors": PALETTE[: model["n_class"]].tolist(),
        "unlabeled_color": UNLABELED_COLOR.tolist(),
        "mesh": {
            **{key: mesh[key] for key in ("stem", "faces", "vertices", "split", "obj_sha256")},
            "positions": verts.ravel().tolist(),
            "triangles": faces.ravel().tolist(),
        },
        "predicted": {"labels": record["face_labels"], "face_counts": record["predicted_face_counts"]},
        "ground_truth": truth,
        "model": model,
        "preprocessing": record["preprocessing"],
        "software": record["software"],
        "benchmark": benchmark_context(Path(path), model["checkpoint_sha256"]),
        "dataset_note": DATASET_NOTES.get(model.get("label_kind")),
        "preset_questions": [question for question, class_name in PRESET_QUESTIONS.get(model.get("label_kind"), [])
                             if class_name in record["class_names"]],
        "limitations": MOCK_LIMITATIONS if record["mock"] else LIMITATIONS,
    }


router = APIRouter()


@router.get("/", include_in_schema=False)
def viewer() -> FileResponse:
    # Revalidated on every load, so the browser never shows an older copy of the page.
    return FileResponse(VIEWER_PAGE, headers={"Cache-Control": "no-cache"})


@router.get("/health")
def health(request: Request) -> dict:
    chat = request.app.state.chat
    return {
        "status": "ok",
        "predictions": len(find_predictions(request.app.state.outputs_dir)),
        "chat": chat is not None,
        "chat_provider": chat.provider if chat else None,  # the viewer labels mock and real chats apart
        "chat_model": chat.model if chat else None,
        "chat_max_requests": CHAT_MAX_STEPS,
        "documents": chat is not None and chat.search is not None,
    }


@router.get("/predictions")
def predictions(request: Request) -> list[dict]:
    listed = []
    for prediction_id, path in find_predictions(request.app.state.outputs_dir).items():
        record = json.loads(path.read_text(encoding="utf-8"))
        listed.append({"id": prediction_id, "stem": record["mesh"]["stem"],
                       "checkpoint": record["model"]["checkpoint"], "mock": record["mock"]})
    return listed


def served_prediction(request: Request, prediction_id: str) -> dict:
    # Only ids found by the scan are served, so a request cannot name another file.
    found = find_predictions(request.app.state.outputs_dir)
    if prediction_id not in found:
        raise HTTPException(status_code=404, detail=f"no prediction {prediction_id!r}")
    try:
        return load_prediction(found[prediction_id])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"a file the prediction needs is missing: {exc}")
    except PredictionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/predictions/{prediction_id:path}")
def prediction(prediction_id: str, request: Request) -> dict:
    return served_prediction(request, prediction_id)


@router.post("/chat")
def chat(body: ChatRequest, request: Request) -> dict:
    backend = request.app.state.chat
    if backend is None:
        raise HTTPException(status_code=503, detail="The chat needs ANTHROPIC_API_KEY or PERPLEXITY_API_KEY in .env. "
                                                    "The viewer works without it.")
    prediction = served_prediction(request, body.prediction_id)
    try:
        result = run_chat(backend.client, backend.model, body.question, prediction, backend.search, backend.titles,
                          max_steps=CHAT_MAX_STEPS)
    except LLMError as exc:
        log.warning("chat failed: %s (%s)", exc.failure, exc)
        raise HTTPException(status_code=502, detail=f"the model returned no usable answer ({exc.failure})")
    except anthropic.APIError as exc:
        log.warning("chat failed: Anthropic API error: %s", exc)
        raise HTTPException(status_code=502, detail="the call to the Anthropic API failed")
    log.info("chat prediction=%s tool_calls=%d citations=%d model=%s seconds=%.1f", body.prediction_id,
             len(result.tool_calls), len(result.citations), result.model, result.latency_s)
    return {
        "answer": result.answer,
        # The two kinds of evidence stay in separate fields, and the viewer shows them apart.
        "document_citations": [asdict(citation) for citation in result.citations],
        "geometry": result.geometry,
        "tool_calls": [{"step": c.step, "tool": c.tool, "input": c.input, "output_summary": c.output_summary}
                       for c in result.tool_calls],
        "documents_available": backend.search is not None,
        "model": result.model,
        "mock": backend.mock,
        "provider": backend.provider,
        "usage": result.usage,
    }


def create_app(outputs_dir: Path = OUTPUTS_DIR, chat: ChatBackend | None = None) -> FastAPI:
    """The tests pass a chat backend built from fakes; otherwise it is loaded at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        if app.state.chat is None:
            app.state.chat = load_chat_backend()
        yield

    app = FastAPI(title="MeshPilot", lifespan=lifespan)
    app.state.outputs_dir = Path(outputs_dir)
    app.state.chat = chat
    app.include_router(router)
    return app


app = create_app()
